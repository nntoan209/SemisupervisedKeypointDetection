import os
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel
from torch.optim import AdamW
from tqdm import tqdm

from dataloader.aflw import get_test_loader, get_train_loader
from losses.loss import AdaptiveWingLoss, KeypointMSELoss
from network.mean_teacher_network import MeanTeacherNetwork
from models.model import PoseModel
from utils import LinearWarmupCosineAnnealingLR
from metrics.nme import NME
from utils import AverageMeter, ema_decay_scheduler, consistency_loss_weight_scheduler
from codec.utils import flip_heatmaps, rotate_image

class EMATrainer:
    def __init__(self, config):
        self.config = config
        
        self.device = torch.device(self.config.device)
        
        # dataloader
        self.labeled_train_loader_1 = get_train_loader(batch_size=self.config.labeled_batch_size, type='labeled')
        self.labeled_train_loader_2 = get_train_loader(batch_size=self.config.labeled_batch_size, type='labeled')
        
        self.unlabeled_train_loader_1 = get_train_loader(batch_size=self.config.unlabeled_batch_size, type='unlabeled')
        self.unlabeled_train_loader_2 = get_train_loader(batch_size=self.config.unlabeled_batch_size, type='unlabeled')
        
        self.len_loader = max(len(self.labeled_train_loader_1), len(self.unlabeled_train_loader_1))
        self.test_loader = get_test_loader(batch_size=self.config.test_batch_size)
        
        # loss functions
        if self.config.supervised_loss == 'awing':
            self.supervised_criterion = AdaptiveWingLoss(alpha=self.config.alpha,
                                                        omega=self.config.omega,
                                                        epsilon=self.config.epsilon,
                                                        theta=self.config.theta,
                                                        use_target_weight=self.config.use_target_weight)
        elif self.config.supervised_loss == 'mse':
            self.supervised_criterion = KeypointMSELoss()
            
        if self.config.consistency_loss == 'mse':
            self.consistency_criterion = KeypointMSELoss()
            
        # evaluation metric
        self.evaluator = NME(normalize_item=config.normalize_item)
            
        # model network
        self.network = MeanTeacherNetwork(student_model=PoseModel(backbone_type=self.config.backbone_type,
                                                                  backbone_cfg=self.config.backbone_cfg,
                                                                  neck_type=self.config.neck_type,
                                                                  neck_cfg=self.config.neck_cfg,
                                                                  head_type=self.config.head_type,
                                                                  head_cfg=self.config.head_cfg,
                                                                  codec_type=self.config.codec_type,
                                                                  codec_cfg=self.config.codec_cfg,
                                                                  test_cfg=self.config.test_cfg),
                                          teacher_model=PoseModel(backbone_type=self.config.backbone_type,
                                                                  backbone_cfg=self.config.backbone_cfg,
                                                                  neck_type=self.config.neck_type,
                                                                  neck_cfg=self.config.neck_cfg,
                                                                  head_type=self.config.head_type,
                                                                  head_cfg=self.config.head_cfg,
                                                                  codec_type=self.config.codec_type,
                                                                  codec_cfg=self.config.codec_cfg,
                                                                  test_cfg=self.config.test_cfg))
        self.network = DataParallel(self.network).to(self.device)
        
        # optimizer
        self.optimizer = AdamW(params=self.network.module.student_model.parameters(),
                               lr=self.config.lr,
                               weight_decay=self.config.weight_decay)
        self.lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=self.optimizer,
                                                          warmup_epochs=self.config.warmup_epoch * self.len_loader,
                                                          max_epochs=self.config.joint_epoch * self.len_loader,
                                                          warmup_start_lr=5e-6,
                                                          eta_min=1e-5)
        self.current_epoch = 0

    def _eval_epoch(self, epoch):
        self.network.eval()
        pbar = tqdm(self.test_loader, total=len(self.test_loader),
                    desc=f'Eval epoch {epoch+1}',
                    ncols=0)
        loss_meter_student = AverageMeter()
        loss_meter_teacher = AverageMeter()

        nme_meter_student = AverageMeter()
        nme_meter_teacher = AverageMeter()

        for batch in pbar:
            
            batch_image = batch['img'].to(self.device)
            batch_heatmap = batch['heatmap'].to(self.device)
            batch_target_weights = batch['keypoints_weight'].to(self.device)
            
            num_item = batch_image.shape[0]
            
            with torch.no_grad():
                t_heatmap_pred, s_heatmap_pred = self.network(batch_image)
                t_keypoints_pred, _, s_keypoints_pred, _ = self.network.module.predict(items=batch, cuda=True)

                # loss for model
                loss_student = self.supervised_criterion(s_heatmap_pred, batch_heatmap,
                                                         batch_target_weights)
                loss_teacher = self.supervised_criterion(t_heatmap_pred, batch_heatmap,
                                                         batch_target_weights)
                
                loss_meter_student.update(val=loss_student.item(),
                                          weight=num_item)
                loss_meter_teacher.update(val=loss_teacher.item(),
                                          weight=num_item)
                
                # nme for model
                nme_student = self.evaluator(s_keypoints_pred, batch)
                nme_teacher = self.evaluator(t_keypoints_pred, batch)
                
                nme_meter_student.update(val=nme_student.item(),
                                         weight=num_item)
                nme_meter_teacher.update(val=nme_teacher.item(),
                                         weight=num_item)

                pbar.set_postfix({
                    'supervised loss student/teacher': [round(loss_meter_student.average(), 5),
                                                        round(loss_meter_teacher.average(), 5)],
                    'nme student/teacher': [round(nme_meter_student.average(), 5),
                                            round(nme_meter_teacher.average(), 5)],
                })
                
        result = {'test/supervised loss student': round(loss_meter_student.average(), 5),
                  'test/nme student': round(nme_meter_student.average(), 5),
                  'test/supervised loss teacher': round(loss_meter_teacher.average(), 5),
                  'test/nme teacher': round(nme_meter_teacher.average(), 5),
                  }
        
        return result

    def _train_joint_epoch(self, epoch):
        print()
        self.network.train()
        
        pbar = tqdm(enumerate(range(self.len_loader)), total=self.len_loader,
                    desc=f'Training epoch {epoch + 1}/{self.config.joint_epoch}',
                    ncols=0)
        
        supervised_dataloader_1 = iter(self.labeled_train_loader_1)
        supervised_dataloader_2 = iter(self.labeled_train_loader_2)
        
        unsupervised_dataloader_1 = iter(self.unlabeled_train_loader_1)
        unsupervised_dataloader_2 = iter(self.unlabeled_train_loader_2)
        
        self.current_epoch = epoch

        supervised_loss_meter = AverageMeter()
        labeled_consistency_loss_meter = AverageMeter()
        unlabeled_consistency_loss_meter = AverageMeter()
        
        for i, _ in pbar:
            self.optimizer.zero_grad()
            
            labeled_batch_1 = next(supervised_dataloader_1)
            labeled_batch_2 = next(supervised_dataloader_2)
            
            unlabeled_batch_1 = next(unsupervised_dataloader_1)
            unlabeled_batch_2 = next(unsupervised_dataloader_2)
            
            num_labeled_item = self.config.labeled_batch_size
            num_unlabeled_item = self.config.unlabeled_batch_size
            
            labeled_batch_image_1 = labeled_batch_1['img'].to(self.device)
            labeled_batch_image_2 = labeled_batch_2['img'].to(self.device)
            
            labeled_batch_heatmap_1 = labeled_batch_1['heatmap'].to(self.device)

            unlabeled_batch_image_1 = unlabeled_batch_1['img'].to(self.device)     
            unlabeled_batch_image_2 = unlabeled_batch_2['img'].to(self.device)   
            
            # Unlabeled images
            # forward through 2 models
            unlabeled_batch_heatmap_pred_1 = self.network.module.student_model(unlabeled_batch_image_1)
            unlabeled_batch_heatmap_pred_2 = self.network.module.teacher_model(unlabeled_batch_image_2)
            
            # rotate and flip the heatmaps to recover the original heatmap
            recovered_unlabeled_batch_heatmap_pred_1 = unlabeled_batch_heatmap_pred_1.clone()
            # iter through the batch size
            for idx in range(num_unlabeled_item): 
                # get the predicted heatmap
                heatmap = recovered_unlabeled_batch_heatmap_pred_1[idx]
                # rotate the heatmap
                heatmap = rotate_image(image=heatmap,
                                       angle=-unlabeled_batch_1['rotation'][idx].item())
                # flip the heatmap
                if unlabeled_batch_1['flip'][idx]:
                    heatmap = flip_heatmaps(heatmaps=torch.unsqueeze(heatmap, 0),
                                            flip_indices=self.config.flip_indices,
                                            shift_heatmap=True).squeeze()
                recovered_unlabeled_batch_heatmap_pred_1[idx] = heatmap
                
            recovered_unlabeled_batch_heatmap_pred_2 = unlabeled_batch_heatmap_pred_2.clone()
            # iter through the batch size
            for idx in range(num_unlabeled_item):
                # get the predicted heatmap
                heatmap = recovered_unlabeled_batch_heatmap_pred_2[idx]
                # rotate the heatmap
                heatmap = rotate_image(image=heatmap,
                                       angle=-unlabeled_batch_2['rotation'][idx].item())
                # flip the heatmap
                if unlabeled_batch_2['flip'][idx]:
                    heatmap = flip_heatmaps(heatmaps=torch.unsqueeze(heatmap, 0),
                                            flip_indices=self.config.flip_indices,
                                            shift_heatmap=True).squeeze()
                recovered_unlabeled_batch_heatmap_pred_2[idx] = heatmap
                
            unlabeled_consistency_loss = self.consistency_criterion(recovered_unlabeled_batch_heatmap_pred_1,
                                                                    recovered_unlabeled_batch_heatmap_pred_2)
            unlabeled_consistency_loss_meter.update(val=unlabeled_consistency_loss.item(),
                                                    weight=num_unlabeled_item)
            
            # Labeled images
            # forward through 2 models 
            labeled_batch_heatmap_pred_1 = self.network.module.student_model(labeled_batch_image_1)
            labeled_batch_heatmap_pred_2 = self.network.module.teacher_model(labeled_batch_image_2)
            
            # rotate and flip the heatmaps to recover the original heatmap
            recovered_labeled_batch_heatmap_pred_1 = labeled_batch_heatmap_pred_1.clone()
            # iter through the batch size
            for idx in range(num_labeled_item):
                heatmap = recovered_labeled_batch_heatmap_pred_1[idx]
                # rotate the heatmap
                heatmap = rotate_image(image=heatmap,
                                       angle=-labeled_batch_1['rotation'][idx].item())
                # flip the heatmap
                if labeled_batch_1['flip'][idx]:
                    heatmap = flip_heatmaps(heatmaps=torch.unsqueeze(heatmap, 0),
                                            flip_indices=self.config.flip_indices,
                                            shift_heatmap=True).squeeze()
                recovered_labeled_batch_heatmap_pred_1[idx] = heatmap
                
            recovered_labeled_batch_heatmap_pred_2 = labeled_batch_heatmap_pred_2.clone()
            # iter through the batch size
            for idx in range(num_labeled_item):
                heatmap = recovered_labeled_batch_heatmap_pred_2[idx]
                # rotate the heatmap
                heatmap = rotate_image(image=heatmap,
                                       angle=-labeled_batch_2['rotation'][idx].item())
                # flip the heatmap
                if labeled_batch_2['flip'][idx]:
                    heatmap = flip_heatmaps(heatmaps=torch.unsqueeze(heatmap, 0),
                                            flip_indices=self.config.flip_indices,
                                            shift_heatmap=True).squeeze()
                recovered_labeled_batch_heatmap_pred_2[idx] = heatmap
            
            labeled_consistency_loss = self.consistency_criterion(recovered_labeled_batch_heatmap_pred_1,
                                                                  recovered_labeled_batch_heatmap_pred_2)
            labeled_consistency_loss_meter.update(val=labeled_consistency_loss.item(),
                                                  weight=num_labeled_item)
        
            supervised_loss = self.supervised_criterion(labeled_batch_heatmap_pred_1,
                                                        labeled_batch_heatmap_1,
                                                        labeled_batch_1['keypoints_weight'].to(self.device))
            supervised_loss_meter.update(val=supervised_loss.item(),
                                         weight=num_labeled_item)
            
            consistency_loss = unlabeled_consistency_loss + labeled_consistency_loss
            
            consistency_loss_weight = consistency_loss_weight_scheduler(final_value=self.config.final_consistency_loss_weight,
                                                                        max_step=self.config.consistency_loss_weight_ramp_up_epoch * self.len_loader,
                                                                        step=epoch * self.len_loader + i)
            loss_total = supervised_loss + consistency_loss * consistency_loss_weight
            
            # back propagation
            loss_total.backward()
            clip_grad_norm_(self.network.module.student_model.parameters(), 5)

            self.optimizer.step()
            
            ema_decay = ema_decay_scheduler(self.config.start_ema_decay,
                                            self.config.end_ema_decay,
                                            max_step=self.config.ema_ramp_up_epoch * self.len_loader,
                                            step=epoch * self.len_loader + i)
            
            # update weights for teacher
            self.network.module._update_teacher_ema(ema_decay)  
            
            # update the learning rate
            self.lr_scheduler.step()
            lr = self.lr_scheduler.get_last_lr()[0]
            for i in range(len(self.optimizer.param_groups)):
                self.optimizer.param_groups[i]['lr'] = lr

            pbar.set_postfix({
                'supervised loss': round(supervised_loss_meter.average(), 5),
                'label consistency loss': round(labeled_consistency_loss_meter.average(), 5),
                'unlabel consistency loss': round(unlabeled_consistency_loss_meter.average(),5),
                'lr': self.lr_scheduler.get_last_lr()[0]
            })

    def save_checkpoint(self, epoch, dir='checkpoint_last.pt', type='latest'):
        checkpoint_path = os.path.join(self.config.snapshot_dir, dir)
        checkpoint = {'state_dict': self.network.module.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict(),
                      'epoch': self.current_epoch,
                      }
        torch.save(checkpoint, checkpoint_path)
        print(f"-----> save {type} checkpoint at epoch {epoch+1}")

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config.snapshot_dir, 'checkpoint_last.pt')
        checkpoint = torch.load(checkpoint_path)
        self.network.module.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.current_epoch = checkpoint['epoch']
        print("----> load checkpoint")
        
    def load_checkpoint_from_pt(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.network.module.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.current_epoch = checkpoint['epoch']
        print("----> load checkpoint")

    def train(self, resume=False):        
        if resume:
            self.load_checkpoint()
            
        best_nme_teacher = 1e9
            
        for epoch in range(self.current_epoch, self.config.labeled_epoch + self.config.joint_epoch):  
            self._train_joint_epoch(epoch)
                
            result = self._eval_epoch(epoch)
            nme_teacher = result['test/nme teacher']
            if nme_teacher <= best_nme_teacher:
                best_nme_teacher = nme_teacher
                self.save_checkpoint(epoch=epoch, dir='checkpoint_best.pt', type='best')
                
            self.save_checkpoint(epoch=epoch, dir='checkpoint_last.pt', type='latest')
            