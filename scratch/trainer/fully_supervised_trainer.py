import os
import torch
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel
from tqdm import tqdm

from dataloader.aflw import get_test_loader, get_train_loader
from losses.loss import AdaptiveWingLoss, KeypointMSELoss
from models.model import PoseModel
from optimizer.optimizer import build_optimizer
from optimizer.lr_scheduler import LinearWarmupCosineAnnealingLR
from metrics.nme import NME
from utils import AverageMeter

class FullySupervisedTrainer:
    def __init__(self, config):
        self.config = config
        
        self.device = torch.device(self.config.device)
        
        # dataloader
        self.train_loader = get_train_loader(batch_size=self.config.labeled_batch_size, type='labeled', drop_last=False)
        self.len_loader = len(self.train_loader)
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
            
        # evaluation metric
        self.evaluator = NME(normalize_item=config.normalize_item)
            
        # model
        self.model = PoseModel(backbone_type=self.config.backbone_type,
                               backbone_cfg=self.config.backbone_cfg,
                               neck_type=self.config.neck_type,
                               neck_cfg=self.config.neck_cfg,
                               head_type=self.config.head_type,
                               head_cfg=self.config.head_cfg,
                               codec_type=self.config.codec_type,
                               codec_cfg=self.config.codec_cfg,
                               test_cfg=self.config.test_cfg)
        self.model = DataParallel(self.model).to(self.device)
        
        # optimizer
        self.optimizer = build_optimizer(optimizer_cfg=self.config.optimizer_cfg,
                                         model=self.model)
        
        self.lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=self.optimizer,
                                                          warmup_epochs=self.config.warmup_epoch * self.len_loader,
                                                          max_epochs=self.config.joint_epoch * self.len_loader,
                                                          warmup_start_lr_factor=self.config.start_factor,
                                                          eta_min=1e-6)
        self.current_epoch = 0

    def _eval_epoch(self, epoch):
        self.model.eval()
        pbar = tqdm(self.test_loader, total=len(self.test_loader),
                    desc=f'Eval epoch {epoch+1}',
                    ncols=0)
        loss_meter = AverageMeter()
        nme_meter= AverageMeter()

        for batch in pbar:
            
            batch_image = batch['img'].to(self.device)
            batch_heatmap = batch['heatmap'].to(self.device)
            batch_target_weights = batch['keypoints_weight'].to(self.device)
            
            num_item = batch_image.shape[0]
            
            with torch.no_grad():
                heatmap_pred = self.model(batch_image)
                keypoints_pred, keypoints_score= self.model.module.predict(items=batch, cuda=True)

                # loss for model
                loss = self.supervised_criterion(heatmap_pred, batch_heatmap,
                                                 batch_target_weights)
                
                loss_meter.update(val=loss.item(),
                                  weight=num_item)
                
                # nme for model
                nme = self.evaluator(keypoints_pred, batch)
                
                nme_meter.update(val=nme.item(),
                                 weight=num_item)

                pbar.set_postfix({
                    'supervised loss': [round(loss_meter.average(), 5)],
                    'nme': [round(nme_meter.average(), 5)]
                })
                
        result = {'test/supervised loss': round(loss_meter.average(), 5),
                  'test/nme': round(nme_meter.average(), 5),
                  }
        3
        return result

    def _train_joint_epoch(self, epoch):
        print()
        self.model.train()
        
        pbar = tqdm(enumerate(range(self.len_loader)), total=self.len_loader,
                    desc=f'Training epoch {epoch + 1}/{self.config.joint_epoch}',
                    ncols=0)
        
        supervised_dataloader = iter(self.train_loader)
        
        self.current_epoch = epoch

        supervised_loss_meter = AverageMeter()
        
        for i, _ in pbar:
            self.optimizer.zero_grad()
            
            batch = next(supervised_dataloader)
            
            batch_image = batch['img'].to(self.device)
            
            batch_heatmap = batch['heatmap'].to(self.device)   
            
            num_item = batch_image.shape[0]
            
            # Labeled images
            # forward through model
            batch_heatmap_pred = self.model(batch_image)
        
            supervised_loss = self.supervised_criterion(batch_heatmap_pred,
                                                        batch_heatmap,
                                                        batch['keypoints_weight'].to(self.device))
            supervised_loss_meter.update(val=supervised_loss.item(),
                                         weight=num_item)
            
            loss_total = supervised_loss
            
            # back propagation
            loss_total.backward()
            clip_grad_norm_(self.model.parameters(), 4)

            self.optimizer.step()
            
            # update the learning rate
            self.lr_scheduler.step()

            pbar.set_postfix({
                'supervised loss': round(supervised_loss_meter.average(), 5),
                'top lr': self.lr_scheduler.get_last_lr()[-1],
                'bottom lr': self.lr_scheduler.get_last_lr()[0]
            })

    def save_checkpoint(self, epoch, dir='checkpoint_last.pt', type='latest'):
        checkpoint_path = os.path.join(self.config.snapshot_dir, dir)
        checkpoint = {'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict(),
                      'epoch': self.current_epoch,
                      }
        torch.save(checkpoint, checkpoint_path)
        print(f"-----> save {type} checkpoint at epoch {epoch+1}")

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config.snapshot_dir, 'checkpoint_last.pt')
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.current_epoch = checkpoint['epoch']
        print("----> load checkpoint")
        
    def load_checkpoint_from_pt(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.current_epoch = checkpoint['epoch']
        print("----> load checkpoint")

    def train(self, resume=False):        
        if resume:
            self.load_checkpoint()
            
        best_nme = 1e9
            
        for epoch in range(self.current_epoch, self.config.labeled_epoch + self.config.joint_epoch):  
            self._train_joint_epoch(epoch)
                
            result = self._eval_epoch(epoch)
            current_nme = result['test/nme']
            if current_nme <= best_nme:
                best_nme = current_nme
                self.save_checkpoint(epoch=epoch, dir='checkpoint_best.pt', type='best')
                
            self.save_checkpoint(epoch=epoch, dir='checkpoint_last.pt', type='latest')
            