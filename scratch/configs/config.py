from easydict import EasyDict as edict
import os
import json

def get_config(new=True):
    C = edict()
    config = C

    """ PATH CONFIG """
    C.images_path = r'data/custom_aflw/images'
    C.test_annotations_path = r'data/custom_aflw/annotations/face_landmarks_custom_aflw_test.json'
    C.labeled_train_annotations_path = r'data/custom_aflw/annotations/face_landmarks_custom_aflw_labeled_train.json'
    C.unlabeled_train_annotations_path = r'data/custom_aflw/annotations/face_landmarks_custom_aflw_unlabeled_train.json'
    
    C.backbone_pretrained = "vitpose-b-multi-coco.pth"
    
    C.snapshot_dir = r'./ema_log'
        
    """ DATA CONFIG """
    C.input_size = [256, 192]
    C.heatmap_size = (64, 48)
    
    C.labeled_batch_size = 2
    C.unlabeled_batch_size = 2
    C.test_batch_size = 2
    C.num_workers = 2
    
    C.flip_prob = 0.5
    C.rotate_prob = 0.7
    C.rotate_factor = 45
    
    C.flip_indices = [1, 0, 2, 3, 4]
    
    """ MODEL CONFIG """
    C.device = "cuda"
    
    C.codec_type = "MSRAHeatmap"
    C.codec_cfg = {"input_size":C.input_size,
                   "heatmap_size":C.heatmap_size}

    C.backbone_type = "ViT"
    C.backbone_cfg ={"img_size":(256, 192),
                   "patch_size":16,
                   "embed_dim":768,
                   "depth":12,
                   "num_heads":12,
                   "ratio":1,
                   "use_checkpoint":False,
                   "mlp_ratio":4,
                   "qkv_bias":True,
                   "drop_path_rate":0.3,
                   "pretrained": C.backbone_pretrained}
    
    C.neck_type = None
    C.neck_cfg= None
    
    C.head_type = "ViPNASHead"
    C.head_cfg = {"in_channels":768,
                "out_channels":5,
                "deconv_out_channels":(256, 256),
                "deconv_num_groups":(256, 256),
                "deconv_kernel_sizes":(3, 3),
                "conv_kernel_sizes":3}
    
    C.test_cfg = {'flip_test': True,
                  'shift_heatmap': True}
            
    """ LOSS CONFIG """
    C.supervised_loss = 'awing'
    C.alpha = 2.1
    C.omega = 14
    C.epsilon = 1
    C.theta = 0.5
    
    C.use_target_weight = False
    C.dataset_keypoint_weights = [1., 1., 1., 1., 1.]
    
    C.consistency_loss = 'mse'
    
    C.consistency_loss_weight = 1
    
    """ METRICS CONFIG """
    C.normalize_item = 'bbox_size'
    
    """ OPTIMIZER CONFIG """
    C.lr = 2e-3
    C.weight_decay = 1e-4
    
    C.start_ema_decay = 0.5
    C.end_ema_decay = 0.99
    C.ema_linear_epoch = 10
    
    """ TRAIN CONFIG """
    C.joint_epoch = 50
    C.labeled_epoch = 0
    
    C.seed = 420
    
    if new:
        if not os.path.isdir(C.snapshot_dir):
            os.makedirs(C.snapshot_dir)
        with open(os.path.join(C.snapshot_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    
    return config
