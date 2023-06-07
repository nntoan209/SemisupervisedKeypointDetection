from easydict import EasyDict as edict
import os
import json

def log_config(cfg):
    if not os.path.isdir(cfg.snapshot_dir):
        os.makedirs(cfg.snapshot_dir)
    with open(os.path.join(cfg.snapshot_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=4)

def get_config(new=True):
    C = edict()
    config = C

    """ PATH CONFIG """
    C.images_path = r'data/custom_aflw/images'
    C.test_annotations_path = r'data/custom_aflw/annotations/face_landmarks_custom_aflw_test.json'
    C.labeled_train_annotations_path = r'data/custom_aflw/annotations/face_landmarks_custom_aflw_labeled_train.json'
    C.unlabeled_train_annotations_path = r'data/custom_aflw/annotations/face_landmarks_custom_aflw_unlabeled_train.json'
    
    C.backbone_pretrained = "hrnetv2_w18_aflw_256x256_dark-219606c0_20210125.pth"
    # C.backbone_pretrained = "vitpose-b-multi-coco.pth"
    
    C.snapshot_dir = r'./ema_log'
        
    """ DATA CONFIG """
    C.input_size = [256, 256]
    C.heatmap_size = (64, 64)
    
    C.labeled_batch_size = 2
    C.unlabeled_batch_size = 2
    C.test_batch_size = 2
    C.num_workers = 2
    
    C.flip_prob = 0.5
    C.rotate_prob = 0.7
    C.rotate_factor = 45
    
    C.flip_indices = [1, 0, 2, 3, 4]
    
    """ MODEL CONFIG """
    C.device = "cuda:0"
    
    C.codec_type = "MSRAHeatmap"
    C.codec_cfg = {"input_size":C.input_size,
                   "heatmap_size":C.heatmap_size}

    # # config for ViT backbone
    # C.backbone_type = "ViT"
    # C.backbone_cfg ={"img_size":(192, 256),
    #                "patch_size":16,
    #                "embed_dim":768,
    #                "depth":12,
    #                "num_heads":12,
    #                "ratio":1,
    #                "use_checkpoint":False,
    #                "mlp_ratio":4,
    #                "qkv_bias":True,
    #                "drop_path_rate":0.3,
    #                "pretrained": C.backbone_pretrained}
    # C.neck_type = None
    # C.neck_cfg = None
    # C.head_type = "HeatmapHead"
    # C.head_cfg = {"in_channels":768,
    #               "out_channels":5,
    #               "deconv_out_channels":(256, 256),
    #               "deconv_kernel_sizes": (4, 4)}
    
    # config for HRNet backbone
    C.extra={
            "stage1":dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            "stage2":dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            "stage3":dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            "stage4":dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
                multiscale_output=True),
            "upsample":dict(mode='bilinear', align_corners=False),
            }
    C.backbone_type = "HRNet"
    C.backbone_cfg ={"extra": C.extra,
                     "pretrained": C.backbone_pretrained}
    C.neck_type = "FeatureMapProcessor"
    C.neck_cfg = dict()
    C.head_type = "HeatmapHead"
    C.head_cfg = {"in_channels":270,
                  "out_channels":5,
                  "conv_out_channels": (270, ),
                  "conv_kernel_sizes": (1, ),
                  "deconv_out_channels": None
                 }
    
    C.test_cfg = {'flip_test': True,
                  'shift_heatmap': True}
            
    """ LOSS CONFIG """
    C.supervised_loss = 'awing'
    C.alpha = 2.1
    C.omega = 14
    C.epsilon = 1
    C.theta = 0.5
    
    C.use_target_weight = True
    C.dataset_keypoint_weights = [1., 1., 1., 1.25, 1.25]
    
    C.consistency_loss = 'mse'
    C.final_consistency_loss_weight = 3
    C.consistency_loss_weight_ramp_up_epoch = 5
    
    """ METRICS CONFIG """
    C.normalize_item = 'bbox_size'
    
    """ OPTIMIZER CONFIG """
    C.lr = 2e-3
    C.weight_decay = 1e-4
    
    C.start_ema_decay = 0.98
    C.end_ema_decay = 0.999
    C.ema_ramp_up_epoch = 5
    
    """ TRAIN CONFIG """
    C.joint_epoch = 50
    C.labeled_epoch = 0
    
    C.warmup_epoch = 5
    C.seed = 420
    
    # if new:
    #     log_config(C)
    
    return config
