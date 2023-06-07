import shutil
import os
import sys
sys.dont_write_bytecode = True

from trainer.semi_supervised_trainer import EMATrainer
from trainer.fully_supervised_trainer import FullySupervisedTrainer
from argparse import ArgumentParser
from configs.config import get_config, log_config
import traceback
import torch
import numpy as np

parser = ArgumentParser(description='Mean teacher network for AFLW')
parser.add_argument("--trainer", default='ema', choices=['ema', 'fully_supervised'], help='type of trainer')
parser.add_argument("--resume", action='store_true', default=False, help='resume training from last checkpoint')
parser.add_argument("--epoch", type=int, default=50, help='total number of epochs')
parser.add_argument("--rampup", type=int, default=10, help='number of ramp up epoch for learning rate, [consistency loss weight, ema decay rate]')
parser.add_argument("--batchsize", type=int, default=8, help='batch size')
parser.add_argument("--startemadecay", type=float, default=0.99, help='initial ema decay')
args = parser.parse_args()

if __name__ == "__main__":
    if args.resume:
        cfg = get_config(new=False)
    else:
        cfg = get_config(new=True)
        
    cfg.joint_epoch = args.epoch
    
    cfg.consistency_loss_weight_ramp_up_epoch = args.rampup
    cfg.ema_ramp_up_epoch = args.rampup
    cfg.warmup_epoch = args.rampup
    
    cfg.labeled_batch_size = args.batchsize
    cfg.unlabeled_batch_size = args.batchsize
    cfg.test_batch_size = args.batchsize
    
    cfg.start_ema_decay = args.startemadecay
    
    log_config(cfg)
        
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("----------------------------------------------------------------------------------------------------")
        
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.seed)
    
    if args.trainer == 'ema':
        trainer = EMATrainer(cfg)
    elif args.trainer == 'fully_supervised':
        trainer = FullySupervisedTrainer(cfg)
    
    try:
        trainer.train(args.resume)
    except Exception:
        print(traceback.print_exc())
        shutil.rmtree(cfg.snapshot_dir)
    except KeyboardInterrupt:
        shutil.rmtree(cfg.snapshot_dir)
