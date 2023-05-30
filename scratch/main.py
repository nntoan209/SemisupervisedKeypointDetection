import shutil

from trainer.trainer import EMATrainer
from argparse import ArgumentParser
from configs.config import get_config
import traceback
import torch
import numpy as np

parser = ArgumentParser(description='Mean teacher network for AFLW')
parser.add_argument("--resume", action='store_true', default=False, help='resume training from last checkpoint')
args = parser.parse_args()

if __name__ == "__main__":
    if args.resume:
        cfg = get_config(new=False)
    else:
        cfg = get_config(new=True)
    for k, v in cfg.items():
        print(f"{k}: {v}")
        
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(cfg.seed)
    
    trainer = EMATrainer(cfg)
    
    try:
        trainer.train(args.resume)
    except Exception:
        print(traceback.print_exc())
        shutil.rmtree(cfg.snapshot_dir)
    except KeyboardInterrupt:
        shutil.rmtree(cfg.snapshot_dir)
