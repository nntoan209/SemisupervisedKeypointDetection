from torch.utils.data import Dataset, DataLoader
from utils import InfiniteDataLoader
from dataloader.transforms import *
from codec.msra_heatmap import MSRAHeatmap
from configs.config import get_config
config = get_config()
CODECS = {"MSRAHeatmap": MSRAHeatmap}
codec = CODECS[config.codec_type](**config.codec_cfg)

import os
import json
import numpy as np

class AFLWDataClass(Dataset):
    def __init__(self,
                 images_path,
                 annotations_path,
                 transforms=None,
                 type='labeled'):
        super(AFLWDataClass, self).__init__()
        self.images_path = images_path
        with open(annotations_path, "r") as file:
            annotations_file = json.load(file)
            self.images = annotations_file['images']
            self.annotations = annotations_file['annotations']
        self.transforms = transforms
        self.type = type
        
    def __getitem__(self, index):
        image = self.images[index]
        annotation = self.annotations[index]
        
        item = dict()
        # initialize the item with the available information
        item['img_path'] = os.path.join(self.images_path, image['file_name'])
        if self.type == 'labeled':
            _keypoints = np.array(annotation['keypoints'], dtype=np.float32).reshape(1, -1, 3)
            item['keypoints'] = _keypoints[..., :2]
            item['keypoints_visible'] = np.minimum(1, _keypoints[..., 2])
            
        item['bbox'] = annotation['bbox']
        
        # pass the item through the pipeline
        for transform in self.transforms:
            item = transform(item)
            
        return item
        
    def __len__(self):
        return len(self.images)
    
def get_train_loader(type: str='labeled'):
    if type == 'labeled':        
        transforms = [LoadImage(),
                      BBoxTransform(),
                      RandomFlip(prob=config.flip_prob,
                                 type='labeled'),
                      TopDownAffine(input_size=config.input_size,
                                    type='labeled'),
                      RandomBboxTransform(scale_prob=0,
                                          rotate_prob=config.rotate_prob,
                                          rotate_factor=config.rotate_factor),
                      GenerateTarget(codec=codec),
                      ImageToTensor(),
                      ColorJittering(),
                      Normalize(mean=[123.675, 116.28, 103.53],
                                    std=[58.395, 57.12, 57.375])]
    
        dataset = AFLWDataClass(images_path=config.images_path,
                                annotations_path=config.labeled_train_annotations_path,
                                transforms=transforms,
                                type='labeled')
        batch_size = config.labeled_batch_size
    elif type == 'unlabeled':
        transforms = [LoadImage(),
                      BBoxTransform(),
                      RandomFlip(prob=config.flip_prob,
                                 type='unlabeled'),
                      TopDownAffine(input_size=config.input_size,
                                    type='unlabeled'),
                      RandomBboxTransform(scale_prob=0,
                                          rotate_prob=config.rotate_prob,
                                          rotate_factor=config.rotate_factor),
                      ImageToTensor(),
                      ColorJittering(),
                      Normalize(mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375])]
    
        dataset = AFLWDataClass(images_path=config.images_path,
                                annotations_path=config.unlabeled_train_annotations_path,
                                transforms=transforms,
                                type='unlabeled')
        batch_size = config.unlabeled_batch_size
        
    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=batch_size,
                                    num_workers=config.num_workers,
                                    drop_last=True,
                                    shuffle=False,
                                    pin_memory=True
                                    )
    return dataloader

def get_test_loader():
    transforms = [LoadImage(),
                  BBoxTransform(),
                  TopDownAffine(input_size=config.input_size,
                                type='labeled'),
                  GenerateTarget(codec=codec),
                  ImageToTensor(),
                  Normalize(mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375])]
    
    dataset = AFLWDataClass(images_path=config.images_path,
                            annotations_path=config.test_annotations_path,
                            transforms=transforms,
                            type='labeled')
    
    dataloader = DataLoader(dataset,
                            batch_size=config.test_batch_size,
                            num_workers=config.num_workers,
                            drop_last=True,
                            shuffle=False,
                            pin_memory=True
                            )
    return dataloader
