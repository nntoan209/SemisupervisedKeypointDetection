from torch.utils.data import Dataset, DataLoader
from utils import InfiniteDataLoader
from dataloader.transforms import *
from codec.msra_heatmap import MSRAHeatmap

CODECS = {"MSRAHeatmap": MSRAHeatmap}

import os
import json
import numpy as np

class AFLWDataClass(Dataset):
    def __init__(self,
                 images_path,
                 annotations_path,
                 dataset_keypoint_weights,
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
        self.dataset_keypoint_weights = dataset_keypoint_weights
        
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
            item['keypoints_weight'] = np.array(self.dataset_keypoint_weights)
            
        item['bbox'] = annotation['bbox']
        
        # pass the item through the pipeline
        for transform in self.transforms:
            item = transform(item)
            
        return item
        
    def __len__(self):
        return len(self.images)
    
def get_train_loader(config, type: str='labeled', drop_last=True):
    if type == 'labeled':        
        codec = CODECS[config.codec_type](**config.codec_cfg)
        
        transforms = [LoadImage(),
                      BBoxTransform(),
                      RandomFlip(prob=config.flip_prob,
                                 type='labeled'),
                      RandomBboxTransform(scale_prob=0,
                                          rotate_prob=config.rotate_prob,
                                          rotate_factor=config.rotate_factor),
                      TopDownAffine(input_size=config.input_size,
                                    type='labeled'),
                      GenerateTarget(codec=codec),
                      ImageToTensor(),
                      ColorJittering(),
                      Normalize(mean=[123.675, 116.28, 103.53],
                                    std=[58.395, 57.12, 57.375])]
    
        dataset = AFLWDataClass(images_path=config.images_path,
                                annotations_path=config.labeled_train_annotations_path,
                                dataset_keypoint_weights=config.dataset_keypoint_weights,
                                transforms=transforms,
                                type='labeled')
    elif type == 'unlabeled':
        transforms = [LoadImage(),
                      BBoxTransform(),
                      RandomFlip(prob=config.flip_prob,
                                 type='unlabeled'),
                      RandomBboxTransform(scale_prob=0,
                                          rotate_prob=config.rotate_prob,
                                          rotate_factor=config.rotate_factor),
                      TopDownAffine(input_size=config.input_size,
                                    type='unlabeled'),
                      ImageToTensor(),
                      ColorJittering(),
                      Normalize(mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375])]
    
        dataset = AFLWDataClass(images_path=config.images_path,
                                annotations_path=config.unlabeled_train_annotations_path,
                                dataset_keypoint_weights=config.dataset_keypoint_weights,
                                transforms=transforms,
                                type='unlabeled')
        
    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=config.labeled_batch_size,
                                    num_workers=config.num_workers,
                                    drop_last=drop_last,
                                    shuffle=False,
                                    pin_memory=True
                                    )
    return dataloader

def get_test_loader(config):
    codec = CODECS[config.codec_type](**config.codec_cfg)
    
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
                            dataset_keypoint_weights=config.dataset_keypoint_weights,
                            transforms=transforms,
                            type='labeled')
    
    dataloader = DataLoader(dataset,
                            batch_size=config.labeled_batch_size,
                            num_workers=config.num_workers,
                            drop_last=False,
                            shuffle=True,
                            pin_memory=True
                            )
    return dataloader
