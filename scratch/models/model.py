import torch
import torch.nn as nn
from codec.utils import flip_heatmaps
from metrics.nme import *
from models.backbones.hrnet import HRNet
from models.backbones.vit import ViT
from models.neck import FeatureMapProcessor
from models.heads.heatmap_head import HeatmapHead
from models.heads.vipnas_head import ViPNASHead
from codec.msra_heatmap import MSRAHeatmap

BACKBONES = {"HRNet": HRNet, "ViT": ViT}
NECKS = {"FeatureMapProcessor": FeatureMapProcessor}
HEADS = {"HeatmapHead": HeatmapHead, "ViPNASHead": ViPNASHead}
CODECS = {"MSRAHeatmap": MSRAHeatmap}

class PoseModel(nn.Module):
    def __init__(self,
                 backbone_type,
                 backbone_cfg,
                 neck_type,
                 neck_cfg,
                 head_type,
                 head_cfg,
                 codec_type,
                 codec_cfg,
                 test_cfg: dict):
        super(PoseModel, self).__init__()
        self.backbone = BACKBONES[backbone_type](**backbone_cfg)
        if neck_type == None:
            self.neck = nn.Identity()
        else:
            self.neck = NECKS[neck_type](**neck_cfg)
        self.head = HEADS[head_type](**head_cfg)
        self.codec = CODECS[codec_type](**codec_cfg)
        self.test_cfg = test_cfg
        
    def _extract_features(self, inputs):
        """
        Extract the features by passing through backbone and neck
        """
        x = self.backbone(inputs)
        x = self.neck(x)
        
        return x
    
    def forward(self, inputs):
        """
        Forward the features through the head to get the heatmap
        """
        x = self._extract_features(inputs)
        x = self.head(x)

        return x
    
    def predict(self, items, cuda=True):
        """
        Predict the keypoints by combining the 2 versions of heatmap: the curernt version
        and the flipped version to improve the accuracy
        """
        self.eval()
        with torch.no_grad():
            if self.test_cfg.get('flip_test', False):
                _feats = self._extract_features(items['img'].to(torch.device("cuda:0")) if cuda else items['img'])
                _feats_flip = self._extract_features(items['img'].flip(-1).to(torch.device("cuda:0")) if cuda else items['img'].flip(-1))
                _batch_heatmaps = self.head(_feats)
                _batch_heatmaps_flip = flip_heatmaps(self.head(_feats_flip),
                                                    flip_indices=[1, 0, 2, 3, 4],
                                                    shift_heatmap=self.test_cfg.get('shift_heatmap', False))
                batch_heatmaps = (_batch_heatmaps + _batch_heatmaps_flip) * 0.5
            else:
                _feats = self._extract_features(items['img'].to(torch.device("cuda:0")) if cuda else items['img'])
                batch_heatmaps = self.head(_feats)

        batch_keypoints = []
        batch_scores = []
        input_size = (items['input_size'][0][0].item(), items['input_size'][1][0].item())
        for i, heatmap in enumerate(batch_heatmaps):
            heatmap_np = heatmap.detach().cpu().numpy()
            keypoints, scores = self.codec.decode(heatmap_np)
            # convert the keypoints to the original image space
            keypoints = keypoints / input_size \
                * items['scale'][i].detach().cpu().numpy() \
                + items['center'][i].detach().cpu().numpy() \
                - 0.5 * items['scale'][i].detach().cpu().numpy()
            
            batch_keypoints.append(keypoints)
            batch_scores.append(scores)     
                    
        return (np.stack(batch_keypoints, axis=0), np.stack(batch_scores, axis=0))
    