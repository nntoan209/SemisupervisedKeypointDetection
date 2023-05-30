import cv2
import torch
import numpy as np
from scipy.stats import truncnorm
from torchvision.transforms import ColorJitter
from configs.config import get_config
config = get_config()


class LoadImage():
    def __call__(self, item: dict) -> dict:
        item['img'] = cv2.imread(item['img_path'])
        item['img_shape'] = list(item['img'].shape[:2])
        item['ori_shape'] = list(item['img'].shape[:2])
        
        return item
    
    
class BBoxTransform():
    def __init__(self,
                 scale_padding=1.25):
        self.scale_padding = scale_padding
        
    def __call__(self, item: dict) -> dict:
        x, y, w, h = item['bbox']
        img_h, img_w = item['img_shape']
        x1 = np.clip(x, 0, img_w - 1)
        y1 = np.clip(y, 0, img_h - 1)
        x2 = np.clip(x + w, 0, img_w - 1)
        y2 = np.clip(y + h, 0, img_h - 1)
        item['bbox'] = np.array([x1, y1, x2, y2], dtype=np.float32).reshape(1, 4)
        
        # center scale
        x1, y1, x2, y2 = np.hsplit(item['bbox'], [1, 2, 3])
        center = np.hstack([x1 + x2, y1 + y2]) * 0.5
        scale = np.hstack([x2 - x1, y2 - y1]) * self.scale_padding
        item['center'] = center
        item['scale'] = scale
        
        return item
    

class RandomFlip():
    def __init__(self,
                 prob,
                 type='labeled'):
        self.prob = prob
        self.type = type

    def _flip_img(self,
                  img: np.ndarray):
        return np.flip(img, axis=1)
    
    def _flip_bbox(self,
                   bbox: np.ndarray,
                   center: np.ndarray,
                   image_size: tuple):
        bbox_flipped = bbox.copy()
        center_flipped = center.copy()
        w, _ = image_size
        
        bbox_flipped[..., ::2] = w - bbox[..., ::2] - 1
        center_flipped[..., 0] = w - center[..., 0] - 1
        
        return bbox_flipped, center_flipped
    
    def _flip_keypoints(self,
                        keypoints: np.ndarray,
                        keypoints_visible: np.ndarray,
                        image_size: tuple,
                        flip_indices: list
                        ):
        # swap the symmetric keypoint pairs
        keypoints = keypoints[..., flip_indices, :]
        keypoints_visible = keypoints_visible[..., flip_indices]
        
        # flip the keypoints
        w, h = image_size
        keypoints[..., 0] = w - 1 - keypoints[..., 0]
        
        return keypoints, keypoints_visible
        
    def __call__(self, item: dict) -> dict:
        self.flip = False
        if np.random.rand() < self.prob:
            self.flip = True
        item['flip'] = self.flip
        if item['flip']:
            h, w = item['img_shape']
            # flip image
            item['img'] = self._flip_img(item['img'])
            
            # flip bbox
            item['bbox'], item['center'] = self._flip_bbox(item['bbox'], item['center'], (w, h))
            
            # flip keypoints
            if self.type == 'labeled':
                item['keypoints'], item['keypoints_visible'] = self._flip_keypoints(item['keypoints'],
                                                                                    item['keypoints_visible'],
                                                                                    image_size=(w, h),
                                                                                    flip_indices=config.flip_indices)
            
        return item


class RandomBboxTransform():
    def __init__(self,
                 scale_factor: tuple = (0.8, 1.2),
                 scale_prob: float = 0.7,
                 rotate_factor: float = 45,
                 rotate_prob: float = 0.6):
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.rotate_factor = rotate_factor
        self.rotate_prob = rotate_prob
        
    def _get_transform_params(self):
        # get scaling params
        scale_min, scale_max = self.scale_factor
        mu = (scale_max + scale_min) * 0.5
        sigma = (scale_max - scale_min) * 0.5
        scale = truncnorm.rvs(-1., 1., size=(1, 1)) * sigma + mu
        scale = np.where(
            np.random.rand(1, 1) < self.scale_prob, scale, 1.)
        
        # get rotation parameters
        rotate = truncnorm.rvs(-1., 1., size=(1, )) * self.rotate_factor
        rotate = np.where(
            np.random.rand(1) < self.rotate_prob, rotate, 0.)
        
        return scale, rotate
    
    def __call__(self, item: dict) -> dict:
        scale, rotate = self._get_transform_params()
        item['scale'] *= scale
        item['rotation'] = rotate
        
        return item
    

class TopDownAffine():
    def __init__(self,
                 input_size: tuple,
                 type='labeled'):
        self.input_size = input_size
        self.type = type
    
    @staticmethod
    def _fix_aspect_ratio(bbox_scale: np.ndarray, aspect_ratio: float):
        w, h = np.hsplit(bbox_scale, [1])
        bbox_scale = np.where(w > h * aspect_ratio,
                              np.hstack([w, w / aspect_ratio]),
                              np.hstack([h * aspect_ratio, h]))
        return bbox_scale
    
    @staticmethod
    def _rotate_point(pt: np.ndarray, angle_rad: float):
        """
        Rotate a point by an angle
        """
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        rot_mat = np.array([[cs, -sn], [sn, cs]])
        return rot_mat @ pt
    
    @staticmethod
    def _get_3rd_point(a: np.ndarray, b:np.ndarray):
        """To calculate the affine matrix, three pairs of points are required. This
        function is used to get the 3rd point, given 2D points a & b.
        """
        direction = a - b
        c = b + np.r_[-direction[1], direction[0]]
        return c
    
    def _get_warp_matrix(self,
                        center: np.ndarray,
                        scale: np.ndarray,
                        rot: float,
                        output_size: tuple,
                        inv: bool = False
                        ):
        
        src_w = scale[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.deg2rad(rot)
        src_dir = self._rotate_point(np.array([0., src_w * -0.5]), rot_rad)
        dst_dir = np.array([0., dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        src[2, :] = self._get_3rd_point(src[0, :], src[1, :])

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        dst[2, :] = self._get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return warp_mat        
   
    def __call__(self, item: dict) -> dict:
        w, h = self.input_size
        warp_size = (int(w), int(h))
        item['input_size'] = [w, h]
        
        # reshape bbox to fixed aspect ratio
        item['scale'] = self._fix_aspect_ratio(item['scale'], aspect_ratio=w / h)
        
        # get center, scale and rotation
        center = item['center'][0]
        scale = item['scale'][0]
        rot = item.get('rotation', [0.])[0]
        
        # get the warp matrix
        warp_mat = self._get_warp_matrix(center, scale, rot, output_size=(w, h))
        
        # transform the image
        item['img'] = cv2.warpAffine(item['img'], warp_mat, warp_size, flags=cv2.INTER_LANCZOS4)
        
        # transform the keypoints
        if self.type == 'labeled':
            transformed_keypoints = item['keypoints'].copy()
            transformed_keypoints[..., :2] = cv2.transform(item['keypoints'][..., :2], warp_mat)
            item['transformed_keypoints'] = transformed_keypoints
        
        return item
    

class GenerateTarget():
    def __init__(self,
                 codec):
        self.codec = codec
        
    def __call__(self, item: dict):
        heatmap, keypoint_weights = self.codec.encode(item['transformed_keypoints'],
                                                      item['keypoints_visible'])
        
        keypoint_weights *= np.array(config.dataset_keypoint_weights, dtype=np.float32)
        item['heatmap'] = heatmap
        item['keypoint_weights'] = keypoint_weights
        
        return item
    
class ImageToTensor():
    def __call__(self, item: dict):
        
        if len(item['img'].shape) < 3:
            item['img'] = np.expand_dims(item['img'], -1)

        img = np.ascontiguousarray(item['img'])
        tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        # convert the channels from BGR to RGB
        tensor = tensor[[2, 1, 0], ...]
        
        item['img'] = tensor
        
        return item
    
class ColorJittering():
    def __init__(self,
                 brightness=0.1,
                 contrast=0.05,
                 saturation=0.05):
        self.colorjitter = ColorJitter(brightness=brightness,
                                       contrast=contrast,
                                       saturation=saturation)
    
    def __call__(self, item: dict):
        item['img'] = self.colorjitter(item['img'])
        
        return item
        
        
class Normalize():
    def __init__(self,
                 mean: list,
                 std: list):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def __call__(self, item: dict):
        item['img'] = (item['img'] - self.mean) / self.std
        
        return item
    