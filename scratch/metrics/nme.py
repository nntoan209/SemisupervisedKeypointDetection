import numpy as np
import torch

class NME:
    def __init__(self, normalize_item: str='bbox_size'):
        self.normalize_item = normalize_item
        
    def __call__(self, batch_keypoints: np.ndarray, items: dict):
        """
        Arguments:
            batch_keypoints: shape [B, 1, K, D]
                B: batch_size
                K: number of keypoints
                D: dimension of each keypoints (2)
            items: contains the information of the items in the batch
        Returns:
            results: array of shape [B, ]
        """
        batch_size, _, num_keypoints, _ = batch_keypoints.shape
        gt_batch_keypoints = items['keypoints'].detach().cpu().numpy()
        normalize_factor = get_normalize_factor(items=items,
                                                normalize_item=self.normalize_item)
        displacement = batch_keypoints - gt_batch_keypoints
        distances = np.linalg.norm(displacement, axis=-1).reshape(batch_size, num_keypoints)
        results = np.sum(distances, axis=1) / normalize_factor
        
        return torch.sum(results)
    
def get_normalize_factor(items, normalize_item):
    """
    returns: array of shape [B, ] or a scalar if all items have the same normalize factor
    """
    if normalize_item == None:
        return 1
    if normalize_item == 'bbox_size':
        batch_size = items['bbox'].shape[0]
        bbox_shapes = torch.abs(items['bbox'][:, :, :2] - items['bbox'][:, :, 2:]).reshape(batch_size, 2)
        return torch.prod(bbox_shapes, dim=1)
    
    if normalize_item == 'image_size':
        image_sizes = torch.stack((items['img_shape'][0], items['img_shape'][1]), dim=1) 
        return torch.prod(image_sizes, dim=1)
    
    if normalize_item == 'input_size':
        # all items must have the same size
        input_size_0 = items['input_size'][0][0].item()
        input_size_1 = items['input_size'][1][0].item()
        return input_size_0 * input_size_1 
    