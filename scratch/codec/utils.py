import torch
import cv2

def flip_heatmaps(heatmaps: torch.Tensor,
                  flip_indices: list = [1, 0, 2, 3, 4],
                  shift_heatmap: bool = False):
    
    heatmaps = heatmaps.flip(-1)
    if flip_indices is not None:
        assert len(flip_indices) == heatmaps.shape[1]
        heatmaps = heatmaps[:, flip_indices]
        
    if shift_heatmap:
        heatmaps[..., 1:] = heatmaps[..., :-1].clone()
    
    return heatmaps

def rotate_image(image, angle):
    """
    image: tensor shape [C, W, H]
    angle: float
    """
    clone_image = image.clone()
    
    # Get the height and width of the image
    channels, height, width = clone_image.shape[:3]

    # Calculate the center point of rotation
    center_x = width / 2
    center_y = height / 2

    # Create the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    
    # Apply the rotation to the image using affine transformation
    rotated_image = torch.zeros_like(clone_image)
    for channel in range(channels):
        rotated_image[channel] = torch.Tensor(cv2.warpAffine(clone_image[channel].detach().cpu().numpy(), rotation_matrix, (width, height), flags=cv2.INTER_LANCZOS4))

    # Return the rotated image
    return rotated_image

    