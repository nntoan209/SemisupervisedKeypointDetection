import numpy as np
import cv2
from itertools import product

class MSRAHeatmap():
    def __init__(self,
                 input_size: tuple,
                 heatmap_size: tuple,
                 sigma: float = 2,
                 blur_kernel_size: int = 11):
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.blur_kernel_size = blur_kernel_size
        self.scale_factor = (np.array(input_size) / heatmap_size).astype(np.float32)
    
    @staticmethod
    def _generate_unbiased_gaussian_heatmaps(
        heatmap_size: tuple,
        keypoints: np.ndarray,
        keypoints_visible: np.ndarray,
        sigma: float):

        N, K, _ = keypoints.shape
        W, H = heatmap_size

        heatmaps = np.zeros((K, H, W), dtype=np.float32)
        keypoint_weights = keypoints_visible.copy()

        # 3-sigma rule
        radius = sigma * 3

        # xy grid
        x = np.arange(0, W, 1, dtype=np.float32)
        y = np.arange(0, H, 1, dtype=np.float32)[:, None]

        for n, k in product(range(N), range(K)):
            # skip unlabled keypoints
            if keypoints_visible[n, k] < 0.5:
                continue

            mu = keypoints[n, k]
            # check that the gaussian has in-bounds part
            left, top = mu - radius
            right, bottom = mu + radius + 1

            if left >= W or top >= H or right < 0 or bottom < 0:
                keypoint_weights[n, k] = 0
                continue

            gaussian = np.exp(-((x - mu[0])**2 + (y - mu[1])**2) / (2 * sigma**2))

            _ = np.maximum(gaussian, heatmaps[k], out=heatmaps[k])

        return heatmaps, keypoint_weights
    
    def encode(self,
               keypoints: np.ndarray,
               keypoints_visible: np.ndarray):
        heatmaps, keypoint_weights = self._generate_unbiased_gaussian_heatmaps(heatmap_size=self.heatmap_size,
                                                                               keypoints=keypoints / self.scale_factor,
                                                                               keypoints_visible=keypoints_visible,
                                                                               sigma=self.sigma)
        
        return heatmaps, keypoint_weights
    
    @staticmethod
    def _get_heatmap_maximum(heatmaps: np.ndarray):
        if heatmaps.ndim == 3:
            K, H, W = heatmaps.shape
            B = None
            heatmaps_flatten = heatmaps.reshape(K, -1)
        else:
            B, K, H, W = heatmaps.shape
            heatmaps_flatten = heatmaps.reshape(B * K, -1)

        y_locs, x_locs = np.unravel_index(
            np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
        locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
        vals = np.amax(heatmaps_flatten, axis=1)
        locs[vals <= 0.] = -1

        if B:
            locs = locs.reshape(B, K, 2)
            vals = vals.reshape(B, K)

        return locs, vals
    
    @staticmethod
    def _gaussian_blur(heatmaps,
                       blur_kernel_size):
        border = (blur_kernel_size - 1) // 2
        K, H, W = heatmaps.shape

        for k in range(K):
            origin_max = np.max(heatmaps[k])
            dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[k].copy()
            dr = cv2.GaussianBlur(dr, (blur_kernel_size, blur_kernel_size), 0)
            heatmaps[k] = dr[border:-border, border:-border].copy()
            heatmaps[k] *= origin_max / np.max(heatmaps[k])
        return heatmaps
    
    def _refine_keypoints_dark(self,
                               keypoints: np.ndarray,
                               heatmaps: np.ndarray,
                               blur_kernel_size: int):
        N, K = keypoints.shape[:2]
        H, W = heatmaps.shape[1:]

        # modulate heatmaps
        heatmaps = self._gaussian_blur(heatmaps, blur_kernel_size)
        np.maximum(heatmaps, 1e-10, heatmaps)
        np.log(heatmaps, heatmaps)

        for n, k in product(range(N), range(K)):
            x, y = keypoints[n, k, :2].astype(int)
            if 1 < x < W - 2 and 1 < y < H - 2:
                dx = 0.5 * (heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1])
                dy = 0.5 * (heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x])

                dxx = 0.25 * (
                    heatmaps[k, y, x + 2] - 2 * heatmaps[k, y, x] +
                    heatmaps[k, y, x - 2])
                dxy = 0.25 * (
                    heatmaps[k, y + 1, x + 1] - heatmaps[k, y - 1, x + 1] -
                    heatmaps[k, y + 1, x - 1] + heatmaps[k, y - 1, x - 1])
                dyy = 0.25 * (
                    heatmaps[k, y + 2, x] - 2 * heatmaps[k, y, x] +
                    heatmaps[k, y - 2, x])
                derivative = np.array([[dx], [dy]])
                hessian = np.array([[dxx, dxy], [dxy, dyy]])
                if dxx * dyy - dxy**2 != 0:
                    hessianinv = np.linalg.inv(hessian)
                    offset = -hessianinv @ derivative
                    offset = np.squeeze(np.array(offset.T), axis=0)
                    keypoints[n, k, :2] += offset
        return keypoints
    
    def decode(self, encoded_heatmap: np.ndarray):
        heatmaps = encoded_heatmap.copy()
        K, H, W = heatmaps.shape

        keypoints, scores = self._get_heatmap_maximum(heatmaps)

        # Unsqueeze the instance dimension for single-instance results
        keypoints, scores = keypoints[None], scores[None]

        keypoints = self._refine_keypoints_dark(keypoints,
                                                heatmaps,
                                                blur_kernel_size=self.blur_kernel_size)

        # Restore the keypoint scale
        keypoints = keypoints * self.scale_factor

        return keypoints, scores