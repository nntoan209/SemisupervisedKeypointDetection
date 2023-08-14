import torch
import torch.nn as nn
import torch.nn.functional as F

class KeypointMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel

    def forward(self,
                output,
                target,
                target_weights = None,
                mask = None):
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """

        _mask = self._get_mask(target, target_weights, mask)
        if _mask is None:
            loss = F.mse_loss(output, target)
        else:
            _loss = F.mse_loss(output, target, reduction='none')
            loss = (_loss * _mask).mean()

        return loss

    def _get_mask(self,
                  target,
                  target_weights,
                  mask):
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert (mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1
                for d_m, d_t in zip(mask.shape, target.shape))), (
                    f'mask and target have mismatched shapes {mask.shape} v.s.'
                    f'{target.shape}')

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape +
                                        (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any()
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask
    
    
class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.

    Args:
        alpha (float), omega (float), epsilon (float), theta (float)
            are hyper-parameters.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
    """

    def __init__(self,
                 alpha=2.1,
                 omega=14,
                 epsilon=1,
                 theta=0.5,
                 use_target_weight=False,):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.use_target_weight = use_target_weight

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
        """
        H, W = pred.shape[2:4]
        delta = (target - pred).abs()

        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * (torch.pow(
            self.theta / self.epsilon,
            self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega *
            torch.log(1 + 1e-12 +
                      torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C)

        return torch.mean(losses)

    def forward(self,
                output,
                target,
                target_weights = None):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, H, W]): Output heatmaps.
            target (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            target_weights = target_weights.view(target_weights.shape +
                                                 (1, ) * ndim_pad)
            loss = self.criterion(output * target_weights,
                                  target * target_weights)
        else:
            loss = self.criterion(output, target)

        return loss
