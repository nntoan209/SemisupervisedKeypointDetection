import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMapProcessor(nn.Module):
    def __init__(
        self,
        scale_factor: float = 1.0,
        apply_relu: bool = False,
        align_corners: bool = False,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.apply_relu = apply_relu
        self.align_corners = align_corners

    def forward(self, inputs):

        if not isinstance(inputs, (tuple, list)):
            sequential_input = False
            inputs = [inputs]
        else:
            sequential_input = True
            inputs = self._concat(inputs)

        if self.apply_relu:
            inputs = [F.relu(x) for x in inputs]

        if self.scale_factor != 1.0:
            inputs = self._rescale(inputs)

        if not sequential_input:
            inputs = inputs[0]

        return inputs
    
    def _resize(self, 
                input,
                size=None,
                scale_factor=None,
                mode='nearest',
                align_corners=None):
        if isinstance(size, torch.Size):
            size = tuple(int(x) for x in size)

        return F.interpolate(input, size, scale_factor, mode, align_corners)

    def _concat(self, inputs):
        size = inputs[0].shape[-2:]
        resized_inputs = [
            self._resize(
                x,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners) for x in inputs
        ]
        return [torch.cat(resized_inputs, dim=1)]

    def _rescale(self, inputs):
        rescaled_inputs = [
            self._resize(
                x,
                scale_factor=self.scale_factor,
                mode='bilinear',
                align_corners=self.align_corners,
            ) for x in inputs
        ]
        return rescaled_inputs
