from torch import nn

class ViPNASHead(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels: int,
                 deconv_out_channels= (144, 144, 144),
                 deconv_kernel_sizes = (4, 4, 4),
                 deconv_num_groups = (16, 16, 16),
                 conv_out_channels = None,
                 conv_kernel_sizes = None,
                 final_layer: dict = dict(kernel_size=1)):

        super(ViPNASHead, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')
            if deconv_num_groups is None or len(deconv_out_channels) != len(
                    deconv_num_groups):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_num_groups" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_num_groups}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
                layer_groups=deconv_num_groups,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            if conv_kernel_sizes is None or len(conv_out_channels) != len(
                    conv_kernel_sizes):
                raise ValueError(
                    '"conv_out_channels" and "conv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {conv_out_channels} and '
                    f'{conv_kernel_sizes}')

            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes)
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        if final_layer is not None:
            cfg = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1)
            cfg.update(final_layer)
            self.final_layer = nn.Conv2d(**cfg)
        else:
            self.final_layer = nn.Identity()

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels,
                            layer_kernel_sizes,
                            layer_groups) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size, groups in zip(layer_out_channels,
                                                     layer_kernel_sizes,
                                                     layer_groups):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                groups=groups,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(nn.ConvTranspose2d(**cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)
    
    def forward(self, feats):
        if isinstance(feats, list):
            x = feats[-1]
        else:
            x = feats

        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)

        return x
