import os
import torch
import torch.nn as nn


BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, dilation=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.mid_channels = out_channels // self.expansion
        self.conv1 = conv3x3(in_channels, self.mid_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(self.mid_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.mid_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.mid_channels = out_channels // self.expansion
        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.mid_channels, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.mid_channels, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(self.mid_channels, out_channels, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HRModule(nn.Module):
    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 upsample_cfg,
                 multi_scale_output=True):
        super(HRModule, self).__init__()
        self._check_branches(
            num_branches, num_blocks, in_channels, num_channels)

        self.in_channels = in_channels
        self.upsample_cfg = upsample_cfg
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, num_blocks,
                        in_channels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = 'NUM_BRANCHES({}) <> in_channels({})'.format(
                num_branches, len(in_channels))
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.in_channels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.in_channels[branch_index],
                            num_channels[branch_index] * block.expansion,
                            stride,
                            downsample=downsample))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.in_channels[branch_index],
                                num_channels[branch_index] * block.expansion))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(in_channels[j],
                                  in_channels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(in_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i),
                                    mode=self.upsample_cfg['mode'],
                                    align_corners=self.upsample_cfg['align_corners'])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = in_channels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(in_channels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = in_channels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(in_channels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRNet(nn.Module):

    def __init__(self,
                 extra,
                 in_channels=3,
                 frozen_stages=-1):
        
        super(HRNet, self).__init__()
        self.pretrained_layers = ['*']
        self.extra = extra
        self.frozen_stages = frozen_stages
        self.upsample_cfg = self.extra.get('upsample', {
            'mode': 'bilinear',
            'align_corners': False
        })

        # stem net
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        
        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, stage1_out_channels,
                                       num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = blocks_dict[block_type]
        num_channels = [
            channel * block.expansion for channel in num_channels
        ]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = blocks_dict[block_type]
        num_channels = [
            channel * block.expansion for channel in num_channels
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = blocks_dict[block_type]
        num_channels = [
            channel * block.expansion for channel in num_channels
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multi_scale_output=self.stage4_cfg.get('multiscale_output', False))

        self._freeze_stages()

    def _make_transition_layer(self,
                               num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, in_channels,
                    multi_scale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = blocks_dict[layer_config['block']]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    self.upsample_cfg,
                    reset_multi_scale_output)
            )
            in_channels = modules[-1].in_channels

        return nn.Sequential(*modules), in_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return tuple(y_list)
    
    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.bn1.eval()
            self.bn2.eval()

            for m in [self.conv1, self.bn1, self.conv2, self.bn2]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if i == 1:
                m = getattr(self, 'layer1')
            else:
                m = getattr(self, f'stage{i}')

            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i < 4:
                m = getattr(self, f'transition{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None, verbose=False):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)
            
        if not (pretrained is None) and not (os.path.isfile(pretrained)):
            raise FileNotFoundError('pretrained file is specified but can not find the file')

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for _name, m in pretrained_state_dict['state_dict'].items():
                if _name.startswith('backbone'):
                    name = '.'.join(_name.split('.')[1:])
                    if name.split('.')[0] in self.pretrained_layers \
                    or self.pretrained_layers[0] == '*':
                        if name in parameters_names or name in buffers_names:
                            if verbose:
                                print(
                                    '=> init {} from {}'.format(name, pretrained)
                                )
                            need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
