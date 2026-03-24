# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from typing import Sequence

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, ModuleList, Sequential

from mmseg.registry import MODELS
from mmpretrain.models.utils.norm import GRN, build_norm_layer
from .base_backbone import BaseBackbone
class ConvNeXtBlock(BaseModule):

    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, padding=3),
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)

        self.linear_pw_conv = linear_pw_conv
        self.norm = build_norm_layer(norm_cfg, in_channels)

        mid_channels = int(mlp_ratio * in_channels)
        if self.linear_pw_conv:
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)

        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = MODELS.build(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        if use_grn:
            self.grn = GRN(mid_channels)
        else:
            self.grn = None

        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        def _inner_forward(x):
            shortcut = x
            x = self.depthwise_conv(x)

            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_last')
                x = self.pointwise_conv2(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                x = self.pointwise_conv1(x)
                x = self.act(x)
                if self.grn is not None:
                    x = self.grn(x, data_format='channel_first')
                x = self.pointwise_conv2(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x



class SpectralBranch(nn.Module):

    def __init__(self, spec_channels=9):
        super().__init__()
        self.spec_channels = spec_channels 
        init_out_len = self.spec_channels 
        

        self.init_conv = nn.Conv1d(1, 14, kernel_size=3, stride=1, padding=1)
        self.init_bn = nn.BatchNorm1d(14)
        self.init_relu = nn.ReLU(inplace=True)
        
        
        self.depthwise_convs = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(5):
            self.depthwise_convs.append(
                nn.Conv1d(14, 14, kernel_size=3, stride=1, padding=1, groups=14)
            )
            self.pointwise_convs.append(
                nn.Conv1d(14, 14, kernel_size=1, stride=1, padding=0)
            )
            self.bns.append(nn.BatchNorm1d(14))
        
        
        self.final_conv = nn.Conv1d(84, 84, kernel_size=init_out_len, stride=1, padding=0) 
        self.final_bn = nn.BatchNorm1d(84) 
        
        
        self.channel_compress = nn.Conv2d(84, 3, kernel_size=1, stride=1, padding=1)

    def forward(self, x):

        BS, C_spec, H, W = x.shape
               
        x = x.permute(0, 2, 3, 1).contiguous() 
        x = x.view(BS, -1, C_spec) 
        x = x.unsqueeze(2)  
        x = x.reshape(-1, 1, C_spec)
        
        x1 = self.init_conv(x) 
        x1 = self.init_bn(x1)
        x1 = self.init_relu(x1)
        
        features = [x1]
        current = x1
        for i in range(5):
            dw = self.depthwise_convs[i](current)
            pw = self.pointwise_convs[i](dw)
            bn = self.bns[i](pw)
            current = nn.functional.relu(bn + current)
            features.append(current)
        
        concat_x = torch.cat(features, dim=1) 
        

        out = self.final_conv(concat_x) 
        out = self.final_bn(out)
        out = nn.functional.relu(out)

        out = out.squeeze(2)
        out = out.view(BS, H, W, 84) 
        out = out.permute(0, 3, 1, 2).contiguous()

        out = self.channel_compress(out)  # (BS, 3, H, W)
        
        return out


@MODELS.register_module()
class ConvNeXtSpectral(BaseBackbone):

    arch_settings = {
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        }
    }

    def __init__(self,
                 arch='base',
                 in_channels=3, 
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=-1,
                 frozen_stages=0,
                 gap_before_final_norm=False,  
                 with_cp=False,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm', 'BatchNorm2d'],
                         val=1., bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)


        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from {set(self.arch_settings)}'
            arch = self.arch_settings[arch]
        self.depths = arch['depths']
        self.channels = arch['channels']
        self.num_stages = len(self.depths)  

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must be sequence or int'
        for i, idx in enumerate(out_indices):
            if idx < 0:
                out_indices[i] = self.num_stages + idx
                assert out_indices[i] >= 0, f'Invalid out_indices {idx}'
        self.out_indices = out_indices

        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm


        self.spectral_branch = SpectralBranch()
        

        self.stem = nn.Sequential(
            nn.Conv2d(

                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_norm_layer(norm_cfg, self.channels[0]),
        )
        

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        block_idx = 0


        self.downsample_layers = ModuleList()
        self.stages = nn.ModuleList()


        self.downsample_layers.append(self.stem)


        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:

                downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)


            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth

            self.stages.append(stage)


            if i in self.out_indices:
                norm_layer = build_norm_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)

        self._freeze_stages()

    def forward(self, x):

        if isinstance(x,list):
            x=x[0]

        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.dim() == 4:
            squeeze_batch = False
        else:
            raise ValueError(f'Expected input of 3 or 4 dimensions, but got {x.dim()}')

            

        spectral_feat = self.spectral_branch(x)  # (BS, 3, 512, 512)
        

        x = spectral_feat
        outs = []
        for i, stage in enumerate(self.stages):
            x = self.downsample_layers[i](x)
            x = stage(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = x.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(x))

        return tuple(outs)

    def _freeze_stages(self):

        if self.frozen_stages <= 0:
            return
        
        if self.frozen_stages >= 1:
            self.spectral_branch.eval()
            for param in self.spectral_branch.parameters():
                param.requires_grad = False
        
        for i in range(min(self.frozen_stages, self.num_stages + 1)):  # +1 for stem
            if i < len(self.downsample_layers):
                downsample_layer = self.downsample_layers[i]
                downsample_layer.eval()
                for param in downsample_layer.parameters():
                    param.requires_grad = False
            
            if i < len(self.stages):
                stage = self.stages[i]
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):

        if not param_name.startswith(prefix):
            return 12, 14 
            
        param_name = param_name[len(prefix):]
        
        if param_name.startswith('spectral_branch'):
            layer_id = 0
            max_layer_id = 13
        elif param_name.startswith('stem'):
            layer_id = 1
            max_layer_id = 13
        elif param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            layer_id = 2 + stage_id * 3
            max_layer_id = 13
        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2]) if len(param_name.split('.')) > 2 else 0
            if stage_id == 0:
                layer_id = 3 + block_id
            elif stage_id == 1:
                layer_id = 6 + block_id
            elif stage_id == 2:
                layer_id = 9 + block_id
            else:  # stage_id == 3
                layer_id = 12 + block_id
            max_layer_id = 15
        else:  # norm层
            layer_id = 15
            max_layer_id = 16
            
        return layer_id, max_layer_id
