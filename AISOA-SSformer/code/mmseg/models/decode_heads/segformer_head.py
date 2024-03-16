# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import math
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr
from mmseg.models.backbones.attention import *
from IPython import embed

import torch.nn as nn

import torch
class TAPI():
    def __init__(self, decay, dynamic_weight=True, n=20):
        self.decay = decay
        self.dynamic_weight = dynamic_weight
        self.prev_params = None
        self.n = n
        self.param_history = None
        self.history_index = 0
        
    def compute_dynamic_weight(self, old, new):
        if self.prev_params is not None:
            # 计算参数的变化率
            change = new - self.prev_params
            # 使用变化率的绝对值作为权重，也可以考虑其他形式的调整
            weight = torch.abs(change)
            # 对权重进行归一化
            weight /= weight.sum()
            return weight
        else:
            # 第一次调用时，返回均匀权重
            return torch.ones_like(new) / len(new)

    def update_average(self, old, new):
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        if self.param_history is None:
            # 初始化参数历史记录
            self.param_history = [[torch.zeros_like(p.data) for _ in range(self.n)]
                                  for p in current_model.parameters()]

        for i, (current_params, ema_params) in enumerate(zip(current_model.parameters(), ema_model.parameters())):
            # 更新参数历史记录
            self.param_history[i].pop(0)
            self.param_history[i].append(current_params.data.clone())

            # 计算加权平均
            weights = torch.linspace(1, self.n, steps=self.n)
            weights /= weights.sum()  # 对权重进行归一化
            weighted_avg_param = sum([w * p for w, p in zip(weights, self.param_history[i])])

            # 使用加权平均值更新EMA参数
            ema_params.data = self.update_average(ema_params.data, weighted_avg_param) 
            
class SparseLinear(nn.Module):
    def __init__(self, input_dim, output_dim, sparsity):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity

        # Create a dense weight matrix
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        # Initialize weights and biases
        self.reset_parameters()

        # Apply sparsity
        self.apply_sparsity()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def apply_sparsity(self):
        with torch.no_grad():
            mask = (torch.rand(self.weight.size()) > self.sparsity).float()
            self.weight *= mask

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class SGUP(nn.Module):
    """
    SGUP with Sparse Linear Layers
    """
    def __init__(self, input_dim=2048, embed_dim=768, sparsity=0.5,ema_decay=0.99):
        super().__init__()
        self.proj = SparseLinear(input_dim, embed_dim, sparsity)
        self.ema = TAPI(decay=ema_decay)
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        self.ema.update_model_average(self, self)
        
        return x

@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = SGUP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = SGUP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = SGUP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = SGUP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )
        
        self.ca = SFAM(embedding_dim,embedding_dim)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.ca(x)
        x = self.linear_pred(x)

        return x
