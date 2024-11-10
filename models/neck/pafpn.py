#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from ..layers.conv import BaseConv, DWConv
from ..blocks.csp import CSPLayer

# from yolox pafpn
## modificatoin 
## remove width
class PAFPN(nn.Module):
    """
    YOLOv3 model without backbone. Input features are directly passed to the neck layers.
    """

    def __init__(
        self,
        depth=1.0,
        in_stages = ("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.in_features = in_stages
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            in_channels[2], in_channels[1], 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            2 * in_channels[1],
            in_channels[1],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            in_channels[1], in_channels[0], 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            2 * in_channels[0],
            in_channels[0],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            in_channels[0], in_channels[0], 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            2 * in_channels[0],
            in_channels[1],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            in_channels[1], in_channels[1], 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            2 * in_channels[1],
            in_channels[2],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, backbone_features):
        """
        Args:
            x2, x1, x0: Feature maps from external backbone.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        features = [backbone_features[f] for f in self.in_features]
        [x2, x1, x0] = features
        
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32s
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16
        
        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8
        
        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16
        
        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        
        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
