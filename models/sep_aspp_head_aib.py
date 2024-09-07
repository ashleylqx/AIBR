# Copyright (c) OpenMMLab. All rights reserved.
import math
from numbers import Number
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.aspp_head import ASPPHead, ASPPModule


class DepthwiseSeparableASPPModule(ASPPModule):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

    
@MODELS.register_module()
class DepthwiseSeparableASPPHeadAIB_V4(ASPPHead):
    """Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation.

    This head is the implementation of `DeepLabV3+
    <https://arxiv.org/abs/1802.02611>`_.

    Args:
        c1_in_channels (int): The input channels of c1 decoder. If is 0,
            the no decoder will be used.
        c1_channels (int): The intermediate channels of c1 decoder.
    """

    def __init__(self, c1_in_channels, c1_channels, **kwargs):
        super().__init__(**kwargs)
        assert c1_in_channels >= 0
        self.aspp_modules = DepthwiseSeparableASPPModule(
            dilations=self.dilations,
            in_channels=self.in_channels,
            channels=self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        if c1_in_channels > 0:
            self.c1_bottleneck = ConvModule(
                c1_in_channels,
                c1_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.c1_bottleneck = None
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs, num_sample=1):
        """Forward function."""
        if num_sample>1:
            inputs = [inputs[i].view(num_sample, -1, inputs[i].size(1), inputs[i].size(2), inputs[i].size(3))
                      for i in range(len(inputs))]

            outs = []
            for idx in range(num_sample):
                inputs_sample = [inputs[i][idx]
                                 for i in range(len(inputs))]
                current_out = self.forward_per_sample(inputs_sample)
                outs.append(current_out)
            return torch.mean(torch.stack(outs), dim=0)

        else:
            return self.forward_per_sample(inputs)       # only return out

    def forward_per_sample(self, inputs):
        x = self._transform_inputs(inputs)
        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)
        output = self.bottleneck(aspp_outs)
        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output

    # forward_train
    def loss(self, inputs, batch_data_samples, train_cfg, num_sample=1):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs, num_sample)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    # forward_test
    def predict(self, inputs, batch_img_metas, test_cfg, num_sample=1):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        seg_logits = self.forward(inputs, num_sample)
        return self.predict_by_feat(seg_logits, batch_img_metas)

