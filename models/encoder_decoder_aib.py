# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import math
import numpy as np
import cv2
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.structures import SegDataSample
from mmengine.structures import PixelData


@MODELS.register_module()
class EncoderDecoderAIB_V4(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     inference(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 num_sample_train: int = 1,
                 num_sample_test: int = 4,
                 va_beta: float = 1e-2):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.num_sample_train = num_sample_train
        self.num_sample_test = num_sample_test
        self.va_beta = va_beta
        # self.output_dir = 'work_dirs/debug/format_results_bdd100k'
        # self.output_dir = 'work_dirs/bdd100k_attention/deeplabv3plus_r101_aib_v4-d8_4xb2-40k_cityscapes-512x1024'
        self.output_dir = 'work_dirs/nightcity-fine_attention/deeplabv3plus_r50_aib_v4-d8_4xb2-80k_cityscapes_nightcity-512x1024'
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)
    
    def extract_feat(self, inputs, num_sample, return_att=True):
        """Extract features from images."""
        current_out = self.backbone(inputs, num_sample, return_att=return_att) # same as aib_v2
        x = current_out[0]
        if self.with_neck:
            x = self.neck(x)
        if return_att:
            att_mask = current_out[1]
            return x, att_mask
        return x, current_out[1], current_out[2]

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, _, _ = self.extract_feat(inputs, self.num_sample_test, return_att=False) # same as aib_v2
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg, num_sample=self.num_sample_test)

        return seg_logits
    
    def encode_decode_returnatt(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, att_mask = self.extract_feat(inputs, self.num_sample_test, return_att=True) # same as aib_v2
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg, num_sample=self.num_sample_test)
        
        assert att_mask.size(0) == self.num_sample_test
        assert att_mask.size(1) == inputs.size(0)
        # assert att_mask.size(2) == self.backbone.att_dim_w
        # assert att_mask.size(3) == self.backbone.att_dim_h
        # att_mask = att_mask.view(num_sample, img.size(0), self.backbone.att_dim, self.backbone.att_dim)
        # only visualize the first sample of attention maps
        att_mask_list = [att_mask[0, bs] for bs in range(inputs.size(0))]
        att_mask = torch.stack(att_mask_list).unsqueeze(1) # (bs, att_dim, att_dim)->(bs, 1, att_dim, att_dim)
        att_mask = resize(
            input=att_mask,
            size=inputs.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        return seg_logits, att_mask

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList, num_sample=1) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg, num_sample)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList, num_sample=1) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg, num_sample)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg, num_sample)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    # previous forward_train?
    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x, mu, std = self.extract_feat(inputs, self.num_sample_train, return_att=False)
        latent_losses = self.kl_div(mu, std) # same as aib_v2
        att_dict = {'att_max': mu.mean(0).max(),
                    'att_min': mu.mean(0).min()} # same as aib_v1
        
        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples, num_sample=self.num_sample_train)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples, num_sample=self.num_sample_train)
            losses.update(loss_aux)

        losses.update(latent_losses) # same as aib_v2
        losses.update(att_dict) # same as aib_v1

        return losses

    def kl_div(self, mus, sigma):
        """Compute segmentation loss."""
        loss = dict()
        kl_div = -0.5 * (1+2*sigma.log()-mus.pow(2)-sigma.pow(2))
        loss['loss_info'] = self.va_beta * torch.mean(kl_div).div(math.log(2))
        return loss
    
    # previous forward_test?
    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)

        return self.postprocess_result(seg_logits, data_samples)
    
        # For attention map visualization
        # seg_logits, att_masks = self.inference_returnatt(inputs, batch_img_metas)
        # return self.postprocess_attention(seg_logits, att_masks, data_samples)
    
    def postprocess_attention(self,
                           seg_logits: Tensor,
                           att_masks: Tensor,
                           data_samples: OptSampleList = None) -> SampleList:
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            if C > 1:
                i_seg_pred = i_seg_logits.argmax(dim=0, keepdim=True)
            else:
                i_seg_logits = i_seg_logits.sigmoid()
                i_seg_pred = (i_seg_logits >
                              self.decode_head.threshold).to(i_seg_logits)
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

            # Process attention masks
            att_maps = self.postprocess_prediction(att_masks[i][0].cpu().numpy())
            att_maps = Image.fromarray(att_maps.astype(np.uint8))
            basename = data_samples[i].img_path.split('/')[-1]
            basename = basename.split('.')[0]
            png_filename = os.path.join(self.output_dir, f'{basename}.png')
            att_maps.save(png_filename)

        return data_samples


    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        # x = self.extract_feat_test(inputs)
        # x = self.extract_feat(inputs, self.num_sample_test, return_att=False) # aib_v1
        x, _, _ = self.extract_feat(inputs, self.num_sample_test, return_att=False) # aib_v2
        # return self.decode_head.forward(x)
        return self.decode_head.predict(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits
    
    def slide_inference_returnatt(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        att_masks = inputs.new_zeros((batch_size, 1, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit, crop_att_mask = self.encode_decode_returnatt(crop_img, batch_img_metas)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                att_masks += F.pad(crop_att_mask,
                               (int(x1), int(att_masks.shape[3] - x2), int(y1),
                                int(att_masks.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat
        att_masks = att_masks / count_mat

        return seg_logits, att_masks

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits
    
    def whole_inference_returnatt(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits, att_mask = self.encode_decode_returnatt(inputs, batch_img_metas)

        return seg_logits, att_mask

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit
    
    def inference_returnatt(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit, att_mask = self.slide_inference_returnatt(inputs, batch_img_metas)
        else:
            seg_logit, att_mask = self.whole_inference_returnatt(inputs, batch_img_metas)

        return seg_logit, att_mask

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
    def aug_test_returnatt(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit, att_mask = self.inference_returnatt(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit, cur_att_mask = self.inference_returnatt(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
            att_mask += cur_att_mask
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)

        att_mask /= len(inputs)
        att_mask = att_mask.cpu().numpy()
        att_mask_list = []
        # unravel batch dim
        for bidx in range(att_mask.shape[0]):
            # postporcessing
            tmp = self.postprocess_prediction(att_mask[bidx])  # only one att_mask for each sample?
            att_mask_list.append(tmp)

        return seg_pred, att_mask_list

    def postprocess_prediction(self, prediction, size=None, print_info=True):
        """
        Postprocess saliency maps by resizing and applying gaussian blurringself.
        args:
            prediction: numpy array with saliency postprocess_prediction
            size: original (H,W) of the image
        returns:
            numpy array with saliency map normalized 0-255 (int8)
        """
        if print_info:
            print('max %.4f min %.4f' % (
            np.max(prediction), np.min(prediction)))  # l1 norm is much larger than l2? but maps are similar

        prediction = prediction - np.min(prediction)

        # prediction = prediction - np.mean(prediction)
        # prediction[prediction<0] = 0

        # print('max %.4f min %.4f'%(np.max(prediction), np.min(prediction))) # l1 norm is much larger than l2? but maps are similar
        if np.max(prediction) != 0:
            saliency_map = (prediction / np.max(prediction) * 255).astype(np.uint8)
        else:
            saliency_map = prediction.astype(np.uint8)

        # resize back to original size
        saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), 0)
        if size is not None:
            saliency_map = cv2.resize(saliency_map, (size[1], size[0]), interpolation=cv2.INTER_CUBIC)

        # clip again
        # saliency_map = np.clip(saliency_map, 0, 255)
        if np.max(saliency_map) != 0:
            saliency_map = saliency_map.astype('float') / np.max(saliency_map) * 255.
        else:
            print('Zero saliency map.')

        return saliency_map