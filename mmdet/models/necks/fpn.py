# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType


@MODELS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)



# from .mymodule.darknet import BaseConv
# from .mymodule.rcrnet import _ASPPModule
# from .mymodule.non_local_dot_product import NONLocalBlock3D
from .mymodule.convgru import ConvGRUCell
import torch
from torch import nn
import pdb
from mmcv.cnn import ConvModule
from mmengine.model import xavier_init, constant_init
import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):
    def __init__(self, planes, reduce_ratio=8):
        super(NonLocalBlock, self).__init__()

        inter_planes = planes // reduce_ratio
        self.query_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.key_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.value_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        proj_query = proj_query.contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = proj_key.contiguous().view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = proj_value.contiguous().view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma * out + x
        return out

class Align(nn.Module):
    def __init__(self,
                 out_channels=256,
                 kernel_size=3):
        super(Align, self).__init__()
        self.out_channels = out_channels

        self.depthwise_conv = nn.Conv2d(self.out_channels * 2, self.out_channels * 2, kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2, bias=False, groups=self.out_channels * 2)
        self.pointwise_conv = nn.Conv2d(self.out_channels * 2, 2, kernel_size=1, bias=False)

        self._init_weight()
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m.weight, 1.0)
                constant_init(m.bias, 0)
    def forward(self, low_feature, high_feature):
        b, c, h, w = low_feature.size() 
        high_feature_up = F.interpolate(high_feature, size=(h, w), mode='bilinear', align_corners=True)
        concat_feature = torch.cat([low_feature, high_feature_up], dim=1)
        high_offset = self.pointwise_conv(self.depthwise_conv(concat_feature))
        high_feature_new = self.grid_sample(high_feature, high_offset, (h, w))
        out = low_feature + high_feature_new
        return out
    def grid_sample(self, input, offset, size):
        b, _, h, w = input.size()
        out_h, out_w = size
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(b, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + offset.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid, align_corners=True)
        return output




@MODELS.register_module()
class FPNSeq(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        num_frame: int, 
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        
        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        
        ########运动信息
        self.num_frame = num_frame
        # self.aspp = _ASPPModule(256, out_channels, 16)
        self.convgru_forward = ConvGRUCell(out_channels, out_channels, 3)
        self.convgru_backward = ConvGRUCell(out_channels, out_channels, 3)
        self.bidirection_conv = nn.Conv2d(out_channels*2, out_channels, 3, 1, 1)
        
        ####text
        self.textatt = TextGuidedAttention2(dim = out_channels, text_channel = 512, amplify=4)
        self.nonlocalblock = NonLocalBlock(out_channels)
        self.align = Align(out_channels, 3)

        ###ablation
        # self.selfatte = SelfAttentionWeighting(out_channels)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # pdb.set_trace()
        res = []
        for input in inputs: ######[[(1,256,128,128),(1,512,64,64),(1,1024,32,32),(1,2048,16,16)],[],[],[],[]]
            # build laterals
            laterals = [
                lateral_conv(input[i + self.start_level])
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]

            # build top-down path
            used_backbone_levels = len(laterals)
            for i in range(used_backbone_levels - 1, 0, -1):
                # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
                #  it cannot co-exist with `size` in `F.interpolate`.
                if 'scale_factor' in self.upsample_cfg:
                    # fix runtime error of "+=" inplace operation in PyTorch 1.10
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], **self.upsample_cfg)
                else:
                    prev_shape = laterals[i - 1].shape[2:]
                    laterals[i - 1] = laterals[i - 1] + F.interpolate(
                        laterals[i], size=prev_shape, **self.upsample_cfg)

            # build outputs
            # part 1: from original levels
            outs = [
                self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            ]
            # part 2: add extra levels
            if self.num_outs > len(outs):
                # use max pool to get more levels on top of outputs
                # (e.g., Faster R-CNN, Mask R-CNN)
                if not self.add_extra_convs:
                    for i in range(self.num_outs - used_backbone_levels):
                        outs.append(F.max_pool2d(outs[-1], 1, stride=2))
                # add conv layers on top of original feature maps (RetinaNet)
                else:
                    if self.add_extra_convs == 'on_input':
                        extra_source = input[self.backbone_end_level - 1]
                    elif self.add_extra_convs == 'on_lateral':
                        extra_source = laterals[-1]
                    elif self.add_extra_convs == 'on_output':
                        extra_source = outs[-1]
                    else:
                        raise NotImplementedError
                    outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                    for i in range(used_backbone_levels + 1, self.num_outs):
                        if self.relu_before_extra_convs:
                            outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                        else:
                            outs.append(self.fpn_convs[i](outs[-1]))
            res.append(outs)
        
        results = []  #[[(1,256,64,64),(1,256,32,32),(1,256,16,16),(1,256,8,8),(1,256,4,4)],[],[],[],[]]
        res = list(map(list, zip(*res))) 
        # pdb.set_trace()
        
        for feat in res: #64是一个list，32是一个list，16是一个list，8是一个list，4是一个list
            feats_time = torch.stack(feat, dim=2) # [1,256,5,32,32]
            
            feat = feats_time[:,:,0,:,:] # [1,256,32,32] 第一帧特征
            feats_forward = []
            # forward
            for i in range(self.num_frame):
                feat = self.convgru_forward(feats_time[:,:,i,:,:], feat) # [1,256,32,32]
                feats_forward.append(feat)

            # backward
            feat = feats_forward[-1] #最后一帧特征
            feats_backward = []
            for i in range(self.num_frame):
                feat = self.convgru_backward(feats_forward[self.num_frame-1-i], feat)
                feats_backward.append(feat)
            feats_backward = feats_backward[::-1] #反转，让最后一帧还在最后
            
            feats = []
            for i in range(self.num_frame):
                feat = torch.tanh(self.bidirection_conv(torch.cat((feats_forward[i], feats_backward[i]), dim=1)))
                feats.append(feat)  

            c_feat = feats[-1]
            c_feat = self.nonlocalblock(c_feat)
            c_feat,txt_fea = self.textatt(c_feat)
            results.append(c_feat) # [(1,256,64,64),(1,256,32,32),(1,256,16,16),(1,256,8,8),(1,256,4,4)]
        
        _,txt_fea = self.textatt(results[0])

        return results,txt_fea

        # r = []
        # P3 = self.align(results[3], results[4])
        # P4 = self.align(results[2], P3)
        # P5 = self.align(results[1], P4)
        # P6 = self.align(results[0], P5)
        # r.append(P6)
        # r.append(P5)
        # r.append(P4)
        # r.append(P3)
        # return r, results




class TextGuidedAttention(nn.Module):
    def __init__(self, dim = 256, text_channel = 512, amplify=4):
        super(TextGuidedAttention, self).__init__()
        
        # Hidden layer dimension
        d = int(dim * amplify)
        
        # 用于映射视觉特征到一个新的表示空间
        self.mlp_vis = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, text_channel, 1, bias=False)  # 映射到文本特征维度
        )

        # 学习视觉特征与文本特征的相似性（注意力权重）
        self.similarity_mlp = nn.Conv2d(text_channel, 1, 1, bias=False)  # 输出注意力权重

        # 对视觉特征进行重新加权
        self.reweight = nn.Conv2d(dim, dim, 1, bias=False)
        
        ######clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text = clip.tokenize([
    # Target1
    "The target is usually brighter than the surrounding background, with a higher temperature, making it stand out in the image.",
    "The target usually appears as a small dot, round or oval shape, with relatively blurred edges, though it may be irregular due to noise.",
    "The target occupies only a few pixels in the image, typically ranging from several to a dozen pixels.",
    "The target may show slight movement, often revealing a continuous trajectory in multi-frame images.",
    "The target often has a contrast with the background, making it more prominent in the image.",
    "The target's brightness intensity may slightly fluctuate across frames but overall the target region will still be brighter than the background.",
    "The target typically as a thermal source  has distinct temperature characteristics, with a significant difference from the background temperature.",
    "The target may maintain a certain stability, but slight jitter may occur due to noise and background interference in multi-frame images.",
    "The target is prone to being obscured, due to the low brightness and the weak signal compared to background noise.",
    "The target is usually isolated in the image, with no other significant feature points around it."
    # Target2
    # "The target shape is irregular due to background noise and surrounding interference factors, showing unstable edge contours.",
    # "The target occupies only a few pixels in the image, typically ranging from several to a dozen pixels.",
    # "The target is prone to being obscured, due to the signal of target is weak compared to background noise.",
    # "The target has a slightly higher brightness compared to the background, but the brightness difference may be very subtle and hard to detect.",
    # "The target has blurred edges and lack clear boundaries, is easy to blend in with the background.",
    # "The target has a surface texture similar to the background, appearing gray and blurry, indicating its temperature.",
    # "The target temperature is slightly higher than the surrounding pixel temperature and gradually attenuates toward the surroundings.",
    # "The target shows slight movement, often revealing a simple trajectory in multi-frame images, but may be slightly jittery due to background interference.",
    # "The target is usually isolated in the image, with no other significant feature points around it.",
    # "The target is often specific types of objects, such as heat sources, aircraft, vehicles, etc., that contrast with the background and are usually parts of interest in the scene."
    ]).to(self.device)  #["a target", "a car", "a cat"]
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.device)  # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
        
    # 文本特征映射到图像特征空间
    def forward(self, visual_feats):
        with torch.no_grad():
            text_features = self.clip.encode_text(self.text) # 10 × text_channel
        txt_fea = text_features.clone()
        # 将视觉特征映射到文本特征维度
        vis_proj = self.mlp_vis(visual_feats)  # B × text_channel × H × W   [1/3,512,64,64]
        
        # 扩展文本特征
        text_features = text_features.mean(dim=0)  # shape: [512]
        text_features = text_features.view(1, text_features.shape[0], 1, 1)  # [1 512 1 1]
        text_features = text_features.expand_as(vis_proj)  # 广播到 1/3 × 512 × 64 × 64
        
        # 计算相似性 (逐像素点乘)
        similarity = vis_proj * text_features  # B × text_channel × H × W [1 512 64 64]
        attention_map = self.similarity_mlp(similarity)  # B × 1 × H × W
        # 归一化为注意力权重
        attention_map = F.softmax(attention_map.flatten(2), dim=-1).view_as(attention_map)  # B × 1 × H × W
        # 对原始视觉特征残差加权
        weighted_feats = attention_map * visual_feats + visual_feats  # B × C × H × W
        weighted_feats = self.reweight(weighted_feats)  # 再次通过 1x1 卷积调整通道关系

        return weighted_feats, txt_fea


class TextGuidedAttention2(nn.Module):
    def __init__(self, dim = 256, text_channel = 512, amplify=4):
        super(TextGuidedAttention2, self).__init__()

        self.mlp_vis = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256) 
        )


        # 学习视觉特征与文本特征的相似性（注意力权重）
        self.similarity_mlp = nn.Conv2d(dim, 1, 1, bias=False)  # 输出注意力权重

        # 对视觉特征进行重新加权
        self.reweight = nn.Conv2d(dim, dim, 1, bias=False)
        
        ######clip
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text = clip.tokenize([
    # Target1
    # "The target is usually brighter than the surrounding background, with a higher temperature, making it stand out in the image.",
    # "The target usually appears as a small dot, round or oval shape, with relatively blurred edges, though it may be irregular due to noise.",
    # "The target occupies only a few pixels in the image, typically ranging from several to a dozen pixels.",
    # "The target may show slight movement, often revealing a continuous trajectory in multi-frame images.",
    # "The target often has a contrast with the background, making it more prominent in the image.",
    # "The target's brightness intensity may slightly fluctuate across frames but overall the target region will still be brighter than the background.",
    # "The target typically as a thermal source  has distinct temperature characteristics, with a significant difference from the background temperature.",
    # "The target may maintain a certain stability, but slight jitter may occur due to noise and background interference in multi-frame images.",
    # "The target is prone to being obscured, due to the low brightness and the weak signal compared to background noise.",
    # "The target is usually isolated in the image, with no other significant feature points around it."
    # Target2
    # "The target shape is irregular due to background noise and surrounding interference factors, showing unstable edge contours.",
    # "The target occupies only a few pixels in the image, typically ranging from several to a dozen pixels.",
    # "The target is prone to being obscured, due to the signal of target is weak compared to background noise.",
    # "The target has a slightly higher brightness compared to the background, but the brightness difference may be very subtle and hard to detect.",
    # "The target has blurred edges and lack clear boundaries, is easy to blend in with the background.",
    # "The target has a surface texture similar to the background, appearing gray and blurry, indicating its temperature.",
    # "The target temperature is slightly higher than the surrounding pixel temperature and gradually attenuates toward the surroundings.",
    # "The target shows slight movement, often revealing a simple trajectory in multi-frame images, but may be slightly jittery due to background interference.",
    # "The target is usually isolated in the image, with no other significant feature points around it.",
    # "The target is often specific types of objects, such as heat sources, aircraft, vehicles, etc., that contrast with the background and are usually parts of interest in the scene."
    # Target3
    "The target is usually brighter than the surrounding background, with a higher temperature, making it stand out in the image.",
    "The target shape is irregular due to background noise and surrounding interference, showing blurred edge contours.",
    "The target occupies only a few pixels in the image, typically ranging from several to a dozen pixels.",
    "The target may show slight movement, often revealing a simple trajectory in frames, but may be jittery due to background interference.",
    "The target often has a contrast with the background, making it more prominent in the image.",
    "The target's brightness intensity may slightly fluctuate across frames but overall the target region will still be brighter than the background.",
    "The target typically as a thermal source has distinct temperature characteristics, with a significant difference from the background temperature.",
    "The target is usually isolated in the image, with no other significant feature points around it.",
    "The target is prone to being obscured, due to the low brightness and the weak signal compared to background noise.",
    "The target may maintain a certain stability in multi-frame images, but slight jitter may occur due to noise and background interference.",
    "The target has a surface texture similar to the background, appearing gray and blurry, indicating its temperature.",
    "The target is specific types of objects, such as heat sources, aircraft, vehicles, etc., that are parts of interest in the scene.",
    "The target typically has a low signal-to-noise ratio, making it harder to detect from background noise."
    ]).to(self.device)  #["a target", "a car", "a cat"]
        self.clip, self.preprocess = clip.load("ViT-B/32", device=self.device)  # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'


        ###Ablation -concat
        # self.convcat = nn.Conv2d(512, 256, 1, bias=False)
        ### Ablation -attention to produce weight

        ###Ablation -Learnable weight
        # self.learnable_weight = nn.Parameter(torch.randn(1, dim, 1, 1), requires_grad=True)
        # self.weight_mlp = nn.Conv2d(dim, dim, kernel_size=1, bias=False)  # 调整矩阵维度
    # 文本特征映射到图像特征空间
    def forward(self, visual_feats):
        # pdb.set_trace()
        with torch.no_grad():
            text_features = self.clip.encode_text(self.text) # 10 × text_channel
        txt_fea = text_features.clone()
        # 将视觉特征映射到文本特征维度
        text_features = text_features.to(torch.float32)
        text_features = self.mlp_vis(text_features)  # 10 256
        # 扩展文本特征
        text_features = text_features.mean(dim=0)  # shape: [256]
        text_features = text_features.view(1, text_features.shape[0], 1, 1)  # [1 256 1 1]
        text_features = text_features.expand_as(visual_feats)  # 广播到 1/3 × 256 × 64 × 64



        ###Ablation -concat
        # visual_feats = torch.cat((visual_feats, text_features), dim=1)  # B × (C+text_channel) × H × W
        # weighted_feats = self.convcat(visual_feats)
        ##Ablation -attention to produce weight


        # Ablation -learnable weight
        # weight_matrix = F.relu(self.weight_mlp(self.learnable_weight.expand_as(visual_feats)))  # 确保非负
        # weighted_feats = weight_matrix * visual_feats


        
        # 计算相似性 (逐像素点乘)
        similarity = visual_feats * text_features  # B × text_channel × H × W [1 256 64 64]
        attention_map = self.similarity_mlp(similarity)  # B × 1 × H × W
        # 归一化为注意力权重
        attention_map = F.softmax(attention_map.flatten(2), dim=-1).view_as(attention_map)  # B × 1 × H × W
        # 对原始视觉特征残差加权
        weighted_feats = attention_map * visual_feats + visual_feats  # B × C × H × W
        weighted_feats = self.reweight(weighted_feats)  # 再次通过 1x1 卷积调整通道关系

        return weighted_feats, txt_fea














###Ablation 自注意力加权
class SelfAttentionWeighting(nn.Module):
    def __init__(self, dim=256):
        super(SelfAttentionWeighting, self).__init__()
        self.query_conv = nn.Conv2d(dim, dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(dim, dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        
        # Query, Key, Value projections
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # B × (H*W) × (C/8)
        key = self.key_conv(x).view(B, -1, H * W)  # B × (C/8) × (H*W)
        value = self.value_conv(x).view(B, -1, H * W)  # B × C × (H*W)
        
        # Attention Map (softmax on similarity matrix)
        attention = F.softmax(torch.bmm(query, key), dim=-1)  # B × (H*W) × (H*W)
        
        # Weighted Sum of Values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B × C × (H*W)
        out = out.view(B, C, H, W)  # Reshape to B × C × H × W
        
        # Weighted features
        out = self.gamma * out + x
        return out
    
if __name__ == '__main__':
    model = SelfAttentionWeighting()
    x = torch.randn(3, 256, 64, 64)
    out = model(x)
    print(out.size())