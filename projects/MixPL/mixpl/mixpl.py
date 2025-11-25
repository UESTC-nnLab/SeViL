# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor 
import torch.nn.functional as F

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmdet.models.detectors import SemiBaseDetector
from mmdet.structures.bbox import bbox_project
from torch.nn import functional as F
import numpy as np
import math
import os.path as osp
import pdb
import clip
try:
    import sklearn.mixture as skm
except ImportError:
    skm = None

import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F




@MODELS.register_module()
class MixPL(SemiBaseDetector):
    """Base class for semi-supervised detectors."""

    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.cache_inputs = []
        self.text_feature = None
        # self.dynamicmatching = MultiModalDynamicMatching(512, 256, 512)
        self.recover_text = RecoverText()
        self.recover_image = RecoverImage()
         # 加载 CLIP 模型和预处理函数
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)  # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """ 
        # pdb.set_trace() 
        losses = dict()
        loss = self.loss_by_gt_instances(multi_batch_inputs['sup'], multi_batch_data_samples['sup'])
        losses.update(loss)  ###监督下的损失（学生模型）


        origin_batch_pseudo_data_samples, batch_info, features, txt_fea = self.get_pseudo_instances(
            multi_batch_inputs['unsup_teacher'], multi_batch_data_samples['unsup_teacher']) #教师模型产生弱增强后的伪标签
        self.text_feature = txt_fea ###文本特征
        

        # pdb.set_trace()
        # #####mask loss
        mask_txt_fea, mask_txt = mask_text(self.text_feature)
        mask_img_fea, mask_img = mask_image(features[0])
        recov_txt_fea = self.recover_text(mask_txt_fea, features[0])
        recov_img_fea = self.recover_image(self.text_feature, mask_img_fea)
        # #####recover loss
        # # recover_txt_loss = F.smooth_l1_loss(recov_txt_fea, self.text_feature) # smooth l1 loss  
        recover_txt_loss = F.mse_loss(recov_txt_fea, self.text_feature.to(torch.float32))
        # # losses.update({'rec_txt_loss': recover_txt_loss})
        recover_img_loss = F.mse_loss(recov_img_fea, features[0])
        # # losses.update({'rec_img_loss': recover_img_loss})
        recover_loss = recover_txt_loss + recover_img_loss
        ###ablation
        # recover_loss = recover_img_loss
        losses.update({'rec_loss': recover_loss})
        
        
        #######
        multi_batch_data_samples['unsup_student'] = self.project_pseudo_instances(
            origin_batch_pseudo_data_samples, multi_batch_data_samples['unsup_student']) # 伪标签投影到学生模型的输入上

        batch_unsup_inputs = copy.deepcopy(multi_batch_inputs['unsup_student'])
        batch_unsup_data_samples = copy.deepcopy(multi_batch_data_samples['unsup_student'])

        

        # pdb.set_trace()
        batch_unsup_inputs, batch_unsup_data_samples = self.merge(
            *zip(*list(map(self.erase, *self.split(batch_unsup_inputs, batch_unsup_data_samples)))))  ####optical erase


        losses.update(**self.loss_by_pseudo_instances(
                batch_unsup_inputs, batch_unsup_data_samples))

        return losses
        

    def loss_by_pseudo_instances(self,
                                batch_inputs: Tensor,
                                batch_data_samples: SampleList,
                                batch_info: Optional[dict] = None) -> dict:
       
        batch_data_samples = filter_gt_instances(
            batch_data_samples, score_thr=self.semi_train_cfg.cls_pseudo_thr)  
        ##### cal per image pseudo instances
        # img_num = len(batch_data_samples)
        # pseudo_instances_num = sum([data_samples.gt_instances.bboxes.size(0) for data_samples in batch_data_samples])
        # pseudo_instances_num = round(pseudo_instances_num / img_num, 3)
        # with open('ratio.txt', 'a') as f:
        #     f.write(f',{pseudo_instances_num}\n')

        
        #pseudo begin####
        # 从batch_inputs中获取最后一帧图像
        keyframe = batch_inputs[:,-1,:,:,:] #keyframe: (3, 3, 512, 512)        
        # 对每一个batch中的图像进行处理
        for i in range(keyframe.shape[0]):
            score_list = []
            pse_boxes = batch_data_samples[i].gt_instances.bboxes #boxes: (n, 4)
            keyframe_i = keyframe[i].unsqueeze(0) #keyframe_i: (1, 3, 512, 512)
            pse_scores = batch_data_samples[i].gt_instances.scores #scores: (n)  
            # 根据 box的位置信息，将box进行crop出来并和文本特征计算相似度
            for j in range(pse_boxes.shape[0]):
                # pdb.set_trace()
                box = pse_boxes[j,:].unsqueeze(0) #box: (1, 4)
                score = pse_scores[j].unsqueeze(0) #score: (1)
                boxitem = box[0].cpu().numpy()
                #将box从keyframe_i中对应位置crop出来
                crop_box = keyframe_i[:,:,np.int32(boxitem[1]):np.int32(boxitem[3]),np.int32(boxitem[0]):np.int32(boxitem[2])] #crop_box: (1, 3, x, y)
                crop_box = F.interpolate(crop_box, size=(224, 224), mode='bilinear', align_corners=False) #crop_box: (1, 3, 224, 224)
                crop_feature = self.model.encode_image(crop_box)
                # 计算文本特征和crop_feature的相似度求均值
                similarity = F.cosine_similarity(crop_feature, self.text_feature, dim=-1).mean()

                # print('score:',score)
                # print('similarity:',similarity)

                #平衡score和相似度 #结合score和相似度计算最终的可信度指数
                metric = 2 * score * similarity/(score + similarity)
                # print('metric:',metric)
                score_list.append(metric)
                
                
                 # 计算空间方差
                # crop_region = crop_box.squeeze(0).permute(1, 2, 0).cpu().numpy()  # 提取像素值
                # mean_intensity = crop_region.mean() # 0.2249632
                # spatial_variance = ((crop_region - mean_intensity) ** 2).mean() # 0.032116167
                # # CI计算公式
                # gamma, delta, alpha = 1.0, 1.0, 1.0  # 权重参数
                # ci = (similarity ** gamma * score ** delta) / (1 + spatial_variance ** alpha) * 100 # 0.0026
                # score_list.append(ci)
            if len(score_list) != 0:
                score_list = torch.cat(score_list, dim=0)
                batch_data_samples[i].gt_instances.simi = score_list
            else:
                batch_data_samples[i].gt_instances.simi = torch.tensor([]).to(self.data_preprocessor.device)

        simi_list = []
        for data_samples in batch_data_samples:
           score = data_samples.gt_instances['simi']
           simi_list.append(score)
        simis = torch.cat(simi_list, dim=0)

        dynamic_thr = self.gmm_policy(scores=simis)
        # print('dynamic_thr:',dynamic_thr)
        batch_data_samples =self.filter_gt_by_simi(batch_data_samples, score_thr=dynamic_thr)
        #### pseudo end #####

        losses = self.student.loss(batch_inputs, batch_data_samples)
        pseudo_instances_num = min([len(data_samples.gt_instances) for data_samples in batch_data_samples])
        unsup_weight = self.semi_train_cfg.unsup_weight if pseudo_instances_num >= self.semi_train_cfg.least_num else 0.
        return rename_loss_dict('unsup_', reweight_loss_dict(losses, unsup_weight))
    
    def filter_gt_by_maxsize(self,batch_data_samples: SampleList,
                                 wh_thr: tuple) -> SampleList:
        for data_samples in batch_data_samples:
            bboxes = data_samples.gt_instances.bboxes
            if bboxes.shape[0] > 0:
                w = bboxes[:, 2] - bboxes[:, 0]
                h = bboxes[:, 3] - bboxes[:, 1]
                data_samples.gt_instances = data_samples.gt_instances[
                    (w <= wh_thr[0]) & (h <= wh_thr[1])]
        return batch_data_samples
    
    def filter_gt_by_simi(self, batch_data_samples: SampleList,
                                  score_thr: float) -> SampleList:
        for data_samples in batch_data_samples:
            assert 'scores' in data_samples.gt_instances, \
                'there does not exit scores in instances'
            if data_samples.gt_instances.bboxes.shape[0] > 0:
                data_samples.gt_instances = data_samples.gt_instances[
                    data_samples.gt_instances.simi > score_thr]
        return batch_data_samples
    

    def gmm_policy(self, scores, given_gt_thr=0.3, policy='high'):
        """The policy of choosing pseudo label.

        The previous GMM-B policy is used as default.
        1. Use the predicted bbox to fit a GMM with 2 center.
        2. Find the predicted bbox belonging to the positive
            cluster with highest GMM probability.
        3. Take the class score of the finded bbox as gt_thr.

        Args:
            scores (nd.array): The scores.

        Returns:
            float: Found gt_thr.

        """
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)
        gmm_scores = gmm.score_samples(scores)
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (
                    scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr

        return pos_thr
    def bgmm_policy(self,scores, given_gt_thr=0.3, policy='high'):
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        bgmm = BayesianGaussianMixture(n_components=2, covariance_type='full',n_init=2)
        bgmm.fit(scores)
        gmm_assignment = bgmm.predict(scores)
        gmm_scores = bgmm.score_samples(scores)
        
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
            else:
                pos_thr = given_gt_thr
        return pos_thr
    def merge(self, inputs_list, batch_data_samples):
        # pdb.set_trace()
        batch_size = len(inputs_list)
        h, w = 0, 0
        for i in range(batch_size):
            img_h, img_w = batch_data_samples[i].img_shape
            h, w = max(h, img_h), max(w, img_w)
        h, w = max(h, math.ceil(h / 32) * 32), max(w, math.ceil(w / 32) * 32)
        num_frame = inputs_list[0].shape[0]
        batch_inputs = torch.zeros((batch_size,num_frame, 3, h, w)).to(self.data_preprocessor.device)
        for i in range(batch_size):
            img_h, img_w = batch_data_samples[i].img_shape
            # batch_inputs[i, :, :img_h, :img_w] = inputs_list[i]
            batch_inputs[i,:, :, :img_h, :img_w] = inputs_list[i]
            batch_data_samples[i].set_metainfo({'batch_input_shape': (h, w)})
            batch_data_samples[i].set_metainfo({'pad_shape': (h, w)})
        return batch_inputs, batch_data_samples

    def split(self, batch_inputs, batch_data_samples):
        inputs_list = []
        for i in range(len(batch_inputs)):
            inputs = batch_inputs[i]
            data_samples = batch_data_samples[i]
            img_h, img_w = data_samples.img_shape 
            inputs_list.append(inputs[..., :img_h, :img_w]) ## 将输入按照图像形状进行切片，并添加到输入列表中
            data_samples.pop('batch_input_shape') #
            data_samples.pop('pad_shape')
        return inputs_list, batch_data_samples


    def update_cache(self, batch_inputs: Tensor):
        inputs_list = [batch_inputs[i].unsqueeze(0) for i in range(batch_inputs.size(0))]
        cache_size = self.semi_train_cfg.cache_size
        self.cache_inputs.extend(inputs_list)
        self.cache_inputs = self.cache_inputs[-cache_size:]#如果缓存输入列表的长度超过了缓存大小 cache_size，则对缓存输入列表进行截断，保留最近的 cache_size 个元素
    

    def erase(self, inputs, data_samples):  #inputs: (5, 3, 256, 256)
        inputscopy = inputs.clone()
        inputstemp = inputs.clone()
        inputs = inputs[-1, :, :, :]
        def _get_patches(img_shape): #inputs = inputs[-1, :, :, :]  ##只对关键帧进行erase
            patches = []
            n_patches = np.random.randint(
                self.semi_train_cfg.erase_patches[0], self.semi_train_cfg.erase_patches[1])
            for _ in range(n_patches):
                ratio = np.random.random() * \
                        (self.semi_train_cfg.erase_ratio[1] - self.semi_train_cfg.erase_ratio[0]) + \
                        self.semi_train_cfg.erase_ratio[0]
                ph, pw = int(img_shape[0] * ratio), int(img_shape[1] * ratio)
                px1 = np.random.randint(0, img_shape[1] - pw)
                py1 = np.random.randint(0, img_shape[0] - ph)
                px2, py2 = px1 + pw, py1 + ph
                patches.append([px1, py1, px2, py2])
            return torch.tensor(patches).to(self.data_preprocessor.device)
        erase_patches = _get_patches(data_samples.img_shape)
        for patch in erase_patches:
            px1, py1, px2, py2 = patch
            inputs[:, py1:py2, px1:px2] = 0
        bboxes = data_samples.gt_instances.bboxes
        left_top = torch.maximum(bboxes[:, None, :2], erase_patches[:, :2])
        right_bottom = torch.minimum(bboxes[:, None, 2:], erase_patches[:, 2:])
        wh = torch.clamp(right_bottom - left_top, 0)
        inter_areas = wh[:, :, 0] * wh[:, :, 1]
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        bboxes_erased_ratio = inter_areas.sum(-1) / (bbox_areas + 1e-7)
        valid_inds = bboxes_erased_ratio < self.semi_train_cfg.erase_thr
        data_samples.gt_instances = data_samples.gt_instances[valid_inds]
        inputscopy[-1, :, :, :] = inputs
        assert inputscopy.shape == inputstemp.shape
        return inputscopy, data_samples


def mask_text(text_features, mask_prob=0.15):
    """
    对文本特征进行随机掩码。
    :param text_features: 文本特征，shape = [N, D]，其中 N 是单词数，D 是单词向量维度。
    :param mask_prob: 掩码概率。
    :return: 掩盖后的文本特征，掩码矩阵。
    """
    N, D = text_features.shape
    # 创建掩码矩阵
    mask = torch.rand(N) < mask_prob  # 随机掩盖 mask_prob 比例的单词
    mask = mask.to(text_features.device)  # 确保掩码矩阵与输入特征在同一设备上

    # 掩盖文本：用零向量代替被掩盖的单词
    masked_text_features = text_features.clone()
    masked_text_features[mask] = 0  # 被掩盖部分设置为零

    return masked_text_features, mask


def mask_image(image_features, mask_prob=0.25, block_size=2):
    """
    对图像特征进行随机掩码。
    :param image_features: 图像特征，shape = [B, C, H, W]。
    :param mask_prob: 掩码概率。
    :param block_size: 掩盖区域的块大小（正方形）。
    :return: 掩盖后的图像特征，掩码矩阵。
    """
    B, C, H, W = image_features.shape

    # 生成掩码矩阵
    mask = torch.rand(B, H // block_size, W // block_size) < mask_prob  # 每个块是否掩盖
    mask = mask.repeat_interleave(block_size, dim=1).repeat_interleave(block_size, dim=2)  # 扩展到块大小
    mask = mask.unsqueeze(1).to(image_features.device)  # 添加通道维度并与输入对齐

    # 掩盖图像：用零值代替被掩盖的区域
    masked_image_features = image_features.clone()
    masked_image_features *= ~mask  # 掩盖部分设置为零

    return masked_image_features, mask



class RecoverText(nn.Module):
    def __init__(self, text_dim=512, visual_dim=256, hidden_dim=512):
        super(RecoverText, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)  # 文本投影
        self.visual_proj_q = nn.Conv2d(visual_dim, hidden_dim, kernel_size=1)  # 视觉投影 Q
        self.visual_proj_kv = nn.Conv2d(visual_dim, hidden_dim, kernel_size=1)  # 视觉投影 K, V
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 动态融合权重

    def forward(self, text_features, visual_features):
        B, C, H, W = visual_features.size()
        N, D = text_features.size()

        text_features = text_features.to(torch.float32)
        Q = self.text_proj(text_features)  # [N, hidden_dim]
        K = self.visual_proj_q(visual_features).flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]
        V = self.visual_proj_kv(visual_features).flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]

        # 注意力计算
        attention_weights = torch.softmax(torch.matmul(Q.unsqueeze(0), K.transpose(1, 2)) / (D ** 0.5), dim=-1)  # [B, N, H*W]

        adjusted_text_features = torch.matmul(attention_weights, V)  # [B, N, hidden_dim]
        adjusted_text_features = adjusted_text_features.mean(dim=0)  # [N, hidden_dim]

        recover_text_features = self.alpha * adjusted_text_features + text_features
        return recover_text_features

class RecoverImage(nn.Module):
    def __init__(self, text_dim=512, visual_dim=256):
        super(RecoverImage, self).__init__()
        self.text_proj = nn.Linear(text_dim, visual_dim)
        # self.recov = nn.Sequential(
            # nn.Conv2d(visual_dim*2, visual_dim, 3, 1, 1),
            # nn.ReLU(),
            # nn.Conv2d(visual_dim, visual_dim, 3, 1, 1),
        # )    
        self.recov = nn.Sequential(
            BaseConv(visual_dim*2, visual_dim, 3, 1, act='relu'),
            BaseConv(visual_dim, visual_dim, 3, 1, act='relu'),)

    def forward(self, text_features, visual_features):
        text_features = text_features.to(torch.float32)
        text_features = self.text_proj(text_features)  # 10 256
        # 扩展文本特征
        text_features = text_features.mean(dim=0)  # shape: [256]
        text_features = text_features.view(1, text_features.shape[0], 1, 1)  # [1 256 1 1]
        # pdb.set_trace()
        text_features = text_features.expand_as(visual_features)  # 广播到 1/3 × 256 × 64 × 64
        visual_features = torch.cat([visual_features, text_features], dim=1)  # [1 512 64 64]
        visual_features = self.recov(visual_features)
        return visual_features




class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="relu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "sigmoid":
        module = nn.Sigmoid()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module




class MultiModalDynamicMatching(nn.Module):
    def __init__(self, text_dim, visual_dim, hidden_dim):
        super(MultiModalDynamicMatching, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)  # 文本投影
        self.visual_proj_q = nn.Conv2d(visual_dim, hidden_dim, kernel_size=1)  # 视觉投影 Q
        self.visual_proj_kv = nn.Conv2d(visual_dim, hidden_dim, kernel_size=1)  # 视觉投影 K, V
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 动态融合权重

    def forward(self, text_features, visual_features):
        """
        输入:
        - text_features: 文本特征, shape = [N, D]
        - visual_features: 图像特征, shape = [B, C, H, W]
        输出:
        - dynamic_text_features: 动态调整的文本特征, shape = [N, D]
        """
        B, C, H, W = visual_features.size()
        N, D = text_features.size()

        # 视觉特征的上下文投影
        # print(text_features.dtype) torch.float16
        text_features = text_features.to(torch.float32)
        Q = self.text_proj(text_features)  # [N, hidden_dim]
        K = self.visual_proj_q(visual_features).flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]
        V = self.visual_proj_kv(visual_features).flatten(2).permute(0, 2, 1)  # [B, H*W, hidden_dim]

        # 注意力计算
        attention_weights = torch.softmax(torch.matmul(Q.unsqueeze(0), K.transpose(1, 2)) / (D ** 0.5), dim=-1)  # [B, N, H*W]

        # 动态调整文本特征
        adjusted_text_features = torch.matmul(attention_weights, V)  # [B, N, hidden_dim]
        adjusted_text_features = adjusted_text_features.mean(dim=0)  # [N, hidden_dim]

        # 融合原始文本特征
        # dynamic_text_features = self.alpha * adjusted_text_features + (1 - self.alpha) * text_features
        dynamic_text_features = self.alpha * adjusted_text_features + text_features
        return dynamic_text_features



if __name__ == '__main__':
    model = MultiModalDynamicMatching(512, 256, 512)
    text_features = torch.randn(10, 512)
    visual_features = torch.randn(1, 256, 64, 64)
    output = model(text_features, visual_features)
    print(output.size())