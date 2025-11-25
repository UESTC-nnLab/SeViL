
import torch
import numpy as np


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



import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRecoveryWithVisualContext(nn.Module):
    def __init__(self, text_dim=512, visual_dim=256, hidden_dim=512, num_text_descriptions=10):
        super(TextRecoveryWithVisualContext, self).__init__()

        # 将视觉特征映射到与文本特征相同的空间
        self.visual_proj = nn.Conv2d(visual_dim, text_dim, kernel_size=1)  # [B, C, H, W] -> [B, D, H, W]

        # 用于恢复掩码文本特征的解码器
        self.text_recovery = nn.Conv2d(text_dim, text_dim, kernel_size=1)  # [B, D, H, W] -> [B, D, H, W]

        # 文本特征（N, D），假设是提前获取的（如从CLIP等模型中获得）
        self.text_embeddings = nn.Parameter(torch.randn(num_text_descriptions, text_dim))  # [N, D]
        
        # 视觉特征到文本特征的交互映射
        self.interaction_layer = nn.Conv2d(text_dim + text_dim, text_dim, kernel_size=1)

        self.num_text_descriptions = num_text_descriptions

    def forward(self, masked_text_features, visual_features):
        """
        :param masked_text_features: 掩码后的文本特征 [N, D] (在 batch 中按需重复)
        :param visual_features: 视觉特征 [B, C, H, W]
        :return: 恢复后的文本特征 [N, D]
        """
        B, C, H, W = visual_features.shape
        N, D = masked_text_features.shape  # N个文本特征，每个文本特征是D维

        # Step 1: 将视觉特征投影到文本特征空间
        visual_features_proj = self.visual_proj(visual_features)  # [B, D, H, W]

        # Step 2: 将文本特征（[N, D]）扩展到与视觉特征相同的空间，以便计算相似性
        # 扩展文本特征到 [B, N, D, H, W]，每个文本描述的特征在每个位置上共享
        text_features_expanded = self.text_embeddings.view(1, N, D, 1, 1).expand(B, -1, -1, H, W)  # [B, N, D, H, W]

        # Step 3: 计算文本和视觉特征之间的相似性
        visual_features_expanded = visual_features_proj.view(B, 1, D, H, W).expand(-1, N, -1, -1, -1)  # [B, N, D, H, W]

        # 使用点积计算相似性，得到每个文本描述与视觉特征的匹配度
        similarity_map = torch.einsum('bnchw,bndhw->bnchw', text_features_expanded, visual_features_expanded)  # [B, N, H, W]

        # Step 4: 使用 Softmax 归一化相似度映射，得到 attention map
        attention_map = F.softmax(similarity_map.view(B, -1), dim=-1).view(B, N, H, W)  # [B, N, H, W]

        # Step 5: 恢复文本特征
        # 使用 attention map 来调整掩码文本特征
        # 在文本特征恢复的过程中，视觉特征通过相似性权重影响恢复的强度
        recovered_text_features = masked_text_features.view(1, N, D, 1, 1).expand(B, -1, -1, H, W)  # [B, N, D, H, W]
        
        # 将视觉特征通过注意力调整后的权重与文本特征进行交互
        combined_features = torch.cat([recovered_text_features, visual_features_proj], dim=1)  # [B, 2D, H, W]
        interaction_features = self.interaction_layer(combined_features)  # [B, D, H, W]

        # 聚合得到恢复的文本特征 [N, D]
        final_text_features = interaction_features.mean(dim=(2, 3))  # [B, D] -> 聚合空间维度
        final_text_features = final_text_features.view(N, D)  # [N, D]

        return final_text_features







# 示例用法
if __name__ == "__main__":
    # # 假设文本特征，N = 10, D = 512
    # text_features = torch.randn(10, 512)
    # masked_text, text_mask = mask_text(text_features, mask_prob=0.2)
    # print("原始文本特征：", text_features.shape)
    # print("掩盖后的文本特征：", masked_text.shape)
    # # print("文本掩码矩阵：", text_mask)

    # # 假设图像特征，B = 2, C = 256, H = 32, W = 32
    # image_features = torch.randn(2, 256, 32, 32)
    # masked_image, image_mask = mask_image(image_features, mask_prob=0.3, block_size=4)
    # print("原始图像特征：", image_features.shape)
    # print("掩盖后的图像特征：", masked_image.shape)
    # # print("图像掩码矩阵：", image_mask)

    # # 恢复文本特征
    model = TextRecoveryWithVisualContext(text_dim=512, visual_dim=256, hidden_dim=512, num_text_descriptions=10)
    text_features = torch.randn(10, 512)
    visual_features = torch.randn(1, 256, 64, 64)
    recovered_text_features = model(text_features, visual_features)
