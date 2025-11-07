"""
Skeleton-Aware Tokenizer for Human Pose Point Clouds
基于原始dVAE架构的骨架专用分词器

回归原始dVAE设计:
1. 使用FPS+KNN分组策略
2. Gumbel Softmax量化
3. 简单有效的损失函数 (重建损失 + KL散度)
4. 成熟稳定的训练流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .build import MODELS
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils.logger import *
from utils import misc

# 导入原始dVAE的核心组件
try:
    from knn_cuda import KNN
    knn = KNN(k=4, transpose_mode=False)
except ImportError:
    print("Warning: knn_cuda not available, using PyTorch fallback")
    knn = None

# KNN工具函数 (从dvae.py复制)
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


# 原始dVAE的分组类 (从dvae.py复制并适配)
class Group(nn.Module):
    """
    原始dVAE的分组策略
    将输入点云(B×N×3)划分为多个局部点云组(B×G×M×3)
    通过FPS(最远点采样)选择G个中心点,使用KNN为每个中心点找到K个最近邻
    中心化处理，使每个组的中心点为原点
    """
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


# 原始dVAE的编码器 (从dvae.py复制)
class Encoder(nn.Module):
    """
    将局部点云组编码为特征向量
    """
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


# 原始dVAE的DGCNN (从dvae.py复制)
class DGCNN(nn.Module):
    def __init__(self, encoder_channel, output_channel):
        super().__init__()
        self.input_trans = nn.Conv1d(encoder_channel, 128, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 256),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 512),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 1024),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer5 = nn.Sequential(nn.Conv1d(2304, output_channel, kernel_size=1, bias=False),
                                nn.GroupNorm(4, output_channel),
                                nn.LeakyReLU(negative_slope=0.2)
                                )

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        """
        原始dVAE的图特征提取实现
        Args:
            coor_q: B 3 N # 全局特征的中心点
            x_q: B C N # 全局特征的点云特征
            coor_k: B 3 N # 局部特征的中心点
            x_k: B C N # 局部特征的点云特征
        """
        k = 4
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)

        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, f, coor):
        # f: B G C, coor: B G 3
        feature_list  = []
        coor = coor.transpose(1, 2).contiguous()         # B 3 N
        f = f.transpose(1, 2).contiguous()               # B C N
        f = self.input_trans(f)             # B 128 N

        f = self.get_graph_feature(coor, f, coor, f) # B 256 N k
        f = self.layer1(f)                           # B 256 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 256 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f) # B 512 N k
        f = self.layer2(f)                           # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 512 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f) # B 1024 N k
        f = self.layer3(f)                           # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 512 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f) # B 1024 N k
        f = self.layer4(f)                           # B 1024 N k
        f = f.max(dim=-1, keepdim=False)[0]          # B 1024 N
        feature_list.append(f)

        f = torch.cat(feature_list, dim=1) # B 2304 N
        f = self.layer5(f)                 # B output_channel N
        f = f.transpose(1, 2).contiguous() # B N output_channel
        return f


# 原始dVAE的解码器 (从dvae.py复制)
class Decoder(nn.Module):
    """
    将编码后的特征向量解码为粗糙点云和精细点云
    """
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2
        self.num_coarse = self.num_fine // 4
        assert num_fine % 4 == 0

        self.mlp = nn.Sequential(
            nn.Linear(encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(encoder_channel + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2)

    def forward(self, feature_global):
        bs, g, c = feature_global.shape
        feature_global = feature_global.reshape(bs * g, c)

        coarse = self.mlp(feature_global).reshape(bs * g, self.num_coarse, 3)

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size**2, -1)
        point_feat = point_feat.reshape(bs * g, self.num_fine, 3).transpose(2, 1)

        seed = self.folding_seed.unsqueeze(2).expand(bs * g, -1, self.num_coarse, -1)
        seed = seed.reshape(bs * g, -1, self.num_fine).to(feature_global.device)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_fine)
        feat = torch.cat([feature_global, point_feat, seed], dim=1)

        fine = self.final_conv(feat) + point_feat
        fine = fine.transpose(1, 2).contiguous()

        coarse = coarse.reshape(bs, g, self.num_coarse, 3)
        fine = fine.reshape(bs, g, self.num_fine, 3)

        return coarse, fine


@MODELS.register_module()
class SkeletonTokenizer(nn.Module):
    """
    基于原始dVAE架构的骨架分词器

    回归简单有效的设计:
    1. 使用原始dVAE的所有组件
    2. 只保留重建损失和KL散度损失
    3. 稳定的Gumbel Softmax量化
    4. 成熟的训练流程
    """
    def __init__(self, config, **kwargs):
        super().__init__()

        # 从配置中读取参数 (兼容不同的配置格式)
        self.num_group = getattr(config, 'num_group', 8)
        self.group_size = getattr(config, 'group_size', 90)
        self.num_tokens = getattr(config, 'codebook_size', 512)
        self.encoder_dims = getattr(config, 'encoder_dims', 384)
        self.tokens_dims = getattr(config, 'tokens_dims', 256)
        self.decoder_dims = getattr(config, 'decoder_dims', 384)

        # 计算目标点数
        self.target_points = self.num_group * self.group_size

        print_log(f'[SkeletonTokenizer] 回归原始dVAE架构', logger='SkeletonTokenizer')
        print_log(f'[SkeletonTokenizer] 分组数: {self.num_group}, 每组点数: {self.group_size}', logger='SkeletonTokenizer')
        print_log(f'[SkeletonTokenizer] 目标点数: {self.target_points}', logger='SkeletonTokenizer')
        print_log(f'[SkeletonTokenizer] 码本大小: {self.num_tokens}', logger='SkeletonTokenizer')
        print_log(f'[SkeletonTokenizer] 编码器维度: {self.encoder_dims}', logger='SkeletonTokenizer')

        # 原始dVAE的核心组件
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.dgcnn_1 = DGCNN(encoder_channel=self.encoder_dims, output_channel=self.num_tokens)
        self.codebook = nn.Parameter(torch.randn(self.num_tokens, self.tokens_dims))
        self.dgcnn_2 = DGCNN(encoder_channel=self.tokens_dims, output_channel=self.decoder_dims)
        self.decoder = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size)

        # 损失函数 (只保留必要的)
        self.loss_func_cdl1 = ChamferDistanceL1()

        print_log(f'[SkeletonTokenizer] 初始化完成 - 使用原始dVAE架构', logger='SkeletonTokenizer')

    def forward(self, inp, temperature=1., hard=False, **kwargs):
        """
        前向传播 - 完全按照原始dVAE的流程
        """
        # 1. 分组
        neighborhood, center = self.group_divider(inp)

        # 2. 编码
        logits = self.encoder(neighborhood)   # B G C

        # 3. 量化 (使用Gumbel Softmax)
        logits = self.dgcnn_1(logits, center) # B G N
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=2, hard=hard)
        sampled = torch.einsum('b g n, n c -> b g c', soft_one_hot, self.codebook)

        # 4. 解码
        feature = self.dgcnn_2(sampled, center)
        coarse, fine = self.decoder(feature)

        # 5. 重建完整点云 (按照原始dVAE的方式)
        with torch.no_grad():
            whole_fine = (fine + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse = (coarse + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)

        assert fine.size(2) == self.group_size
        ret = (whole_coarse, whole_fine, coarse, fine, neighborhood, logits)

        return ret

    def recon_loss(self, ret, gt):
        """
        重建损失计算 - 修复维度不匹配问题
        处理coarse(20点)与group_gt(80点)的维度差异
        """
        whole_coarse, whole_fine, coarse, fine, group_gt, _ = ret

        bs, g, _, _ = coarse.shape

        # 重塑tensors
        coarse = coarse.reshape(bs*g, -1, 3).contiguous()  # [B*G, num_coarse, 3]
        fine = fine.reshape(bs*g, -1, 3).contiguous()      # [B*G, group_size, 3]
        group_gt = group_gt.reshape(bs*g, -1, 3).contiguous()  # [B*G, group_size, 3]

        # 解决方案1: 只使用fine进行损失计算，coarse作为中间表示
        # 这符合原始dVAE设计：coarse是粗糙重建，fine是精细重建
        try:
            loss_fine_block = self.loss_func_cdl1(fine, group_gt)
            # 对于coarse，我们采样group_gt到相应大小进行比较
            num_coarse = coarse.shape[1]
            if num_coarse < group_gt.shape[1]:
                # 随机采样group_gt到coarse大小
                indices = torch.randperm(group_gt.shape[1], device=group_gt.device)[:num_coarse]
                group_gt_sampled = group_gt[:, indices, :]
                loss_coarse_block = self.loss_func_cdl1(coarse, group_gt_sampled)
            else:
                loss_coarse_block = self.loss_func_cdl1(coarse, group_gt)
            
            loss_recon = loss_coarse_block + loss_fine_block
            
        except Exception as e:
            print_log(f'[SkeletonTokenizer] ChamferDistance失败，使用L2损失: {e}', logger='SkeletonTokenizer')
            # 备用方案：使用MSE损失
            loss_fine_block = F.mse_loss(fine, group_gt)
            
            # 对coarse使用采样后的group_gt
            num_coarse = coarse.shape[1]
            if num_coarse < group_gt.shape[1]:
                indices = torch.randperm(group_gt.shape[1], device=group_gt.device)[:num_coarse]
                group_gt_sampled = group_gt[:, indices, :]
                loss_coarse_block = F.mse_loss(coarse, group_gt_sampled)
            else:
                loss_coarse_block = F.mse_loss(coarse, group_gt)
                
            loss_recon = loss_coarse_block + loss_fine_block

        return loss_recon

    def get_loss(self, ret, gt):
        """
        损失计算 - 完全按照原始dVAE的方式
        只保留重建损失和KL散度损失
        """
        # 1. 重建损失
        loss_recon = self.recon_loss(ret, gt)

        # 2. KL散度损失 (原始dVAE方式)
        logits = ret[-1]  # B G N
        softmax = F.softmax(logits, dim=-1)
        mean_softmax = softmax.mean(dim=1)
        log_qy = torch.log(mean_softmax + 1e-8)
        log_uniform = torch.log(torch.tensor([1. / self.num_tokens], device=gt.device))
        loss_klv = F.kl_div(log_qy, log_uniform.expand(log_qy.size(0), log_qy.size(1)),
                           None, None, 'batchmean', log_target=True)

        return loss_recon, loss_klv