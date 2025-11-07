"""
时空图卷积骨架Tokenizer
基于25个关节点的直接处理，使用语义分组和时空图卷积
不再需要720点的密集化扩展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 导入模型注册器
try:
    from .build import MODELS
except ImportError:
    # 如果作为独立脚本运行，创建一个简单的注册器
    class SimpleRegistry:
        def register_module(self):
            def decorator(cls):
                return cls
            return decorator
    MODELS = SimpleRegistry()

class SkeletonGraph:
    """NTU RGB+D 25关节点的骨架图结构定义"""
    
    def __init__(self):
        # NTU RGB+D 25关节点连接关系 (修正后的正确连接顺序)
        self.skeleton_edges = [
            # 头部和脊柱连接
            (3, 2),   # 头顶点 → 颈椎上段
            (2, 20),  # 颈椎上段 → 锁骨/肩关节区域
            (20, 1),  # 锁骨/肩关节区域 → 胸椎段
            (1, 0),   # 胸椎段 → 腰椎/骨盆衔接处
            
            # 左臂连接 (从肩膀到手)
            (20, 4),  # 锁骨/肩关节区域 → 左上臂关节
            (4, 5),   # 左上臂关节 → 左肘关节
            (5, 6),   # 左肘关节 → 左腕关节
            (6, 22),  # 左腕关节 → 左手指近端关节
            (6, 7),   # 左腕关节 → 左手掌关节
            (7, 21),  # 左手掌关节 → 左手指远端关节
            
            # 右臂连接 (从肩膀到手)
            (20, 8),  # 锁骨/肩关节区域 → 右上臂关节
            (8, 9),   # 右上臂关节 → 右肘关节
            (9, 10),  # 右肘关节 → 右腕关节
            (10, 24), # 右腕关节 → 右手指近端关节
            (10, 11), # 右腕关节 → 右手掌关节
            (11, 23), # 右手掌关节 → 右手指远端关节
            
            # 左腿连接 (从骨盆到脚)
            (0, 12),  # 腰椎/骨盆衔接处 → 左髋关节
            (12, 13), # 左髋关节 → 左膝关节
            (13, 14), # 左膝关节 → 左踝关节
            (14, 15), # 左踝关节 → 左脚趾近端关节
            
            # 右腿连接 (从骨盆到脚)
            (0, 16),  # 腰椎/骨盆衔接处 → 右髋关节
            (16, 17), # 右髋关节 → 右膝关节
            (17, 18), # 右膝关节 → 右踝关节
            (18, 19), # 右踝关节 → 右脚趾远端关节
        ]
        
        # 转换为0-based索引
        self.edges = [(i-1, j-1) for i, j in self.skeleton_edges]
        
        # 细粒度语义分组：5个主要身体区域（左右分离）
        self.semantic_groups = {
            'head_spine': [0, 1, 2, 3, 20],               # 头部+脊柱 (5个关节)
            'left_arm': [4, 5, 6, 7, 21, 22],             # 左臂+左手 (6个关节)
            'right_arm': [8, 9, 10, 11, 23, 24],          # 右臂+右手 (6个关节)
            'left_leg': [12, 13, 14, 15],                 # 左腿 (4个关节)
            'right_leg': [16, 17, 18, 19]                 # 右腿 (4个关节)
        }
        
        self.num_joints = 25
        self.adjacency_matrix = self._build_adjacency_matrix()
        
    def _build_adjacency_matrix(self):
        """构建邻接矩阵"""
        adj = torch.zeros(self.num_joints, self.num_joints)
        
        # 添加边连接
        for i, j in self.edges:
            if 0 <= i < self.num_joints and 0 <= j < self.num_joints:
                adj[i, j] = 1
                adj[j, i] = 1  # 无向图
        
        # 添加自连接
        adj += torch.eye(self.num_joints)
        
        # 归一化
        degree = adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # 避免除零
        adj = adj / degree
        
        return adj

class ST_GCN_Layer(nn.Module):
    """时空图卷积层"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        
        # 空间图卷积
        self.spatial_gcn = nn.Conv2d(in_channels, out_channels * kernel_size, 1)
        
        # 时间卷积
        padding = (kernel_size - 1) // 2
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, 
                                     (kernel_size, 1), (stride, 1), (padding, 0))
        
        # 残差连接
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_matrix):
        """
        x: (B, C, T, V) - Batch, Channels, Time, Vertices(joints)
        adj_matrix: (V, V) - 邻接矩阵
        """
        residual = self.residual(x)

        # 空间图卷积
        B, C, T, V = x.size()

        # 先应用图卷积，再应用空间卷积
        # 重塑为图卷积格式
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()  # (B, T, V, C)
        x_reshaped = x_reshaped.view(B*T, V, C)

        # 应用图卷积 A * X
        x_gcn = torch.matmul(adj_matrix.to(x.device), x_reshaped)  # (B*T, V, C)

        # 恢复原始形状
        x_gcn = x_gcn.view(B, T, V, C).permute(0, 3, 1, 2)  # (B, C, T, V)

        # 应用空间卷积
        x = self.spatial_gcn(x_gcn)  # (B, out_channels*K, T, V)

        # 重新整形以分离kernel维度
        out_channels = x.size(1) // self.kernel_size
        x = x.view(B, self.kernel_size, out_channels, T, V)

        # 聚合不同kernel的结果
        x = x.sum(dim=1)  # (B, out_channels, T, V)

        # 时间卷积
        x = self.temporal_conv(x)

        # 残差连接和归一化
        x = self.bn(x + residual)
        x = self.relu(x)
        x = self.dropout(x)

        return x

class SemanticGroupProcessor(nn.Module):
    """语义分组处理器"""
    
    def __init__(self, group_joints, in_channels, out_channels):
        super().__init__()
        self.group_joints = group_joints
        self.num_joints = len(group_joints)
        
        # 为该分组构建子图邻接矩阵
        self.register_buffer('sub_adj', self._build_sub_adjacency())
        
        # ST-GCN层
        self.st_gcn_layers = nn.ModuleList([
            ST_GCN_Layer(in_channels, 64),
            ST_GCN_Layer(64, 128), 
            ST_GCN_Layer(128, out_channels)
        ])
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _build_sub_adjacency(self):
        """为当前分组构建子图邻接矩阵"""
        # 创建完整骨架图
        skeleton_graph = SkeletonGraph()
        full_adj = skeleton_graph.adjacency_matrix

        # 提取子图，保持连接性
        sub_adj = full_adj[self.group_joints][:, self.group_joints].clone()

        # 确保子图连通性：如果子图不连通，添加必要的连接
        num_joints = len(self.group_joints)

        # 检查连通性并修复
        if num_joints > 1:
            # 添加自连接
            sub_adj += torch.eye(num_joints)

            # 如果某些节点没有连接，连接到第一个节点（作为根节点）
            degree = sub_adj.sum(dim=1)
            isolated_nodes = (degree == 1).nonzero(as_tuple=True)[0]  # 只有自连接的节点

            for node in isolated_nodes:
                if node != 0:  # 不是根节点
                    sub_adj[0, node] = 1
                    sub_adj[node, 0] = 1

        # 重新归一化
        degree = sub_adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1  # 避免除零
        sub_adj = sub_adj / degree

        return sub_adj
        
    def forward(self, x):
        """
        x: (B, C, T, V) 其中V是该分组的关节数
        """
        for layer in self.st_gcn_layers:
            x = layer(x, self.sub_adj)
        
        # 全局特征聚合
        features = self.global_pool(x)  # (B, C, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, C)
        
        return features

class VectorQuantizer(nn.Module):
    """向量量化模块（码本）"""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.9):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # 码本
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs):
        """
        inputs: (B, embedding_dim)
        """
        # 计算距离
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self.embedding.weight.t()))
        
        # 找到最近的码字
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings,
                               device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight)

        # 损失计算
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()

        # 确保返回的indices是1D张量
        indices = encoding_indices.squeeze(1)  # 从(B,1)变为(B,)
        if indices.dim() == 0:  # 如果是标量，转为1D张量
            indices = indices.unsqueeze(0)

        return quantized, loss, indices

class IndependentSemanticCodebooks(nn.Module):
    """独立语义组码本"""

    def __init__(self, semantic_groups, tokens_per_group=128, token_dim=64):
        super().__init__()

        self.semantic_groups = semantic_groups
        self.tokens_per_group = tokens_per_group
        self.token_dim = token_dim

        # 为每个语义组创建独立码本
        self.group_codebooks = nn.ModuleDict()
        for group_name in semantic_groups.keys():
            self.group_codebooks[group_name] = VectorQuantizer(
                num_embeddings=tokens_per_group,
                embedding_dim=token_dim,
                commitment_cost=0.25
            )

    def forward(self, group_features_dict):
        """
        group_features_dict: {group_name: (B, token_dim)}
        返回: {group_name: (quantized, loss, indices)}
        """
        results = {}
        total_loss = 0.0

        for group_name, features in group_features_dict.items():
            if group_name in self.group_codebooks:
                quantized, loss, indices = self.group_codebooks[group_name](features)
                results[group_name] = {
                    'quantized': quantized,
                    'loss': loss,
                    'indices': indices
                }
                total_loss += loss

        return results, total_loss

    def get_token_sequence(self, group_results):
        """
        将各组的token索引组合成序列，用于LLM
        返回: (B, num_groups) 的token序列
        """
        batch_size = None
        token_sequence = []
        device = None

        # 按固定顺序排列各组token
        group_order = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']

        # 首先确定batch_size和device
        for group_name in group_order:
            if group_name in group_results:
                indices = group_results[group_name]['indices']

                # 确保indices是张量且至少是1D
                if not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(indices)
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)

                if batch_size is None:
                    batch_size = indices.size(0)
                    device = indices.device
                break

        # 如果没有找到有效的组结果，返回空张量
        if batch_size is None:
            return torch.empty(0, len(group_order))

        # 处理每个组的token
        for group_name in group_order:
            if group_name in group_results:
                indices = group_results[group_name]['indices']

                # 确保indices是正确格式的张量
                if not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(indices, device=device)
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)

                # 确保batch维度匹配
                if indices.size(0) != batch_size:
                    if indices.size(0) == 1:
                        indices = indices.expand(batch_size)
                    else:
                        # 如果维度不匹配，用第一个值填充
                        indices = torch.full((batch_size,), indices[0].item(), device=device)

                # 为每个组添加偏移量，确保token ID不重复
                group_offset = group_order.index(group_name) * self.tokens_per_group
                offset_indices = indices + group_offset
                token_sequence.append(offset_indices)
            else:
                # 如果某组缺失，用特殊token填充
                padding_token = torch.full((batch_size,), -1, device=device)
                token_sequence.append(padding_token)

        if token_sequence:
            return torch.stack(token_sequence, dim=1)  # (B, num_groups)
        else:
            return torch.empty(0, len(group_order))

@MODELS.register_module()
class GCNSkeletonTokenizer(nn.Module):
    """
    基于时空图卷积的骨架Tokenizer
    直接处理25个关节点，使用语义分组
    """
    
    def __init__(self, config=None, **kwargs):
        super().__init__()
        
        # 处理配置参数 - 兼容不同的参数传递方式
        if config is not None:
            # 如果传递了config对象
            if hasattr(config, '__dict__'):
                config_dict = config.__dict__
            else:
                config_dict = config
        else:
            config_dict = {}
        
        # 合并所有参数
        params = {**config_dict, **kwargs}
        
        self.num_tokens = params.get('num_tokens', 512)
        self.token_dim = params.get('token_dim', 256)
        self.temporal_length = params.get('temporal_length', 1)
        
        # 骨架图结构
        self.skeleton_graph = SkeletonGraph()
        
        # 输入嵌入
        self.input_embedding = nn.Linear(3, 64)  # xyz坐标 -> 特征维度
        
        # 语义分组处理器 - 统一输出维度
        self.group_processors = nn.ModuleDict()
        group_token_dim = 64  # 每个组的token维度

        for group_name, joints in self.skeleton_graph.semantic_groups.items():
            self.group_processors[group_name] = SemanticGroupProcessor(
                joints, 64, group_token_dim
            )

        # 独立语义组码本
        self.semantic_codebooks = IndependentSemanticCodebooks(
            semantic_groups=self.skeleton_graph.semantic_groups,
            tokens_per_group=128,  # 每个组128个token
            token_dim=group_token_dim
        )

        # 全局特征融合（用于重构）
        total_group_dim = len(self.skeleton_graph.semantic_groups) * group_token_dim
        self.global_fusion = nn.Sequential(
            nn.Linear(total_group_dim, self.token_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, self.token_dim)
        )
        
        # 增强重构头 - 多层深度网络，更好的特征提取能力
        self.reconstruction_head = nn.Sequential(
            # 第一阶段：特征扩展
            nn.Linear(self.token_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第二阶段：特征细化
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第三阶段：空间感知
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            # 第四阶段：关节重构
            nn.Linear(256, 25 * 3)  # 重建25个关节点的xyz坐标
        )
        
        # 残差连接控制：通过sigmoid门限限制残差的放大
        # 初始值设为 -5.0 让初期训练几乎完全依赖码本
        self.residual_gate = nn.Parameter(torch.tensor(-5.0))  # sigmoid(-5) ≈ 0.007
        self.residual_reg_weight = params.get('residual_reg_weight', 0.1)  # 提高到0.1
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        
    def forward(self, inp=None, temperature=1.0, hard=False, return_recon=True, skeleton_data=None, eval=False, **kwargs):
        """
        inp: (B, 25, 3) 或 (B, T, 25, 3) - 25个关节点的xyz坐标（主要输入参数，与原框架一致）
        skeleton_data: 兼容参数名，等同于inp
        temperature: 温度参数（为了兼容原框架，在GCN中不使用）
        hard: 硬采样标志（为了兼容原框架，在GCN中不使用）
        eval: 验证模式标志（为了兼容原框架）
        """
        # 兼容不同的输入参数名（优先使用inp以匹配原框架）
        if inp is not None:
            skeleton_data = inp
        elif skeleton_data is not None:
            pass  # 使用skeleton_data
        else:
            raise ValueError("Either inp or skeleton_data must be provided")

        # 处理不同的输入维度
        if skeleton_data.dim() == 3:
            # (B, 25, 3) -> (B, 1, 25, 3)
            skeleton_data = skeleton_data.unsqueeze(1)
        elif skeleton_data.dim() == 4:
            # (B, T, 25, 3) 已经是正确格式
            pass
        else:
            raise ValueError(f"Expected 3D or 4D input, got {skeleton_data.dim()}D")

        batch_size, temporal_length, num_joints, coord_dim = skeleton_data.shape

        # 输入嵌入：处理每一帧
        x = skeleton_data.reshape(batch_size * temporal_length, num_joints, coord_dim)
        x = self.input_embedding(x)  # (B*T, 25, 64)
        x = x.reshape(batch_size, temporal_length, num_joints, -1)

        # 为时空卷积准备数据格式 (B, C, T, V)
        x = x.permute(0, 3, 1, 2)  # (B, 64, T, 25)
        
        # 分组处理
        group_features_dict = {}
        for group_name, joints in self.skeleton_graph.semantic_groups.items():
            # 提取该分组的关节特征
            group_x = x[:, :, :, joints]  # (B, 64, T, num_group_joints)

            # 通过分组处理器
            group_feat = self.group_processors[group_name](group_x)
            group_features_dict[group_name] = group_feat

        # 独立语义组量化
        group_results, total_vq_loss = self.semantic_codebooks(group_features_dict)

        # 全局特征融合（用于重构）
        group_quantized_features = []
        for group_name in self.skeleton_graph.semantic_groups.keys():
            if group_name in group_results:
                group_quantized_features.append(group_results[group_name]['quantized'])

        combined_features = torch.cat(group_quantized_features, dim=1)  # (B, total_group_dim)
        global_features = self.global_fusion(combined_features)

        if return_recon:
            input_flat = skeleton_data.reshape(batch_size, temporal_length, -1)  # (B, T, 75)
            base_recon_flat = self.reconstruction_head(global_features)  # (B, 75)

            if temporal_length > 1:
                base_recon_flat = base_recon_flat.unsqueeze(1).repeat(1, temporal_length, 1)  # (B, T, 75)
            else:
                base_recon_flat = base_recon_flat.unsqueeze(1)  # (B, 1, 75)

            residual_scale = torch.sigmoid(self.residual_gate)
            final_flat = base_recon_flat + residual_scale * (input_flat - base_recon_flat)

            final_reconstructed = final_flat.reshape(batch_size, temporal_length, 25, 3)
            base_reconstructed = base_recon_flat.reshape(batch_size, temporal_length, 25, 3)

            if temporal_length == 1:
                final_reconstructed = final_reconstructed.squeeze(1)
                base_reconstructed = base_reconstructed.squeeze(1)
                original = skeleton_data.squeeze(1)
            else:
                original = skeleton_data

            # 如果是验证模式，返回兼容原框架的元组格式
            if eval or hard:
                # 返回 (coarse_points, fine_points, ...) 格式以兼容验证代码
                # 对于骨架数据，coarse和fine都返回重建的骨架
                return (final_reconstructed, final_reconstructed, None, None, None, None)

            # 只在非验证模式下生成token序列
            token_sequence = self.semantic_codebooks.get_token_sequence(group_results)

            return {
                'quantized': global_features,
                'reconstructed': final_reconstructed,
                'vq_loss': total_vq_loss,
                'token_sequence': token_sequence,
                'group_results': group_results,
                'original': original,
                'base_reconstructed': base_reconstructed,
                'residual_scale': residual_scale
            }
        else:
            # 如果是验证模式但不需要重建，返回原始数据
            if eval or hard:
                return (skeleton_data, skeleton_data, None, None, None, None)

            # 只在非验证模式下生成token序列
            token_sequence = self.semantic_codebooks.get_token_sequence(group_results)

            return {
                'quantized': global_features,
                'vq_loss': total_vq_loss,
                'token_sequence': token_sequence,
                'group_results': group_results
            }
    
    def get_loss(self, ret, gt):
        """计算总损失 - 与框架接口一致"""
        # ret 是 forward 的输出，gt 是真实骨架数据
        forward_output = ret
        vq_loss = forward_output['vq_loss']  # 码本量化损失
        
        if 'reconstructed' in forward_output:
            reconstructed = forward_output['reconstructed']
            original = forward_output['original']
            
            # 重构损失（对应原始的loss_1）
            recon_loss = self.mse_loss(reconstructed, original)
            
            # 返回格式与框架一致：(重构损失, VQ损失)
            # 框架会通过kldweight配置来组合这两个损失
            residual_scale = forward_output.get('residual_scale')
            residual_reg = torch.tensor(0.0, device=reconstructed.device)
            if residual_scale is not None:
                residual_reg = self.residual_reg_weight * (residual_scale ** 2)

            total_loss = recon_loss + residual_reg
            return total_loss, vq_loss
        else:
            # 如果没有重构，返回零重构损失和VQ损失
            return torch.tensor(0.0, device=vq_loss.device), vq_loss
    
    def encode(self, skeleton_data):
        """编码：骨架 -> 离散token序列"""
        with torch.no_grad():
            output = self.forward(skeleton_data, return_recon=False)
            return output['token_sequence']  # (B, num_groups)

    def decode(self, token_sequence):
        """解码：离散token序列 -> 骨架"""
        with torch.no_grad():
            batch_size = token_sequence.size(0)
            group_order = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']

            # 从token序列重建各组特征
            group_features = []
            for i, group_name in enumerate(group_order):
                if i < token_sequence.size(1):
                    # 移除偏移量获取原始token索引
                    group_offset = i * self.semantic_codebooks.tokens_per_group
                    original_indices = token_sequence[:, i] - group_offset

                    # 从对应组的码本获取特征
                    if group_name in self.semantic_codebooks.group_codebooks:
                        quantized_features = self.semantic_codebooks.group_codebooks[group_name].embedding(original_indices)
                        group_features.append(quantized_features)

            # 融合特征
            if group_features:
                combined_features = torch.cat(group_features, dim=1)
                global_features = self.global_fusion(combined_features)

                base = self.reconstruction_head(global_features)
                reconstructed = base.reshape(batch_size, 25, 3)

                return reconstructed
            else:
                return torch.zeros(batch_size, 25, 3)

    # ----- 诊断 / 工具方法 -----
    def analyze_codebook_usage(self, skeleton_data, topk=10, device='cpu'):
        """
        分析一批输入上的码本使用情况与残差贡献。

        返回一个字典，包含每个语义组的token频次(hist)、总体vq_loss均值、
        以及残差分量范数的统计（mean/std）。用于训练/导出后快速诊断码本是否collapse。
        """
        was_training = self.training
        self.eval()

        if isinstance(skeleton_data, np.ndarray):
            skeleton_data = torch.tensor(skeleton_data, dtype=torch.float32)

        if skeleton_data.dim() == 3:
            skeleton_data = skeleton_data.unsqueeze(1)

        skeleton_data = skeleton_data.to(device)

        with torch.no_grad():
            out = self.forward(skeleton_data, return_recon=True)

        # group_results 是字典
        group_results = out.get('group_results', {})
        usage = {}
        for gname, gres in group_results.items():
            indices = gres['indices']
            if isinstance(indices, torch.Tensor):
                arr = indices.detach().cpu().numpy().ravel()
            else:
                arr = np.array(indices).ravel()
            # 统计前 topk 出现的token
            vals, counts = np.unique(arr, return_counts=True)
            order = np.argsort(-counts)
            top_vals = vals[order][:topk] if len(order) > 0 else np.array([])
            top_counts = counts[order][:topk] if len(order) > 0 else np.array([])
            usage[gname] = {
                'unique_tokens': int(len(vals)),
                'topk_tokens': top_vals.tolist(),
                'topk_counts': top_counts.tolist(),
                'total': int(arr.size)
            }

        vq_loss = out.get('vq_loss')
        vq_loss_val = float(vq_loss.detach().cpu().mean()) if isinstance(vq_loss, torch.Tensor) else (float(vq_loss) if vq_loss is not None else None)

        residual_norms = None
        residual_scale = out.get('residual_scale')
        base_recon = out.get('base_reconstructed')
        final_recon = out.get('reconstructed')

        if base_recon is not None and final_recon is not None:
            base_np = base_recon.detach().cpu().numpy()
            final_np = final_recon.detach().cpu().numpy()
            residual_component = final_np - base_np
            if residual_component.ndim == 4:  # (B, T, 25, 3)
                flat = residual_component.reshape(residual_component.shape[0], -1)
            else:  # (B, 25, 3)
                flat = residual_component.reshape(residual_component.shape[0], -1)
            norms = np.linalg.norm(flat, axis=1)
            residual_norms = {
                'mean': float(np.mean(norms)),
                'std': float(np.std(norms)),
                'min': float(np.min(norms)),
                'max': float(np.max(norms))
            }
        residual_scale_val = None
        if isinstance(residual_scale, torch.Tensor):
            residual_scale_val = float(residual_scale.detach().cpu().item())

        if was_training:
            self.train()

        return {
            'usage': usage,
            'vq_loss_mean': vq_loss_val,
            'residual_norms': residual_norms,
            'residual_scale': residual_scale_val
        }

def create_gcn_skeleton_tokenizer(config=None, **kwargs):
    """创建GCN骨架Tokenizer的工厂函数"""
    
    # 默认配置
    default_config = {
        'num_tokens': 512,
        'token_dim': 256,
        'temporal_length': 1
    }
    
    if config:
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = config
        default_config.update(config_dict)
    
    default_config.update(kwargs)
    
    model = GCNSkeletonTokenizer(config=default_config)
    
    print(f"Created GCNSkeletonTokenizer with Independent Semantic Codebooks:")
    print(f"  - Global tokens: {default_config['num_tokens']}")
    print(f"  - Token dim: {default_config['token_dim']}")
    print(f"  - Semantic groups: {len(model.skeleton_graph.semantic_groups)}")
    print(f"  - Tokens per group: {model.semantic_codebooks.tokens_per_group}")
    print(f"  - Total vocabulary size: {len(model.skeleton_graph.semantic_groups) * model.semantic_codebooks.tokens_per_group}")
    print(f"  - Group names: {list(model.skeleton_graph.semantic_groups.keys())}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

if __name__ == "__main__":
    # 测试代码
    model = create_gcn_skeleton_tokenizer()
    
    # 模拟输入：批次大小为4，25个关节点，xyz坐标
    test_input = torch.randn(4, 25, 3)
    
    print(f"Input shape: {test_input.shape}")
    
    # 前向传播
    output = model(test_input)
    
    print(f"Quantized features shape: {output['quantized'].shape}")
    print(f"Reconstructed shape: {output['reconstructed'].shape}")
    print(f"Token sequence shape: {output['token_sequence'].shape}")
    print(f"Number of semantic groups: {len(output['group_results'])}")

    # 显示各组的token信息
    for group_name, group_result in output['group_results'].items():
        print(f"  {group_name}: token indices shape {group_result['indices'].shape}")

    # 计算损失
    total_loss, recon_loss = model.get_loss(output, test_input)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"VQ loss: {output['vq_loss'].item():.4f}")

    # 测试编码解码
    token_sequence = model.encode(test_input)
    decoded_skeleton = model.decode(token_sequence)
    print(f"Encoded token sequence shape: {token_sequence.shape}")
    print(f"Decoded skeleton shape: {decoded_skeleton.shape}")
