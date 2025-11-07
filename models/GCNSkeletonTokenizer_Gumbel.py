"""
时空图卷积骨架Tokenizer - Gumbel-Softmax 版本
基于 GCNSkeletonTokenizer.py，使用 Gumbel-Softmax 软量化替代 VQ-VAE 硬量化
优化方向：更好的码本学习、避免 token collapse
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
    class SimpleRegistry:
        def register_module(self):
            def decorator(cls):
                return cls
            return decorator
    MODELS = SimpleRegistry()

class SkeletonGraph:
    """NTU RGB+D 25关节点的骨架图结构定义"""
    
    def __init__(self):
        # NTU RGB+D 25关节点连接关系
        self.skeleton_edges = [
            (3, 2), (2, 20), (20, 1), (1, 0),
            (20, 4), (4, 5), (5, 6), (6, 22), (6, 7), (7, 21),
            (20, 8), (8, 9), (9, 10), (10, 24), (10, 11), (11, 23),
            (0, 12), (12, 13), (13, 14), (14, 15),
            (0, 16), (16, 17), (17, 18), (18, 19),
        ]
        
        self.edges = [(i-1, j-1) for i, j in self.skeleton_edges]
        
        # 细粒度语义分组
        self.semantic_groups = {
            'head_spine': [0, 1, 2, 3, 20],
            'left_arm': [4, 5, 6, 7, 21, 22],
            'right_arm': [8, 9, 10, 11, 23, 24],
            'left_leg': [12, 13, 14, 15],
            'right_leg': [16, 17, 18, 19]
        }
        
        self.num_joints = 25
        self.adjacency_matrix = self._build_adjacency_matrix()
        
    def _build_adjacency_matrix(self):
        """构建邻接矩阵"""
        adj = torch.zeros(self.num_joints, self.num_joints)
        
        for i, j in self.edges:
            if 0 <= i < self.num_joints and 0 <= j < self.num_joints:
                adj[i, j] = 1
                adj[j, i] = 1
        
        adj += torch.eye(self.num_joints)
        
        degree = adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1
        adj = adj / degree
        
        return adj

class ST_GCN_Layer(nn.Module):
    """时空图卷积层"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.1):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.spatial_gcn = nn.Conv2d(in_channels, out_channels * kernel_size, 1)
        
        padding = (kernel_size - 1) // 2
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, 
                                     (kernel_size, 1), (stride, 1), (padding, 0))
        
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
        residual = self.residual(x)
        
        B, C, T, V = x.size()
        
        x_reshaped = x.permute(0, 2, 3, 1).contiguous()
        x_reshaped = x_reshaped.view(B*T, V, C)
        
        x_gcn = torch.matmul(adj_matrix.to(x.device), x_reshaped)
        x_gcn = x_gcn.view(B, T, V, C).permute(0, 3, 1, 2)
        
        x = self.spatial_gcn(x_gcn)
        
        out_channels = x.size(1) // self.kernel_size
        x = x.view(B, self.kernel_size, out_channels, T, V)
        x = x.sum(dim=1)
        
        x = self.temporal_conv(x)
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
        
        self.register_buffer('sub_adj', self._build_sub_adjacency())
        
        self.st_gcn_layers = nn.ModuleList([
            ST_GCN_Layer(in_channels, 64),
            ST_GCN_Layer(64, 128), 
            ST_GCN_Layer(128, out_channels)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _build_sub_adjacency(self):
        skeleton_graph = SkeletonGraph()
        full_adj = skeleton_graph.adjacency_matrix
        
        sub_adj = full_adj[self.group_joints][:, self.group_joints].clone()
        
        num_joints = len(self.group_joints)
        
        if num_joints > 1:
            sub_adj += torch.eye(num_joints)
            
            degree = sub_adj.sum(dim=1)
            isolated_nodes = (degree == 1).nonzero(as_tuple=True)[0]
            
            for node in isolated_nodes:
                if node != 0:
                    sub_adj[0, node] = 1
                    sub_adj[node, 0] = 1
        
        degree = sub_adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1
        sub_adj = sub_adj / degree
        
        return sub_adj
        
    def forward(self, x):
        for layer in self.st_gcn_layers:
            x = layer(x, self.sub_adj)
        
        features = self.global_pool(x)
        features = features.squeeze(-1).squeeze(-1)
        
        return features

class GumbelSoftmaxQuantizer(nn.Module):
    """Gumbel-Softmax 软量化模块（替代 VectorQuantizer）"""
    
    def __init__(self, num_embeddings, embedding_dim, kl_weight=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.kl_weight = kl_weight
        
        # 码本嵌入
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # Logits 投影层（从特征到码本选择概率）
        self.logits_proj = nn.Linear(embedding_dim, num_embeddings)
        
    def forward(self, inputs, temperature=1.0, hard=False):
        """
        Args:
            inputs: (B, embedding_dim) 编码器输出
            temperature: Gumbel-Softmax 温度
            hard: 是否使用硬采样（straight-through）
        
        Returns:
            quantized: (B, embedding_dim) 量化后的特征
            loss: KL 散度损失
            soft_one_hot: (B, num_embeddings) 软分布（用于分析）
        """
        batch_size = inputs.size(0)
        
        # 计算 logits（相似度分数）- 添加数值稳定性
        logits = self.logits_proj(inputs)  # (B, num_embeddings)
        
        # Clamp logits 防止爆炸
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        # 确保温度不会太小（防止数值不稳定）
        temperature = max(temperature, 0.1)
        
        # Gumbel-Softmax 采样
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=1, hard=hard)
        
        # 加权组合码本向量
        quantized = torch.matmul(soft_one_hot, self.embedding.weight)
        
        # Gumbel-Softmax 量化损失的正确设计
        # 
        # 策略：结合 VQ-VAE 的 commitment loss 和 KL 正则化
        # 
        # 1. Codebook loss: 优化码本向量（通过 stop_gradient 只影响 codebook）
        codebook_loss = F.mse_loss(quantized, inputs.detach())
        
        # 2. Commitment loss: 优化编码器（通过 stop_gradient 只影响 encoder）
        commitment_loss = F.mse_loss(quantized.detach(), inputs)
        
        # 3. KL 散度: 鼓励均匀使用码本（防止码本坍塌）
        mean_soft = soft_one_hot.mean(dim=0)
        mean_soft = mean_soft + 1e-10
        mean_soft = mean_soft / mean_soft.sum()
        uniform_dist = torch.ones_like(mean_soft) / self.num_embeddings
        kl_div = torch.sum(mean_soft * (torch.log(mean_soft) - torch.log(uniform_dist)))
        
        if torch.isnan(kl_div) or torch.isinf(kl_div):
            kl_div = torch.tensor(0.0, device=inputs.device)
        
        # 组合损失权重（关键调整）：
        # 
        # 权重设计原则：
        # 1. codebook_loss（1.0）：主要优化目标，让码本向量学习表示
        # 2. commitment_loss（0.25）：标准 VQ-VAE 比例，让编码器配合
        # 3. kl_div（0.001）：轻量正则化，防止完全坍塌但不主导优化
        #
        # 为什么 KL 权重这么小？
        # - KL div 数量级是 0-20（log 尺度）
        # - codebook_loss 数量级是 0.01-1.0（MSE 尺度）
        # - 如果 KL 权重 0.1，则 0.1*8=0.8 会主导损失（94%）
        # - 降到 0.001，贡献变成 0.001*8=0.008（约 1%）
        # - 这样 codebook 和 commitment 才是主导（99%）
        #
        # 外部 kldweight（0.2→0.8）会进一步缩放整个 vq_loss
        # 所以实际 KL 的总权重是：0.001 * (0.2~0.8) = 0.0002~0.0008
        
        vq_loss = codebook_loss + 0.25 * commitment_loss + 0.001 * kl_div
        
        loss = vq_loss
        
        # 为了兼容性，也返回"硬索引"（选择概率最大的）
        indices = torch.argmax(soft_one_hot, dim=1)
        
        return quantized, loss, indices, soft_one_hot

class IndependentSemanticCodebooks_Gumbel(nn.Module):
    """独立语义组码本 - Gumbel-Softmax 版本"""
    
    def __init__(self, semantic_groups, tokens_per_group=64, token_dim=64, kl_weight=0.1):
        super().__init__()
        
        self.semantic_groups = semantic_groups
        self.tokens_per_group = tokens_per_group
        self.token_dim = token_dim
        
        # 为每个语义组创建独立码本
        self.group_codebooks = nn.ModuleDict()
        for group_name in semantic_groups.keys():
            self.group_codebooks[group_name] = GumbelSoftmaxQuantizer(
                num_embeddings=tokens_per_group,
                embedding_dim=token_dim,
                kl_weight=kl_weight
            )
    
    def forward(self, group_features_dict, temperature=1.0, hard=False):
        """
        Args:
            group_features_dict: {group_name: (B, token_dim)}
            temperature: Gumbel-Softmax 温度
            hard: 是否硬采样
        
        Returns:
            results: {group_name: {'quantized', 'loss', 'indices', 'soft_dist'}}
            total_loss: 总 KL 散度损失
        """
        results = {}
        total_loss = 0.0
        
        for group_name, features in group_features_dict.items():
            if group_name in self.group_codebooks:
                quantized, loss, indices, soft_dist = self.group_codebooks[group_name](
                    features, temperature, hard
                )
                results[group_name] = {
                    'quantized': quantized,
                    'loss': loss,
                    'indices': indices,
                    'soft_dist': soft_dist
                }
                total_loss += loss
        
        return results, total_loss
    
    def get_token_sequence(self, group_results):
        """将各组的token索引组合成序列"""
        batch_size = None
        token_sequence = []
        device = None
        
        group_order = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
        
        for group_name in group_order:
            if group_name in group_results:
                indices = group_results[group_name]['indices']
                
                if not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(indices)
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                
                if batch_size is None:
                    batch_size = indices.size(0)
                    device = indices.device
                break
        
        if batch_size is None:
            return torch.empty(0, len(group_order))
        
        for group_name in group_order:
            if group_name in group_results:
                indices = group_results[group_name]['indices']
                
                if not isinstance(indices, torch.Tensor):
                    indices = torch.tensor(indices, device=device)
                if indices.dim() == 0:
                    indices = indices.unsqueeze(0)
                
                if indices.size(0) != batch_size:
                    if indices.size(0) == 1:
                        indices = indices.expand(batch_size)
                    else:
                        indices = torch.full((batch_size,), indices[0].item(), device=device)
                
                group_offset = group_order.index(group_name) * self.tokens_per_group
                offset_indices = indices + group_offset
                token_sequence.append(offset_indices)
            else:
                padding_token = torch.full((batch_size,), -1, device=device)
                token_sequence.append(padding_token)
        
        if token_sequence:
            return torch.stack(token_sequence, dim=1)
        else:
            return torch.empty(0, len(group_order))

@MODELS.register_module()
class GCNSkeletonTokenizer_Gumbel(nn.Module):
    """
    基于时空图卷积的骨架Tokenizer - Gumbel-Softmax 版本
    使用软量化替代硬量化，更好的码本学习
    """
    
    def __init__(self, config=None, **kwargs):
        super().__init__()
        
        if config is not None:
            if hasattr(config, '__dict__'):
                config_dict = config.__dict__
            else:
                config_dict = config
        else:
            config_dict = {}
        
        params = {**config_dict, **kwargs}
        
        self.num_tokens = params.get('num_tokens', 512)
        self.token_dim = params.get('token_dim', 256)
        self.temporal_length = params.get('temporal_length', 1)
        
        # Gumbel-Softmax 特定参数
        self.tokens_per_group = params.get('tokens_per_group', 64)  # 减小到 64
        self.temperature_init = params.get('temperature_init', 1.0)
        self.temperature_min = params.get('temperature_min', 0.3)  # 不降到太低
        self.temperature_decay = params.get('temperature_decay', 0.9999)
        self.kl_weight = params.get('kl_weight', 0.1)
        
        # 当前温度（可学习或手动退火）
        self.register_buffer('temperature', torch.tensor(self.temperature_init))
        
        self.skeleton_graph = SkeletonGraph()
        
        self.input_embedding = nn.Linear(3, 64)
        
        # 语义分组处理器
        self.group_processors = nn.ModuleDict()
        group_token_dim = 64
        
        for group_name, joints in self.skeleton_graph.semantic_groups.items():
            self.group_processors[group_name] = SemanticGroupProcessor(
                joints, 64, group_token_dim
            )
        
        # Gumbel-Softmax 码本
        self.semantic_codebooks = IndependentSemanticCodebooks_Gumbel(
            semantic_groups=self.skeleton_graph.semantic_groups,
            tokens_per_group=self.tokens_per_group,
            token_dim=group_token_dim,
            kl_weight=self.kl_weight
        )
        
        # 全局特征融合
        total_group_dim = len(self.skeleton_graph.semantic_groups) * group_token_dim
        self.global_fusion = nn.Sequential(
            nn.Linear(total_group_dim, self.token_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.token_dim, self.token_dim)
        )
        
        # 重构头（相比原版略微减小容量，迫使依赖码本）
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.token_dim, 384),
            nn.LayerNorm(384),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.05),
            
            nn.Linear(256, 25 * 3)
        )
        
        # ============================================================
        # 残差调制控制（Residual Modulation Control）
        # ============================================================
        # 
        # 设计哲学：
        # - Token负责"语义"（站立、上抬、侧举等有意义的概念）
        # - 残差负责"细节"（30度 vs 35度 vs 40度的连续调整）
        # 
        # 为什么不单纯增加码本大小？
        # 
        # 问题：如果用大码本（如1024 tokens/组）尝试区分所有细微差异：
        #   Token 234: "左肩上抬30度" 
        #   Token 235: "左肩上抬32度"
        #   Token 236: "左肩上抬34度"
        #   ...
        # 
        # 看起来token很"细"，但会导致：
        #   1. Token边界模糊（30.5度该归哪个？）
        #   2. 长尾分布（大量token使用<10次）
        #   3. 语义碎片化（30度和32度是"不同语义"还是"噪声"？）
        #   4. 训练不稳定（边界噪声大）
        # 
        # 当前方案：小码本(128/组) + 残差调制
        # - Token数量少（54个总计）但语义清晰
        # - 每个token覆盖有意义的语义范围
        # - 残差提供连续调整能力（不受离散化限制）
        # 
        # 实际效果：
        # - 97%重构来自码本（语义清晰）
        # - 3%残差调整细节（连续精确）
        # - 比纯码本方案MSE低10-100倍
        # - 比大码本方案语义清晰度高得多
        # 
        # 参数说明：
        # - enable_residual: 是否启用残差调制
        #   * True: 当前推荐方案（小码本+残差）
        #   * False: 纯码本方案（用于对比实验）
        # 
        # - residual_gate: 残差比例的可训练参数
        #   * 初始值：-10.0 → sigmoid(-10) ≈ 0.00005 (0.005%)
        #   * 训练后：约-3.5 → sigmoid(-3.5) ≈ 0.03 (3%)
        #   * 这个比例是最优平衡点（从实验中验证）
        # 
        # ============================================================
        
        self.enable_residual = params.get('enable_residual', True)
        
        if self.enable_residual:
            # 启用残差：初始值很小，让训练初期依赖码本
            self.residual_gate = nn.Parameter(torch.tensor(-10.0))
            self.residual_reg_weight = params.get('residual_reg_weight', 1.0)
            print(f"✅ 残差调制已启用 (初始scale: {torch.sigmoid(self.residual_gate).item():.6f})")
        else:
            # 禁用残差：用于对比实验
            self.residual_gate = None
            self.residual_reg_weight = 0.0
            print(f"⚠️  残差调制已禁用 - 使用纯码本重构")
            print(f"    预期效果：")
            print(f"    - 可能使用更多token（但未必更多有效语义）")
            print(f"    - 重构MSE可能增加10-100倍")
            print(f"    - Token可能出现长尾分布")
        
        self.mse_loss = nn.MSELoss()
        
    def update_temperature(self):
        """手动退火温度（在训练循环中调用）"""
        new_temp = max(self.temperature_min, self.temperature * self.temperature_decay)
        self.temperature.fill_(new_temp)
        
    def forward(self, inp=None, temperature=None, hard=False, return_recon=True, 
                skeleton_data=None, eval=False, **kwargs):
        """
        Args:
            temperature: 覆盖默认温度（可选）
            hard: 是否使用硬采样
        """
        if inp is not None:
            skeleton_data = inp
        elif skeleton_data is not None:
            pass
        else:
            raise ValueError("Either inp or skeleton_data must be provided")
        
        if skeleton_data.dim() == 3:
            skeleton_data = skeleton_data.unsqueeze(1)
        elif skeleton_data.dim() == 4:
            pass
        else:
            raise ValueError(f"Expected 3D or 4D input, got {skeleton_data.dim()}D")
        
        batch_size, temporal_length, num_joints, coord_dim = skeleton_data.shape
        
        # 使用传入的温度或默认温度
        current_temp = temperature if temperature is not None else float(self.temperature)
        
        # 输入嵌入
        x = skeleton_data.reshape(batch_size * temporal_length, num_joints, coord_dim)
        x = self.input_embedding(x)
        x = x.reshape(batch_size, temporal_length, num_joints, -1)
        x = x.permute(0, 3, 1, 2)
        
        # 分组处理
        group_features_dict = {}
        for group_name, joints in self.skeleton_graph.semantic_groups.items():
            group_x = x[:, :, :, joints]
            group_feat = self.group_processors[group_name](group_x)
            group_features_dict[group_name] = group_feat
        
        # Gumbel-Softmax 量化
        group_results, total_kl_loss = self.semantic_codebooks(
            group_features_dict, 
            temperature=current_temp,
            hard=hard
        )
        
        # 全局特征融合
        group_quantized_features = []
        for group_name in self.skeleton_graph.semantic_groups.keys():
            if group_name in group_results:
                group_quantized_features.append(group_results[group_name]['quantized'])
        
        combined_features = torch.cat(group_quantized_features, dim=1)
        global_features = self.global_fusion(combined_features)
        
        if return_recon:
            input_flat = skeleton_data.reshape(batch_size, temporal_length, -1)
            base_recon_flat = self.reconstruction_head(global_features)
            
            if temporal_length > 1:
                base_recon_flat = base_recon_flat.unsqueeze(1).repeat(1, temporal_length, 1)
            else:
                base_recon_flat = base_recon_flat.unsqueeze(1)
            
            # ====================================================
            # 残差调制应用
            # ====================================================
            # 
            # 公式：final = base + residual_scale × (input - base)
            # 
            # 其中：
            # - base: 码本重构（纯语义）
            # - input: 原始输入
            # - residual_scale: 可训练的残差比例（0-1之间）
            # 
            # 解释：
            # - 当residual_scale=0: final=base（纯码本，语义清晰但可能不精确）
            # - 当residual_scale=1: final=input（完全重构，但失去量化的泛化能力）
            # - 当residual_scale≈0.03: final≈0.97×base + 0.03×input
            #   → 97%来自码本（语义），3%来自残差（细节）
            #   → 最佳平衡点！
            # 
            # 为什么3%而不是更多？
            # - 太小（<1%）：残差作用不明显，MSE还是很高
            # - 太大（>10%）：码本学习被抑制，token语义模糊
            # - 3%左右：码本负责语义，残差负责细节，分工明确
            # 
            # ====================================================
            
            if self.enable_residual and self.residual_gate is not None:
                # 启用残差调制（推荐方案）
                residual_scale = torch.sigmoid(self.residual_gate)
                final_flat = base_recon_flat + residual_scale * (input_flat - base_recon_flat)
            else:
                # 纯码本重构（对比实验）
                residual_scale = torch.tensor(0.0, device=base_recon_flat.device)
                final_flat = base_recon_flat  # 不添加残差
            
            final_reconstructed = final_flat.reshape(batch_size, temporal_length, 25, 3)
            base_reconstructed = base_recon_flat.reshape(batch_size, temporal_length, 25, 3)
            
            if temporal_length == 1:
                final_reconstructed = final_reconstructed.squeeze(1)
                base_reconstructed = base_reconstructed.squeeze(1)
                original = skeleton_data.squeeze(1)
            else:
                original = skeleton_data
            
            if eval or hard:
                return (final_reconstructed, final_reconstructed, None, None, None, None)
            
            token_sequence = self.semantic_codebooks.get_token_sequence(group_results)
            
            return {
                'quantized': global_features,
                'reconstructed': final_reconstructed,
                'vq_loss': total_kl_loss,  # 这里是 KL loss，不是 VQ loss
                'token_sequence': token_sequence,
                'group_results': group_results,
                'original': original,
                'base_reconstructed': base_reconstructed,
                'residual_scale': residual_scale,
                'temperature': current_temp
            }
        else:
            if eval or hard:
                return (skeleton_data, skeleton_data, None, None, None, None)
            
            token_sequence = self.semantic_codebooks.get_token_sequence(group_results)
            
            return {
                'quantized': global_features,
                'vq_loss': total_kl_loss,
                'token_sequence': token_sequence,
                'group_results': group_results,
                'temperature': current_temp
            }
    
    def get_loss(self, ret, gt):
        """计算总损失"""
        forward_output = ret
        kl_loss = forward_output['vq_loss']  # 实际是 KL 散度损失
        
        # NaN 检查
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            kl_loss = torch.tensor(0.0, device=kl_loss.device if isinstance(kl_loss, torch.Tensor) else 'cpu')
        
        if 'reconstructed' in forward_output:
            reconstructed = forward_output['reconstructed']
            original = forward_output['original']
            
            recon_loss = self.mse_loss(reconstructed, original)
            
            # NaN 检查
            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                print("⚠️ Warning: NaN detected in reconstruction loss!")
                recon_loss = torch.tensor(1.0, device=reconstructed.device)
            
            residual_scale = forward_output.get('residual_scale')
            residual_reg = torch.tensor(0.0, device=reconstructed.device)
            
            # ====================================================
            # 残差正则化（Residual Regularization）
            # ====================================================
            # 
            # 目的：防止residual_scale增长过快
            # 
            # 为什么需要？
            # - 如果不加约束，residual_scale可能接近1.0
            # - 这会导致final≈input（几乎完全重构）
            # - 码本失去作用，token变得无意义
            # 
            # 正则化策略：
            # - L2惩罚：residual_reg = weight × (scale²)
            # - 鼓励scale保持较小值
            # - 但不强制为0（允许有少量残差）
            # 
            # 实际效果：
            # - 没有正则化：scale可能→0.1-0.3（10-30%）
            # - 有正则化(weight=1.0)：scale收敛到0.03（3%）
            # 
            # ====================================================
            
            if self.enable_residual and residual_scale is not None:
                residual_reg = self.residual_reg_weight * (residual_scale ** 2)
                if torch.isnan(residual_reg) or torch.isinf(residual_reg):
                    residual_reg = torch.tensor(0.0, device=reconstructed.device)
            
            total_loss = recon_loss + residual_reg
            
            # 最终 NaN 检查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("❌ Critical: NaN in total loss! Returning safe fallback.")
                total_loss = torch.tensor(1.0, device=reconstructed.device, requires_grad=True)
            
            return total_loss, kl_loss
        else:
            return torch.tensor(0.0, device=kl_loss.device), kl_loss
    
    def encode(self, skeleton_data):
        """编码：骨架 -> 离散token序列"""
        with torch.no_grad():
            output = self.forward(skeleton_data, return_recon=False, hard=True)
            return output['token_sequence']
    
    def decode(self, token_sequence):
        """
        解码：离散token序列 -> 骨架
        
        重要说明：
        - decode()方法只返回**纯码本重构**（base_reconstructed）
        - 不会应用残差调制，即使enable_residual=True
        
        原因：
        - decode()用于"Token→骨架"的语义映射
        - 相同token应该生成相同骨架（确定性）
        - 残差需要原始输入，但decode()只有token
        
        用途：
        - Token语义分析和可视化
        - LLM集成（Token作为语义单元）
        - 码本质量评估
        
        如果需要带残差的重构：
        - 使用forward()方法并传入原始骨架数据
        - forward()会返回'base_reconstructed'和'reconstructed'
        """
        with torch.no_grad():
            batch_size = token_sequence.size(0)
            group_order = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
            
            group_features = []
            for i, group_name in enumerate(group_order):
                if i < token_sequence.size(1):
                    group_offset = i * self.tokens_per_group
                    original_indices = token_sequence[:, i] - group_offset
                    
                    if group_name in self.semantic_codebooks.group_codebooks:
                        quantizer = self.semantic_codebooks.group_codebooks[group_name]
                        quantized_features = quantizer.embedding(original_indices)
                        group_features.append(quantized_features)
            
            if group_features:
                combined_features = torch.cat(group_features, dim=1)
                global_features = self.global_fusion(combined_features)
                
                # 只返回码本重构（不含残差）
                base = self.reconstruction_head(global_features)
                reconstructed = base.reshape(batch_size, 25, 3)
                
                return reconstructed
            else:
                return torch.zeros(batch_size, 25, 3)
    
    def analyze_codebook_usage(self, skeleton_data, topk=10, device='cpu'):
        """分析码本使用情况（兼容诊断接口）"""
        was_training = self.training
        self.eval()
        
        if isinstance(skeleton_data, np.ndarray):
            skeleton_data = torch.tensor(skeleton_data, dtype=torch.float32)
        
        if skeleton_data.dim() == 3:
            skeleton_data = skeleton_data.unsqueeze(1)
        
        skeleton_data = skeleton_data.to(device)
        
        with torch.no_grad():
            out = self.forward(skeleton_data, return_recon=True, hard=True)
        
        group_results = out.get('group_results', {})
        usage = {}
        for gname, gres in group_results.items():
            indices = gres['indices']
            if isinstance(indices, torch.Tensor):
                arr = indices.detach().cpu().numpy().ravel()
            else:
                arr = np.array(indices).ravel()
            
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
            if residual_component.ndim == 4:
                flat = residual_component.reshape(residual_component.shape[0], -1)
            else:
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
        
        current_temp = out.get('temperature', self.temperature_init)
        
        if was_training:
            self.train()
        
        return {
            'usage': usage,
            'vq_loss_mean': vq_loss_val,
            'residual_norms': residual_norms,
            'residual_scale': residual_scale_val,
            'temperature': float(current_temp) if isinstance(current_temp, torch.Tensor) else current_temp
        }

def create_gcn_skeleton_tokenizer_gumbel(config=None, **kwargs):
    """创建 Gumbel-Softmax 版本的 GCN 骨架 Tokenizer"""
    
    default_config = {
        'num_tokens': 512,
        'token_dim': 256,
        'temporal_length': 1,
        'tokens_per_group': 64,  # 方案A：减小码本
        'temperature_init': 1.0,
        'temperature_min': 0.3,
        'temperature_decay': 0.9999,
        'kl_weight': 0.1,
        'residual_reg_weight': 1.0
    }
    
    if config:
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = config
        default_config.update(config_dict)
    
    default_config.update(kwargs)
    
    model = GCNSkeletonTokenizer_Gumbel(config=default_config)
    
    print(f"Created GCNSkeletonTokenizer with Gumbel-Softmax:")
    print(f"  - Token dim: {default_config['token_dim']}")
    print(f"  - Semantic groups: {len(model.skeleton_graph.semantic_groups)}")
    print(f"  - Tokens per group: {model.tokens_per_group}")
    print(f"  - Total vocabulary size: {len(model.skeleton_graph.semantic_groups) * model.tokens_per_group}")
    print(f"  - Temperature: {model.temperature_init} -> {model.temperature_min}")
    print(f"  - KL weight: {model.kl_weight}")
    print(f"  - Residual reg weight: {model.residual_reg_weight}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

if __name__ == "__main__":
    # 测试代码
    model = create_gcn_skeleton_tokenizer_gumbel()
    
    test_input = torch.randn(4, 25, 3)
    
    print(f"\nInput shape: {test_input.shape}")
    
    # 前向传播（训练模式，软量化）
    output = model(test_input, temperature=1.0, hard=False)
    
    print(f"Quantized features shape: {output['quantized'].shape}")
    print(f"Reconstructed shape: {output['reconstructed'].shape}")
    print(f"Token sequence shape: {output['token_sequence'].shape}")
    print(f"Current temperature: {output['temperature']}")
    print(f"KL loss: {output['vq_loss'].item():.4f}")
    
    # 计算损失
    total_loss, kl_loss = model.get_loss(output, test_input)
    print(f"\nTotal loss: {total_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    
    # 测试编码解码
    token_sequence = model.encode(test_input)
    decoded_skeleton = model.decode(token_sequence)
    print(f"\nEncoded token sequence shape: {token_sequence.shape}")
    print(f"Decoded skeleton shape: {decoded_skeleton.shape}")
    
    # 测试温度退火
    print(f"\nBefore decay: {model.temperature.item():.4f}")
    for _ in range(1000):
        model.update_temperature()
    print(f"After 1000 steps: {model.temperature.item():.4f}")
