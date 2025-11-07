# -*- coding: utf-8 -*-
"""
MARS+Transformer骨架GIF动画可视化脚本 (PyTorch版本)
生成多个样本相邻几帧的3D骨架动画GIF
支持Ground Truth vs 预测结果的动态对比
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os
import warnings
warnings.filterwarnings('ignore')

# 尝试导入PyTorch (用于直接模型推理)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    print("✓ PyTorch可用，支持直接模型推理")
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch不可用，仅支持预保存的预测结果")

# Microsoft Kinect 19关节点骨架连接定义 (转换为0-based索引)
skeleton_connections = [
    (2, 3),   # head-neck
    (2, 18),  # neck-spineshoulder
    (18, 4),  # spineshoulder-leftshoulder
    (4, 5),   # leftshoulder-leftelbow
    (5, 6),   # leftelbow-leftwrist
    (18, 7),  # spineshoulder-rightshoulder
    (7, 8),   # rightshoulder-rightelbow
    (8, 9),   # rightelbow-rightwrist
    (18, 1),  # spineshoulder-spinemid
    (1, 0),   # spinemid-spinebase
    (0, 10),  # spinebase-hipleft
    (10, 11), # hipleft-kneeleft
    (11, 12), # kneeleft-ankleleft
    (12, 13), # ankleleft-footleft
    (0, 14),  # spinebase-hipright
    (14, 15), # hipright-kneeright
    (15, 16), # kneeright-ankleright
    (16, 17)  # ankleright-footright
]

# PyTorch模型定义（简化版，仅用于推理）
class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力块"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        return x * y

class ResidualSEBlock(nn.Module):
    """残差块结合SE注意力"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se_block = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se_block(out)
        out += identity
        out = self.relu(out)
        return out

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)

class LightweightTransformerBlock(nn.Module):
    """轻量级Transformer注意力块"""
    def __init__(self, d_model, num_heads=4, dff=256, dropout=0.1):
        super(LightweightTransformerBlock, self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class EnhancedMARSBackbone(nn.Module):
    """创建增强的MARS主干网络 - 支持多尺度特征融合"""
    def __init__(self, input_channels=5, multi_scale=True):
        super(EnhancedMARSBackbone, self).__init__()
        
        self.multi_scale = multi_scale
        
        self.initial_conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.initial_bn1 = nn.BatchNorm2d(32)
        self.initial_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.initial_bn2 = nn.BatchNorm2d(32)
        
        self.res_se_1 = ResidualSEBlock(32, 64)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        
        self.res_se_2 = ResidualSEBlock(64, 128)
        self.spatial_att_1 = SpatialAttention()
        
        self.res_se_3 = ResidualSEBlock(128, 256)
        self.spatial_att_2 = SpatialAttention()
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        
        # 输出特征维度: 64 + 128 + 256 = 448 (多尺度) 或 256 (单尺度)
        self.output_dim = 448 if multi_scale else 256
    
    def forward(self, x):
        x = self.relu(self.initial_bn1(self.initial_conv1(x)))
        x = self.relu(self.initial_bn2(self.initial_conv2(x)))
        
        # 第一个残差SE块 - 保存特征1
        feat1 = self.res_se_1(x)
        x = self.maxpool1(feat1)
        
        # 第二个残差SE块 + 空间注意力 - 保存特征2
        feat2 = self.res_se_2(x)
        feat2 = self.spatial_att_1(feat2)
        
        # 第三个残差SE块 + 空间注意力 - 保存特征3
        feat3 = self.res_se_3(feat2)
        feat3 = self.spatial_att_2(feat3)
        
        if self.multi_scale:
            # 多尺度特征融合
            feat1_pool = self.global_avg_pool(feat1).flatten(1)  # (batch, 64)
            feat2_pool = self.global_avg_pool(feat2).flatten(1)  # (batch, 128)
            feat3_pool = self.global_avg_pool(feat3).flatten(1)  # (batch, 256)
            
            # 拼接多尺度特征: 64 + 128 + 256 = 448
            output = torch.cat([feat1_pool, feat2_pool, feat3_pool], dim=1)
        else:
            # 单尺度特征（仅使用最深层特征）
            output = self.global_avg_pool(feat3).flatten(1)  # (batch, 256)
        
        return output

class TransformerRegressionHead(nn.Module):
    """创建Transformer增强的回归头 - 简洁高效版本（与models/skeleton_extractor.py保持一致）"""
    def __init__(self, input_dim=256, output_dim=57):
        super(TransformerRegressionHead, self).__init__()
        
        # 简洁的特征投影（单层，足够高效）
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 重塑为序列格式参数
        self.seq_len = 8
        self.d_model = 64
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=self.seq_len)
        
        # 保持原先的2层Transformer（简洁高效）
        self.transformer_1 = LightweightTransformerBlock(self.d_model, num_heads=4, dff=128)
        self.transformer_2 = LightweightTransformerBlock(self.d_model, num_heads=4, dff=128)
        
        # 单一平均池化（简洁有效）
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 优化的回归头 - 渐进式降维设计
        self.regression_head = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征投影
        x = self.feature_projection(x)  # (batch, 512)
        
        # 重塑为序列格式
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.d_model)  # (batch, 8, 64)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 2层Transformer处理（保持简洁）
        x = self.transformer_1(x)
        x = self.transformer_2(x)
        
        # 单一平均池化
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_avg_pool(x).squeeze(-1)  # (batch, d_model)
        
        # 优化的渐进式回归头
        output = self.regression_head(x)
        
        return output

class MARSTransformerModel(nn.Module):
    """完整的MARS+Transformer骨架提取模型 - 支持多尺度特征融合"""
    def __init__(self, input_channels=5, output_dim=57, multi_scale=True):
        super(MARSTransformerModel, self).__init__()
        
        # MARS CNN主干（支持多尺度特征融合）
        self.backbone = EnhancedMARSBackbone(input_channels, multi_scale=multi_scale)
        
        # 根据是否使用多尺度确定输入维度
        input_dim = self.backbone.output_dim  # 448 (多尺度) 或 256 (单尺度)
        
        # Transformer回归头（自动适配输入维度）
        self.regression_head = TransformerRegressionHead(input_dim, output_dim)
    
    def forward(self, x):
        # CNN特征提取（多尺度或单尺度）
        features = self.backbone(x)
        
        # Transformer回归
        output = self.regression_head(features)
        
        return output

def load_data():
    """加载数据"""
    print("加载测试数据和预测结果...")
    
    # 加载Ground Truth标签
    labels_test = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/labels_test.npy')
    print(f"✓ Ground Truth数据: {labels_test.shape}")
    
    # 尝试加载PyTorch预测结果
    pred_files = [
        'predictions_mars_transformer_torch.npy',  # PyTorch预测结果
        'predictions_mars_transformer_torch_live.npy',  # PyTorch实时预测结果
        'predictions_mars_transformer.npy',  # TensorFlow预测结果
        'predictions_mars_transformer_live.npy',  # TensorFlow实时预测结果
        'predictions_skeleton_extraction.npy',  # 备用文件
        'Pred_test_transformer_100.npy',  # 其他备用文件
        'Pred_test_100.npy'
    ]
    
    predictions = None
    used_file = None
    
    for pred_file in pred_files:
        if os.path.exists(pred_file):
            try:
                predictions = np.load(pred_file)
                used_file = pred_file
                print(f"✓ 预测结果数据: {predictions.shape} (来源: {pred_file})")
                break
            except Exception as e:
                print(f"⚠️ 加载 {pred_file} 失败: {e}")
                continue
    
    if predictions is None:
        print(f"❌ 未找到有效的预测文件")
        print(f"   尝试的文件: {pred_files}")
        return None, None
    
    # 验证数据形状匹配
    if len(labels_test) != len(predictions):
        print(f"⚠️ 数据长度不匹配: GT({len(labels_test)}) vs Pred({len(predictions)})")
        min_len = min(len(labels_test), len(predictions))
        labels_test = labels_test[:min_len]
        predictions = predictions[:min_len]
        print(f"✓ 已截断到相同长度: {min_len}")
    
    return labels_test, predictions

def predict_with_torch_model(model_path='mars_transformer_best.pth', feature_path='/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_test.npy'):
    """使用PyTorch模型直接进行预测"""
    if not HAS_TORCH:
        print("❌ PyTorch不可用，无法进行模型推理")
        return None
        
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
        
    if not os.path.exists(feature_path):
        print(f"❌ 特征文件不存在: {feature_path}")
        return None
    
    try:
        # 配置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔄 使用设备: {device}")
        
        # 加载模型（默认启用多尺度特征融合）
        print(f"🔄 加载PyTorch模型: {model_path}")
        print("   配置: 多尺度特征融合 (64+128+256=448维)")
        model = MARSTransformerModel(input_channels=5, output_dim=57, multi_scale=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("✓ 模型加载成功")
        
        # 加载特征数据
        print(f"🔄 加载测试特征: {feature_path}")
        features = np.load(feature_path)
        print(f"✓ 特征数据: {features.shape}")
        
        # 转换数据格式：(N, H, W, C) -> (N, C, H, W)
        features = np.transpose(features, (0, 3, 1, 2))
        features_tensor = torch.FloatTensor(features).to(device)
        
        print("🔄 开始预测...")
        predictions = []
        batch_size = 32
        
        with torch.no_grad():
            for i in range(0, len(features_tensor), batch_size):
                batch = features_tensor[i:i+batch_size]
                outputs = model(batch)
                predictions.append(outputs.cpu().numpy())
                
                # 显示进度
                if (i // batch_size + 1) % 50 == 0:
                    print(f"  处理批次: {i//batch_size + 1}/{(len(features_tensor) + batch_size - 1)//batch_size}")
        
        predictions = np.concatenate(predictions, axis=0)
        print(f"✓ 预测完成: {predictions.shape}")
        
        # 保存预测结果
        output_file = 'predictions_mars_transformer_torch_gif_live.npy'
        np.save(output_file, predictions)
        print(f"✓ 预测结果已保存: {output_file}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ 模型推理失败: {e}")
        return None

def parse_joints(joints_data):
    """解析关节数据格式: (x1...x19, y1...y19, z1...z19)"""
    if joints_data.shape == (57,):
        x_coords = joints_data[0:19]
        y_coords = joints_data[19:38]  
        z_coords = joints_data[38:57]
        return np.column_stack((x_coords, y_coords, z_coords))
    else:
        raise ValueError(f"无效的关节数据形状: {joints_data.shape}")

def plot_skeleton_frame(joints, ax, color='blue', linewidth=2, alpha=1.0, marker_size=80):
    """绘制单帧骨架"""
    ax.clear()
    
    # 绘制关节点
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
              c=color, s=marker_size, alpha=alpha, edgecolors='black', linewidths=0.5)
    
    # 绘制骨架连接线
    for connection in skeleton_connections:
        if connection[0] < len(joints) and connection[1] < len(joints):
            joint1 = joints[connection[0]]
            joint2 = joints[connection[1]]
            ax.plot([joint1[0], joint2[0]], 
                   [joint1[1], joint2[1]], 
                   [joint1[2], joint2[2]], 
                   color=color, alpha=alpha, linewidth=linewidth)
    
    # 设置固定的坐标轴范围（基于所有帧的数据范围）
    return ax

def get_data_bounds(ground_truth_frames, prediction_frames):
    """计算所有帧的数据边界"""
    all_joints = []
    
    for gt, pred in zip(ground_truth_frames, prediction_frames):
        all_joints.append(parse_joints(gt))
        all_joints.append(parse_joints(pred))
    
    all_joints = np.vstack(all_joints)
    
    x_min, x_max = all_joints[:, 0].min(), all_joints[:, 0].max()
    y_min, y_max = all_joints[:, 1].min(), all_joints[:, 1].max()
    z_min, z_max = all_joints[:, 2].min(), all_joints[:, 2].max()
    
    # 计算统一的范围
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    max_range = max(x_range, y_range, z_range)
    
    # 添加边距
    margin = max_range * 0.2
    
    # 计算中心点
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    
    # 返回统一的边界
    half_range = max_range / 2 + margin
    bounds = {
        'xlim': (x_center - half_range, x_center + half_range),
        'ylim': (y_center - half_range, y_center + half_range),
        'zlim': (z_center - half_range, z_center + half_range)
    }
    
    return bounds

def create_comparison_gif(ground_truth_frames, prediction_frames, sample_indices, output_path, 
                         fps=2, duration_per_frame=0.8):
    """创建对比GIF动画"""
    
    print(f"🎬 开始创建PyTorch对比GIF: {len(ground_truth_frames)} 帧")
    
    # 计算数据边界
    bounds = get_data_bounds(ground_truth_frames, prediction_frames)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle('MARS+Transformer Skeleton Animation (PyTorch): Ground Truth vs Prediction', 
                fontsize=16, fontweight='bold')
    
    # 左侧: Ground Truth
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold', color='blue')
    
    # 右侧: PyTorch Prediction
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('PyTorch Prediction', fontsize=14, fontweight='bold', color='red')
    
    # 设置轴属性
    for ax in [ax1, ax2]:
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_xlim(bounds['xlim'])
        ax.set_ylim(bounds['ylim'])
        ax.set_zlim(bounds['zlim'])
        ax.view_init(elev=20, azim=45)
    
    # 添加帧信息文本
    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold')
    
    def animate(frame_idx):
        """动画更新函数"""
        # 获取当前帧数据
        gt_joints = parse_joints(ground_truth_frames[frame_idx])
        pred_joints = parse_joints(prediction_frames[frame_idx])
        
        # 计算误差
        joint_errors = np.sqrt(np.sum((gt_joints - pred_joints) ** 2, axis=1))
        mean_error = np.mean(joint_errors)
        
        # 绘制Ground Truth
        plot_skeleton_frame(gt_joints, ax1, color='blue', linewidth=3, alpha=0.9)
        ax1.set_xlim(bounds['xlim'])
        ax1.set_ylim(bounds['ylim'])
        ax1.set_zlim(bounds['zlim'])
        ax1.view_init(elev=20, azim=45)
        
        # 绘制PyTorch Prediction
        plot_skeleton_frame(pred_joints, ax2, color='red', linewidth=3, alpha=0.9)
        ax2.set_xlim(bounds['xlim'])
        ax2.set_ylim(bounds['ylim'])
        ax2.set_zlim(bounds['zlim'])
        ax2.view_init(elev=20, azim=45)
        
        # 更新帧信息
        sample_idx = sample_indices[frame_idx]
        frame_text.set_text(f'Sample {sample_idx+1:03d} | Frame {frame_idx+1:02d}/{len(ground_truth_frames):02d} | 3D Error: {mean_error:.4f} | PyTorch')
        
        return []
    
    # 创建动画
    print("🔄 渲染PyTorch动画帧...")
    anim = animation.FuncAnimation(fig, animate, frames=len(ground_truth_frames), 
                                 interval=int(duration_per_frame * 1000), blit=False, repeat=True)
    
    # 保存GIF
    print(f"💾 保存PyTorch GIF动画: {output_path}")
    try:
        anim.save(output_path, writer='pillow', fps=fps, dpi=100)
        print(f"✅ PyTorch GIF保存成功: {output_path}")
    except Exception as e:
        print(f"❌ PyTorch GIF保存失败: {e}")
        # 尝试降低质量保存
        try:
            print("🔄 尝试降低质量重新保存...")
            anim.save(output_path, writer='pillow', fps=fps, dpi=80)
            print(f"✅ PyTorch GIF保存成功 (降低质量): {output_path}")
        except Exception as e2:
            print(f"❌ 降质量保存也失败: {e2}")
    
    plt.close()

def create_overlay_gif(ground_truth_frames, prediction_frames, sample_indices, output_path, 
                      fps=2, duration_per_frame=0.8):
    """创建重叠GIF动画"""
    
    print(f"🎬 开始创建PyTorch重叠GIF: {len(ground_truth_frames)} 帧")
    
    # 计算数据边界
    bounds = get_data_bounds(ground_truth_frames, prediction_frames)
    
    # 创建图形
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('MARS+Transformer Skeleton Animation (PyTorch): Overlay Comparison', 
                fontsize=16, fontweight='bold')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_xlim(bounds['xlim'])
    ax.set_ylim(bounds['ylim'])
    ax.set_zlim(bounds['zlim'])
    ax.view_init(elev=20, azim=45)
    
    # 添加图例和帧信息
    legend_text = fig.text(0.02, 0.95, "● Ground Truth (Blue)\n● PyTorch Prediction (Red)", 
                          fontsize=12, fontweight='bold', verticalalignment='top')
    frame_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold')
    
    def animate(frame_idx):
        """动画更新函数"""
        ax.clear()
        
        # 获取当前帧数据
        gt_joints = parse_joints(ground_truth_frames[frame_idx])
        pred_joints = parse_joints(prediction_frames[frame_idx])
        
        # 计算误差
        joint_errors = np.sqrt(np.sum((gt_joints - pred_joints) ** 2, axis=1))
        mean_error = np.mean(joint_errors)
        
        # 绘制Ground Truth (蓝色)
        ax.scatter(gt_joints[:, 0], gt_joints[:, 1], gt_joints[:, 2], 
                  c='blue', s=100, alpha=0.8, edgecolors='black', linewidths=0.5, label='Ground Truth')
        
        for connection in skeleton_connections:
            if connection[0] < len(gt_joints) and connection[1] < len(gt_joints):
                joint1 = gt_joints[connection[0]]
                joint2 = gt_joints[connection[1]]
                ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], 
                       color='blue', alpha=0.8, linewidth=3)
        
        # 绘制PyTorch Prediction (红色，透明)
        ax.scatter(pred_joints[:, 0], pred_joints[:, 1], pred_joints[:, 2], 
                  c='red', s=80, alpha=0.6, edgecolors='darkred', linewidths=0.5, label='PyTorch Prediction')
        
        for connection in skeleton_connections:
            if connection[0] < len(pred_joints) and connection[1] < len(pred_joints):
                joint1 = pred_joints[connection[0]]
                joint2 = pred_joints[connection[1]]
                ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], 
                       color='red', alpha=0.6, linewidth=2)
        
        # 设置轴属性
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_xlim(bounds['xlim'])
        ax.set_ylim(bounds['ylim'])
        ax.set_zlim(bounds['zlim'])
        ax.view_init(elev=20, azim=45)
        
        # 更新帧信息
        sample_idx = sample_indices[frame_idx]
        frame_text.set_text(f'Sample {sample_idx+1:03d} | Frame {frame_idx+1:02d}/{len(ground_truth_frames):02d} | 3D Error: {mean_error:.4f} | PyTorch')
        
        return []
    
    # 创建动画
    print("🔄 渲染PyTorch重叠动画帧...")
    anim = animation.FuncAnimation(fig, animate, frames=len(ground_truth_frames), 
                                 interval=int(duration_per_frame * 1000), blit=False, repeat=True)
    
    # 保存GIF
    print(f"💾 保存PyTorch重叠GIF动画: {output_path}")
    try:
        anim.save(output_path, writer='pillow', fps=fps, dpi=100)
        print(f"✅ PyTorch重叠GIF保存成功: {output_path}")
    except Exception as e:
        print(f"❌ PyTorch重叠GIF保存失败: {e}")
        # 尝试降低质量保存
        try:
            print("🔄 尝试降低质量重新保存...")
            anim.save(output_path, writer='pillow', fps=fps, dpi=80)
            print(f"✅ PyTorch重叠GIF保存成功 (降低质量): {output_path}")
        except Exception as e2:
            print(f"❌ 降质量保存也失败: {e2}")
    
    plt.close()

def main(use_live_prediction=True, frames_per_gif=8, num_gifs=5, fps=2):
    """主函数"""
    print("MARS+Transformer骨架GIF动画可视化工具 (PyTorch版本)")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = 'visualizations/skeleton_extractor_gif_new'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ 已创建输出目录: {output_dir}")
    
    # 加载Ground Truth
    ground_truth = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/labels_test.npy')
    print(f"✓ Ground Truth数据: {ground_truth.shape}")
    
    predictions = None
    
    # 尝试使用实时PyTorch模型预测
    if use_live_prediction and HAS_TORCH:
        print("\n🚀 尝试使用mars_transformer_best.pth进行实时预测...")
        predictions = predict_with_torch_model('mars_transformer_best.pth')
        
        if predictions is not None:
            print("✅ 使用PyTorch实时模型预测成功!")
        else:
            print("⚠️ PyTorch实时预测失败，尝试加载预保存的结果...")
    
    # 如果实时预测失败，使用预保存的结果
    if predictions is None:
        print("\n📁 加载预保存的预测结果...")
        ground_truth, predictions = load_data()
        if ground_truth is None or predictions is None:
            print("❌ 数据加载失败，退出程序")
            return
    
    print(f"\n🎬 开始生成 {num_gifs} 个PyTorch GIF动画，每个包含 {frames_per_gif} 帧")
    print(f"动画参数: FPS={fps}, 每帧时长={1000/fps:.0f}ms")
    print("-" * 60)
    
    total_samples = len(ground_truth)
    step = max(1, total_samples // (num_gifs * frames_per_gif))
    
    for gif_idx in range(num_gifs):
        print(f"\n🎥 生成第 {gif_idx+1}/{num_gifs} 个PyTorch GIF动画...")
        
        # 计算起始索引
        start_idx = gif_idx * frames_per_gif * step
        
        # 选择连续的帧
        frame_indices = []
        gt_frames = []
        pred_frames = []
        
        for frame_idx in range(frames_per_gif):
            sample_idx = start_idx + frame_idx * step
            if sample_idx < total_samples:
                frame_indices.append(sample_idx)
                gt_frames.append(ground_truth[sample_idx])
                pred_frames.append(predictions[sample_idx])
        
        if len(frame_indices) < frames_per_gif:
            print(f"⚠️ 样本不足，只能生成 {len(frame_indices)} 帧")
        
        print(f"   样本索引: {[idx+1 for idx in frame_indices]}")
        
        # 生成PyTorch对比GIF
        comparison_path = os.path.join(output_dir, f'skeleton_torch_comparison_{gif_idx+1:02d}.gif')
        create_comparison_gif(gt_frames, pred_frames, frame_indices, comparison_path, fps=fps)
        
        # 生成PyTorch重叠GIF
        overlay_path = os.path.join(output_dir, f'skeleton_torch_overlay_{gif_idx+1:02d}.gif')
        create_overlay_gif(gt_frames, pred_frames, frame_indices, overlay_path, fps=fps)
    
    # 生成汇总信息
    print("-" * 60)
    print(f"✅ PyTorch GIF动画生成完成!")
    print(f"输出目录: {output_dir}/")
    print(f"生成文件:")
    
    gif_files = [f for f in os.listdir(output_dir) if f.endswith('.gif')]
    for i, gif_file in enumerate(sorted(gif_files), 1):
        file_path = os.path.join(output_dir, gif_file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  {i:2d}. {gif_file} ({file_size:.2f} MB)")
    
    # 创建README文件
    readme_path = os.path.join(output_dir, 'README_torch.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("MARS+Transformer骨架GIF动画说明 (PyTorch版本)\n")
        f.write("=" * 50 + "\n\n")
        f.write("文件类型:\n")
        f.write("- skeleton_torch_comparison_XX.gif: 左右对比动画 (Ground Truth vs PyTorch Prediction)\n")
        f.write("- skeleton_torch_overlay_XX.gif: 重叠对比动画 (蓝色GT + 红色PyTorch Prediction)\n\n")
        f.write(f"动画参数:\n")
        f.write(f"- 帧数: {frames_per_gif} 帧/GIF\n")
        f.write(f"- 帧率: {fps} FPS\n")
        f.write(f"- 总GIF数量: {num_gifs}\n")
        f.write(f"- 生成时间: {np.datetime64('now')}\n\n")
        f.write("模型信息:\n")
        f.write("- 架构: MARS+Transformer (PyTorch版本)\n")
        f.write("- 权重文件: mars_transformer_best.pth\n")
        f.write("- 框架: PyTorch\n")
        f.write("- 输入格式: (N, C, H, W) - PyTorch标准格式\n")
        f.write("- 输出维度: 57 (19个关节点的3D坐标)\n")
    
    print(f"✓ PyTorch说明文件已保存: {readme_path}")
    print(f"\n🚀 MARS+Transformer PyTorch GIF可视化完成!")

if __name__ == "__main__":
    import sys
    
    # 默认参数
    use_live = True
    frames_per_gif = 8
    num_gifs = 8
    fps = 2
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.lower() in ['--no-live', '-n']:
                use_live = False
                print("📁 强制使用预保存的预测结果")
            elif arg.startswith('--frames='):
                frames_per_gif = int(arg.split('=')[1])
                print(f"🎬 设置每个GIF帧数: {frames_per_gif}")
            elif arg.startswith('--gifs='):
                num_gifs = int(arg.split('=')[1])
                print(f"🎥 设置GIF数量: {num_gifs}")
            elif arg.startswith('--fps='):
                fps = int(arg.split('=')[1])
                print(f"⏱️ 设置帧率: {fps} FPS")
            elif arg.lower() in ['--help', '-h']:
                print("MARS+Transformer PyTorch GIF动画可视化工具")
                print("用法:")
                print("  python skeleton_gif_visualization_torch.py                    # 使用默认参数")
                print("  python skeleton_gif_visualization_torch.py --no-live         # 只使用预保存结果")
                print("  python skeleton_gif_visualization_torch.py --frames=10       # 设置每个GIF帧数")
                print("  python skeleton_gif_visualization_torch.py --gifs=3          # 设置GIF数量")
                print("  python skeleton_gif_visualization_torch.py --fps=3           # 设置帧率")
                print("  python skeleton_gif_visualization_torch.py --help            # 显示帮助")
                print("\n默认参数:")
                print(f"  frames_per_gif={frames_per_gif}, num_gifs={num_gifs}, fps={fps}")
                exit(0)
    
    print(f"🎬 PyTorch动画参数: {frames_per_gif}帧/GIF, {num_gifs}个GIF, {fps}FPS")
    main(use_live_prediction=use_live, frames_per_gif=frames_per_gif, num_gifs=num_gifs, fps=fps)