# -*- coding: utf-8 -*-
"""
MARS+Transformer骨架可视化脚本 (PyTorch版本)
生成Ground Truth vs 预测结果的3D骨架对比图
支持直接使用PyTorch模型进行预测和可视化
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

# 简化的PyTorch模型定义（仅用于推理）
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
    """创建增强的MARS主干网络"""
    def __init__(self, input_channels=5):
        super(EnhancedMARSBackbone, self).__init__()
        
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
    
    def forward(self, x):
        x = self.relu(self.initial_bn1(self.initial_conv1(x)))
        x = self.relu(self.initial_bn2(self.initial_conv2(x)))
        
        x = self.res_se_1(x)
        x = self.maxpool1(x)
        
        x = self.res_se_2(x)
        x = self.spatial_att_1(x)
        
        x = self.res_se_3(x)
        x = self.spatial_att_2(x)
        
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        
        return x

class TransformerRegressionHead(nn.Module):
    """创建Transformer增强的回归头"""
    def __init__(self, input_dim=256, output_dim=57):
        super(TransformerRegressionHead, self).__init__()
        
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.seq_len = 8
        self.d_model = 64
        
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=self.seq_len)
        self.transformer_1 = LightweightTransformerBlock(self.d_model, num_heads=4, dff=128)
        self.transformer_2 = LightweightTransformerBlock(self.d_model, num_heads=4, dff=128)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.final_layers = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        x = self.feature_projection(x)
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.d_model)
        x = self.pos_encoding(x)
        x = self.transformer_1(x)
        x = self.transformer_2(x)
        x = x.transpose(1, 2)
        x = self.global_avg_pool(x).squeeze(-1)
        output = self.final_layers(x)
        return output

class MARSTransformerModel(nn.Module):
    """完整的MARS+Transformer骨架提取模型"""
    def __init__(self, input_channels=5, output_dim=57):
        super(MARSTransformerModel, self).__init__()
        self.backbone = EnhancedMARSBackbone(input_channels)
        self.regression_head = TransformerRegressionHead(256, output_dim)
    
    def forward(self, x):
        features = self.backbone(x)
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
        'predictions_mars_transformer.npy',  # TensorFlow预测结果
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

def predict_with_torch_model(model_path='mars_transformer_best.pth', feature_path='/home/uo/myProject/HumanPoint-BERTdata/MARS/featuremap_test.npy'):
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
        
        # 加载模型
        print(f"🔄 加载PyTorch模型: {model_path}")
        model = MARSTransformerModel(input_channels=5, output_dim=57)
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
        
        predictions = np.concatenate(predictions, axis=0)
        print(f"✓ 预测完成: {predictions.shape}")
        
        # 保存预测结果
        output_file = 'predictions_mars_transformer_torch_live.npy'
        np.save(output_file, predictions)
        print(f"✓ 预测结果已保存: {output_file}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ 模型推理失败: {e}")
        return None

def plot_3d_skeleton(joints, ax, title, color='blue', linewidth=2, alpha=1.0):
    """绘制3D骨架"""
    # 数据格式: (x1,x2,...,x19, y1,y2,...,y19, z1,z2,...,z19)
    if joints.shape == (57,):
        # 重新组织数据为 (19, 3) 格式
        x_coords = joints[0:19]    # x坐标: 0-18
        y_coords = joints[19:38]   # y坐标: 19-37  
        z_coords = joints[38:57]   # z坐标: 38-56
        joints = np.column_stack((x_coords, y_coords, z_coords))
    
    # 绘制关节点
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], 
              c=color, s=80, alpha=alpha, edgecolors='black', linewidths=0.5)
    
    # 绘制骨架连接线
    for connection in skeleton_connections:
        if connection[0] < len(joints) and connection[1] < len(joints):
            joint1 = joints[connection[0]]
            joint2 = joints[connection[1]]
            ax.plot([joint1[0], joint2[0]], 
                   [joint1[1], joint2[1]], 
                   [joint1[2], joint2[2]], 
                   color=color, alpha=alpha, linewidth=linewidth)
    
    # 设置图形属性
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    # 根据实际数据范围设置坐标轴，保持xyz比例一致
    x_min, x_max = joints[:, 0].min(), joints[:, 0].max()
    y_min, y_max = joints[:, 1].min(), joints[:, 1].max()
    z_min, z_max = joints[:, 2].min(), joints[:, 2].max()
    
    # 计算各轴的范围
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    # 使用最大范围作为统一比例尺
    max_range = max(x_range, y_range, z_range)
    margin = max_range * 0.2  # 统一的边距
    
    # 计算每个轴的中心点
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2
    
    # 设置相同的范围，以各轴中心为基准
    half_range = max_range / 2 + margin
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)
    ax.set_zlim(z_center - half_range, z_center + half_range)
    
    # 设置视角
    ax.view_init(elev=20, azim=45)

def create_skeleton_comparison(ground_truth, prediction, sample_idx, output_dir):
    """创建单个样本的骨架对比图"""
    
    # 解析数据格式: (x1...x19, y1...y19, z1...z19)
    def parse_joints(joints_data):
        x_coords = joints_data[0:19]
        y_coords = joints_data[19:38]  
        z_coords = joints_data[38:57]
        return np.column_stack((x_coords, y_coords, z_coords))
    
    # 计算3D误差
    gt_joints = parse_joints(ground_truth)
    pred_joints = parse_joints(prediction)
    
    # 计算每个关节点的3D欧几里得距离误差
    joint_errors = np.sqrt(np.sum((gt_joints - pred_joints) ** 2, axis=1))
    mean_error = np.mean(joint_errors)
    
    # 创建图形
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f'Sample {sample_idx+1:02d} - Skeleton Comparison (PyTorch) (Mean 3D Error: {mean_error:.4f})', 
                fontsize=16, fontweight='bold')
    
    # Ground Truth骨架
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_skeleton(ground_truth, ax1, 'Ground Truth', color='blue', linewidth=3)
    
    # 预测骨架
    ax2 = fig.add_subplot(122, projection='3d')
    plot_3d_skeleton(prediction, ax2, 'PyTorch Prediction', color='red', linewidth=3)
    
    # 添加误差信息
    error_text = f"3D Error Statistics (PyTorch):\n"
    error_text += f"Mean: {mean_error:.4f}\n"
    error_text += f"Max: {np.max(joint_errors):.4f}\n"
    error_text += f"Min: {np.min(joint_errors):.4f}\n"
    error_text += f"Std: {np.std(joint_errors):.4f}"
    
    fig.text(0.02, 0.02, error_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # 保存图片
    filename = os.path.join(output_dir, f'skeleton_torch_sample_{sample_idx+1:02d}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ 已生成PyTorch版本: {filename} (3D Error: {mean_error:.4f})")
    
    return mean_error

def create_overlay_comparison(ground_truth, prediction, sample_idx, output_dir):
    """创建重叠对比图"""
    
    # 解析数据格式
    def parse_joints(joints_data):
        x_coords = joints_data[0:19]
        y_coords = joints_data[19:38]  
        z_coords = joints_data[38:57]
        return np.column_stack((x_coords, y_coords, z_coords))
    
    # 计算误差
    gt_joints = parse_joints(ground_truth)
    pred_joints = parse_joints(prediction)
    joint_errors = np.sqrt(np.sum((gt_joints - pred_joints) ** 2, axis=1))
    mean_error = np.mean(joint_errors)
    
    # 创建重叠图
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(f'Sample {sample_idx+1:02d} - PyTorch Overlay Comparison', fontsize=16, fontweight='bold')
    
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制Ground Truth (蓝色)
    plot_3d_skeleton(ground_truth, ax, '', color='blue', linewidth=3, alpha=0.8)
    
    # 绘制预测结果 (红色，透明)
    plot_3d_skeleton(prediction, ax, '', color='red', linewidth=2, alpha=0.6)
    
    # 设置标题和图例
    ax.set_title(f'Ground Truth (Blue) vs PyTorch Prediction (Red)\nMean 3D Error: {mean_error:.4f}', 
                fontsize=14, fontweight='bold')
    
    # 添加图例 (使用文本标注代替)
    ax.text2D(0.02, 0.95, "● Ground Truth (Blue)", transform=ax.transAxes, 
              color='blue', fontsize=12, fontweight='bold')
    ax.text2D(0.02, 0.90, "● PyTorch Prediction (Red)", transform=ax.transAxes, 
              color='red', fontsize=12, fontweight='bold')
    
    # 保存图片
    filename = os.path.join(output_dir, f'overlay_torch_sample_{sample_idx+1:02d}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ 已生成PyTorch重叠图: {filename}")
    
    return mean_error

def main(use_live_prediction=True):
    """主函数"""
    print("MARS+Transformer骨架可视化工具 (PyTorch版本)")
    print("=" * 60)
    
    # 创建输出目录 (使用torch特定名称)
    output_dir = 'visualizations/skeleton_extractor'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ 已创建输出目录: {output_dir}")
    
    # 加载Ground Truth  /home/uo/myProject/HumanPoint-BERT/data/MARS/labels_test.npy
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
    
    # 生成前10个样本的可视化
    print(f"\n开始生成前10个样本的骨架可视化 (PyTorch版本)...")
    print("-" * 50)
    
    total_error = 0
    sample_errors = []
    
    for i in range(min(10, len(ground_truth))):
        # 生成对比图
        error = create_skeleton_comparison(ground_truth[i], predictions[i], i, output_dir)
        
        # 生成重叠图
        create_overlay_comparison(ground_truth[i], predictions[i], i, output_dir)
        
        sample_errors.append(error)
        total_error += error
    
    # 生成汇总信息
    print("-" * 50)
    print(f"✅ PyTorch可视化完成!")
    print(f"生成文件数量: {len(os.listdir(output_dir))} 张图片")
    print(f"平均3D误差: {total_error/10:.4f}")
    print(f"最佳样本: Sample {np.argmin(sample_errors)+1:02d} (误差: {min(sample_errors):.4f})")
    print(f"最差样本: Sample {np.argmax(sample_errors)+1:02d} (误差: {max(sample_errors):.4f})")
    
    # 创建误差总结文件
    summary_file = os.path.join(output_dir, 'error_summary_torch.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("MARS+Transformer骨架预测误差总结 (PyTorch版本)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"模型: MARS+Transformer PyTorch (mars_transformer_best.pth)\n")
        f.write(f"测试时间: {np.datetime64('now')}\n\n")
        
        for i, error in enumerate(sample_errors):
            f.write(f"Sample {i+1:02d}: {error:.6f}\n")
        
        f.write(f"\n统计信息:\n")
        f.write(f"平均误差: {total_error/10:.6f}\n")
        f.write(f"标准差: {np.std(sample_errors):.6f}\n")
        f.write(f"最小误差: {min(sample_errors):.6f}\n")
        f.write(f"最大误差: {max(sample_errors):.6f}\n")
        
        # 添加性能对比信息
        f.write(f"\n模型信息:\n")
        f.write(f"框架: PyTorch\n")
        f.write(f"架构: MARS CNN + Transformer Attention\n")
        f.write(f"输入格式: (N, 5, 8, 8) - PyTorch格式\n")
        f.write(f"输出维度: 57 (19个关节点的3D坐标)\n")
    
    print(f"✓ PyTorch误差总结已保存: {summary_file}")
    print(f"\n📁 所有文件已保存到: {output_dir}/")
    print(f"🚀 MARS+Transformer PyTorch可视化完成!")

if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    use_live = True
    if len(sys.argv) > 1:
        if sys.argv[1].lower() in ['--no-live', '-n']:
            use_live = False
            print("📁 强制使用预保存的预测结果")
        elif sys.argv[1].lower() in ['--help', '-h']:
            print("MARS+Transformer PyTorch可视化工具")
            print("用法:")
            print("  python skeleton_visualization_torch.py           # 优先使用实时模型预测")
            print("  python skeleton_visualization_torch.py --no-live # 只使用预保存结果")
            print("  python skeleton_visualization_torch.py --help    # 显示帮助")
            exit(0)
    
    main(use_live_prediction=use_live)