#!/usr/bin/env python3
"""
MARS + 轻量级Transformer骨架提取模型 (PyTorch版本)
结合原始MARS CNN架构和Transformer注意力机制
保持原始数据格式(n,8,8,5)不变
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import math

# 设置matplotlib后端
plt.switch_backend('Agg')

# GPU配置
def configure_gpu():
    """配置GPU使用"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return device
    else:
        print("❌ 未检测到GPU，将使用CPU")
        return torch.device('cpu')

# SE注意力模块
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
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1)
        # Scale
        return x * y

# 残差块 + SE注意力
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
        
        # 残差连接的投影层
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

# 空间注意力模块
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

# 位置编码
class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
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

# 轻量级Transformer块
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
        # 自注意力
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

# 增强的MARS主干网络（多尺度特征融合版本）
class EnhancedMARSBackbone(nn.Module):
    """创建增强的MARS主干网络 - 支持多尺度特征融合"""
    def __init__(self, input_channels=5, multi_scale=True):
        super(EnhancedMARSBackbone, self).__init__()
        
        self.multi_scale = multi_scale
        
        # 初始特征提取
        self.initial_conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.initial_bn1 = nn.BatchNorm2d(32)
        self.initial_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.initial_bn2 = nn.BatchNorm2d(32)
        
        # 残差SE块
        self.res_se_1 = ResidualSEBlock(32, 64)
        self.maxpool1 = nn.MaxPool2d(2, 2)  # 8x8 -> 4x4
        
        self.res_se_2 = ResidualSEBlock(64, 128)
        self.spatial_att_1 = SpatialAttention()
        
        self.res_se_3 = ResidualSEBlock(128, 256)
        self.spatial_att_2 = SpatialAttention()
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.relu = nn.ReLU(inplace=True)
        
        # 输出特征维度: 64 + 128 + 256 = 448 (多尺度) 或 256 (单尺度)
        self.output_dim = 448 if multi_scale else 256
    
    def forward(self, x):
        # 初始特征提取
        x = self.relu(self.initial_bn1(self.initial_conv1(x)))
        x = self.relu(self.initial_bn2(self.initial_conv2(x)))
        
        # 第一个残差SE块 - 保存特征1
        feat1 = self.res_se_1(x)  # (batch, 64, 8, 8) 或 (batch, 64, 4, 4) after pooling
        x = self.maxpool1(feat1)  # (batch, 64, 4, 4)
        
        # 第二个残差SE块 + 空间注意力 - 保存特征2
        feat2 = self.res_se_2(x)  # (batch, 128, 4, 4)
        feat2 = self.spatial_att_1(feat2)
        
        # 第三个残差SE块 + 空间注意力 - 保存特征3
        feat3 = self.res_se_3(feat2)  # (batch, 256, 4, 4)
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

# Transformer增强的回归头
class TransformerRegressionHead(nn.Module):
    """创建Transformer增强的回归头 - 简洁高效版本"""
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

# 完整的MARS+Transformer模型
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
        
        print(f"✓ 模型初始化完成 - 多尺度融合: {multi_scale}, 输入维度: {input_dim}")
    
    def forward(self, x):
        # CNN特征提取（多尺度或单尺度）
        features = self.backbone(x)
        
        # Transformer回归
        output = self.regression_head(features)
        
        return output

# 数据集类
class RadarSkeletonDataset(Dataset):
    """雷达骨架数据集"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 数据加载和预处理
def load_and_preprocess_data():
    """加载并预处理数据"""
    print("🔄 加载数据...")
    
    # 加载特征数据
    featuremap_train = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_train.npy')
    featuremap_validate = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_validate.npy')
    featuremap_test = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/featuremap_test.npy')
    
    print(f"训练数据: {featuremap_train.shape}")
    print(f"验证数据: {featuremap_validate.shape}")
    print(f"测试数据: {featuremap_test.shape}")
    
    # 加载标签数据
    labels_train = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/labels_train.npy')
    labels_validate = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/labels_validate.npy')
    labels_test = np.load('/home/uo/myProject/HumanPoint-BERT/data/MARS/labels_test.npy')
    
    print(f"标签数据: {labels_train.shape}")
    
    # 转换数据格式：(N, H, W, C) -> (N, C, H, W)
    featuremap_train = np.transpose(featuremap_train, (0, 3, 1, 2))
    featuremap_validate = np.transpose(featuremap_validate, (0, 3, 1, 2))
    featuremap_test = np.transpose(featuremap_test, (0, 3, 1, 2))
    
    return (featuremap_train, featuremap_validate, featuremap_test, 
            labels_train, labels_validate, labels_test)

def create_data_loaders(train_features, train_labels, val_features, val_labels, 
                       test_features, test_labels, batch_size=32):
    """创建数据加载器"""
    train_dataset = RadarSkeletonDataset(train_features, train_labels)
    val_dataset = RadarSkeletonDataset(val_features, val_labels)
    test_dataset = RadarSkeletonDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, num_epochs=250):
    """训练模型"""
    print("🚀 开始训练MARS+Transformer模型...")
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, 
                                                    factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    
    # 早停和模型保存
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += F.l1_loss(outputs, batch_labels).item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                val_mae += F.l1_loss(outputs, batch_labels).item()
        
        # 计算平均损失
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:3d}/{num_epochs} - "
              f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f} - "
              f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停和模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            try:
                # 先保存到临时文件，然后重命名，避免保存过程中被中断
                torch.save(model.state_dict(), 'mars_transformer_best_tmp.pth')
                import os
                if os.path.exists('mars_transformer_best.pth'):
                    os.remove('mars_transformer_best.pth')
                os.rename('mars_transformer_best_tmp.pth', 'mars_transformer_best.pth')
                print(f"✓ 保存最佳模型 (Val Loss: {val_loss:.6f})")
            except Exception as e:
                print(f"⚠️ 模型保存失败: {e}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 保存最终模型
    torch.save(model.state_dict(), 'mars_transformer_final.pth')
    print("✓ 保存最终模型")
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, device):
    """评估模型"""
    print("📊 评估模型性能...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_features)
            
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())
    
    # 合并所有结果
    predictions = np.concatenate(all_predictions, axis=0)
    ground_truth = np.concatenate(all_labels, axis=0)
    
    # 计算评估指标
    mae = mean_absolute_error(ground_truth, predictions)
    mse = mean_squared_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    
    print(f"测试集性能:")
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    
    # 保存预测结果
    np.save('predictions_mars_transformer_torch.npy', predictions)
    print("✓ 预测结果已保存: predictions_mars_transformer_torch.npy")
    
    return predictions, ground_truth, mae, mse, rmse

def main():
    """主函数"""
    print("MARS+Transformer骨架提取模型 (PyTorch版本)")
    print("=" * 60)
    
    # 配置设备
    device = configure_gpu()
    
    # 加载数据
    (train_features, val_features, test_features, 
     train_labels, val_labels, test_labels) = load_and_preprocess_data()
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_features, train_labels, val_features, val_labels, 
        test_features, test_labels, batch_size=32
    )
    
    # 创建模型（启用多尺度特征融合）
    print("\n🔧 模型配置:")
    print("  - 多尺度特征融合: 启用 (64 + 128 + 256 = 448维)")
    print("  - Transformer层数: 2层")
    print("  - 回归头: 渐进式降维设计")
    model = MARSTransformerModel(input_channels=5, output_dim=57, multi_scale=True).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 训练模型 (100轮)
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, num_epochs=250)
    
    # 加载最佳模型进行评估
    try:
        model.load_state_dict(torch.load('mars_transformer_best.pth', map_location=device))
        print("✓ 成功加载最佳模型")
    except Exception as e:
        print(f"⚠️ 加载最佳模型失败: {e}")
        print("使用当前训练后的模型进行评估")
    
    # 评估模型
    predictions, ground_truth, mae, mse, rmse = evaluate_model(model, test_loader, device)
    
    # 绘制训练历史
    # plot_training_history(train_losses, val_losses)
    
    print("\n🎉 MARS+Transformer训练完成 (PyTorch版本)!")
    print(f"✓ 最佳模型: mars_transformer_best.pth")
    print(f"✓ 最终模型: mars_transformer_final.pth")
    print(f"✓ 预测结果: predictions_mars_transformer_torch.npy")
    print(f"✓ 训练历史: mars_transformer_training_history_torch.png")

if __name__ == "__main__":
    main()