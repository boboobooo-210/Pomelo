"""
MARS雷达数据集加载器 - 简化版
用于加载雷达点云数据和对应的骨架标签，专注于基本的数据转换
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import logging


class MARSRadarDatasetSimple(Dataset):
    """MARS雷达数据集 - 简化版"""
    
    def __init__(self, 
                 data_path: str,
                 subset: str = 'train',
                 normalize: bool = True):
        """
        Args:
            data_path: 数据根目录路径 (data/MARS/)
            subset: 数据子集 ('train', 'test', 'validate')
            normalize: 是否归一化
        """
        self.data_path = data_path
        self.subset = subset
        self.normalize = normalize
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 加载数据
        self._load_data()
        
        # 数据统计
        self.logger.info(f"Loaded {subset} dataset: {len(self)} samples")
        
    def _load_data(self):
        """加载数据文件"""
        # 构建文件路径
        featuremap_file = os.path.join(self.data_path, f'featuremap_{self.subset}.npy')
        labels_file = os.path.join(self.data_path, f'labels_{self.subset}.npy')
        
        # 检查文件是否存在
        if not os.path.exists(featuremap_file):
            raise FileNotFoundError(f"Feature map file not found: {featuremap_file}")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        # 加载数据
        raw_featuremaps = np.load(featuremap_file)  # [N, 8, 8, 5]
        raw_labels = np.load(labels_file)           # [N, 57]
        
        # 数据形状检查
        assert len(raw_featuremaps) == len(raw_labels), \
            f"Feature maps and labels length mismatch: {len(raw_featuremaps)} vs {len(raw_labels)}"
        
        # 处理特征图：从(N,8,8,5)处理为(N,64,3)
        if raw_featuremaps.shape[1:] == (8, 8, 5):
            # 1. 合并中间两个维度: (N,8,8,5) -> (N,64,5)
            reshaped_features = raw_featuremaps.reshape(-1, 64, 5)
            # 2. 切片去掉多普勒特征: (N,64,5) -> (N,64,3)
            self.featuremaps = reshaped_features[:, :, :3]
            self.logger.info("Converted featuremaps from (N,8,8,5) to (N,64,3)")
        elif raw_featuremaps.shape[1:] == (64, 3):
            self.featuremaps = raw_featuremaps
            self.logger.info("Featuremaps already in (N,64,3) format")
        else:
            raise ValueError(f"Unexpected feature map shape: {raw_featuremaps.shape[1:]} (expected (8,8,5) or (64,3))")
        
        # 处理标签：从MARS格式(N,3,19)转换为模型期望格式(N,19,3)
        if raw_labels.shape[1:] == (57,):
            # MARS原始格式：[x1,x2,...,x19, y1,y2,...,y19, z1,z2,...,z19]
            # 需要转换为：[x1,y1,z1, x2,y2,z2, ..., x19,y19,z19]

            # 1. 重塑为(N,3,19)
            labels_reshaped = raw_labels.reshape(-1, 3, 19)
            # 2. 转置为(N,19,3)
            labels_transposed = labels_reshaped.transpose(0, 2, 1)
            # 3. 重新展平为(N,57)，但现在是正确的格式
            self.labels = labels_transposed.reshape(-1, 57)

            self.logger.info("Converted labels from MARS format (N,3,19) to model format (N,19,3)")
            self.logger.info("Original format: [x1,x2,...,x19, y1,y2,...,y19, z1,z2,...,z19]")
            self.logger.info("Converted format: [x1,y1,z1, x2,y2,z2, ..., x19,y19,z19]")
        else:
            raise ValueError(f"Unexpected label shape: {raw_labels.shape[1:]} (expected (57,))")
        
        # 验证最终数据形状
        expected_feature_shape = (64, 3)  # 64个点，每个点3维坐标
        expected_label_shape = (57,)      # 57维标签向量
        
        if self.featuremaps.shape[1:] != expected_feature_shape:
            raise ValueError(f"Final feature map shape mismatch: {self.featuremaps.shape[1:]} vs {expected_feature_shape}")
        if self.labels.shape[1:] != expected_label_shape:
            raise ValueError(f"Final label shape mismatch: {self.labels.shape[1:]} vs {expected_label_shape}")
        
        # 数据预处理
        if self.normalize:
            self._normalize_data()

        self.logger.info(f"Feature maps shape: {self.featuremaps.shape}")
        self.logger.info(f"Labels shape: {self.labels.shape}")
        
    def _normalize_data(self):
        """数据归一化 - 修正版本，保持骨架结构"""
        # 归一化特征图 (3维空间坐标)
        spatial_coords = self.featuremaps  # [N, 64, 3]
        spatial_mean = np.mean(spatial_coords, axis=(0, 1), keepdims=True)
        spatial_std = np.std(spatial_coords, axis=(0, 1), keepdims=True) + 1e-8
        self.featuremaps = (spatial_coords - spatial_mean) / spatial_std

        # 修正：对骨架标签使用更合适的归一化方法
        # 将57维向量重塑为19x3，然后进行空间归一化
        original_shape = self.labels.shape  # [N, 57]

        # 重塑为 [N, 19, 3]
        labels_3d = self.labels.reshape(-1, 19, 3)

        # 计算每个样本的中心点和尺度
        # 使用所有样本的统计信息来保持一致性
        all_coords = labels_3d.reshape(-1, 3)  # [N*19, 3]

        # 计算全局中心点（所有关节的平均位置）
        global_center = np.mean(all_coords, axis=0, keepdims=True)  # [1, 3]

        # 计算全局尺度（使用标准差的平均值作为统一尺度）
        centered_coords = all_coords - global_center
        global_scale = np.mean(np.std(centered_coords, axis=0)) + 1e-8

        # 对每个样本进行归一化：中心化 + 统一缩放
        labels_3d_normalized = (labels_3d - global_center) / global_scale

        # 重塑回57维
        self.labels = labels_3d_normalized.reshape(original_shape)

        # 保存归一化参数
        self.norm_params = {
            'spatial_mean': spatial_mean,
            'spatial_std': spatial_std,
            'skeleton_center': global_center,
            'skeleton_scale': global_scale
        }
        
    def __len__(self):
        return len(self.featuremaps)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取原始数据
        featuremap = self.featuremaps[idx].copy()  # [64, 3]
        skeleton = self.labels[idx].copy()         # [57]

        # 转换为tensor并直接返回元组
        featuremap_tensor = torch.from_numpy(featuremap).float()
        skeleton_tensor = torch.from_numpy(skeleton).float()

        return featuremap_tensor, skeleton_tensor
    
    def get_statistics(self):
        """获取数据集统计信息"""
        stats = {
            'num_samples': len(self),
            'feature_shape': self.featuremaps.shape[1:],
            'label_shape': self.labels.shape[1:],
            'feature_range': {
                'min': np.min(self.featuremaps),
                'max': np.max(self.featuremaps),
                'mean': np.mean(self.featuremaps),
                'std': np.std(self.featuremaps)
            },
            'skeleton_range': {
                'min': np.min(self.labels),
                'max': np.max(self.labels),
                'mean': np.mean(self.labels),
                'std': np.std(self.labels)
            }
        }
        return stats


def create_mars_radar_dataloaders_simple(data_path: str, 
                                         batch_size: int = 32,
                                         num_workers: int = 4,
                                         normalize: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建MARS雷达数据加载器 - 简化版"""
    
    # 创建数据集
    train_dataset = MARSRadarDatasetSimple(
        data_path=data_path,
        subset='train',
        normalize=normalize
    )
    
    val_dataset = MARSRadarDatasetSimple(
        data_path=data_path,
        subset='validate',
        normalize=normalize
    )
    
    test_dataset = MARSRadarDatasetSimple(
        data_path=data_path,
        subset='test',
        normalize=normalize
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
