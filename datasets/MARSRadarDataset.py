"""
MARS雷达数据集加载器
用于加载雷达点云数据和对应的骨架标签
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import logging


class MARSRadarDataset(Dataset):
    """MARS雷达数据集"""
    
    def __init__(self, 
                 data_path: str,
                 subset: str = 'train',
                 transform=None,
                 normalize: bool = True,
                 augment: bool = False):
        """
        Args:
            data_path: 数据根目录路径 (data/MARS/)
            subset: 数据子集 ('train', 'test', 'validate')
            transform: 数据变换
            normalize: 是否归一化
            augment: 是否数据增强
        """
        self.data_path = data_path
        self.subset = subset
        self.transform = transform
        self.normalize = normalize
        self.augment = augment
        
        # 加载数据
        self._load_data()
        
        # 数据统计
        self.logger = logging.getLogger(__name__)
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

        # 处理标签：从MARS格式(N,57)转换为标准格式
        if raw_labels.shape[1:] == (57,):
            # MARS原始格式: [x1,x2,...,x19, y1,y2,...,y19, z1,z2,...,z19]
            # 目标格式: [x1,y1,z1, x2,y2,z2, ..., x19,y19,z19]

            # 重塑为(N,3,19): 3个坐标维度，每个维度19个关节值
            reshaped_labels = raw_labels.reshape(-1, 3, 19)
            # 转置为(N,19,3): 19个关节，每个关节3个坐标
            transposed_labels = reshaped_labels.transpose(0, 2, 1)
            # 重新展平为(N,57)但现在是正确的格式
            self.labels = transposed_labels.reshape(-1, 57)

            self.logger.info("Labels converted from MARS format (N,3,19) to standard format (N,19,3)")
            self.logger.info(f"Original shape: {raw_labels.shape}, Final shape: {self.labels.shape}")
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
        """数据归一化"""
        # 归一化特征图 (3维空间坐标)
        spatial_coords = self.featuremaps  # [N, 64, 3]
        spatial_mean = np.mean(spatial_coords, axis=(0, 1), keepdims=True)
        spatial_std = np.std(spatial_coords, axis=(0, 1), keepdims=True) + 1e-8
        self.featuremaps = (spatial_coords - spatial_mean) / spatial_std

        # 归一化标签 (57维向量)
        label_mean = np.mean(self.labels, axis=0, keepdims=True)
        label_std = np.std(self.labels, axis=0, keepdims=True) + 1e-8
        self.labels = (self.labels - label_mean) / label_std

        # 保存归一化参数
        self.norm_params = {
            'spatial_mean': spatial_mean,
            'spatial_std': spatial_std,
            'label_mean': label_mean,
            'label_std': label_std
        }
        

        
    def __len__(self):
        return len(self.featuremaps)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取原始数据
        featuremap = self.featuremaps[idx].copy()  # [64, 5]
        skeleton = self.labels[idx].copy()         # [57]

        # 数据增强（暂时不使用）
        # if self.augment and self.subset == 'train':
        #     featuremap, skeleton = self._augment_data(featuremap, skeleton)

        # 应用变换
        if self.transform:
            featuremap = self.transform(featuremap)

        # 转换为tensor
        sample = {
            'featuremap': torch.from_numpy(featuremap).float(),
            'skeleton': torch.from_numpy(skeleton).float(),
            'index': idx
        }

        return sample
    

    
    def get_statistics(self):
        """获取数据集统计信息"""
        stats = {
            'num_samples': len(self),
            'feature_shape': self.featuremaps.shape[1:],
            'label_shape': self.labels.shape[1:],
            'feature_range': {
                'spatial': {
                    'min': np.min(self.featuremaps),
                    'max': np.max(self.featuremaps),
                    'mean': np.mean(self.featuremaps),
                    'std': np.std(self.featuremaps)
                }
            },
            'skeleton_range': {
                'min': np.min(self.labels),
                'max': np.max(self.labels),
                'mean': np.mean(self.labels),
                'std': np.std(self.labels)
            }
        }
        return stats


def create_mars_radar_dataloaders(data_path: str, 
                                 batch_size: int = 32,
                                 num_workers: int = 4,
                                 normalize: bool = True,
                                 augment: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建MARS雷达数据加载器"""
    
    # 创建数据集
    train_dataset = MARSRadarDataset(
        data_path=data_path,
        subset='train',
        normalize=normalize,
        augment=augment
    )
    
    val_dataset = MARSRadarDataset(
        data_path=data_path,
        subset='validate',
        normalize=normalize,
        augment=False
    )
    
    test_dataset = MARSRadarDataset(
        data_path=data_path,
        subset='test',
        normalize=normalize,
        augment=False
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
