import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class MARS(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS  # 550个点
        
        # 根据subset确定数据文件
        if self.subset == 'train':
            self.data_file = os.path.join(self.data_root, 'Augmented_labels_train.npy')
        elif self.subset == 'test':
            self.data_file = os.path.join(self.data_root, 'Augmented_labels_test.npy')
        elif self.subset == 'val':
            self.data_file = os.path.join(self.data_root, 'Augmented_labels_validate.npy')
        else:
            raise ValueError(f"Unknown subset: {self.subset}")
        
        self.sample_points_num = config.get('npoints', self.npoints)
        self.whole = config.get('whole', False)
        self.augment = config.get('augment', True)  # 是否进行数据增强
        
        print_log(f'[DATASET] Loading MARS {self.subset} dataset from {self.data_file}', logger='MARS')
        print_log(f'[DATASET] Sample out {self.sample_points_num} points', logger='MARS')
        
        # 加载数据
        self.data = np.load(self.data_file).astype(np.float32)
        print_log(f'[DATASET] {self.data.shape[0]} instances were loaded', logger='MARS')
        print_log(f'[DATASET] Data shape: {self.data.shape}', logger='MARS')
        
        # 生成索引排列用于随机采样
        self.permutation = np.arange(self.npoints)
        
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        if m > 0:
            pc = pc / m
        return pc
    
    def random_sample(self, pc, num):
        """随机采样点云"""
        if pc.shape[0] >= num:
            np.random.shuffle(self.permutation[:pc.shape[0]])
            pc = pc[self.permutation[:num]]
        else:
            # 如果点数不足，进行重复采样
            indices = np.random.choice(pc.shape[0], num, replace=True)
            pc = pc[indices]
        return pc
    
    def __getitem__(self, idx):
        # 获取骨架点云数据
        data = self.data[idx].copy()
        
        # 智能采样策略
        if self.sample_points_num < data.shape[0]:
            # 如果需要采样更少的点，进行随机采样（数据增强）
            data = self.random_sample(data, self.sample_points_num)
        elif self.sample_points_num > data.shape[0]:
            # 如果需要更多点，进行重复采样
            indices = np.random.choice(data.shape[0], self.sample_points_num, replace=True)
            data = data[indices]
        else:
            # 如果点数相等，根据训练阶段决定是否进行数据增强
            if self.subset == 'train' and self.augment:
                # 训练时可以选择随机打乱点的顺序作为轻微增强
                data = self.random_sample(data, self.sample_points_num)
            # 验证和测试时保持原始顺序
        
        # 标准化点云
        data = self.pc_norm(data)
        
        # 转换为torch张量
        data = torch.from_numpy(data).float()
        
        # 为了与现有框架兼容，返回taxonomy_id和model_id
        # 对于MARS数据集，我们使用通用的标识符
        taxonomy_id = 'human_skeleton'
        model_id = f'mars_{idx:06d}'
        
        return taxonomy_id, model_id, data
    
    def __len__(self):
        return len(self.data)
