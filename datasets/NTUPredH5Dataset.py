"""
NTU-Pred H5数据集加载器
专门用于加载data/NTU-Pred目录下的增强H5文件

特点：
1. 直接加载H5文件中的720点增强骨架数据
2. 支持多帧数据的智能选择（中间帧/随机帧）
3. 保持与原有训练框架的兼容性
4. 高效的数据加载和预处理
5. 内置解剖学分组支持
"""

import os
import torch
import numpy as np
import h5py
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
import logging
import re
from pathlib import Path


# 动作分类定义（与原NTU数据集保持一致）
SINGLE_DAILY_ACTIONS = list(range(1, 41)) + list(range(61, 103))
REHABILITATION_ACTIONS = list(range(41, 50)) + list(range(103, 106))
INTERACTION_ACTIONS = list(range(50, 61)) + list(range(107, 121))

# 移除康复动作从单人日常动作中
for action in REHABILITATION_ACTIONS:
    if action in SINGLE_DAILY_ACTIONS:
        SINGLE_DAILY_ACTIONS.remove(action)

# 适合骨架学习的动作 (单人动作，包括日常和康复)
SKELETON_LEARNING_ACTIONS = SINGLE_DAILY_ACTIONS + REHABILITATION_ACTIONS
DVAE_ACTIONS = SKELETON_LEARNING_ACTIONS


@DATASETS.register_module()
class NTUPredH5Dataset(data.Dataset):
    """
    NTU-Pred H5数据集加载器
    直接加载预处理的H5文件，无需实时数据增强
    """
    
    def __init__(self, config):
        # 兼容字典和对象两种输入格式
        if isinstance(config, dict):
            # 字典格式，使用 get 方法
            self.data_root = config.get('DATA_PATH', './data/NTU-Pred')
            self.subset = config.get('subset', 'train')
            self.npoints = config.get('N_POINTS', 720)
            self.target_points = config.get('npoints', 720)
            self.action_filter = config.get('action_filter', 'dvae')
            self.whole = config.get('whole', False)
            self.augment = config.get('augment', True)
            
            # H5特定配置
            self.data_key = config.get('data_key', 'enhanced_data')
            self.frame_selection = config.get('frame_selection', 'middle')
            self.coordinate_conversion = config.get('coordinate_conversion', True)
            self.validation_checks = config.get('validation_checks', True)
            
            # 解剖学分组配置
            self.num_group = config.get('num_group', 7)
        else:
            # 对象格式，使用属性访问
            self.data_root = config.DATA_PATH
            self.subset = config.subset
            self.npoints = config.get('N_POINTS', 720) if hasattr(config, 'get') else getattr(config, 'N_POINTS', 720)
            self.target_points = config.get('npoints', 720) if hasattr(config, 'get') else getattr(config, 'npoints', 720)
            self.action_filter = config.get('action_filter', 'dvae') if hasattr(config, 'get') else getattr(config, 'action_filter', 'dvae')
            self.whole = config.get('whole', False) if hasattr(config, 'get') else getattr(config, 'whole', False)
            self.augment = config.get('augment', True) if hasattr(config, 'get') else getattr(config, 'augment', True)
            
            # H5特定配置
            self.data_key = config.get('data_key', 'enhanced_data') if hasattr(config, 'get') else getattr(config, 'data_key', 'enhanced_data')
            self.frame_selection = config.get('frame_selection', 'middle') if hasattr(config, 'get') else getattr(config, 'frame_selection', 'middle')
            self.coordinate_conversion = config.get('coordinate_conversion', True) if hasattr(config, 'get') else getattr(config, 'coordinate_conversion', True)
            self.validation_checks = config.get('validation_checks', True) if hasattr(config, 'get') else getattr(config, 'validation_checks', True)
            
            # 解剖学分组配置
            self.num_group = config.get('num_group', 7) if hasattr(config, 'get') else getattr(config, 'num_group', 7)
        self.anatomical_regions = self._init_anatomical_regions()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        print_log(f'[DATASET] Loading NTU-Pred H5 {self.subset} dataset', logger='NTUPredH5')
        print_log(f'[DATASET] Data root: {self.data_root}', logger='NTUPredH5')
        print_log(f'[DATASET] Target points: {self.target_points}', logger='NTUPredH5')
        print_log(f'[DATASET] Action filter: {self.action_filter}', logger='NTUPredH5')
        print_log(f'[DATASET] Frame selection: {self.frame_selection}', logger='NTUPredH5')
        print_log(f'[DATASET] Anatomical groups: {self.num_group}', logger='NTUPredH5')
        
        # 加载数据
        self._load_h5_data()
        
        print_log(f'[DATASET] {len(self.data_files)} H5 files loaded', logger='NTUPredH5')
        
        # 生成索引排列用于随机采样
        if self.target_points != self.npoints:
            self.permutation = np.arange(self.npoints)
        else:
            self.permutation = None
    
    def _init_anatomical_regions(self):
        """初始化解剖学分组区域定义"""
        return {
            'joints': {'range': (0, 25), 'name': '原始关节点'},
            'torso_lower': {'range': (25, 88), 'name': '下躯干区域'},
            'head_neck': {'range': (88, 115), 'name': '头颈区域'},
            'left_arm': {'range': (115, 250), 'name': '左臂区域'},
            'right_arm': {'range': (250, 385), 'name': '右臂区域'},
            'left_leg': {'range': (385, 548), 'name': '左腿区域'},
            'right_leg': {'range': (548, 720), 'name': '右腿区域'}
        }
    
    def _load_h5_data(self):
        """加载H5数据文件列表"""
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"H5数据目录不存在: {self.data_root}")
        
        # 收集所有H5文件
        h5_files = []
        for file in os.listdir(self.data_root):
            if file.endswith('.h5'):
                h5_files.append(file)
        
        print_log(f'[DATASET] Found {len(h5_files)} H5 files', logger='NTUPredH5')
        
        # 根据动作过滤器筛选文件
        valid_files = []
        for file in h5_files:
            action_id = self._parse_action_id(file)
            if self._is_valid_action(action_id):
                valid_files.append(file)
        
        print_log(f'[DATASET] {len(valid_files)} files match action filter: {self.action_filter}', 
                 logger='NTUPredH5')
        
        # 数据集划分
        self.data_files = self._split_dataset(valid_files)
        print_log(f'[DATASET] {self.subset.upper()} split: {len(self.data_files)} files', 
                 logger='NTUPredH5')
        
        # 预加载文件信息（可选，用于验证）
        if self.validation_checks:
            self._validate_h5_files()
    
    def _parse_action_id(self, filename):
        """从文件名解析动作ID"""
        try:
            # NTU文件名格式: S***C***P***R***A***.h5
            match = re.search(r'A(\d{3})\.h5$', filename)
            if match:
                return int(match.group(1))
            else:
                print_log(f'[DATASET] Cannot parse action ID from {filename}', logger='NTUPredH5')
                return 0
        except Exception as e:
            print_log(f'[DATASET] Error parsing action ID from {filename}: {e}', logger='NTUPredH5')
            return 0
    
    def _parse_subject_id(self, filename):
        """从文件名解析受试者ID"""
        try:
            match = re.search(r'P(\d{3})', filename)
            if match:
                return int(match.group(1))
            else:
                print_log(f'[DATASET] Cannot parse subject ID from {filename}', logger='NTUPredH5')
                return 1
        except Exception as e:
            print_log(f'[DATASET] Error parsing subject ID from {filename}: {e}', logger='NTUPredH5')
            return 1
    
    def _is_valid_action(self, action_id):
        """判断动作是否符合过滤条件"""
        if self.action_filter == 'dvae':
            return action_id in DVAE_ACTIONS
        elif self.action_filter == 'daily':
            return action_id in SINGLE_DAILY_ACTIONS
        elif self.action_filter == 'rehab':
            return action_id in REHABILITATION_ACTIONS
        elif self.action_filter == 'all':
            return True
        else:
            return action_id in DVAE_ACTIONS
    
    def _split_dataset(self, valid_files):
        """数据集划分"""
        # NTU RGB+D标准划分
        train_subjects = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]
        test_subjects = [3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40]
        val_subjects = [1, 2, 4, 5]  # 从训练集分出验证集
        train_subjects = [s for s in train_subjects if s not in val_subjects]
        
        train_files = []
        val_files = []
        test_files = []
        
        for file in valid_files:
            subject_id = self._parse_subject_id(file)
            
            if subject_id in train_subjects:
                train_files.append(file)
            elif subject_id in val_subjects:
                val_files.append(file)
            elif subject_id in test_subjects:
                test_files.append(file)
            else:
                train_files.append(file)  # 默认分到训练集
        
        print_log(f'[DATASET] Data split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}',
                 logger='NTUPredH5')
        
        if self.subset == 'train':
            return train_files
        elif self.subset == 'val':
            return val_files
        elif self.subset == 'test':
            return test_files
        else:
            print_log(f'[DATASET] Unknown subset: {self.subset}, using train', logger='NTUPredH5')
            return train_files
    
    def _validate_h5_files(self):
        """验证H5文件的完整性"""
        print_log(f'[DATASET] Validating {len(self.data_files)} H5 files...', logger='NTUPredH5')
        
        valid_files = []
        invalid_files = []
        
        for file in self.data_files:
            file_path = os.path.join(self.data_root, file)
            try:
                with h5py.File(file_path, 'r') as f:
                    if self.data_key in f:
                        data_shape = f[self.data_key].shape
                        # 检查数据形状：应该是 (frames, points, 3)
                        if len(data_shape) == 3 and data_shape[1] == self.npoints and data_shape[2] == 3:
                            valid_files.append(file)
                        else:
                            print_log(f'[DATASET] Invalid data shape in {file}: {data_shape}', 
                                     logger='NTUPredH5')
                            invalid_files.append(file)
                    else:
                        print_log(f'[DATASET] Missing data key "{self.data_key}" in {file}', 
                                 logger='NTUPredH5')
                        invalid_files.append(file)
            except Exception as e:
                print_log(f'[DATASET] Error validating {file}: {e}', logger='NTUPredH5')
                invalid_files.append(file)
        
        self.data_files = valid_files
        
        print_log(f'[DATASET] Validation complete: {len(valid_files)} valid, {len(invalid_files)} invalid',
                 logger='NTUPredH5')
        
        if invalid_files:
            print_log(f'[DATASET] Invalid files: {invalid_files[:5]}{"..." if len(invalid_files) > 5 else ""}',
                     logger='NTUPredH5')
    
    def _read_h5_file(self, file_path):
        """读取H5文件数据"""
        try:
            with h5py.File(file_path, 'r') as f:
                if self.data_key not in f:
                    print_log(f'[DATASET] Missing data key "{self.data_key}" in {file_path}', 
                             logger='NTUPredH5')
                    return None
                
                data = f[self.data_key][:]  # [frames, points, 3]
                
                # 帧选择策略
                if len(data) == 0:
                    return None
                elif len(data) == 1:
                    selected_data = data[0]
                elif self.frame_selection == 'middle':
                    mid_idx = len(data) // 2
                    selected_data = data[mid_idx]
                elif self.frame_selection == 'random':
                    rand_idx = np.random.randint(len(data))
                    selected_data = data[rand_idx]
                else:  # 'middle' as default
                    mid_idx = len(data) // 2
                    selected_data = data[mid_idx]
                
                # 坐标转换（如果需要）
                if self.coordinate_conversion:
                    # NTU原始格式可能需要坐标轴转换
                    # 这里假设H5文件已经是正确的坐标格式
                    pass
                
                # 数据验证
                if selected_data.shape[0] != self.npoints or selected_data.shape[1] != 3:
                    print_log(f'[DATASET] Invalid data shape in {file_path}: {selected_data.shape}', 
                             logger='NTUPredH5')
                    return None
                
                # 检查是否有无效点（全零点）
                zero_points = np.all(selected_data == 0, axis=1).sum()
                if zero_points > self.npoints * 0.5:  # 超过50%的点为零
                    print_log(f'[DATASET] Too many zero points in {file_path}: {zero_points}/{self.npoints}', 
                             logger='NTUPredH5')
                    return None
                
                return selected_data.astype(np.float32)
                
        except Exception as e:
            print_log(f'[DATASET] Error reading {file_path}: {e}', logger='NTUPredH5')
            return None
    
    def _get_anatomical_groups(self, data):
        """将数据按解剖学区域分组"""
        groups = []
        for region_name, region_info in self.anatomical_regions.items():
            start_idx, end_idx = region_info['range']
            region_data = data[start_idx:end_idx].copy()
            groups.append(region_data)
        return groups
    
    def pc_norm(self, pc):
        """点云标准化"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        if m > 0:
            pc = pc / m
        return pc
    
    def random_sample(self, pc, num):
        """随机采样点云"""
        if pc.shape[0] >= num:
            if self.permutation is not None:
                np.random.shuffle(self.permutation)
                pc = pc[self.permutation[:num]]
            else:
                indices = np.random.choice(pc.shape[0], num, replace=False)
                pc = pc[indices]
        else:
            indices = np.random.choice(pc.shape[0], num, replace=True)
            pc = pc[indices]
        return pc
    
    def __getitem__(self, idx):
        """获取单个样本"""
        file_name = self.data_files[idx]
        file_path = os.path.join(self.data_root, file_name)
        
        # 读取H5数据
        data = self._read_h5_file(file_path)
        if data is None:
            # 如果读取失败，返回一个随机样本
            print_log(f'[DATASET] Failed to load {file_name}, using random sample', 
                     logger='NTUPredH5')
            data = np.random.randn(self.npoints, 3).astype(np.float32)
        
        # 点数调整（如果需要）
        if self.target_points != self.npoints:
            data = self.random_sample(data, self.target_points)
        
        # 数据增强
        if self.augment:
            # 随机旋转
            theta = np.random.uniform(0, 2 * np.pi)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
            data = np.dot(data, rotation_matrix.T)
            
            # 随机缩放
            scale = np.random.uniform(0.8, 1.2)
            data = data * scale
            
            # 添加噪声
            noise = np.random.normal(0, 0.01, data.shape)
            data = data + noise
        
        # 标准化
        data = self.pc_norm(data)
        
        # 转换为torch张量
        data = torch.from_numpy(data).float()
        
        # 解析文件信息
        action_id = self._parse_action_id(file_name)
        subject_id = self._parse_subject_id(file_name)
        
        # 兼容原有框架的返回格式
        taxonomy_id = 'human_skeleton'
        model_id = f'ntu_h5_{idx:06d}'
        
        # 返回额外信息用于调试和分析
        extra_info = {
            'file_name': file_name,
            'action_id': action_id,
            'subject_id': subject_id,
            'anatomical_groups': self._get_anatomical_groups(data.numpy()),
            'region_info': self.anatomical_regions
        }
        
        return taxonomy_id, model_id, data
    
    def __len__(self):
        return len(self.data_files)
    
    def get_dataset_info(self):
        """获取数据集信息"""
        return {
            'dataset_name': 'NTU-Pred H5',
            'subset': self.subset,
            'total_files': len(self.data_files),
            'data_root': self.data_root,
            'points_per_sample': self.target_points,
            'anatomical_groups': self.num_group,
            'action_filter': self.action_filter,
            'frame_selection': self.frame_selection,
            'augmentation': self.augment
        }
    
    def get_sample_files(self, num_samples=5):
        """获取样本文件列表用于调试"""
        return self.data_files[:num_samples]


# 用于调试和测试的辅助函数
def test_ntu_pred_h5_dataset():
    """测试H5数据集加载器"""
    import argparse
    from types import SimpleNamespace
    
    # 创建测试配置
    config = SimpleNamespace(
        DATA_PATH='/home/uo/myProject/HumanPoint-BERT/data/NTU-Pred',
        subset='train',
        N_POINTS=720,
        npoints=720,
        action_filter='dvae',
        augment=False,
        whole=False,
        data_key='enhanced_data',
        frame_selection='middle',
        coordinate_conversion=True,
        validation_checks=True,
        num_group=7
    )
    
    print("创建NTU-Pred H5数据集...")
    try:
        dataset = NTUPredH5Dataset(config)
        print(f"数据集创建成功！")
        print(f"数据集信息: {dataset.get_dataset_info()}")
        
        if len(dataset) > 0:
            print(f"加载第一个样本...")
            taxonomy_id, model_id, data = dataset[0]
            print(f"  taxonomy_id: {taxonomy_id}")
            print(f"  model_id: {model_id}")
            print(f"  data shape: {data.shape}")
            print(f"  data type: {data.dtype}")
            print(f"  data range: [{data.min():.4f}, {data.max():.4f}]")
            
            print(f"样本文件: {dataset.get_sample_files()}")
        
        print("测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_ntu_pred_h5_dataset()
