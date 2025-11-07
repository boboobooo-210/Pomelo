"""
NTU RGB+D 原始骨架数据集 (Raw .skeleton files)
专门处理原始.skeleton文件的数据集类，支持动作过滤和数据集分割
用于GCN骨架Tokenizer训练
"""

import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
import struct
import logging
from sklearn.model_selection import train_test_split


# 动作分类定义
SINGLE_DAILY_ACTIONS = list(range(1, 41)) + list(range(61, 103))  # A1-A40, A61-A102 (除康复动作)
REHABILITATION_ACTIONS = list(range(41, 50)) + list(range(103, 106))  # A41-A49, A103-A105
INTERACTION_ACTIONS = list(range(50, 61)) + list(range(107, 121))  # A50-A60, A107-A120

# 移除康复动作从单人日常动作中
for action in REHABILITATION_ACTIONS:
    if action in SINGLE_DAILY_ACTIONS:
        SINGLE_DAILY_ACTIONS.remove(action)

# 过滤掉交互动作的主要动作集合
MAIN_ACTIONS = SINGLE_DAILY_ACTIONS + REHABILITATION_ACTIONS


@DATASETS.register_module()
class NTU_Skeleton_Raw(data.Dataset):
    """
    NTU RGB+D 原始骨架数据集
    专门处理.skeleton文件，支持动作过滤和标准数据集分割
    """
    
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset  # 'train', 'val', 'test'
        self.npoints = config.N_POINTS  # 25个关节点
        
        # 动作过滤设置
        self.action_filter = config.get('action_filter', {})
        self.excluded_actions = self.action_filter.get('exclude', [])
        
        # 数据集分割设置
        self.dataset_split = config.get('dataset_split', {
            'train_ratio': 0.6,
            'val_ratio': 0.2,
            'test_ratio': 0.2
        })
        
        # 其他设置
        self.sample_points_num = config.get('npoints', self.npoints)
        self.whole = config.get('whole', False)
        self.augment = config.get('augment', True)
        self.normalize = config.get('normalize', True)
        self.single_frame = config.get('single_frame', True)
        
        # 内存优化设置
        self.max_files = config.get('max_files', None)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        print_log(f'[NTU_Skeleton_Raw] Loading {self.subset} dataset from {self.data_root}', logger='NTU_Skeleton_Raw')
        print_log(f'[NTU_Skeleton_Raw] Action filter exclude: {self.excluded_actions}', logger='NTU_Skeleton_Raw')
        print_log(f'[NTU_Skeleton_Raw] Dataset split: {self.dataset_split}', logger='NTU_Skeleton_Raw')
        print_log(f'[NTU_Skeleton_Raw] Max files limit: {self.max_files}', logger='NTU_Skeleton_Raw')
        
        # 加载数据
        self._load_data()
        
        print_log(f'[NTU_Skeleton_Raw] {len(self.data)} instances were loaded for {self.subset}', logger='NTU_Skeleton_Raw')
        
        # 生成索引排列用于随机采样
        self.permutation = np.arange(self.npoints)
        
    def _load_data(self):
        """加载NTU RGB+D骨架数据"""
        self.data = []
        self.labels = []
        self.file_names = []
        
        # 获取所有.skeleton文件
        print(f'[NTU_Skeleton_Raw] 正在扫描目录: {self.data_root}...', flush=True)
        skeleton_files = []
        for file in os.listdir(self.data_root):
            if file.endswith('.skeleton'):
                skeleton_files.append(file)
        
        print(f'[NTU_Skeleton_Raw] Found {len(skeleton_files)} skeleton files', flush=True)
        
        # 过滤文件：移除交互动作
        print(f'[NTU_Skeleton_Raw] 正在过滤动作类别...', flush=True)
        valid_files = []
        for file in skeleton_files:
            action_id = self._parse_action_id(file)
            if self._is_valid_action(action_id):
                valid_files.append(file)
        
        print(f'[NTU_Skeleton_Raw] {len(valid_files)} files after action filtering', flush=True)
        
        # 应用文件数量限制
        if self.max_files and len(valid_files) > self.max_files:
            np.random.seed(42)  # 确保可重现
            valid_files = np.random.choice(valid_files, self.max_files, replace=False).tolist()
            print_log(f'[NTU_Skeleton_Raw] Limited to {len(valid_files)} files for memory optimization', logger='NTU_Skeleton_Raw')
        
        # 数据集分割
        print(f'[NTU_Skeleton_Raw] 正在划分数据集 (train/val/test)...', flush=True)
        file_splits = self._split_files(valid_files)
        subset_files = file_splits[self.subset]
        
        print(f'[NTU_Skeleton_Raw] {len(subset_files)} files assigned to {self.subset} split', flush=True)
        
        # 加载对应子集的文件
        print(f'[NTU_Skeleton_Raw] 开始加载 {len(subset_files)} 个骨架文件...', flush=True)
        loaded_count = 0
        for idx, file in enumerate(subset_files):
            try:
                skeleton_data = self._read_skeleton_file(os.path.join(self.data_root, file))
                if skeleton_data is not None and len(skeleton_data) > 0:
                    # 转换为25关节点格式
                    point_cloud = self._skeleton_to_pointcloud(skeleton_data)
                    if point_cloud is not None:
                        self.data.append(point_cloud)
                        action_id = self._parse_action_id(file)
                        self.labels.append(action_id)
                        self.file_names.append(file)
                        loaded_count += 1
                        
                        # 更频繁的进度报告
                        if loaded_count % 500 == 0:
                            progress = (idx + 1) / len(subset_files) * 100
                            print(f'[NTU_Skeleton_Raw] 加载进度: {loaded_count}/{len(subset_files)} ({progress:.1f}%)', flush=True)
                        
            except Exception as e:
                print_log(f'[NTU_Skeleton_Raw] Error loading {file}: {e}', logger='NTU_Skeleton_Raw')
                continue
        
        print(f'[NTU_Skeleton_Raw] 数据加载完成！成功加载 {loaded_count} 个样本', flush=True)
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
    def _parse_action_id(self, filename):
        """从文件名解析动作ID"""
        # 文件名格式: S001C001P001R001A001.skeleton
        try:
            action_part = filename.split('A')[1].split('.')[0]
            return int(action_part)
        except:
            return 0
    
    def _is_valid_action(self, action_id):
        """判断动作是否有效（排除交互动作）"""
        # 排除指定的动作ID
        if action_id in self.excluded_actions:
            return False
        
        # 排除交互动作 (A050-A060, A106-A120)
        if action_id in INTERACTION_ACTIONS:
            return False
        
        # 只保留单人动作
        return action_id in MAIN_ACTIONS
    
    def _split_files(self, files):
        """将文件按比例分割为训练、验证、测试集"""
        # 使用固定随机种子确保可重现性
        np.random.seed(42)
        
        # 按动作类别进行分层抽样
        files_by_action = {}
        for file in files:
            action_id = self._parse_action_id(file)
            if action_id not in files_by_action:
                files_by_action[action_id] = []
            files_by_action[action_id].append(file)
        
        train_files = []
        val_files = []
        test_files = []
        
        train_ratio = self.dataset_split['train_ratio']
        val_ratio = self.dataset_split['val_ratio']
        test_ratio = self.dataset_split['test_ratio']
        
        for action_id, action_files in files_by_action.items():
            n_files = len(action_files)
            if n_files == 0:
                continue
                
            # 计算分割点
            n_train = int(n_files * train_ratio)
            n_val = int(n_files * val_ratio)
            n_test = n_files - n_train - n_val
            
            # 确保每个分割至少有一个样本（如果总数允许）
            if n_files >= 3:
                n_train = max(1, n_train)
                n_val = max(1, n_val) 
                n_test = max(1, n_test)
                # 重新调整以确保总和正确
                if n_train + n_val + n_test != n_files:
                    diff = n_files - (n_train + n_val + n_test)
                    n_train += diff
            
            # 随机打乱文件
            np.random.shuffle(action_files)
            
            # 分割文件
            train_files.extend(action_files[:n_train])
            val_files.extend(action_files[n_train:n_train+n_val])
            test_files.extend(action_files[n_train+n_val:])
        
        return {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
    
    def _read_skeleton_file(self, filepath):
        """读取.skeleton文件"""
        try:
            with open(filepath, 'r') as f:
                # 读取帧数
                frame_count = int(f.readline().strip())
                
                frames_data = []
                for frame_idx in range(frame_count):
                    # 读取人体数量
                    body_count = int(f.readline().strip())
                    
                    frame_skeletons = []
                    for body_idx in range(body_count):
                        # 读取人体信息行
                        body_info = f.readline().strip()
                        
                        # 读取关节数量
                        joint_count = int(f.readline().strip())
                        
                        # 读取关节数据
                        joints = []
                        for joint_idx in range(joint_count):
                            joint_line = f.readline().strip().split()
                            if len(joint_line) >= 3:
                                x, y, z = float(joint_line[0]), float(joint_line[1]), float(joint_line[2])
                                joints.append([x, y, z])
                        
                        if len(joints) == 25:  # NTU有25个关节
                            frame_skeletons.append(np.array(joints))
                    
                    if frame_skeletons:
                        # 如果有多个人体，选择第一个
                        frames_data.append(frame_skeletons[0])
                
                return frames_data if frames_data else None
                
        except Exception as e:
            return None
    
    def _skeleton_to_pointcloud(self, skeleton_frames):
        """将骨架序列转换为点云"""
        if not skeleton_frames:
            return None

        # 选择处理策略
        if self.single_frame:
            # 选择中间帧
            if len(skeleton_frames) == 1:
                skeleton = skeleton_frames[0]
            else:
                mid_idx = len(skeleton_frames) // 2
                skeleton = skeleton_frames[mid_idx]
        else:
            # 平均多帧（时序建模用）
            skeleton = np.mean(skeleton_frames, axis=0)

        # 确保是25个关节点
        if skeleton.shape[0] != 25:
            return None

        # 数据清理：移除无效点
        valid_joints = []
        for joint in skeleton:
            # 检查是否为有效关节（不是全零）
            if not (abs(joint[0]) < 1e-6 and abs(joint[1]) < 1e-6 and abs(joint[2]) < 1e-6):
                valid_joints.append(joint)

        if len(valid_joints) < 10:  # 至少需要10个有效关节
            return None

        # 保持25个关节点的结构
        point_cloud = skeleton.copy()

        # 对于无效关节，用有效关节的平均值填充
        valid_joints = np.array(valid_joints)
        mean_joint = np.mean(valid_joints, axis=0)
        
        for i in range(25):
            joint = point_cloud[i]
            if abs(joint[0]) < 1e-6 and abs(joint[1]) < 1e-6 and abs(joint[2]) < 1e-6:
                point_cloud[i] = mean_joint + np.random.normal(0, 0.01, 3)  # 添加小扰动避免重复

        return point_cloud.astype(np.float32)
    
    def pc_norm(self, pc):
        """点云标准化"""
        if not self.normalize:
            return pc
            
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        if m > 0:
            pc = pc / m
        return pc
    
    def __getitem__(self, idx):
        """获取数据项"""
        # 获取骨架点云数据
        data = self.data[idx].copy()
        label = self.labels[idx]

        # 确保数据形状正确 (25, 3)
        assert data.shape == (25, 3), f"Expected shape (25, 3), got {data.shape}"
        
        # 数据增强（仅在训练时）
        if self.augment and self.subset == 'train':
            # 随机旋转（绕z轴）
            theta = np.random.uniform(0, 2 * np.pi)
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            data = np.dot(data, rotation_matrix.T)
            
            # 随机缩放
            scale = np.random.uniform(0.9, 1.1)
            data = data * scale
            
            # 添加噪声
            noise = np.random.normal(0, 0.005, data.shape).astype(np.float32)
            data = data + noise
        
        # 标准化
        data = self.pc_norm(data)

        # 转换为torch张量
        data = torch.from_numpy(data).float()

        # 返回兼容格式
        taxonomy_id = 'ntu_skeleton'
        model_id = f'ntu_{self.subset}_{idx:06d}'

        return taxonomy_id, model_id, data
    
    def __len__(self):
        return len(self.data)
    
    def get_action_statistics(self):
        """获取动作统计信息"""
        from collections import Counter
        action_counts = Counter(self.labels)
        
        # 统计排除的动作
        excluded_count = 0
        total_files = len([f for f in os.listdir(self.data_root) if f.endswith('.skeleton')])
        
        stats = {
            'subset': self.subset,
            'total_samples': len(self.data),
            'unique_actions': len(action_counts),
            'action_distribution': dict(action_counts),
            'excluded_actions': self.excluded_actions,
            'dataset_split': self.dataset_split,
            'total_skeleton_files': total_files
        }
        
        return stats
    
    def get_sample_info(self, idx):
        """获取样本详细信息"""
        return {
            'index': idx,
            'file_name': self.file_names[idx] if idx < len(self.file_names) else 'unknown',
            'action_id': self.labels[idx],
            'data_shape': self.data[idx].shape,
            'subset': self.subset
        }
