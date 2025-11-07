"""
NTU RGB+D 数据集加载器
根据动作类型进行分类：单人日常动作、康复动作、双人互动动作
用于DVAE训练构建人体骨架点云码本
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


# 动作分类定义
SINGLE_DAILY_ACTIONS = list(range(1, 41)) + list(range(61, 103))  # A1-A40, A61-A102 (除康复动作)
REHABILITATION_ACTIONS = list(range(41, 50)) + list(range(103, 106))  # A41-A49, A103-A105
INTERACTION_ACTIONS = list(range(50, 61)) + list(range(107, 121))  # A50-A60, A107-A120 (不使用)

# 移除康复动作从单人日常动作中
for action in REHABILITATION_ACTIONS:
    if action in SINGLE_DAILY_ACTIONS:
        SINGLE_DAILY_ACTIONS.remove(action)

# DVAE训练使用的动作（单人日常 + 康复）
DVAE_ACTIONS = SINGLE_DAILY_ACTIONS + REHABILITATION_ACTIONS


@DATASETS.register_module()
class NTU(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS  # 骨架关节点数，NTU为25个关节
        self.action_filter = config.get('action_filter', 'dvae')  # 'dvae', 'daily', 'rehab', 'all'
        
        self.sample_points_num = config.get('npoints', self.npoints)
        self.whole = config.get('whole', False)
        self.augment = config.get('augment', True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        print_log(f'[DATASET] Loading NTU RGB+D {self.subset} dataset from {self.data_root}', logger='NTU')
        print_log(f'[DATASET] Action filter: {self.action_filter}', logger='NTU')
        print_log(f'[DATASET] Sample out {self.sample_points_num} points', logger='NTU')
        
        # 加载数据
        self._load_data()
        
        print_log(f'[DATASET] {len(self.data)} instances were loaded', logger='NTU')
        
        # 生成索引排列用于随机采样
        self.permutation = np.arange(self.npoints)
        
    def _load_data(self):
        """加载NTU RGB+D骨架数据"""
        self.data = []
        self.labels = []
        
        # 获取所有.skeleton文件
        skeleton_files = []
        for file in os.listdir(self.data_root):
            if file.endswith('.skeleton'):
                skeleton_files.append(file)
        
        print_log(f'[DATASET] Found {len(skeleton_files)} skeleton files', logger='NTU')
        
        # 根据文件名解析动作类别
        valid_files = []
        for file in skeleton_files:
            action_id = self._parse_action_id(file)
            if self._is_valid_action(action_id):
                valid_files.append(file)
        
        print_log(f'[DATASET] {len(valid_files)} files match action filter: {self.action_filter}', logger='NTU')
        
        # 加载有效文件
        for file in valid_files:
            try:
                skeleton_data = self._read_skeleton_file(os.path.join(self.data_root, file))
                if skeleton_data is not None and len(skeleton_data) > 0:
                    # 转换为点云格式 (N, 3) -> (75, 3) for 25 joints
                    point_cloud = self._skeleton_to_pointcloud(skeleton_data)
                    if point_cloud is not None:
                        self.data.append(point_cloud)
                        action_id = self._parse_action_id(file)
                        self.labels.append(action_id)
            except Exception as e:
                print_log(f'[DATASET] Error loading {file}: {e}', logger='NTU')
                continue
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
    def _parse_action_id(self, filename):
        """从文件名解析动作ID"""
        # 文件名格式: S001C001P001R001A001.skeleton
        # 提取动作ID (A后面的数字)
        try:
            action_part = filename.split('A')[1].split('.')[0]
            return int(action_part)
        except:
            return 0
    
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
            print_log(f'[DATASET] Error reading {filepath}: {e}', logger='NTU')
            return None
    
    def _skeleton_to_pointcloud(self, skeleton_frames):
        """将骨架序列转换为点云"""
        if not skeleton_frames:
            return None

        # 选择中间帧或平均多帧
        if len(skeleton_frames) == 1:
            skeleton = skeleton_frames[0]
        else:
            # 选择中间帧
            mid_idx = len(skeleton_frames) // 2
            skeleton = skeleton_frames[mid_idx]

        # 确保是25个关节点
        if skeleton.shape[0] != 25:
            return None

        # 数据清理：移除无效点
        valid_joints = []
        for joint in skeleton:
            if not (joint[0] == 0 and joint[1] == 0 and joint[2] == 0):
                valid_joints.append(joint)

        if len(valid_joints) < 10:  # 至少需要10个有效关节
            return None

        # NTU数据集保持原始关节数量，不进行上采样
        # 如果有效关节少于25个，用最近邻填充到25个
        point_cloud = np.array(valid_joints)

        if len(point_cloud) < 25:
            # 用最近邻填充到25个关节
            while len(point_cloud) < 25:
                # 随机选择一个现有关节进行复制
                idx = np.random.randint(len(point_cloud))
                point_cloud = np.vstack([point_cloud, point_cloud[idx]])
        elif len(point_cloud) > 25:
            # 如果超过25个（理论上不会发生），随机选择25个
            indices = np.random.choice(len(point_cloud), 25, replace=False)
            point_cloud = point_cloud[indices]

        return point_cloud.astype(np.float32)
    
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

        # NTU数据集保持25个关节点，不进行重采样
        # 确保数据形状正确
        if data.shape[0] != 25:
            print(f"Warning: Sample {idx} has {data.shape[0]} joints instead of 25")
            # 如果不是25个关节，进行调整
            if data.shape[0] < 25:
                # 用重复填充到25个
                while data.shape[0] < 25:
                    idx_to_repeat = np.random.randint(data.shape[0])
                    data = np.vstack([data, data[idx_to_repeat:idx_to_repeat+1]])
            else:
                # 随机选择25个
                indices = np.random.choice(data.shape[0], 25, replace=False)
                data = data[indices]
        
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

        # 为了与现有框架兼容，返回taxonomy_id和model_id
        taxonomy_id = 'human_skeleton'
        model_id = f'ntu_{idx:06d}'

        return taxonomy_id, model_id, data
    
    def __len__(self):
        return len(self.data)
    
    def get_action_statistics(self):
        """获取动作统计信息"""
        from collections import Counter
        action_counts = Counter(self.labels)
        
        stats = {
            'total_samples': len(self.data),
            'unique_actions': len(action_counts),
            'action_distribution': dict(action_counts)
        }
        
        return stats
