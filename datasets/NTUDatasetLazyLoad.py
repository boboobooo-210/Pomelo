"""
内存友好的NTU RGB+D数据集加载器
使用延迟加载策略，不预加载所有数据到内存
"""

import os
import torch
import numpy as np
import torch.utils.data as data
from .NTUDatasetAugmented import NTUAugmented, NTU_CONNECTIONS
from utils.logger import print_log


class NTULazyLoad(data.Dataset):
    """
    内存友好的NTU数据集加载器
    只在需要时加载单个样本，不预加载所有数据
    """
    
    def __init__(self, config):
        self.config = config
        self.data_root = config.get('data_root', 'data/NTU-RGB+D')
        self.subset = config.get('subset', 'train')
        self.target_points = config.get('npoints', 720)
        self.action_filter = config.get('action_filter', 'dvae')
        self.augment = config.get('augment', True)
        self.whole = config.get('whole', False)
        self.density_uniform = config.get('density_uniform', True)
        self.min_points_per_bone = config.get('min_points_per_bone', 3)
        
        print_log(f'[LAZY DATASET] Loading NTU RGB+D {self.subset} dataset (lazy loading)', logger='NTULazy')
        print_log(f'[LAZY DATASET] Action filter: {self.action_filter}', logger='NTULazy')
        print_log(f'[LAZY DATASET] Target points: {self.target_points}', logger='NTULazy')
        
        # 只加载文件路径，不加载实际数据
        self._load_file_paths()
        
        print_log(f'[LAZY DATASET] {len(self.file_paths)} files indexed (lazy loading)', logger='NTULazy')
        
        # 预计算骨头点数分配
        self.bone_points_allocation = self._calculate_bone_points_allocation()
        
    def _load_file_paths(self):
        """只加载文件路径，不加载实际数据"""
        self.file_paths = []
        
        # 获取所有.skeleton文件
        skeleton_files = []
        for root, dirs, files in os.walk(self.data_root):
            for file in files:
                if file.endswith('.skeleton'):
                    skeleton_files.append(os.path.relpath(os.path.join(root, file), self.data_root))
        
        print_log(f'[LAZY DATASET] Found {len(skeleton_files)} skeleton files', logger='NTULazy')
        
        # 应用动作过滤器
        if self.action_filter:
            filtered_files = []
            for file in skeleton_files:
                action_id = self._parse_action_id(file)
                if self._should_include_action(action_id):
                    filtered_files.append(file)
            skeleton_files = filtered_files
            print_log(f'[LAZY DATASET] {len(skeleton_files)} files match action filter: {self.action_filter}', logger='NTULazy')
        
        # 数据分割
        total_files = len(skeleton_files)
        train_split = int(0.8 * total_files)
        val_split = int(0.05 * total_files)
        
        if self.subset == 'train':
            split_files = skeleton_files[:train_split]
        elif self.subset == 'val':
            split_files = skeleton_files[train_split:train_split + val_split]
        elif self.subset == 'test':
            split_files = skeleton_files[train_split + val_split:]
        else:
            split_files = skeleton_files
        
        print_log(f'[LAZY DATASET] Data split - Train: {train_split}, Val: {val_split}, Test: {total_files - train_split - val_split}', logger='NTULazy')
        
        self.file_paths = split_files
        print_log(f'[LAZY DATASET] {self.subset.upper()} split: {len(self.file_paths)} files', logger='NTULazy')
    
    def _parse_action_id(self, filename):
        """从文件名解析动作ID"""
        try:
            # NTU文件名格式: SxxxCxxxPxxxRxxxAxxx.skeleton
            parts = filename.split('A')
            if len(parts) >= 2:
                action_part = parts[1].split('.')[0]
                return int(action_part)
        except:
            pass
        return 0
    
    def _should_include_action(self, action_id):
        """根据动作过滤器判断是否包含该动作"""
        if self.action_filter == 'dvae':
            # DVAE相关动作: 单人日常动作和康复动作
            daily_actions = list(range(1, 12))  # 日常动作 1-11
            interaction_actions = list(range(50, 61))  # 交互动作 50-60
            return action_id in daily_actions + interaction_actions
        elif self.action_filter == 'all':
            return True
        else:
            return True
    
    def _calculate_bone_points_allocation(self):
        """计算每根骨头的插值点数分配"""
        if not self.density_uniform:
            available_points = self.target_points - 25
            points_per_bone = max(self.min_points_per_bone, available_points // len(NTU_CONNECTIONS))
            return {i: points_per_bone for i in range(len(NTU_CONNECTIONS))}
        
        # 使用密度均匀策略（简化版本）
        available_points = self.target_points - 25
        total_bones = len(NTU_CONNECTIONS)
        base_points = available_points // total_bones
        extra_points = available_points % total_bones
        
        allocation = {}
        for i in range(total_bones):
            allocation[i] = base_points + (1 if i < extra_points else 0)
            allocation[i] = max(allocation[i], self.min_points_per_bone)
        
        return allocation
    
    def _read_skeleton_file(self, filepath):
        """读取单个骨架文件 - 从NTUAugmented复制"""
        try:
            with open(filepath, 'r') as f:
                frame_count = int(f.readline().strip())
                
                if frame_count == 0:
                    return None
                
                # 读取第一帧数据
                body_count = int(f.readline().strip())
                if body_count == 0:
                    return None
                
                # 读取第一个身体的数据
                body_info = f.readline().strip()
                joint_count = int(f.readline().strip())
                
                joints = []
                for _ in range(joint_count):
                    line = f.readline().strip()
                    coords = line.split()
                    x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
                    joints.append([x, y, z])
                
                return np.array(joints, dtype=np.float32)
        except Exception as e:
            print_log(f'[LAZY DATASET] Error reading {filepath}: {e}', logger='NTULazy')
            return None
    
    def _skeleton_to_augmented_pointcloud(self, skeleton_joints):
        """将骨架转换为增强点云 - 简化版本"""
        if skeleton_joints is None or len(skeleton_joints) == 0:
            return None
        
        # 确保有25个关节点
        if len(skeleton_joints) < 25:
            # 填充到25个点
            padded = np.zeros((25, 3), dtype=np.float32)
            padded[:len(skeleton_joints)] = skeleton_joints
            skeleton_joints = padded
        else:
            skeleton_joints = skeleton_joints[:25]
        
        # 生成增强点云
        point_cloud = [skeleton_joints]  # 原始25个关节点
        
        # 在骨骼连接上插值
        for bone_idx, (start_idx, end_idx) in enumerate(NTU_CONNECTIONS):
            if start_idx < len(skeleton_joints) and end_idx < len(skeleton_joints):
                start_joint = skeleton_joints[start_idx]
                end_joint = skeleton_joints[end_idx]
                
                # 获取该骨头的点数分配
                num_points = self.bone_points_allocation.get(bone_idx, self.min_points_per_bone)
                
                # 在骨头上插值
                if num_points > 0:
                    t_values = np.linspace(0, 1, num_points + 2)[1:-1]  # 排除端点
                    for t in t_values:
                        interpolated = start_joint * (1 - t) + end_joint * t
                        point_cloud.append(interpolated.reshape(1, 3))
        
        # 合并所有点
        point_cloud = np.vstack(point_cloud)
        
        # 确保点数正确
        current_points = len(point_cloud)
        if current_points < self.target_points:
            # 填充
            padding = np.zeros((self.target_points - current_points, 3), dtype=np.float32)
            point_cloud = np.vstack([point_cloud, padding])
        elif current_points > self.target_points:
            # 截断
            point_cloud = point_cloud[:self.target_points]
        
        return point_cloud
    
    def pc_norm(self, pc):
        """点云标准化"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        if m > 0:
            pc = pc / m
        return pc
    
    def __getitem__(self, idx):
        """延迟加载单个样本"""
        file_path = self.file_paths[idx]
        full_path = os.path.join(self.data_root, file_path)
        
        # 实时读取和处理数据
        skeleton_data = self._read_skeleton_file(full_path)
        if skeleton_data is None:
            # 返回零填充数据
            data = np.zeros((self.target_points, 3), dtype=np.float32)
        else:
            # 转换为增强点云
            data = self._skeleton_to_augmented_pointcloud(skeleton_data)
            if data is None:
                data = np.zeros((self.target_points, 3), dtype=np.float32)
        
        # 数据增强
        if self.augment and self.subset == 'train':
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
        
        # 返回与原始数据集兼容的格式
        taxonomy_id = 'human_skeleton'
        model_id = f'ntu_{idx:06d}'
        
        return taxonomy_id, model_id, data
    
    def __len__(self):
        return len(self.file_paths)
