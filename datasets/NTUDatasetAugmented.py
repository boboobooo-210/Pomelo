"""
NTU RGB+D 数据集加载器 - 方案一：数据增强策略
在骨架连接线上插值生成更多点，模拟MARS的做法
"""

import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
import logging


# NTU RGB+D 骨架连接关系
NTU_CONNECTIONS = [
    (3, 2),   # 头顶 - 颈部
    (2, 20),  # 颈部 - 上躯干
    (20, 1),  # 上躯干 - 躯干中
    (1, 0),   # 躯干中 - 躯干下
    
    # 左上肢
    (20, 4),  # 上躯干 - 左肩
    (4, 5),   # 左肩 - 左肘
    (5, 6),   # 左肘 - 左腕
    (6, 22),  # 左腕 - 左手指1
    (6, 7),   # 左腕 - 左手
    (7, 21),  # 左手 - 左手指2
    
    # 右上肢
    (20, 8),  # 上躯干 - 右肩
    (8, 9),   # 右肩 - 右肘
    (9, 10),  # 右肘 - 右腕
    (10, 24), # 右腕 - 右手指1
    (10, 11), # 右腕 - 右手
    (11, 23), # 右手 - 右手指2
    
    # 左下肢
    (0, 12),  # 躯干下 - 左髋
    (12, 13), # 左髋 - 左膝
    (13, 14), # 左膝 - 左踝
    (14, 15), # 左踝 - 左脚
    
    # 右下肢
    (0, 16),  # 躯干下 - 右髋
    (16, 17), # 右髋 - 右膝
    (17, 18), # 右膝 - 右踝
    (18, 19), # 右踝 - 右脚
]

# 动作分类定义
SINGLE_DAILY_ACTIONS = list(range(1, 41)) + list(range(61, 103))
REHABILITATION_ACTIONS = list(range(41, 50)) + list(range(103, 106))
INTERACTION_ACTIONS = list(range(50, 61)) + list(range(107, 121))

# 移除康复动作从单人日常动作中
for action in REHABILITATION_ACTIONS:
    if action in SINGLE_DAILY_ACTIONS:
        SINGLE_DAILY_ACTIONS.remove(action)

# 适合骨架学习的动作 (单人动作，包括日常和康复)
SKELETON_LEARNING_ACTIONS = SINGLE_DAILY_ACTIONS + REHABILITATION_ACTIONS
# 为了向后兼容，保留原名称
DVAE_ACTIONS = SKELETON_LEARNING_ACTIONS


@DATASETS.register_module()
class NTUAugmented(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS  # 原始关节数：25
        self.action_filter = config.get('action_filter', 'dvae')
        
        # 数据增强参数
        self.target_points = config.get('npoints', 720)  # 目标点云数量（默认720，实际可能不同）
        self.density_uniform = config.get('density_uniform', True)  # 是否使用密度均匀策略
        self.min_points_per_bone = config.get('min_points_per_bone', 3)  # 每根骨头最少插值点数

        self.whole = config.get('whole', False)
        self.augment = config.get('augment', True)

        # 恒定填充策略：每根骨头的插值点数固定，便于分组采样
        self.bone_points_allocation = self._calculate_bone_points_allocation()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        print_log(f'[DATASET] Loading NTU RGB+D Augmented {self.subset} dataset', logger='NTUAugmented')
        print_log(f'[DATASET] Action filter: {self.action_filter}', logger='NTUAugmented')
        print_log(f'[DATASET] Density uniform: {self.density_uniform}', logger='NTUAugmented')
        print_log(f'[DATASET] Using constant bone points allocation for consistent grouping', logger='NTUAugmented')
        print_log(f'[DATASET] Bone points allocation: {len(self.bone_points_allocation)} bones', logger='NTUAugmented')
        
        # 加载数据
        self._load_data()
        
        print_log(f'[DATASET] {len(self.data)} instances were loaded', logger='NTUAugmented')
        
        # 生成索引排列用于随机采样
        self.permutation = np.arange(self.target_points)
        
    def _load_data(self):
        """加载NTU RGB+D骨架数据"""
        self.data = []
        self.labels = []
        
        # 获取所有.skeleton文件
        skeleton_files = []
        for file in os.listdir(self.data_root):
            if file.endswith('.skeleton'):
                skeleton_files.append(file)
        
        print_log(f'[DATASET] Found {len(skeleton_files)} skeleton files', logger='NTUAugmented')
        
        # 根据文件名解析动作类别
        valid_files = []
        for file in skeleton_files:
            action_id = self._parse_action_id(file)
            if self._is_valid_action(action_id):
                valid_files.append(file)
        
        print_log(f'[DATASET] {len(valid_files)} files match action filter: {self.action_filter}', logger='NTUAugmented')

        # 数据集划分
        split_files = self._split_dataset(valid_files)
        print_log(f'[DATASET] {self.subset.upper()} split: {len(split_files)} files', logger='NTUAugmented')

        # 加载对应划分的文件
        max_files = len(split_files)
        print_log(f'[DATASET] Loading {max_files} files for {self.subset}', logger='NTUAugmented')
        
        for file in split_files[:max_files]:
            try:
                skeleton_data = self._read_skeleton_file(os.path.join(self.data_root, file))
                if skeleton_data is not None and len(skeleton_data) > 0:
                    # 转换为增强点云格式
                    point_cloud = self._skeleton_to_augmented_pointcloud(skeleton_data)
                    if point_cloud is not None:
                        self.data.append(point_cloud)
                        action_id = self._parse_action_id(file)
                        self.labels.append(action_id)
            except Exception as e:
                print_log(f'[DATASET] Error loading {file}: {e}', logger='NTUAugmented')
                continue
        
        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def _calculate_bone_points_allocation(self):
        """
        计算每根骨头的插值点数分配
        使用弹性填充策略：根据骨骼长度和目标点数动态分配
        """
        if not self.density_uniform:
            # 如果不使用密度均匀策略，则平均分配
            available_points = self.target_points - 25  # 减去原始25个关节点
            points_per_bone = max(self.min_points_per_bone, available_points // len(NTU_CONNECTIONS))
            return {i: points_per_bone for i in range(len(NTU_CONNECTIONS))}

        # 密度均匀策略：根据骨骼长度分配点数
        # 使用标准人体骨骼比例估算各骨骼的相对长度
        bone_length_ratios = self._get_bone_length_ratios()

        # 计算可用于插值的点数
        available_points = self.target_points - 25  # 减去原始25个关节点

        # 根据长度比例分配点数
        total_ratio = sum(bone_length_ratios.values())
        bone_points = {}

        allocated_points = 0
        for i, (bone_idx, ratio) in enumerate(bone_length_ratios.items()):
            if i == len(bone_length_ratios) - 1:
                # 最后一根骨头分配剩余点数
                points = available_points - allocated_points
            else:
                points = int(available_points * ratio / total_ratio)

            # 确保每根骨头至少有最小点数
            points = max(self.min_points_per_bone, points)
            bone_points[bone_idx] = points
            allocated_points += points

        return bone_points

    def _get_bone_length_ratios(self):
        """
        获取各骨骼的相对长度比例
        基于标准人体解剖学比例
        """
        # NTU骨骼连接的相对长度比例（基于人体解剖学）
        bone_ratios = {
            0: 0.8,   # (3,2) 头顶-颈部：较短
            1: 1.2,   # (2,20) 颈部-上躯干：中等
            2: 2.5,   # (20,1) 上躯干-躯干中：较长
            3: 2.0,   # (1,0) 躯干中-躯干下：较长

            # 左上肢
            4: 1.8,   # (20,4) 上躯干-左肩：中等
            5: 3.0,   # (4,5) 左肩-左肘：长（上臂）
            6: 2.8,   # (5,6) 左肘-左腕：长（前臂）
            7: 0.6,   # (6,22) 左腕-左手指1：短
            8: 1.0,   # (6,7) 左腕-左手：短
            9: 0.5,   # (7,21) 左手-左手指2：很短

            # 右上肢
            10: 1.8,  # (20,8) 上躯干-右肩：中等
            11: 3.0,  # (8,9) 右肩-右肘：长（上臂）
            12: 2.8,  # (9,10) 右肘-右腕：长（前臂）
            13: 0.6,  # (10,24) 右腕-右手指1：短
            14: 1.0,  # (10,11) 右腕-右手：短
            15: 0.5,  # (11,23) 右手-右手指2：很短

            # 左下肢
            16: 2.0,  # (0,12) 躯干下-左髋：中等
            17: 4.5,  # (12,13) 左髋-左膝：很长（大腿）
            18: 4.0,  # (13,14) 左膝-左踝：很长（小腿）
            19: 1.2,  # (14,15) 左踝-左脚：短

            # 右下肢
            20: 2.0,  # (0,16) 躯干下-右髋：中等
            21: 4.5,  # (16,17) 右髋-右膝：很长（大腿）
            22: 4.0,  # (17,18) 右膝-右踝：很长（小腿）
            23: 1.2,  # (18,19) 右踝-右脚：短
        }

        return bone_ratios

    def _split_dataset(self, valid_files):
        """
        将数据集划分为训练/验证/测试集
        使用NTU RGB+D的标准划分方法：基于受试者ID
        """
        # NTU RGB+D标准划分：
        # 训练集：受试者 1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38
        # 测试集：受试者 3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40

        train_subjects = [1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38]
        test_subjects = [3,6,7,10,11,12,20,21,22,23,24,26,29,30,32,33,36,37,39,40]

        # 从训练集中分出一部分作为验证集
        val_subjects = [1, 2, 4, 5]  # 取训练集的前4个受试者作为验证集
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
            # 如果受试者ID不在预定义列表中，默认分配到训练集
            else:
                train_files.append(file)

        print_log(f'[DATASET] Data split - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}',
                 logger='NTUAugmented')

        if self.subset == 'train':
            return train_files
        elif self.subset == 'val':
            return val_files
        elif self.subset == 'test':
            return test_files
        else:
            print_log(f'[DATASET] Unknown subset: {self.subset}, using train', logger='NTUAugmented')
            return train_files

    def _parse_subject_id(self, filename):
        """
        从文件名解析受试者ID
        NTU文件名格式: SsssCcccPpppRrrrAaaa.skeleton
        其中Pppp是受试者(performer)ID
        """
        import re
        match = re.match(r'S\d{3}C\d{3}P(\d{3})R\d{3}A\d{3}\.skeleton', filename)
        if match:
            return int(match.group(1))
        else:
            print_log(f'[DATASET] Warning: Cannot parse subject ID from {filename}', logger='NTUAugmented')
            return 1  # 默认返回1
        
    def _parse_action_id(self, filename):
        """从文件名解析动作ID"""
        try:
            action_part = filename.split('A')[1].split('.')[0]
            return int(action_part)
        except:
            return 0
    
    def _is_valid_action(self, action_id):
        """判断动作是否符合过滤条件"""
        if self.action_filter == 'dvae':
            # 'dvae'过滤器：选择适合骨架学习的动作(单人日常+康复)
            # 这些动作同样适合SkeletonTokenizer训练
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
                frame_count = int(f.readline().strip())
                
                frames_data = []
                for frame_idx in range(frame_count):
                    body_count = int(f.readline().strip())
                    
                    frame_skeletons = []
                    for body_idx in range(body_count):
                        body_info = f.readline().strip()
                        joint_count = int(f.readline().strip())
                        
                        joints = []
                        for joint_idx in range(joint_count):
                            joint_line = f.readline().strip().split()
                            if len(joint_line) >= 3:
                                x, y, z = float(joint_line[0]), float(joint_line[1]), float(joint_line[2])
                                joints.append([x, y, z])
                        
                        if len(joints) == 25:
                            frame_skeletons.append(np.array(joints))
                    
                    if frame_skeletons:
                        frames_data.append(frame_skeletons[0])
                
                return frames_data if frames_data else None
                
        except Exception as e:
            return None
    
    def _skeleton_to_augmented_pointcloud(self, skeleton_frames):
        """将骨架序列转换为增强点云"""
        if not skeleton_frames:
            return None
        
        # 选择中间帧
        if len(skeleton_frames) == 1:
            skeleton = skeleton_frames[0]
        else:
            mid_idx = len(skeleton_frames) // 2
            skeleton = skeleton_frames[mid_idx]
        
        if skeleton.shape[0] != 25:
            return None
        
        # 数据清理
        valid_joints = []
        for joint in skeleton:
            if not (joint[0] == 0 and joint[1] == 0 and joint[2] == 0):
                valid_joints.append(joint)
        
        if len(valid_joints) < 10:
            return None
        
        # 确保有25个关节
        skeleton_25 = np.array(valid_joints)
        if len(skeleton_25) < 25:
            while len(skeleton_25) < 25:
                idx = np.random.randint(len(skeleton_25))
                skeleton_25 = np.vstack([skeleton_25, skeleton_25[idx]])
        elif len(skeleton_25) > 25:
            indices = np.random.choice(len(skeleton_25), 25, replace=False)
            skeleton_25 = skeleton_25[indices]
        
        # 进行数据增强：在连接线上插值
        augmented_points = self._interpolate_skeleton(skeleton_25)
        
        return augmented_points.astype(np.float32)
    
    def _interpolate_skeleton(self, skeleton):
        """在骨架连接线上插值生成更多点 - 恒定填充策略"""
        augmented_points = []

        # 添加原始关节点
        for joint in skeleton:
            augmented_points.append(joint)

        # 在每个连接上根据分配的点数插值
        for bone_idx, (start_idx, end_idx) in enumerate(NTU_CONNECTIONS):
            if start_idx < len(skeleton) and end_idx < len(skeleton):
                start_point = skeleton[start_idx]
                end_point = skeleton[end_idx]

                # 获取这根骨头分配的插值点数（恒定数量）
                points_for_this_bone = self.bone_points_allocation.get(bone_idx, self.min_points_per_bone)

                # 在连接线上等距插值
                for i in range(1, points_for_this_bone + 1):
                    t = i / (points_for_this_bone + 1)  # +1是为了不包含端点
                    interpolated_point = start_point + t * (end_point - start_point)
                    augmented_points.append(interpolated_point)

        return np.array(augmented_points)
    
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
            indices = np.random.choice(pc.shape[0], num, replace=True)
            pc = pc[indices]
        return pc
    
    def __getitem__(self, idx):
        # 获取增强后的骨架点云数据
        data = self.data[idx].copy()
        
        # 由于使用恒定填充策略，每个样本的点数应该是固定的
        # 不需要再进行随机采样到目标点数
        
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
        # 对于NTU数据集，我们使用通用的标识符
        taxonomy_id = 'human_skeleton'
        model_id = f'ntu_{idx:06d}'

        return taxonomy_id, model_id, data
    
    def __len__(self):
        return len(self.data)
    
    def get_augmentation_info(self):
        """获取数据增强信息，包含完整的720点分组信息"""
        # 计算实际增强后的点数
        actual_augmented_points = 25 + sum(self.bone_points_allocation.values())
        
        # 获取解剖区域分组信息
        region_info = self.get_branch_grouping_info()

        return {
            'original_joints': 25,
            'connections': len(NTU_CONNECTIONS),
            'bone_points_allocation': self.bone_points_allocation,
            'actual_augmented_points': actual_augmented_points,
            'target_points': self.target_points,
            'augmentation_ratio': actual_augmented_points / 25,
            'density_uniform': self.density_uniform,
            'min_points_per_bone': self.min_points_per_bone,
            'anatomical_regions': region_info['anatomical_regions'],
            'region_count': region_info['region_count'],
            'center_joints': region_info['center_joints']
        }
    
    def get_branch_grouping_info(self):
        """
        获取解剖区域分组信息，便于后续的分组采样
        按照人体解剖结构分为6大功能区域，避免中心关节重复归属
        """
        # 定义6大解剖区域及其包含的关节和连接
        region_definitions = {
            'torso_lower': {
                'joints': [0, 1, 20],  # 躯干下-躯干中-上躯干
                'connections': [2, 3],  # (20,1), (1,0)
                'description': '下躯干区域：0-1-20及中间所有点'
            },
            'head_neck': {
                'joints': [3, 2],  # 头顶-颈部 (不包括20)
                'connections': [0, 1],  # (3,2), (2,20) 但不包括20点
                'description': '头颈区域：3-2-20及中间所有点，不包括20'
            },
            'left_arm': {
                'joints': [4, 5, 6, 7, 21, 22],  # 左臂所有关节 (不包括20)
                'connections': [4, 5, 6, 7, 8, 9],  # 左臂所有连接，但不包括20点
                'description': '左臂区域：20-4-5-6-7-21, 6-22所有点，不包括20'
            },
            'right_arm': {
                'joints': [8, 9, 10, 11, 23, 24],  # 右臂所有关节 (不包括20)
                'connections': [10, 11, 12, 13, 14, 15],  # 右臂所有连接，但不包括20点
                'description': '右臂区域：20-8-9-10-11-23, 10-24所有点，不包括20'
            },
            'left_leg': {
                'joints': [12, 13, 14, 15],  # 左腿所有关节 (不包括0)
                'connections': [16, 17, 18, 19],  # 左腿所有连接，但不包括0点
                'description': '左腿区域：0-12-13-14-15所有点，不包括0'
            },
            'right_leg': {
                'joints': [16, 17, 18, 19],  # 右腿所有关节 (不包括0)
                'connections': [20, 21, 22, 23],  # 右腿所有连接，但不包括0点
                'description': '右腿区域：0-16-17-18-19所有点，不包括0'
            }
        }
        
        # 计算每个区域的点索引范围
        point_ranges = {}
        current_idx = 25  # 前25个是原始关节点
        
        for region_name, region_info in region_definitions.items():
            region_start = current_idx
            region_points = 0
            
            # 计算这个区域包含的插值点数
            for connection_idx in region_info['connections']:
                bone_points = self.bone_points_allocation.get(connection_idx, self.min_points_per_bone)
                region_points += bone_points
            
            region_end = region_start + region_points
            point_ranges[region_name] = {
                'start_idx': region_start,
                'end_idx': region_end,
                'point_count': region_points,
                'joints': region_info['joints'],
                'connections': region_info['connections'],
                'description': region_info['description']
            }
            current_idx = region_end
        
        # 添加中心关节的分组信息
        center_joints = {
            'center_joint_0': [0],    # 躯干下，连接躯干和双腿
            'center_joint_20': [20],  # 上躯干，连接躯干、头颈和双臂
            'peripheral_joints': [1, 2, 3] + list(range(4, 25))  # 其他关节
        }
        
        return {
            'anatomical_regions': point_ranges,
            'center_joints': center_joints,
            'total_points': current_idx,
            'original_joints': 25,
            'region_count': len(point_ranges)
        }
    
    def get_point_region_labels(self):
        """
        获取每个点的区域标签，便于Tokenizer训练时使用
        返回720个点的区域标签数组
        """
        region_info = self.get_branch_grouping_info()
        region_labels = np.zeros(720, dtype=int)
        
        # 定义区域标签映射
        region_name_to_id = {
            'original_joints': 0,    # 原始关节点
            'torso_lower': 1,        # 下躯干区域
            'head_neck': 2,          # 头颈区域  
            'left_arm': 3,           # 左臂区域
            'right_arm': 4,          # 右臂区域
            'left_leg': 5,           # 左腿区域
            'right_leg': 6           # 右腿区域
        }
        
        # 标记原始关节点 (0-24)
        region_labels[:25] = region_name_to_id['original_joints']
        
        # 标记各解剖区域的插值点
        for region_name, info in region_info['anatomical_regions'].items():
            start_idx = info['start_idx']
            end_idx = info['end_idx']
            region_id = region_name_to_id[region_name]
            region_labels[start_idx:end_idx] = region_id
        
        return region_labels, region_name_to_id
    
    def get_detailed_point_info(self):
        """
        获取每个点的详细信息，包括点类型、所属区域、连接关系等
        便于Tokenizer理解点云的结构信息
        """
        region_info = self.get_branch_grouping_info()
        region_labels, region_name_to_id = self.get_point_region_labels()
        
        point_info = []
        
        # 原始关节点信息 (0-24)
        for joint_idx in range(25):
            point_info.append({
                'point_idx': joint_idx,
                'point_type': 'original_joint',
                'joint_id': joint_idx,
                'region_id': region_labels[joint_idx],
                'region_name': 'original_joints',
                'is_center_joint': joint_idx in [0, 20],  # 中心关节
                'bone_connection': None  # 原始关节不属于特定连接
            })
        
        # 插值点信息 (25-719)
        current_idx = 25
        for region_name, info in region_info['anatomical_regions'].items():
            region_id = region_name_to_id[region_name]
            
            # 遍历该区域的每个连接
            for connection_idx in info['connections']:
                start_joint, end_joint = NTU_CONNECTIONS[connection_idx]
                bone_points = self.bone_points_allocation.get(connection_idx, self.min_points_per_bone)
                
                # 为该连接的每个插值点添加信息
                for i in range(bone_points):
                    point_info.append({
                        'point_idx': current_idx,
                        'point_type': 'interpolated',
                        'joint_id': None,  # 插值点没有关节ID
                        'region_id': region_id,
                        'region_name': region_name,
                        'bone_connection': connection_idx,
                        'connection_joints': (start_joint, end_joint),
                        'interpolation_position': (i + 1) / (bone_points + 1)  # 在连接线上的相对位置
                    })
                    current_idx += 1
        
        return point_info
