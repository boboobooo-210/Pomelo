"""
MMFI数据集加载器 - 专门用于VQ-VAE预训练
支持加载dvae_pretrain.npy文件（数据增强后的预训练数据），并自动分配语义标签
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .build import DATASETS
from utils.logger import print_log

# 27个动作的语义标签映射
ACTION_LABELS = {
    'A01': 'Stretching_relaxing',
    'A02': 'Chest_expansion_horizontal', 
    'A03': 'Chest_expansion_vertical',
    'A04': 'Twist_left',
    'A05': 'Twist_right',
    'A06': 'Mark_time',
    'A07': 'Limb_extension_left',
    'A08': 'Limb_extension_right',
    'A09': 'Left_lunge',
    'A10': 'Right_lunge',
    'A11': 'Limb_extension_both',
    'A12': 'Squat',
    'A13': 'Raising_left_hand',
    'A14': 'Raising_right_hand',
    'A15': 'Lunge_toward_left_side',
    'A16': 'Lunge_toward_right_side',
    'A17': 'Waving_left_hand',
    'A18': 'Waving_right_hand',
    'A19': 'Picking_up_things',
    'A20': 'Throwing_toward_left_side',
    'A21': 'Throwing_toward_right_side',
    'A22': 'Kicking_toward_left_side',
    'A23': 'Kicking_toward_right_side',
    'A24': 'Body_extension_left',
    'A25': 'Body_extension_right',
    'A26': 'Jumping_up',
    'A27': 'Bowing'
}

# 动作ID到数字标签的映射
ACTION_ID_TO_LABEL = {action_id: idx for idx, action_id in enumerate(sorted(ACTION_LABELS.keys()))}

@DATASETS.register_module(name='MMFI')
class MMFIDataset(Dataset):
    """
    MMFI数据集 - 用于dVAE训练的骨架数据
    """
    
    def __init__(self, config):
        self.config = config
        self.data_root = config.DATA_PATH
        self.num_points = getattr(config, 'N_POINTS', 650)  # 默认650点
        self.subset = getattr(config, 'subset', 'train')
        
        # 检查是否需要重采样（用于原版dVAE）
        self.target_npoints = getattr(config, 'npoints', None)
        if self.target_npoints:
            print(f"🔄 将重采样点云从 {self.num_points} 到 {self.target_npoints} 个点（用于原版dVAE）")
            self.num_points = self.target_npoints
        
        # 数据分割配置
        self.data_split = {
            'train': {
                'environments': ['E01', 'E02', 'E03'],
                'sessions': 'all',  # E01-E03使用所有S
                'e04_sessions': ['S31', 'S32']  # E04只使用S31和S32
            },
            'test': {
                'environments': ['E04'],
                'sessions': ['S33', 'S34', 'S35', 'S36']
            },
            'val': {
                'environments': ['E04'], 
                'sessions': ['S37', 'S38', 'S39', 'S40']
            }
        }
        
        # 加载数据
        self.data_list = []
        self.labels = []
        self.action_names = []
        
        self._load_data()
        
        print_log(f"MMFI {self.subset} dataset loaded: {len(self.data_list)} samples", logger='MMFI')
        
    def _load_data(self):
        """加载指定分割的数据"""
        split_config = self.data_split[self.subset]
        
        for env in split_config['environments']:
            env_path = os.path.join(self.data_root, env)
            
            if not os.path.exists(env_path):
                continue
                
            # 获取该环境下的所有session
            if env == 'E04':
                # E04的session根据数据集分割来决定
                sessions = split_config.get('sessions', [])
            elif env in ['E01', 'E02', 'E03'] and self.subset == 'train':
                # 训练时，E01-E03使用所有S
                sessions = []
                for item in os.listdir(env_path):
                    if item.startswith('S') and os.path.isdir(os.path.join(env_path, item)):
                        sessions.append(item)
            else:
                # 对于非训练集，E01-E03不参与
                sessions = []
            
            # 如果是训练集且当前环境是E04，添加特定的session
            if self.subset == 'train' and env == 'E04':
                sessions.extend(split_config.get('e04_sessions', []))
            
            for session in sessions:
                session_path = os.path.join(env_path, session)
                
                if not os.path.exists(session_path):
                    continue
                    
                # 遍历所有动作
                for action in sorted(os.listdir(session_path)):
                    if not action.startswith('A'):
                        continue
                        
                    action_path = os.path.join(session_path, action)
                    # 优先使用预训练数据文件（数据增强后），如果不存在则使用原始数据
                    pretrain_data_path = os.path.join(action_path, 'dvae_pretrain.npy')
                    ground_truth_path = os.path.join(action_path, 'ground_truth.npy')
                    
                    # 选择可用的数据文件
                    if os.path.exists(pretrain_data_path):
                        data_path = pretrain_data_path
                        data_type = "pretrain"
                    elif os.path.exists(ground_truth_path):
                        data_path = ground_truth_path
                        data_type = "ground_truth"
                    else:
                        continue  # 如果两个文件都不存在，跳过
                    
                    try:
                        # 加载数据文件
                        data = np.load(data_path)
                        
                        # 检查数据形状
                        if len(data.shape) == 3:
                            frames, points, dims = data.shape
                            print(f"    加载: {env}/{session}/{action} - 形状: {data.shape} ({frames}帧 × {points}点)")
                            
                            # 将每一帧作为独立样本添加到数据集
                            for frame_idx in range(frames):
                                frame_data = data[frame_idx]  # 形状: (650, 3)
                                
                                # 获取语义标签
                                action_label = ACTION_ID_TO_LABEL.get(action, 0)
                                action_name = ACTION_LABELS.get(action, 'Unknown')
                                
                                self.data_list.append({
                                    'data': frame_data,
                                    'path': data_path,
                                    'environment': env,
                                    'session': session,
                                    'action': action,
                                    'frame_idx': frame_idx
                                })
                                self.labels.append(action_label)
                                self.action_names.append(action_name)
                        else:
                            print(f"    ⚠️ 跳过非标准数据形状: {data.shape}")
                            
                    except Exception as e:
                        print_log(f"Error loading {data_path}: {e}", logger='MMFI')
                        continue
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.data_list[idx]
        data = sample['data']  # 已经是单帧数据 (650, 3)
        label = self.labels[idx]
        action_name = self.action_names[idx]
        
        # 数据预处理
        data = self._preprocess_data(data)
        
        # 为了与现有框架兼容，返回taxonomy_id, model_id, data的格式
        taxonomy_id = 'human_skeleton'
        model_id = f"mmfi_{sample['environment']}_{sample['session']}_{sample['action']}_f{sample['frame_idx']:03d}_{idx:06d}"
        
        return taxonomy_id, model_id, data
    
    def _preprocess_data(self, data):
        """数据预处理：单帧点云数据"""
        # 转换为torch tensor
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        # 首先检查并处理NaN/Inf值
        if torch.isnan(data).any() or torch.isinf(data).any():
            print(f"⚠️ 发现NaN或Inf值，将其替换为0")
            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 数据应该已经是 (N, 3) 格式的点云
        if len(data.shape) == 2 and data.shape[1] == 3:
            # 确保点数符合配置要求
            current_points = data.shape[0]
            target_points = self.num_points
            
            if current_points != target_points:
                # 如果点数不匹配，进行重采样
                if current_points > target_points:
                    # 随机采样到目标点数
                    indices = torch.randperm(current_points)[:target_points]
                    data = data[indices]
                else:
                    # 重复采样到目标点数 - 使用重复而不是随机索引避免超出范围
                    repeat_times = (target_points + current_points - 1) // current_points
                    data = data.repeat(repeat_times, 1)[:target_points]
            
            # 再次检查处理后的数据
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"⚠️ 处理后仍有NaN或Inf值，使用零矩阵替换")
                data = torch.zeros_like(data)
            
            # 数据标准化到合理范围 [-2, 2]
            if data.numel() > 0:
                data_abs_max = torch.abs(data).max()
                if data_abs_max > 5.0:  # 如果数据范围过大
                    data = data / data_abs_max * 2.0  # 归一化到[-2, 2]
            
            return data
        else:
            raise ValueError(f"数据形状错误: {data.shape}，期望 (N, 3)")
        
    def get_frame_count_statistics(self):
        """获取帧数统计信息"""
        if not hasattr(self, '_frame_stats'):
            stats = {}
            for sample in self.data_list:
                key = f"{sample['environment']}/{sample['session']}/{sample['action']}"
                if key not in stats:
                    stats[key] = 0
                stats[key] += 1
            self._frame_stats = stats
        return self._frame_stats


