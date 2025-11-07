#!/usr/bin/env python3
"""
AdaptiveSkeletonDVAE 可视化工具
用于可视化实际训练的 AdaptiveSkeletonDVAE 模型的输入输出
"""

import os
import sys
import numpy as np
import torch
import yaml
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.adaptive_skeleton_dvae import AdaptiveSkeletonDVAE
from datasets.build import build_dataset_from_cfg
from utils.config import cfg_from_yaml_file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class AdaptiveSkeletonDVAEVisualizer:
    """AdaptiveSkeletonDVAE 可视化器"""
    
    def __init__(self, config_path, checkpoint_path=None, device='cuda'):
        """
        初始化可视化器
        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径 (可选)
            device: 设备
        """
        self.device = device
        
        # 加载配置
        self.config = cfg_from_yaml_file(config_path)
        print(f"📋 Loaded config from {config_path}")
        
        # 创建模型 - 处理不同的配置结构
        if hasattr(self.config, 'model'):
            model_config = self.config.model
        else:
            # 如果没有model字段，创建一个默认配置
            print("⚠️ No model config found, using default AdaptiveSkeletonDVAE config")
            from types import SimpleNamespace
            model_config = SimpleNamespace()
            model_config.NAME = 'AdaptiveSkeletonDVAE'
            model_config.latent_dim = 512
            model_config.num_tokens = 1024
            model_config.commitment_cost = 0.25
            model_config.loss_type = 'mse'  # 默认使用MSE
        
        self.model = AdaptiveSkeletonDVAE(model_config).to(device)
        
        # 加载检查点
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Loaded model from {checkpoint_path}")
        else:
            print("⚠️ No checkpoint loaded, using random weights")
        
        self.model.eval()
    
    def visualize_single_sample(self, data_sample, save_path=None):
        """
        可视化单个样本的重建结果，包含骨架增强策略的详细展示
        Args:
            data_sample: (N, 3) 点云数据 (650点)
            save_path: 保存路径
        """
        if isinstance(data_sample, np.ndarray):
            data_sample = torch.from_numpy(data_sample).float()
        
        # 添加批次维度
        if len(data_sample.shape) == 2:
            data_sample = data_sample.unsqueeze(0)  # (1, N, 3)
        
        data_sample = data_sample.to(self.device)
        
        with torch.no_grad():
            # 模型推理
            coarse, fine, encoding_indices = self.model(data_sample)
        
        # 转换为numpy
        original = data_sample[0].cpu().numpy()  # (650, 3)
        coarse_recon = coarse[0].cpu().numpy()  # (64, 3)  
        fine_recon = fine[0].cpu().numpy()  # (650, 3)
        
        # 创建更丰富的可视化
        fig = plt.figure(figsize=(24, 16))
        
        # 第一行：骨架增强策略分析
        # 1. 原始17关节骨架结构
        ax1 = fig.add_subplot(241, projection='3d')
        self._plot_skeleton_structure(ax1, original, title='Original 17-Joint Skeleton\n(Extracted from 650 points)', 
                                     show_skeleton=True, show_augmented=False)
        
        # 2. 骨架增强策略可视化 (显示插值点)
        ax2 = fig.add_subplot(242, projection='3d') 
        self._plot_skeleton_structure(ax2, original, title='Skeleton Augmentation Strategy\n(650 points with interpolation)',
                                     show_skeleton=True, show_augmented=True)
        
        # 3. 密集点云展示
        ax3 = fig.add_subplot(243, projection='3d')
        self._plot_point_cloud(ax3, original, title=f'Dense Point Cloud\n({original.shape[0]} points)', 
                              color='blue', size=8)
        
        # 4. Coarse重建 (64点)
        ax4 = fig.add_subplot(244, projection='3d')
        self._plot_point_cloud(ax4, coarse_recon, title=f'Coarse Reconstruction\n({coarse_recon.shape[0]} points)', 
                              color='red', size=30)
        
        # 第二行：重建对比分析
        # 5. Fine重建
        ax5 = fig.add_subplot(245, projection='3d')
        self._plot_point_cloud(ax5, fine_recon, title=f'Fine Reconstruction\n({fine_recon.shape[0]} points)', 
                              color='green', size=8)
        
        # 6. 重建骨架结构
        ax6 = fig.add_subplot(246, projection='3d')
        self._plot_skeleton_structure(ax6, fine_recon, title='Reconstructed Skeleton\n(From 650 points)',
                                     show_skeleton=True, show_augmented=False)
        
        # 7. 叠加对比
        ax7 = fig.add_subplot(247, projection='3d')
        self._plot_overlay_comparison(ax7, original, fine_recon, title='Original vs Reconstructed\n(Overlay)')
        
        # 8. 误差分析
        ax8 = fig.add_subplot(248)
        self._plot_error_analysis(ax8, original, fine_recon)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization to {save_path}")
        
        plt.show()
        
        # 打印统计信息
        mse_coarse = np.mean((original - coarse_recon) ** 2) if original.shape[0] == 64 else "N/A (different sizes)"
        mse_fine = np.mean((original - fine_recon) ** 2)
        print(f"📊 MSE - Coarse: {mse_coarse}, Fine: {mse_fine:.6f}")
        print(f"🔢 VQ Index: {encoding_indices[0].item()}")
        print(f"🦴 Skeleton Analysis:")
        print(f"   Original joints (estimated): 17")
        print(f"   Augmented points: {original.shape[0] - 17} (interpolated)")
        print(f"   Augmentation ratio: {(original.shape[0] - 17) / 17:.1f}x")
        
        return {
            'original': original,
            'coarse': coarse_recon,
            'fine': fine_recon,
            'vq_index': encoding_indices[0].item(),
            'mse_fine': mse_fine
        }
    
    def _plot_point_cloud(self, ax, points, title, color='blue', size=20):
        """绘制点云"""
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=color, s=size, alpha=0.6)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 设置相同的坐标范围
        all_points = points.reshape(-1, 3)
        max_range = np.max(np.abs(all_points)) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    
    def _get_mmfi_skeleton_connections(self):
        """获取MMFI 17关节连接关系"""
        return [
            # 腿部连接
            (1, 2),   # J1-J2: 右侧腰部 -> 右膝盖
            (4, 5),   # J4-J5: 左侧腰部 -> 左膝盖
            (0, 1),   # J0-J1: 躯干中心 -> 右侧腰部
            (0, 4),   # J0-J4: 躯干中心 -> 左侧腰部
            (2, 3),   # J2-J3: 右膝盖 -> 右脚
            (5, 6),   # J5-J6: 左膝盖 -> 左脚
            
            # 躯干到肩膀
            (0, 7),   # J0-J7: 躯干中心 -> 肩膀中心
            
            # 头部连接链
            (10, 9),  # J10-J9: 头顶 -> 头颈部
            (9, 8),   # J9-J8: 头颈部 -> 头颈部
            (8, 7),   # J8-J7: 头颈部 -> 肩膀中心
            
            # 手臂连接链
            (8, 11),  # J8-J11: 头颈部 -> 肩膀
            (11, 12), # J11-J12: 肩膀 -> 左肩膀
            (12, 13), # J12-J13: 左肩膀 -> 左手
            
            (8, 14),  # J8-J14: 头颈部 -> 颈部
            (14, 15), # J14-J15: 颈部 -> 右肩膀
            (15, 16), # J15-J16: 右肩膀 -> 右手
        ]
    
    def _get_joint_colors(self):
        """获取关节点颜色（身体部位编码）"""
        return [
            '#E74C3C',  # J0: SpineBase - 红色 (躯干)
            '#E74C3C',  # J1: 右侧腰部 - 红色 (躯干)
            '#E67E22',  # J2: 右膝盖 - 橘色 (右腿)
            '#E67E22',  # J3: 右脚 - 橘色 (右腿)
            '#E74C3C',  # J4: 左侧腰部 - 红色 (躯干)
            '#9B59B6',  # J5: 左膝盖 - 紫色 (左腿)
            '#9B59B6',  # J6: 左脚 - 紫色 (左腿)
            '#E74C3C',  # J7: 肩膀中心 - 红色 (躯干)
            '#E74C3C',  # J8: 头颈部 - 红色 (躯干)
            '#F39C12',  # J9: 头颈部 - 橙色 (头部)
            '#F39C12',  # J10: 头顶 - 橙色 (头部)
            '#E74C3C',  # J11: 肩膀 - 红色 (躯干)
            '#3498DB',  # J12: 左肩膀 - 蓝色 (左臂)
            '#3498DB',  # J13: 左手 - 蓝色 (左臂)
            '#E74C3C',  # J14: 颈部 - 红色 (躯干)
            '#27AE60',  # J15: 右肩膀 - 绿色 (右臂)
            '#27AE60',  # J16: 右手 - 绿色 (右臂)
        ]
    
    def _extract_skeleton_joints(self, points, method='uniform'):
        """
        从650个点中提取17个主要关节点
        Args:
            points: (650, 3) 密集点云
            method: 提取方法 ('uniform', 'clustering')
        Returns:
            skeleton_joints: (17, 3) 骨架关节点
        """
        if method == 'uniform':
            # 均匀采样17个点
            indices = np.linspace(0, len(points)-1, 17, dtype=int)
            return points[indices]
        elif method == 'clustering':
            # TODO: 使用聚类方法提取关键点
            # 这里简化为均匀采样
            indices = np.linspace(0, len(points)-1, 17, dtype=int) 
            return points[indices]
        else:
            return points[:17]  # 取前17个点
    
    def _plot_skeleton_structure(self, ax, points, title, show_skeleton=True, show_augmented=False):
        """
        绘制骨架结构，展示骨架增强策略
        Args:
            ax: matplotlib轴
            points: (650, 3) 点云数据
            title: 标题
            show_skeleton: 是否显示骨架连接
            show_augmented: 是否显示增强插值点
        """
        # 提取17个主要关节
        skeleton_joints = self._extract_skeleton_joints(points)
        connections = self._get_mmfi_skeleton_connections()
        joint_colors = self._get_joint_colors()
        
        if show_skeleton:
            # 绘制关节点
            ax.scatter(skeleton_joints[:, 0], skeleton_joints[:, 1], skeleton_joints[:, 2],
                      c=joint_colors, s=100, alpha=0.9, edgecolors='black', linewidths=1,
                      label='Original Joints (17)')
            
            # 绘制骨架连接线
            for connection in connections:
                if connection[0] < len(skeleton_joints) and connection[1] < len(skeleton_joints):
                    start_joint = skeleton_joints[connection[0]]
                    end_joint = skeleton_joints[connection[1]]
                    ax.plot([start_joint[0], end_joint[0]], 
                           [start_joint[1], end_joint[1]], 
                           [start_joint[2], end_joint[2]], 
                           color='darkgray', alpha=0.8, linewidth=2, solid_capstyle='round')
        
        if show_augmented:
            # 显示所有650个点，用不同颜色区分原始关节和插值点
            augmented_points = points[17:]  # 假设前17个是关节点，后面是插值点
            
            # 插值点用小圆点显示
            ax.scatter(augmented_points[:, 0], augmented_points[:, 1], augmented_points[:, 2],
                      c='lightcoral', s=15, alpha=0.4, label=f'Interpolated Points ({len(augmented_points)})')
            
            # 原始关节点用大圆点显示
            ax.scatter(skeleton_joints[:, 0], skeleton_joints[:, 1], skeleton_joints[:, 2],
                      c=joint_colors, s=80, alpha=0.9, edgecolors='black', linewidths=1,
                      label='Original Joints (17)')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        
        # 设置坐标范围
        all_points = points.reshape(-1, 3)
        max_range = np.max(np.abs(all_points)) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        # 添加图例
        if show_skeleton or show_augmented:
            ax.legend(loc='upper right', fontsize=8)
    
    def _plot_overlay_comparison(self, ax, original, reconstructed, title):
        """绘制叠加对比图"""
        # 原始点云（蓝色，透明）
        ax.scatter(original[:, 0], original[:, 1], original[:, 2],
                  c='blue', s=15, alpha=0.5, label='Original')
        
        # 重建点云（红色，透明）
        ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
                  c='red', s=15, alpha=0.5, label='Reconstructed')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        # 设置坐标范围
        all_points = np.concatenate([original, reconstructed])
        max_range = np.max(np.abs(all_points)) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    
    def _plot_error_analysis(self, ax, original, reconstructed):
        """绘制误差分析图"""
        # 计算逐点误差
        errors = np.linalg.norm(original - reconstructed, axis=1)
        
        # 绘制误差分布直方图
        ax.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Point-wise L2 Error', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Reconstruction Error Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        
        ax.axvline(mean_error, color='red', linestyle='--', alpha=0.8, 
                  label=f'Mean: {mean_error:.4f}')
        ax.axvline(mean_error + std_error, color='orange', linestyle='--', alpha=0.8,
                  label=f'Mean+Std: {mean_error + std_error:.4f}')
        
        ax.legend(fontsize=8)
        
        # 在标题中添加关键统计信息
        ax.text(0.02, 0.95, f'Max: {max_error:.4f}\nStd: {std_error:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=8)
    
    def _dict_to_namespace(self, config_dict):
        """
        将字典转换为具有属性访问的对象
        Args:
            config_dict: 配置字典
        Returns:
            具有属性访问的配置对象
        """
        from types import SimpleNamespace
        
        if isinstance(config_dict, dict):
            namespace = SimpleNamespace()
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    setattr(namespace, key, self._dict_to_namespace(value))
                else:
                    setattr(namespace, key, value)
            return namespace
        else:
            return config_dict
    def _build_dataset_config(self, dataset_config):
        """
        构建数据集配置，处理_base_引用，并转换为字典格式
        Args:
            dataset_config: 原始配置
        Returns:
            处理后的配置对象（具有属性访问）
        """
        if hasattr(dataset_config, '_base_'):
            # 处理_base_引用
            base_config = dataset_config._base_
            
            # 创建字典配置
            config_dict = {}
            
            # 复制基础配置
            if hasattr(base_config, '__dict__'):
                config_dict.update(base_config.__dict__)
            else:
                config_dict.update(base_config)
            
            # 添加其他配置
            if hasattr(dataset_config, 'others'):
                if hasattr(dataset_config.others, '__dict__'):
                    config_dict.update(dataset_config.others.__dict__)
                else:
                    config_dict.update(dataset_config.others)
            
            # 转换为具有属性访问的对象
            return self._dict_to_namespace(config_dict)
        else:
            # 如果已经是正确的格式，直接返回
            return dataset_config
    
    def visualize_dataset_samples(self, num_samples=5, save_dir=None):
        """
        可视化数据集中的多个样本
        Args:
            num_samples: 样本数量
            save_dir: 保存目录
        """
        # 直接创建MMFI数据集实例，绕过注册系统
        try:
            # 构建数据集配置
            if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'train'):
                dataset_config = self.config.dataset.train
                # 处理_base_引用
                dataset_config = self._build_dataset_config(dataset_config)
            else:
                # 使用默认的MMFI配置
                print("⚠️ No dataset config found, using default MMFI config")
                dataset_config = self._dict_to_namespace({
                    'NAME': 'MMFI',
                    'DATA_PATH': 'data/MMFI',
                    'N_POINTS': 650,
                    'subset': 'train'
                })
            
            # 直接创建MMFI数据集实例
            from datasets.MMFIDataset import MMFIDataset
            dataset = MMFIDataset(dataset_config)
            print(f"📦 Dataset size: {len(dataset)}")
            
        except Exception as e:
            print(f"❌ Failed to create dataset: {e}")
            return []
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        results = []
        for i in range(min(num_samples, len(dataset))):
            print(f"\n🔍 Processing sample {i+1}/{num_samples}")
            
            # 获取数据样本
            data = dataset[i]
            if isinstance(data, dict):
                points = data['points'] if 'points' in data else data['pos']
            elif isinstance(data, (list, tuple)) and len(data) >= 2:
                # MMFIDataset 返回 (taxonomy_id, model_id, data)
                points = data[2] if len(data) > 2 else data[1]
            else:
                points = data
            
            print(f"📐 Sample shape: {points.shape}")
            
            # 可视化
            save_path = os.path.join(save_dir, f'sample_{i+1}_reconstruction.png') if save_dir else None
            result = self.visualize_single_sample(points, save_path)
            results.append(result)
        
        # 统计信息
        mse_values = [r['mse_fine'] for r in results]
        print(f"\n📈 Reconstruction MSE Statistics:")
        print(f"   Mean: {np.mean(mse_values):.6f}")
        print(f"   Std:  {np.std(mse_values):.6f}")
        print(f"   Min:  {np.min(mse_values):.6f}")
        print(f"   Max:  {np.max(mse_values):.6f}")
        
        return results
    
    def visualize_codebook_usage(self, num_samples=100, save_path=None):
        """
        分析码本使用情况
        Args:
            num_samples: 分析的样本数量
            save_path: 保存路径
        """
        # 直接创建MMFI数据集实例
        try:
            # 构建数据集配置
            if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'train'):
                dataset_config = self.config.dataset.train
                # 处理_base_引用
                dataset_config = self._build_dataset_config(dataset_config)
            else:
                # 使用默认的MMFI配置
                print("⚠️ No dataset config found, using default MMFI config")
                dataset_config = self._dict_to_namespace({
                    'NAME': 'MMFI',
                    'DATA_PATH': 'data/MMFI',
                    'N_POINTS': 650,
                    'subset': 'train'
                })
            
            # 直接创建MMFI数据集实例
            from datasets.MMFIDataset import MMFIDataset
            dataset = MMFIDataset(dataset_config)
            
        except Exception as e:
            print(f"❌ Failed to create dataset: {e}")
            return {}
        
        vq_indices = []
        print(f"🔢 Analyzing codebook usage with {num_samples} samples...")
        
        for i in range(min(num_samples, len(dataset))):
            data = dataset[i]
            if isinstance(data, dict):
                points = data['points'] if 'points' in data else data['pos']
            elif isinstance(data, (list, tuple)) and len(data) >= 2:
                # MMFIDataset 返回 (taxonomy_id, model_id, data)
                points = data[2] if len(data) > 2 else data[1]
            else:
                points = data
            
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points).float()
            
            if len(points.shape) == 2:
                points = points.unsqueeze(0)
            
            points = points.to(self.device)
            
            with torch.no_grad():
                _, _, encoding_indices = self.model(points)
                vq_indices.append(encoding_indices[0].item())
        
        # 统计分析
        vq_indices = np.array(vq_indices)
        unique_indices = np.unique(vq_indices)
        
        print(f"📊 Codebook Usage Statistics:")
        print(f"   Total codes used: {len(unique_indices)}/{self.model.codebook_size}")
        print(f"   Usage rate: {len(unique_indices)/self.model.codebook_size*100:.2f}%")
        print(f"   Most frequent code: {np.bincount(vq_indices).argmax()}")
        
        # 绘制直方图
        plt.figure(figsize=(12, 6))
        plt.hist(vq_indices, bins=min(50, len(unique_indices)), alpha=0.7, edgecolor='black')
        plt.title(f'VQ Code Usage Distribution ({num_samples} samples)')
        plt.xlabel('VQ Code Index')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved codebook analysis to {save_path}")
        
        plt.show()
        
        return {
            'indices': vq_indices,
            'unique_count': len(unique_indices),
            'usage_rate': len(unique_indices)/self.model.codebook_size
        }
    
    def visualize_skeleton_augmentation_strategy(self, data_sample, save_path=None):
        """
        专门可视化骨架增强策略的详细过程
        Args:
            data_sample: (650, 3) 点云数据
            save_path: 保存路径
        """
        if isinstance(data_sample, np.ndarray):
            points = data_sample
        else:
            points = data_sample.cpu().numpy() if hasattr(data_sample, 'cpu') else data_sample
        
        # 创建详细的骨架增强策略可视化
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 原始17关节骨架
        ax1 = fig.add_subplot(231, projection='3d')
        skeleton_joints = self._extract_skeleton_joints(points)
        self._plot_skeleton_only(ax1, skeleton_joints, 'Step 1: Original 17 Joints\n(Human Skeleton Structure)')
        
        # 2. 骨架连接线
        ax2 = fig.add_subplot(232, projection='3d')
        self._plot_skeleton_with_connections(ax2, skeleton_joints, 'Step 2: Skeleton Connections\n(16 bone connections)')
        
        # 3. 插值点生成示例（显示几条连接的插值）
        ax3 = fig.add_subplot(233, projection='3d')
        self._plot_interpolation_demo(ax3, skeleton_joints, 'Step 3: Interpolation Strategy\n(Adding points along connections)')
        
        # 4. 完整的增强点云
        ax4 = fig.add_subplot(234, projection='3d')
        self._plot_skeleton_structure(ax4, points, 'Step 4: Complete Augmented Cloud\n(650 points total)',
                                     show_skeleton=True, show_augmented=True)
        
        # 5. 密度分析
        ax5 = fig.add_subplot(235, projection='3d')
        self._plot_density_analysis(ax5, points, skeleton_joints, 'Step 5: Point Density Analysis\n(Distribution along skeleton)')
        
        # 6. 统计分析
        ax6 = fig.add_subplot(236)
        self._plot_augmentation_statistics(ax6, points, skeleton_joints)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved skeleton augmentation analysis to {save_path}")
        
        plt.show()
        
        # 详细统计信息
        print(f"🦴 Skeleton Augmentation Analysis:")
        print(f"   Original joints: 17")
        print(f"   Total points after augmentation: {len(points)}")
        print(f"   Interpolated points: {len(points) - 17}")
        print(f"   Augmentation ratio: {(len(points) - 17) / 17:.1f}x")
        print(f"   Average points per connection: {(len(points) - 17) / 16:.1f}")
        
        return {
            'original_joints': skeleton_joints,
            'total_points': len(points),
            'augmented_points': len(points) - 17,
            'augmentation_ratio': (len(points) - 17) / 17
        }
    
    def _plot_skeleton_only(self, ax, skeleton_joints, title):
        """只绘制骨架关节点"""
        joint_colors = self._get_joint_colors()
        
        ax.scatter(skeleton_joints[:, 0], skeleton_joints[:, 1], skeleton_joints[:, 2],
                  c=joint_colors, s=120, alpha=0.9, edgecolors='black', linewidths=2)
        
        # 添加关节编号
        for i, joint in enumerate(skeleton_joints):
            ax.text(joint[0], joint[1], joint[2], f'J{i}', 
                   fontsize=8, fontweight='bold', ha='center')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        self._set_axis_limits(ax, skeleton_joints)
    
    def _plot_skeleton_with_connections(self, ax, skeleton_joints, title):
        """绘制带连接线的骨架"""
        joint_colors = self._get_joint_colors()
        connections = self._get_mmfi_skeleton_connections()
        
        # 绘制关节点
        ax.scatter(skeleton_joints[:, 0], skeleton_joints[:, 1], skeleton_joints[:, 2],
                  c=joint_colors, s=100, alpha=0.9, edgecolors='black', linewidths=1)
        
        # 绘制连接线，用不同颜色表示不同的连接
        connection_colors = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6', 
                           '#E67E22', '#1ABC9C', '#34495E']
        
        for i, connection in enumerate(connections):
            if connection[0] < len(skeleton_joints) and connection[1] < len(skeleton_joints):
                start_joint = skeleton_joints[connection[0]]
                end_joint = skeleton_joints[connection[1]]
                color = connection_colors[i % len(connection_colors)]
                
                ax.plot([start_joint[0], end_joint[0]], 
                       [start_joint[1], end_joint[1]], 
                       [start_joint[2], end_joint[2]], 
                       color=color, alpha=0.8, linewidth=3, solid_capstyle='round')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        self._set_axis_limits(ax, skeleton_joints)
        
        # 添加连接数量信息
        ax.text2D(0.02, 0.98, f'{len(connections)} connections', 
                 transform=ax.transAxes, fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _plot_interpolation_demo(self, ax, skeleton_joints, title):
        """演示插值过程"""
        connections = self._get_mmfi_skeleton_connections()
        joint_colors = self._get_joint_colors()
        
        # 绘制关节点
        ax.scatter(skeleton_joints[:, 0], skeleton_joints[:, 1], skeleton_joints[:, 2],
                  c=joint_colors, s=80, alpha=0.9, edgecolors='black', linewidths=1)
        
        # 选择几条代表性连接进行插值演示
        demo_connections = connections[:5]  # 前5个连接
        interpolation_points_per_connection = 37  # 平均每个连接的插值点数
        
        for i, connection in enumerate(demo_connections):
            start_joint = skeleton_joints[connection[0]]
            end_joint = skeleton_joints[connection[1]]
            
            # 在连接线上生成插值点
            t_values = np.linspace(0, 1, interpolation_points_per_connection)
            interpolated_points = np.array([
                start_joint + t * (end_joint - start_joint) for t in t_values
            ])
            
            # 绘制连接线
            ax.plot([start_joint[0], end_joint[0]], 
                   [start_joint[1], end_joint[1]], 
                   [start_joint[2], end_joint[2]], 
                   color='gray', alpha=0.6, linewidth=2)
            
            # 绘制插值点
            ax.scatter(interpolated_points[:, 0], interpolated_points[:, 1], interpolated_points[:, 2],
                      c='red', s=20, alpha=0.7, label=f'Interpolated' if i == 0 else "")
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        self._set_axis_limits(ax, skeleton_joints)
        
        if len(demo_connections) > 0:
            ax.legend(loc='upper right', fontsize=10)
    
    def _plot_density_analysis(self, ax, all_points, skeleton_joints, title):
        """绘制点密度分析"""
        # 计算每个点到最近骨架关节的距离
        distances_to_skeleton = []
        for point in all_points:
            min_dist = min([np.linalg.norm(point - joint) for joint in skeleton_joints])
            distances_to_skeleton.append(min_dist)
        
        distances_to_skeleton = np.array(distances_to_skeleton)
        
        # 用颜色编码表示密度（距离骨架的远近）
        scatter = ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
                           c=distances_to_skeleton, s=15, alpha=0.6, cmap='viridis')
        
        # 高亮原始骨架关节
        ax.scatter(skeleton_joints[:, 0], skeleton_joints[:, 1], skeleton_joints[:, 2],
                  c='red', s=80, alpha=1.0, edgecolors='white', linewidths=2,
                  label='Original Joints')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        self._set_axis_limits(ax, all_points)
        
        # 添加颜色条
        plt.colorbar(scatter, ax=ax, shrink=0.6, label='Distance to Skeleton')
    
    def _plot_augmentation_statistics(self, ax, all_points, skeleton_joints):
        """绘制增强策略统计信息"""
        # 计算统计信息
        total_points = len(all_points)
        original_joints = len(skeleton_joints)
        augmented_points = total_points - original_joints
        connections = len(self._get_mmfi_skeleton_connections())
        avg_points_per_connection = augmented_points / connections
        
        # 创建统计图表
        categories = ['Original\nJoints', 'Augmented\nPoints', 'Total\nPoints']
        values = [original_joints, augmented_points, total_points]
        colors = ['#E74C3C', '#3498DB', '#27AE60']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   str(value), ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Skeleton Augmentation Statistics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Points')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加详细统计信息
        stats_text = f"""Augmentation Details:
• Connections: {connections}
• Avg points/connection: {avg_points_per_connection:.1f}
• Augmentation ratio: {augmented_points/original_joints:.1f}x
• Density increase: {total_points/original_joints:.1f}x"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontfamily='monospace', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    def _set_axis_limits(self, ax, points):
        """设置轴限制"""
        max_range = np.max(np.abs(points)) * 1.1
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])


def main():
    parser = argparse.ArgumentParser(description='AdaptiveSkeletonDVAE Visualization Tool')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, 
                       choices=['single', 'dataset', 'codebook', 'skeleton_strategy'], 
                       default='dataset',
                       help='Visualization mode')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='./visualizations/adaptive_dvae',
                       help='Save directory for outputs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = AdaptiveSkeletonDVAEVisualizer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    if args.mode == 'dataset':
        # 可视化数据集样本
        print(f"🎨 Visualizing {args.num_samples} dataset samples...")
        visualizer.visualize_dataset_samples(
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )
    
    elif args.mode == 'codebook':
        # 分析码本使用情况
        print("📈 Analyzing codebook usage...")
        save_path = os.path.join(args.save_dir, 'codebook_usage.png')
        visualizer.visualize_codebook_usage(
            num_samples=args.num_samples,
            save_path=save_path
        )
    
    elif args.mode == 'skeleton_strategy':
        # 可视化骨架增强策略
        print("🦴 Analyzing skeleton augmentation strategy...")
        
        # 获取一个样本进行演示
        try:
            # 构建数据集配置
            if hasattr(visualizer.config, 'dataset') and hasattr(visualizer.config.dataset, 'train'):
                dataset_config = visualizer.config.dataset.train
                dataset_config = visualizer._build_dataset_config(dataset_config)
            else:
                dataset_config = visualizer._dict_to_namespace({
                    'NAME': 'MMFI',
                    'DATA_PATH': 'data/MMFI',
                    'N_POINTS': 650,
                    'subset': 'train'
                })
            
            # 创建数据集实例
            from datasets.MMFIDataset import MMFIDataset
            dataset = MMFIDataset(dataset_config)
            
            # 获取第一个样本进行演示
            data = dataset[0]
            if isinstance(data, dict):
                points = data['points'] if 'points' in data else data['pos']
            elif isinstance(data, (list, tuple)) and len(data) >= 2:
                points = data[2] if len(data) > 2 else data[1]
            else:
                points = data
            
            # 可视化骨架增强策略
            save_path = os.path.join(args.save_dir, 'skeleton_augmentation_strategy.png')
            visualizer.visualize_skeleton_augmentation_strategy(points, save_path)
            
        except Exception as e:
            print(f"❌ Failed to analyze skeleton strategy: {e}")
            print("Using synthetic skeleton data for demonstration...")
            
            # 创建合成的骨架数据进行演示
            synthetic_skeleton = np.random.randn(650, 3) * 0.5
            save_path = os.path.join(args.save_dir, 'skeleton_augmentation_demo.png')
            visualizer.visualize_skeleton_augmentation_strategy(synthetic_skeleton, save_path)
    
    elif args.mode == 'single':
        # 单样本可视化 - 需要手动提供数据
        print("⚠️ Single mode requires manual data input")
        print("   Use visualizer.visualize_single_sample(your_data) directly")
        print("   Or try skeleton_strategy mode to see augmentation analysis")


if __name__ == '__main__':
    main()
