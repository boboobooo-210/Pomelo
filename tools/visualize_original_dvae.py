#!/usr/bin/env python3
"""
原版DiscreteVAE可视化工具
用于可视化原版dVAE模型的输入输出和重建结果
"""

import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.dvae import DiscreteVAE
from datasets.build import build_dataset_from_cfg
from utils.config import cfg_from_yaml_file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


class OriginalDVAEVisualizer:
    """原版DiscreteVAE可视化器"""
    
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
        
        # 创建模型
        if hasattr(self.config, 'model'):
            model_config = self.config.model
        else:
            print("❌ 配置文件中没有model字段")
            return
        
        self.model = DiscreteVAE(model_config).to(device)
        print(f"🔧 Created DiscreteVAE model:")
        print(f"   - Group size: {model_config.group_size}")
        print(f"   - Num groups: {model_config.num_group}")
        print(f"   - Total points: {model_config.group_size * model_config.num_group}")
        print(f"   - Num tokens: {model_config.num_tokens}")
        
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
        可视化单个样本的重建结果
        Args:
            data_sample: (N, 3) 点云数据
            save_path: 保存路径
        """
        if isinstance(data_sample, np.ndarray):
            data_sample = torch.from_numpy(data_sample).float()
        
        # 添加批次维度
        if len(data_sample.shape) == 2:
            data_sample = data_sample.unsqueeze(0)  # (1, N, 3)
        
        data_sample = data_sample.to(self.device)
        
        with torch.no_grad():
            # 原版dVAE前向传播
            # 返回格式: whole_coarse, whole_fine, coarse, fine, group_gt, kl_loss
            ret = self.model(data_sample, temperature=0.1, hard=False)
            
            whole_coarse, whole_fine, coarse, fine, group_gt, kl_loss = ret
        
        # 转换为numpy
        original = data_sample[0].cpu().numpy()  # (N, 3)
        whole_coarse_np = whole_coarse[0].cpu().numpy()  # (N, 3) 
        whole_fine_np = whole_fine[0].cpu().numpy()  # (N, 3)
        
        # coarse和fine是分组格式 (batch, num_group, group_size, 3)
        # 需要重新整形
        bs, num_group, group_size, _ = coarse.shape
        coarse_reshaped = coarse.view(bs, -1, 3)[0].cpu().numpy()  # (num_group*group_size, 3)
        fine_reshaped = fine.view(bs, -1, 3)[0].cpu().numpy()  # (num_group*group_size, 3)
        
        # 创建可视化
        fig = plt.figure(figsize=(20, 12))
        
        # 1. 原始点云
        ax1 = fig.add_subplot(231, projection='3d')
        self._plot_point_cloud(ax1, original, title=f'Original ({original.shape[0]} points)', color='blue')
        
        # 2. Whole Coarse (粗略重建)
        ax2 = fig.add_subplot(232, projection='3d')
        self._plot_point_cloud(ax2, whole_coarse_np, title=f'Whole Coarse ({whole_coarse_np.shape[0]} points)', color='orange')
        
        # 3. Whole Fine (精细重建)
        ax3 = fig.add_subplot(233, projection='3d')
        self._plot_point_cloud(ax3, whole_fine_np, title=f'Whole Fine ({whole_fine_np.shape[0]} points)', color='green')
        
        # 4. 分组Coarse
        ax4 = fig.add_subplot(234, projection='3d')
        self._plot_grouped_points(ax4, coarse_reshaped, num_group, group_size, 
                                 title=f'Grouped Coarse ({num_group}×{group_size} points)')
        
        # 5. 分组Fine
        ax5 = fig.add_subplot(235, projection='3d')
        self._plot_grouped_points(ax5, fine_reshaped, num_group, group_size,
                                 title=f'Grouped Fine ({num_group}×{group_size} points)')
        
        # 6. 对比图
        ax6 = fig.add_subplot(236, projection='3d')
        self._plot_comparison(ax6, original, whole_fine_np, title='Original vs Fine Reconstruction')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved visualization to {save_path}")
        
        plt.show()
        
        # 打印统计信息
        mse_whole_fine = np.mean((original - whole_fine_np) ** 2)
        kl_loss_value = kl_loss.mean().item() if kl_loss is not None else 0.0
        
        print(f"📊 Reconstruction Statistics:")
        print(f"   MSE (Original vs Whole Fine): {mse_whole_fine:.6f}")
        print(f"   KL Loss: {kl_loss_value:.6f}")
        print(f"🔧 Model Info:")
        print(f"   Groups: {num_group}, Group size: {group_size}")
        print(f"   Total reconstructed points: {num_group * group_size}")
        
        return {
            'original': original,
            'whole_coarse': whole_coarse_np,
            'whole_fine': whole_fine_np,
            'coarse_grouped': coarse_reshaped,
            'fine_grouped': fine_reshaped,
            'mse_whole_fine': mse_whole_fine,
            'kl_loss': kl_loss_value
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
        max_range = np.max(np.abs(all_points)) * 1.1 if all_points.size > 0 else 1.0
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
    
    def _plot_grouped_points(self, ax, points, num_group, group_size, title):
        """绘制分组点云，不同组用不同颜色"""
        colors = plt.cm.tab20(np.linspace(0, 1, num_group))
        
        for g in range(num_group):
            start_idx = g * group_size
            end_idx = start_idx + group_size
            if end_idx <= len(points):
                group_points = points[start_idx:end_idx]
                ax.scatter(group_points[:, 0], group_points[:, 1], group_points[:, 2],
                          c=[colors[g]], s=30, alpha=0.7, label=f'Group {g+1}')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        
        # 设置相同的坐标范围
        if len(points) > 0:
            max_range = np.max(np.abs(points)) * 1.1
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
    
    def _plot_comparison(self, ax, original, reconstructed, title):
        """绘制原始和重建的对比"""
        ax.scatter(original[:, 0], original[:, 1], original[:, 2],
                  c='blue', s=20, alpha=0.5, label='Original')
        ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2],
                  c='red', s=15, alpha=0.7, label='Reconstructed')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        # 设置相同的坐标范围
        all_points = np.vstack([original, reconstructed])
        if len(all_points) > 0:
            max_range = np.max(np.abs(all_points)) * 1.1
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])

    def visualize_dataset_samples(self, num_samples=3, save_dir=None):
        """可视化数据集样本"""
        # 直接创建MMFI数据集实例
        try:
            if hasattr(self.config, 'dataset') and hasattr(self.config.dataset, 'train'):
                dataset_config = self.config.dataset.train
                # 处理_base_引用
                dataset_config = self._build_dataset_config(dataset_config)
            else:
                print("⚠️ No dataset config found, using default MMFI config")
                dataset_config = self._dict_to_namespace({
                    'NAME': 'MMFI',
                    'DATA_PATH': 'data/MMFI',
                    'N_POINTS': 650,
                    'subset': 'train',
                    'npoints': self.config.model.group_size * self.config.model.num_group  # 设置目标点数
                })
            
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
            if isinstance(data, (list, tuple)) and len(data) >= 2:
                points = data[2] if len(data) > 2 else data[1]
            else:
                points = data
            
            print(f"📐 Sample shape: {points.shape}")
            
            # 可视化
            save_path = os.path.join(save_dir, f'original_dvae_sample_{i+1}.png') if save_dir else None
            result = self.visualize_single_sample(points, save_path)
            results.append(result)
        
        # 统计信息
        mse_values = [r['mse_whole_fine'] for r in results]
        kl_values = [r['kl_loss'] for r in results]
        
        print(f"\n📈 Reconstruction Statistics (Average):")
        print(f"   Mean MSE: {np.mean(mse_values):.6f}")
        print(f"   Std MSE:  {np.std(mse_values):.6f}")
        print(f"   Mean KL:  {np.mean(kl_values):.6f}")
        print(f"   Std KL:   {np.std(kl_values):.6f}")
        
        return results
    
    def _build_dataset_config(self, dataset_config):
        """构建数据集配置，处理_base_引用"""
        if hasattr(dataset_config, '_base_'):
            base_config = dataset_config._base_
            config_dict = {}
            
            if hasattr(base_config, '__dict__'):
                config_dict.update(base_config.__dict__)
            else:
                config_dict.update(base_config)
            
            if hasattr(dataset_config, 'others'):
                if hasattr(dataset_config.others, '__dict__'):
                    config_dict.update(dataset_config.others.__dict__)
                else:
                    config_dict.update(dataset_config.others)
            
            return self._dict_to_namespace(config_dict)
        else:
            return dataset_config
    
    def _dict_to_namespace(self, config_dict):
        """将字典转换为具有属性访问的对象"""
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


def main():
    parser = argparse.ArgumentParser(description='Original DiscreteVAE Visualization Tool')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Number of samples to visualize')
    parser.add_argument('--save_dir', type=str, default='./visualizations/original_dvae',
                       help='Save directory for outputs')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = OriginalDVAEVisualizer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # 可视化数据集样本
    print(f"🎨 Visualizing {args.num_samples} dataset samples...")
    visualizer.visualize_dataset_samples(
        num_samples=args.num_samples,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
