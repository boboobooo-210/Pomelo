#!/usr/bin/env python3
"""
骨架增强策略可视化演示脚本
演示如何使用 AdaptiveSkeletonDVAE 可视化工具分析骨架增强策略
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from visualize_adaptive_skeleton_dvae import AdaptiveSkeletonDVAEVisualizer
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def create_synthetic_skeleton_data():
    """
    创建合成的骨架数据用于演示
    模拟MMFI数据集的650点人体骨架结构
    """
    # 定义17个主要关节位置 (MMFI标准)
    skeleton_joints = np.array([
        [0.0, 0.0, 0.0],      # J0: 躯干中心
        [0.1, -0.3, 0.0],     # J1: 右侧腰部
        [0.1, -0.7, 0.0],     # J2: 右膝盖
        [0.1, -1.0, 0.0],     # J3: 右脚
        [-0.1, -0.3, 0.0],    # J4: 左侧腰部
        [-0.1, -0.7, 0.0],    # J5: 左膝盖
        [-0.1, -1.0, 0.0],    # J6: 左脚
        [0.0, 0.3, 0.0],      # J7: 肩膀中心
        [0.0, 0.5, 0.0],      # J8: 头颈部
        [0.0, 0.6, 0.0],      # J9: 头颈部
        [0.0, 0.8, 0.0],      # J10: 头顶
        [-0.05, 0.45, 0.0],   # J11: 肩膀
        [-0.2, 0.4, 0.0],     # J12: 左肩膀
        [-0.4, 0.2, 0.0],     # J13: 左手
        [0.05, 0.45, 0.0],    # J14: 颈部
        [0.2, 0.4, 0.0],      # J15: 右肩膀
        [0.4, 0.2, 0.0],      # J16: 右手
    ])
    
    # 定义连接关系
    connections = [
        (1, 2), (4, 5), (0, 1), (0, 4), (2, 3), (5, 6),  # 腿部
        (0, 7),  # 躯干到肩膀
        (10, 9), (9, 8), (8, 7),  # 头部链
        (8, 11), (11, 12), (12, 13),  # 左臂
        (8, 14), (14, 15), (15, 16),  # 右臂
    ]
    
    # 生成增强点云 (在每条连接上插值生成点)
    augmented_points = []
    points_per_connection = 37  # 平均每条连接37个点 (16*37 + 17 ≈ 650)
    
    # 先添加原始关节点
    augmented_points.extend(skeleton_joints)
    
    # 在每条连接上添加插值点
    for connection in connections:
        start_joint = skeleton_joints[connection[0]]
        end_joint = skeleton_joints[connection[1]]
        
        # 在连接线上生成插值点 (不包括端点)
        for i in range(1, points_per_connection + 1):
            t = i / (points_per_connection + 1)
            interpolated_point = start_joint + t * (end_joint - start_joint)
            # 添加小量噪声使其更真实
            noise = np.random.normal(0, 0.01, 3)
            augmented_points.append(interpolated_point + noise)
    
    # 转换为numpy数组并确保是650个点
    augmented_points = np.array(augmented_points)
    if len(augmented_points) > 650:
        augmented_points = augmented_points[:650]
    elif len(augmented_points) < 650:
        # 如果点数不够，随机复制一些点
        remaining = 650 - len(augmented_points)
        random_indices = np.random.choice(len(augmented_points), remaining)
        additional_points = augmented_points[random_indices]
        augmented_points = np.concatenate([augmented_points, additional_points])
    
    return augmented_points

def demo_basic_visualization():
    """基础可视化演示"""
    print("🎨 Creating synthetic skeleton data for demonstration...")
    
    # 创建合成骨架数据
    skeleton_data = create_synthetic_skeleton_data()
    print(f"📊 Generated skeleton with {len(skeleton_data)} points")
    
    # 创建简单的可视化器配置
    print("🔧 Setting up visualizer...")
    
    # 创建一个最小配置用于演示
    from types import SimpleNamespace
    config = SimpleNamespace()
    config.model = SimpleNamespace()
    config.model.NAME = 'AdaptiveSkeletonDVAE'
    config.model.latent_dim = 512
    config.model.num_tokens = 1024
    config.model.commitment_cost = 0.25
    config.model.loss_type = 'mse'
    
    # 临时创建一个配置文件
    import yaml
    import tempfile
    
    config_dict = {
        'model': {
            'NAME': 'AdaptiveSkeletonDVAE',
            'latent_dim': 512,
            'num_tokens': 1024,
            'commitment_cost': 0.25,
            'loss_type': 'mse'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        # 创建可视化器
        visualizer = AdaptiveSkeletonDVAEVisualizer(
            config_path=config_path,
            checkpoint_path=None,  # 不加载检查点
            device='cpu'  # 使用CPU避免CUDA问题
        )
        
        # 演示骨架增强策略可视化
        print("🦴 Analyzing skeleton augmentation strategy...")
        save_dir = './demo_visualizations'
        os.makedirs(save_dir, exist_ok=True)
        
        result = visualizer.visualize_skeleton_augmentation_strategy(
            skeleton_data, 
            save_path=os.path.join(save_dir, 'skeleton_augmentation_demo.png')
        )
        
        print(f"✅ Analysis complete!")
        print(f"   Results saved to: {save_dir}")
        print(f"   Augmentation ratio: {result['augmentation_ratio']:.1f}x")
        
        # 演示单样本重建可视化 (即使没有训练的模型)
        print("\n🔍 Demonstrating single sample visualization...")
        visualizer.visualize_single_sample(
            skeleton_data,
            save_path=os.path.join(save_dir, 'sample_reconstruction_demo.png')
        )
        
        print(f"✅ All demonstrations complete! Check {save_dir} for results.")
        
    finally:
        # 清理临时配置文件
        os.unlink(config_path)

def demo_interactive_skeleton():
    """交互式骨架演示"""
    print("🎪 Interactive skeleton demonstration...")
    
    skeleton_data = create_synthetic_skeleton_data()
    
    # 简单的3D可视化
    fig = plt.figure(figsize=(15, 5))
    
    # 显示原始17关节
    ax1 = fig.add_subplot(131, projection='3d')
    skeleton_joints = skeleton_data[:17]  # 假设前17个点是关节
    ax1.scatter(skeleton_joints[:, 0], skeleton_joints[:, 1], skeleton_joints[:, 2],
               c='red', s=100, alpha=0.8, label='17 Original Joints')
    ax1.set_title('Original 17 Joints')
    ax1.legend()
    
    # 显示所有650点
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(skeleton_data[:, 0], skeleton_data[:, 1], skeleton_data[:, 2],
               c='blue', s=10, alpha=0.6, label='650 Augmented Points')
    ax2.set_title('Augmented Point Cloud (650 points)')
    ax2.legend()
    
    # 叠加显示
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(skeleton_data[:, 0], skeleton_data[:, 1], skeleton_data[:, 2],
               c='lightblue', s=8, alpha=0.4, label='Augmented Points')
    ax3.scatter(skeleton_joints[:, 0], skeleton_joints[:, 1], skeleton_joints[:, 2],
               c='red', s=80, alpha=1.0, edgecolor='black', linewidth=1,
               label='Original Joints')
    ax3.set_title('Combined View')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("📊 Interactive visualization displayed!")

if __name__ == '__main__':
    print("🚀 Starting Skeleton Augmentation Visualization Demo")
    print("=" * 60)
    
    # 选择演示模式
    print("Choose demonstration mode:")
    print("1. Basic visualization (recommended)")
    print("2. Interactive skeleton view")
    print("3. Both")
    
    try:
        choice = input("Enter choice (1-3) [default: 1]: ").strip()
        if not choice:
            choice = '1'
        
        if choice in ['1', '3']:
            print("\n🎨 Running basic visualization demo...")
            demo_basic_visualization()
        
        if choice in ['2', '3']:
            print("\n🎪 Running interactive skeleton demo...")
            demo_interactive_skeleton()
        
        print("\n✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
