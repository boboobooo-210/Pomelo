"""
分析NTU骨架连接关系和数据增强策略
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# 关节名称映射
JOINT_NAMES = {
    0: "躯干下/骨盆",
    1: "躯干中",
    2: "颈部",
    3: "头顶",
    4: "左肩",
    5: "左肘",
    6: "左腕",
    7: "左手",
    8: "右肩",
    9: "右肘",
    10: "右腕",
    11: "右手",
    12: "左髋",
    13: "左膝",
    14: "左踝",
    15: "左脚",
    16: "右髋",
    17: "右膝",
    18: "右踝",
    19: "右脚",
    20: "上躯干/颈部",
    21: "左手指2",
    22: "左手指1",
    23: "右手指2",
    24: "右手指1",
}


def interpolate_skeleton_points(skeleton, connections, points_per_bone=10):
    """
    在骨架连接线上插值生成更多点
    
    Args:
        skeleton: (25, 3) 原始骨架关节点
        connections: 连接关系列表
        points_per_bone: 每根骨头上插值的点数
    
    Returns:
        augmented_skeleton: 增强后的骨架点云
    """
    augmented_points = []
    
    # 添加原始关节点
    for i, joint in enumerate(skeleton):
        augmented_points.append(joint)
    
    # 在每个连接上插值
    for start_idx, end_idx in connections:
        if start_idx < len(skeleton) and end_idx < len(skeleton):
            start_point = skeleton[start_idx]
            end_point = skeleton[end_idx]
            
            # 在连接线上等距插值
            for i in range(1, points_per_bone):
                t = i / points_per_bone
                interpolated_point = start_point + t * (end_point - start_point)
                augmented_points.append(interpolated_point)
    
    return np.array(augmented_points)


def analyze_augmentation_effect():
    """分析数据增强效果"""
    print("🔍 分析NTU骨架数据增强效果")
    print("=" * 50)
    
    # 模拟一个骨架数据
    np.random.seed(42)
    skeleton = np.random.randn(25, 3) * 0.5
    
    print(f"原始骨架点数: {len(skeleton)}")
    print(f"连接关系数: {len(NTU_CONNECTIONS)}")
    
    # 测试不同的插值密度
    for points_per_bone in [5, 10, 15, 20]:
        augmented = interpolate_skeleton_points(skeleton, NTU_CONNECTIONS, points_per_bone)
        print(f"每根骨头插值{points_per_bone}个点: {len(augmented)} 个总点")
    
    # 计算理论最大点数
    max_points = 25 + len(NTU_CONNECTIONS) * 19  # 原始点 + 每个连接最多19个插值点
    print(f"理论最大点数: {max_points}")
    
    return skeleton


def visualize_skeleton_augmentation(skeleton):
    """可视化骨架增强效果"""
    fig = plt.figure(figsize=(15, 5))
    
    # 原始骨架
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], c='red', s=50)
    
    # 绘制连接线
    for start_idx, end_idx in NTU_CONNECTIONS:
        start_point = skeleton[start_idx]
        end_point = skeleton[end_idx]
        ax1.plot([start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                [start_point[2], end_point[2]], 'b-', alpha=0.6)
    
    ax1.set_title('原始骨架 (25点)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 中等增强
    augmented_medium = interpolate_skeleton_points(skeleton, NTU_CONNECTIONS, 10)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(augmented_medium[:, 0], augmented_medium[:, 1], augmented_medium[:, 2], 
               c='blue', s=20, alpha=0.7)
    ax2.set_title(f'中等增强 ({len(augmented_medium)}点)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # 高密度增强
    augmented_high = interpolate_skeleton_points(skeleton, NTU_CONNECTIONS, 20)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(augmented_high[:, 0], augmented_high[:, 1], augmented_high[:, 2], 
               c='green', s=10, alpha=0.6)
    ax3.set_title(f'高密度增强 ({len(augmented_high)}点)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('../docs/ntu_skeleton_augmentation.png', dpi=150, bbox_inches='tight')
    print("📊 可视化结果保存到: ../docs/ntu_skeleton_augmentation.png")
    plt.close()


def compare_with_mars():
    """与MARS数据集对比"""
    print("\n📊 与MARS数据集对比:")
    print("=" * 30)
    
    print("MARS数据集:")
    print("  原始点数: 64")
    print("  增强后: ~650点")
    print("  增强倍数: ~10倍")
    
    print("\nNTU数据集:")
    print("  原始点数: 25")
    print("  连接数: 24")
    
    for points_per_bone in [10, 15, 20, 25]:
        total_points = 25 + 24 * (points_per_bone - 1)
        multiplier = total_points / 25
        print(f"  每骨{points_per_bone}点: {total_points}点 ({multiplier:.1f}倍)")
        
        if total_points >= 650:
            print(f"    ✅ 达到MARS水平")
        elif total_points >= 500:
            print(f"    🔶 接近MARS水平")
        else:
            print(f"    ❌ 低于MARS水平")


def analyze_dvae_requirements():
    """分析DVAE模型要求"""
    print("\n🤖 DVAE模型要求分析:")
    print("=" * 30)
    
    # 常见的DVAE配置
    common_configs = [
        {"points": 256, "groups": 16, "group_size": 16},
        {"points": 512, "groups": 32, "group_size": 16},
        {"points": 1024, "groups": 32, "group_size": 32},
        {"points": 2048, "groups": 64, "group_size": 32},
    ]
    
    print("常见DVAE配置:")
    for config in common_configs:
        print(f"  {config['points']}点: {config['groups']}组 × {config['group_size']}点/组")
    
    print("\nNTU增强后可选配置:")
    for points_per_bone in [15, 20, 25, 30]:
        total_points = 25 + 24 * (points_per_bone - 1)
        
        # 寻找合适的分组方案
        for group_size in [8, 16, 32]:
            if total_points % group_size == 0:
                num_groups = total_points // group_size
                print(f"  {total_points}点: {num_groups}组 × {group_size}点/组 ✅")
            elif total_points > group_size:
                num_groups = total_points // group_size
                remainder = total_points % group_size
                print(f"  {total_points}点: {num_groups}组 × {group_size}点/组 + {remainder}点 (需要padding)")


def main():
    """主分析函数"""
    print("🧬 NTU骨架数据增强策略分析")
    print("=" * 60)
    
    # 分析增强效果
    skeleton = analyze_augmentation_effect()
    
    # 可视化
    visualize_skeleton_augmentation(skeleton)
    
    # 与MARS对比
    compare_with_mars()
    
    # DVAE要求分析
    analyze_dvae_requirements()
    
    print(f"\n🎯 推荐配置:")
    print(f"  方案1: 每骨20点 → 601点 → 19组×32点/组 (padding到608)")
    print(f"  方案2: 每骨25点 → 601点 → 38组×16点/组 (padding到608)")
    print(f"  方案3: 每骨27点 → 649点 → 21组×32点/组 (padding到672)")


if __name__ == '__main__':
    main()
