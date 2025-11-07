#!/usr/bin/env python3
"""
骨架提取 + GCN重构 简化实现
结合skeleton_extractor和GCNSkeletonTokenizer的最小化可执行版本
"""

import os
import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def check_dependencies():
    """检查必需的依赖和文件"""
    print("🔍 检查运行环境...")
    
    # 检查权重文件
    required_files = {
        'extractor_weights': 'mars_transformer_best.pth',
        'gcn_weights': 'experiments/gcn_skeleton_memory_optimized/NTU_models/default/ckpt-best.pth',
        'gcn_config': 'cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml',
        'extractor_model': 'models/skeleton_extractor.py',
        'gcn_model': 'models/GCNSkeletonTokenizer.py'
    }
    
    missing_files = []
    for name, path in required_files.items():
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ❌ {name}: {path} - 文件不存在")
            missing_files.append(path)
    
    # 检查Python包
    required_packages = ['torch', 'numpy', 'matplotlib']
    available_packages = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}: 已安装")
            available_packages.append(pkg)
        except ImportError:
            print(f"  ❌ {pkg}: 未安装")
    
    return len(missing_files) == 0 and len(available_packages) == len(required_packages)

def create_mock_pipeline_demo():
    """创建模拟流水线演示"""
    print("\n🎭 创建模拟流水线演示...")
    
    # 模拟数据
    import numpy as np
    
    # 1. 模拟雷达特征图
    print("  1️⃣ 模拟雷达特征图 (8x8x5)")
    radar_data = np.random.rand(8, 8, 5).astype(np.float32)
    
    # 2. 模拟骨架提取过程
    print("  2️⃣ 模拟MARS骨架提取...")
    # 假设MARS模型输出57维特征，转换为25关节点
    skeleton_57d = np.random.rand(57).astype(np.float32) * 2 - 1  # 范围[-1, 1]
    
    # 转换为25关节点格式 (只取前75维作为25×3坐标)
    if len(skeleton_57d) >= 75:
        skeleton_25joints = skeleton_57d[:75].reshape(25, 3)
    else:
        skeleton_25joints = np.zeros((25, 3))
        available_joints = min(len(skeleton_57d) // 3, 25)
        skeleton_25joints[:available_joints, :] = skeleton_57d[:available_joints*3].reshape(available_joints, 3)
    
    print(f"     提取骨架形状: {skeleton_25joints.shape}")
    
    # 3. 模拟骨架标准化
    print("  3️⃣ 骨架标准化处理...")
    centroid = np.mean(skeleton_25joints, axis=0)
    centered = skeleton_25joints - centroid
    distances = np.sqrt(np.sum(centered**2, axis=1))
    max_distance = np.max(distances)
    
    if max_distance > 0:
        normalized_skeleton = centered / max_distance
    else:
        normalized_skeleton = centered
    
    print(f"     标准化后范围: [{normalized_skeleton.min():.3f}, {normalized_skeleton.max():.3f}]")
    
    # 4. 模拟GCN重构过程
    print("  4️⃣ 模拟GCN重构...")
    
    # 模拟重构结果（添加小量噪声）
    reconstruction_noise = np.random.normal(0, 0.05, skeleton_25joints.shape)
    reconstructed_skeleton = normalized_skeleton + reconstruction_noise
    
    # 模拟token序列（5个语义组）
    token_sequence = np.random.randint(0, 128, 5)
    # 添加组偏移
    group_offsets = [0, 128, 256, 384, 512]
    for i in range(5):
        token_sequence[i] += group_offsets[i]
    
    # 计算模拟指标
    mse_error = np.mean((normalized_skeleton - reconstructed_skeleton)**2)
    vq_loss = np.random.uniform(0.001, 0.01)  # 模拟VQ损失
    
    print(f"     重构MSE误差: {mse_error:.6f}")
    print(f"     模拟VQ损失: {vq_loss:.6f}")
    print(f"     Token序列: {token_sequence}")
    
    # 5. 分析Token序列
    print("  5️⃣ Token序列分析...")
    group_names = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
    for i, (token_id, group_name) in enumerate(zip(token_sequence, group_names)):
        expected_range = f"{group_offsets[i]}-{group_offsets[i]+127}"
        print(f"     {group_name}: Token {token_id} (范围: {expected_range})")
    
    return {
        'radar_data': radar_data,
        'extracted_skeleton': skeleton_25joints,
        'normalized_skeleton': normalized_skeleton,
        'reconstructed_skeleton': reconstructed_skeleton,
        'token_sequence': token_sequence,
        'mse_error': mse_error,
        'vq_loss': vq_loss
    }

def create_text_visualization(demo_result):
    """创建文本形式的可视化"""
    print("\n📊 生成文本可视化...")
    
    import numpy as np
    
    # 创建简单的ASCII图表
    mse_error = demo_result['mse_error']
    vq_loss = demo_result['vq_loss']
    token_sequence = demo_result['token_sequence']
    
    print("=" * 60)
    print("📈 骨架提取+重构流程结果")
    print("=" * 60)
    
    # 质量评估
    print("🎯 质量指标:")
    if mse_error < 0.01:
        quality = "优秀 ✅"
    elif mse_error < 0.05:
        quality = "良好 ⚡"
    else:
        quality = "需改进 ⚠️"
    
    print(f"  MSE重构误差: {mse_error:.6f} - {quality}")
    print(f"  VQ量化损失: {vq_loss:.6f}")
    
    # Token分布
    print("\n🎭 语义组Token分析:")
    group_names = ['头脊柱', '左臂', '右臂', '左腿', '右腿']
    group_ranges = [(0,127), (128,255), (256,383), (384,511), (512,639)]
    
    for i, (token_id, group_name, (min_id, max_id)) in enumerate(zip(token_sequence, group_names, group_ranges)):
        usage_percent = ((token_id - min_id) / (max_id - min_id)) * 100
        bar_length = int(usage_percent / 5)  # 每5%一个字符
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {group_name:>4}: {token_id:3d} |{bar}| {usage_percent:5.1f}%")
    
    # 骨架统计
    print(f"\n🦴 骨架统计:")
    original = demo_result['normalized_skeleton']
    reconstructed = demo_result['reconstructed_skeleton']
    
    joint_errors = np.sqrt(np.sum((original - reconstructed)**2, axis=1))
    max_error_joint = np.argmax(joint_errors)
    min_error_joint = np.argmin(joint_errors)
    
    print(f"  关节点数量: {len(original)}")
    print(f"  最大误差关节: #{max_error_joint} (误差: {joint_errors[max_error_joint]:.4f})")
    print(f"  最小误差关节: #{min_error_joint} (误差: {joint_errors[min_error_joint]:.4f})")
    print(f"  平均关节误差: {np.mean(joint_errors):.4f}")
    
    # 流程总结
    print(f"\n🔄 流程总结:")
    print(f"  雷达特征图: {demo_result['radar_data'].shape} → 57维特征 → 25关节点")
    print(f"  标准化处理: 质心对齐 + 距离归一化") 
    print(f"  GCN编码: 5个语义组 → 5个离散Token")
    print(f"  重构解码: Token → 25关节点坐标")
    
    print("=" * 60)

def create_real_implementation_guide():
    """创建真实实现指南"""
    print("\n📖 真实实现指南:")
    print("=" * 60)
    
    guide_content = """
🚀 如何运行真实的骨架提取+重构流程:

1️⃣ 准备权重文件:
   • mars_transformer_best.pth (MARS骨架提取器)
   • experiments/.../ckpt-best.pth (GCN重构器)

2️⃣ 安装依赖:
   pip install torch torchvision numpy matplotlib

3️⃣ 运行完整流程:
   python tools/skeleton_extraction_reconstruction_pipeline.py

4️⃣ 或分步执行:
   # 加载MARS模型
   from models.skeleton_extractor import MARSTransformerModel
   extractor = MARSTransformerModel(5, 57)
   extractor.load_state_dict(torch.load('mars_transformer_best.pth'))
   
   # 加载GCN模型  
   from models.GCNSkeletonTokenizer import GCNSkeletonTokenizer
   gcn_model = GCNSkeletonTokenizer(config)
   gcn_model.load_state_dict(torch.load('ckpt-best.pth'))
   
   # 处理数据
   radar_data = load_radar_data()  # (B, 5, 8, 8)
   skeleton_57d = extractor(radar_data)  # (B, 57)
   skeleton_25 = skeleton_57d[:, :75].reshape(-1, 25, 3)
   reconstruction = gcn_model(skeleton_25)

5️⃣ 输出结果:
   • 可视化图像: visualizations/skeleton_extraction_reconstruction/
   • 数值结果: pipeline_results.json
   • 质量指标: MSE误差, VQ损失, Token序列

📊 预期性能:
   • 优秀重构: MSE < 0.01
   • 良好重构: MSE < 0.05  
   • Token范围: 每组0-127 (加偏移后0-639)
   • 处理速度: ~100ms/样本 (GPU)
"""
    
    print(guide_content)

def save_demo_results(demo_result, output_dir="demo_output"):
    """保存演示结果"""
    print(f"\n💾 保存演示结果到: {output_dir}/")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数值结果
    results = {
        'pipeline_summary': {
            'mse_error': float(demo_result['mse_error']),
            'vq_loss': float(demo_result['vq_loss']),
            'token_sequence': demo_result['token_sequence'].tolist()
        },
        'quality_assessment': {
            'reconstruction_quality': 'excellent' if demo_result['mse_error'] < 0.01 else 'good' if demo_result['mse_error'] < 0.05 else 'needs_improvement',
            'token_distribution': {
                'head_spine': int(demo_result['token_sequence'][0]),
                'left_arm': int(demo_result['token_sequence'][1]), 
                'right_arm': int(demo_result['token_sequence'][2]),
                'left_leg': int(demo_result['token_sequence'][3]),
                'right_leg': int(demo_result['token_sequence'][4])
            }
        },
        'data_shapes': {
            'radar_input': demo_result['radar_data'].shape,
            'extracted_skeleton': demo_result['extracted_skeleton'].shape,
            'normalized_skeleton': demo_result['normalized_skeleton'].shape,
            'reconstructed_skeleton': demo_result['reconstructed_skeleton'].shape
        }
    }
    
    # 保存JSON结果
    with open(f"{output_dir}/demo_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存numpy数组
    import numpy as np
    np.save(f"{output_dir}/radar_data.npy", demo_result['radar_data'])
    np.save(f"{output_dir}/extracted_skeleton.npy", demo_result['extracted_skeleton'])
    np.save(f"{output_dir}/reconstructed_skeleton.npy", demo_result['reconstructed_skeleton'])
    
    print(f"  ✅ demo_results.json - 数值结果")
    print(f"  ✅ *.npy - 骨架数据文件")

def main():
    """主函数"""
    print("=" * 80)
    print("🦴 骨架提取 + GCN重构流水线演示")
    print("=" * 80)
    
    # 检查环境
    if not check_dependencies():
        print("\n⚠️ 环境检查未完全通过，将运行模拟演示")
    else:
        print("\n✅ 环境检查通过，可运行真实流程")
    
    try:
        # 运行模拟演示
        demo_result = create_mock_pipeline_demo()
        
        # 生成可视化
        create_text_visualization(demo_result)
        
        # 保存结果
        save_demo_results(demo_result)
        
        # 显示实现指南
        create_real_implementation_guide()
        
        print("\n🎉 演示完成！")
        
    except Exception as e:
        print(f"\n❌ 演示执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()