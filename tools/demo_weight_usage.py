#!/usr/bin/env python3
"""
GCN可视化器权重使用流程演示脚本
展示权重从加载到推理的完整过程
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def demonstrate_weight_loading_process():
    """演示权重加载过程"""
    print("=" * 80)
    print("🔄 GCN骨架可视化器权重使用流程演示")
    print("=" * 80)
    
    # 第1步：检查文件路径
    print("\n📁 第1步：检查训练权重和配置文件")
    model_path = "experiments/gcn_skeleton_memory_optimized/checkpoints/ckpt-best.pth"
    config_path = "cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml"
    
    print(f"  权重文件: {model_path}")
    print(f"  配置文件: {config_path}")
    
    if os.path.exists(model_path):
        print("  ✅ 权重文件存在")
    else:
        print("  ❌ 权重文件不存在")
        
    if os.path.exists(config_path):
        print("  ✅ 配置文件存在")
    else:
        print("  ❌ 配置文件不存在")
    
    # 第2步：模拟权重加载
    print("\n🏗️ 第2步：模拟权重加载过程")
    print("  2.1 加载配置文件...")
    try:
        from utils.config import cfg_from_yaml_file
        config = cfg_from_yaml_file(config_path)
        print("      ✅ 配置加载成功")
        print(f"      模型类型: {config.model.get('NAME', 'Unknown')}")
        print(f"      关节数量: {config.model.get('num_joints', 'Unknown')}")
        print(f"      Token维度: {config.model.get('token_dim', 'Unknown')}")
    except Exception as e:
        print(f"      ❌ 配置加载失败: {e}")
        return
    
    print("  2.2 创建模型架构...")
    try:
        from models.GCNSkeletonTokenizer import GCNSkeletonTokenizer
        model = GCNSkeletonTokenizer(config.model)
        print("      ✅ 模型架构创建成功")
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"      总参数量: {total_params:,}")
        print(f"      可训练参数: {trainable_params:,}")
    except Exception as e:
        print(f"      ❌ 模型创建失败: {e}")
        return
    
    print("  2.3 加载训练权重...")
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print("      ✅ 检查点文件加载成功")
            
            # 分析检查点内容
            print(f"      检查点键: {list(checkpoint.keys())}")
            
            if 'base_model' in checkpoint:
                state_dict = checkpoint['base_model']
                print("      使用 'base_model' 权重")
            else:
                state_dict = checkpoint
                print("      使用根级权重")
            
            # 处理分布式训练权重
            new_state_dict = {}
            module_count = 0
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                    module_count += 1
                else:
                    new_state_dict[k] = v
            
            if module_count > 0:
                print(f"      移除了 {module_count} 个 'module.' 前缀")
            
            # 加载权重到模型
            model.load_state_dict(new_state_dict)
            model.eval()
            print("      ✅ 权重加载到模型成功")
            
            # 分析权重结构
            print("      权重结构分析:")
            weight_groups = {}
            for name, param in model.named_parameters():
                group = name.split('.')[0]
                if group not in weight_groups:
                    weight_groups[group] = []
                weight_groups[group].append((name, param.shape))
            
            for group, params in weight_groups.items():
                group_params = sum(p.numel() for _, p in params)
                print(f"        {group}: {len(params)} 层, {group_params:,} 参数")
                
        except Exception as e:
            print(f"      ❌ 权重加载失败: {e}")
            return
    else:
        print("      ❌ 权重文件不存在，跳过加载")
        return

def demonstrate_inference_process():
    """演示推理过程"""
    print("\n🎯 第3步：模拟推理过程")
    
    # 创建模拟骨架数据
    print("  3.1 创建模拟骨架数据...")
    # 模拟NTU RGB+D 25关节点数据
    skeleton = np.random.randn(25, 3).astype(np.float32) * 0.5
    print(f"      输入骨架形状: {skeleton.shape}")
    print(f"      骨架数据范围: [{skeleton.min():.3f}, {skeleton.max():.3f}]")
    
    print("  3.2 数据预处理...")
    # 标准化处理（与训练时一致）
    centroid = np.mean(skeleton, axis=0)
    centered = skeleton - centroid
    distances = np.sqrt(np.sum(centered**2, axis=1))
    max_distance = np.max(distances)
    
    if max_distance > 0:
        normalized = centered / max_distance
    else:
        normalized = centered
    
    print(f"      标准化后形状: {normalized.shape}")
    print(f"      标准化后范围: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    print("  3.3 张量转换...")
    skeleton_tensor = torch.from_numpy(normalized).unsqueeze(0)  # 添加batch维度
    print(f"      张量形状: {skeleton_tensor.shape}")
    print(f"      张量数据类型: {skeleton_tensor.dtype}")
    
    print("  3.4 模拟前向传播...")
    print("      ⚡ 输入嵌入层 (3D坐标 → 64维特征)")
    print("      🧠 语义分组处理:")
    print("         - head_spine: 5个关节 → 64维特征")
    print("         - left_arm: 6个关节 → 64维特征")  
    print("         - right_arm: 6个关节 → 64维特征")
    print("         - left_leg: 4个关节 → 64维特征")
    print("         - right_leg: 4个关节 → 64维特征")
    print("      🎭 向量量化码本:")
    print("         - 每个语义组128个码字")
    print("         - 最近邻匹配 → token ID")
    print("      🔄 特征融合 → 全局特征")
    print("      🏗️ 重建网络 → 25×3 骨架坐标")

def demonstrate_visualization_output():
    """演示可视化输出"""
    print("\n🎨 第4步：可视化输出过程")
    
    print("  4.1 重建质量评估...")
    # 模拟重建误差计算
    original = np.random.randn(25, 3) * 0.5
    reconstructed = original + np.random.randn(25, 3) * 0.1  # 添加少量噪声模拟重建
    
    mse_error = np.mean((original - reconstructed) ** 2)
    print(f"      MSE重建误差: {mse_error:.6f}")
    
    if mse_error < 0.01:
        print("      ✅ 重建质量: 优秀")
    elif mse_error < 0.05:
        print("      ⚠️ 重建质量: 良好")
    else:
        print("      ❌ 重建质量: 需要改进")
    
    print("  4.2 3D可视化生成...")
    print("      📊 创建matplotlib 3D图形")
    print("      🦴 绘制骨架连接关系 (25个关节点)")
    print("      🎨 颜色编码: 蓝色(原始) vs 红色(重建)")
    print("      📐 视角调整: elev=15°, azim=45°")
    
    print("  4.3 输出文件保存...")
    print("      📁 保存路径: visualizations/0_gcn/results_*/")
    print("      🖼️ 文件格式: PNG, DPI=300")
    print("      📝 文件名: gcn_reconstruction_sample_{i}_{name}.png")

def analyze_weight_importance():
    """分析权重重要性"""
    print("\n🔍 第5步：权重重要性分析")
    
    print("  5.1 关键权重组件:")
    weight_components = {
        "input_embedding": "3D坐标到64维特征的线性变换",
        "st_gcn_layers": "时空图卷积核心权重",
        "group_processors": "语义分组的独立处理器",
        "semantic_codebooks": "向量量化的可学习码本",
        "global_fusion": "多组特征融合权重",
        "reconstruction_head": "特征到骨架坐标的重建权重"
    }
    
    for component, description in weight_components.items():
        print(f"      🔧 {component}: {description}")
    
    print("  5.2 权重训练过程:")
    print("      📈 重建损失: MSE(原始, 重建)")
    print("      🎯 VQ损失: Commitment loss + 码本更新")
    print("      ⚖️ 损失平衡: reconstruction_weight + kld_weight * vq_loss")
    print("      🎲 优化器: AdamW, lr=0.001, weight_decay=0.0001")
    
    print("  5.3 权重质量指标:")
    print("      ✅ 收敛性: 损失曲线平滑下降")
    print("      🎯 重建精度: MSE < 0.01 (优秀)")
    print("      🔄 Token使用: 各组码本均匀使用")
    print("      🏃 泛化能力: 测试集性能接近训练集")

def main():
    """主函数"""
    print("🚀 GCN骨架可视化器权重使用完整流程演示")
    
    # 演示各个步骤
    demonstrate_weight_loading_process()
    demonstrate_inference_process()  
    demonstrate_visualization_output()
    analyze_weight_importance()
    
    print("\n" + "=" * 80)
    print("📋 总结：权重使用流程")
    print("=" * 80)
    
    summary_steps = [
        "1️⃣ 加载训练配置 → 创建模型架构",
        "2️⃣ 加载检查点文件 → 提取训练权重", 
        "3️⃣ 处理权重格式 → 加载到模型",
        "4️⃣ 设置评估模式 → 禁用梯度计算",
        "5️⃣ 输入数据预处理 → 标准化和张量转换",
        "6️⃣ 前向传播推理 → 利用训练权重重建",
        "7️⃣ 后处理输出 → 坐标转换和误差计算",
        "8️⃣ 3D可视化生成 → 对比原始和重建骨架"
    ]
    
    for step in summary_steps:
        print(f"  {step}")
    
    print("\n💡 关键要点:")
    print("  • 权重文件包含所有训练好的参数（GCN层、码本、重建网络等）")
    print("  • 推理过程必须与训练时保持数据预处理的一致性")
    print("  • MSE误差直接反映训练权重的重建质量")
    print("  • 可视化结果是评估模型训练效果的重要工具")
    
    print("\n🎯 使用建议:")
    print("  • 确保使用最佳检查点 (ckpt-best.pth) 进行可视化")
    print("  • 仅在测试集上评估，避免过拟合评估")
    print("  • 关注MSE数值和视觉效果的一致性") 
    print("  • 对比不同训练阶段的权重效果")

if __name__ == "__main__":
    main()