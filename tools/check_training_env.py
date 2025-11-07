#!/usr/bin/env python3
"""
GCNSkeletonTokenizer 训练环境检查脚本
检查训练所需的所有文件和配置是否正确
"""

import os
import sys
import importlib.util
from pathlib import Path

def check_file_exists(filepath, description=""):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✅ {filepath} {description}")
        return True
    else:
        print(f"❌ {filepath} {description} - 文件不存在")
        return False

def check_import(module_name, description=""):
    """检查模块是否可以导入"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} {description}")
        return True
    except Exception as e:
        print(f"❌ {module_name} {description} - 导入失败: {e}")
        return False

def check_training_environment():
    """检查训练环境"""
    print("=" * 60)
    print("GCNSkeletonTokenizer 训练环境检查")
    print("=" * 60)
    
    success_count = 0
    total_checks = 0
    
    # 1. 检查核心训练文件
    print("\n📋 1. 核心训练文件检查:")
    core_files = [
        ("main.py", "主训练脚本"),
        ("cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml", "模型配置文件"),
        ("cfgs/dataset_configs/NTU_skeleton_raw.yaml", "数据集配置文件"),
        ("models/GCNSkeletonTokenizer.py", "GCN骨架Tokenizer模型"),
    ]
    
    for filepath, desc in core_files:
        total_checks += 1
        if check_file_exists(filepath, desc):
            success_count += 1
    
    # 2. 检查训练工具文件
    print("\n🔧 2. 训练工具文件检查:")
    tool_files = [
        ("tools/__init__.py", "工具包初始化"),
        ("tools/runner.py", "训练循环逻辑"),
        ("tools/builder.py", "模型和数据集构建"),
    ]
    
    for filepath, desc in tool_files:
        total_checks += 1
        if check_file_exists(filepath, desc):
            success_count += 1
    
    # 3. 检查数据集文件
    print("\n📊 3. 数据集文件检查:")
    dataset_files = [
        ("datasets/__init__.py", "数据集包初始化"),
        ("datasets/build.py", "数据集构建函数"),
        ("datasets/NTUDataset.py", "NTU数据集加载器"),
        ("datasets/NTUSkeletonRawDataset.py", "原始骨架数据加载器"),
        ("datasets/data_transforms.py", "数据变换"),
        ("datasets/io.py", "数据IO工具"),
    ]
    
    for filepath, desc in dataset_files:
        total_checks += 1
        if check_file_exists(filepath, desc):
            success_count += 1
    
    # 4. 检查工具库文件
    print("\n🛠️ 4. 工具库文件检查:")
    util_files = [
        ("utils/config.py", "配置文件解析"),
        ("utils/parser.py", "命令行参数解析"),
        ("utils/logger.py", "日志工具"),
        ("utils/misc.py", "杂项工具"),
        ("utils/dist_utils.py", "分布式训练工具"),
        ("utils/AverageMeter.py", "指标计算"),
        ("utils/metrics.py", "评估指标"),
    ]
    
    for filepath, desc in util_files:
        total_checks += 1
        if check_file_exists(filepath, desc):
            success_count += 1
    
    # 5. 检查模型文件
    print("\n🧠 5. 模型相关文件检查:")
    model_files = [
        ("models/__init__.py", "模型包初始化"),
        ("models/build.py", "模型构建工具"),
        ("models/Tokenizer.py", "基础Tokenizer"),
        ("models/dvae.py", "DVAE模型"),
    ]
    
    for filepath, desc in model_files:
        total_checks += 1
        if check_file_exists(filepath, desc):
            success_count += 1
    
    # 6. 检查Python模块导入
    print("\n🐍 6. Python模块导入检查:")
    import_tests = [
        ("tools", "训练工具包"),
        ("utils.config", "配置解析"),
        ("utils.parser", "参数解析"),
        ("datasets", "数据集包"),
        ("models", "模型包"),
    ]
    
    for module, desc in import_tests:
        total_checks += 1
        if check_import(module, desc):
            success_count += 1
    
    # 7. 检查关键配置
    print("\n⚙️ 7. 配置文件内容检查:")
    try:
        # 检查模型配置
        import yaml
        with open("cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml", 'r') as f:
            model_config = yaml.safe_load(f)
        
        if model_config.get('model', {}).get('NAME') == 'GCNSkeletonTokenizer':
            print("✅ 模型配置正确: GCNSkeletonTokenizer")
            success_count += 1
        else:
            print("❌ 模型配置错误: 未找到GCNSkeletonTokenizer")
        total_checks += 1
        
        # 检查数据集配置
        with open("cfgs/dataset_configs/NTU_skeleton_raw.yaml", 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        if dataset_config.get('NAME') == 'NTU_Skeleton_Raw':
            print("✅ 数据集配置正确: NTU_Skeleton_Raw")
            success_count += 1
        else:
            print("❌ 数据集配置错误: 未找到NTU_Skeleton_Raw")
        total_checks += 1
        
    except Exception as e:
        print(f"❌ 配置文件检查失败: {e}")
        total_checks += 2
    
    # 8. 检查数据路径
    print("\n📁 8. 数据路径检查:")
    data_path = "../HumanPoint-BERT/data/NTU-RGB+D"
    total_checks += 1
    if os.path.exists(data_path):
        print(f"✅ 数据路径存在: {data_path}")
        success_count += 1
        
        # 检查数据文件
        skeleton_files = [f for f in os.listdir(data_path) if f.endswith('.skeleton') or 'skeleton' in f]
        if skeleton_files:
            print(f"✅ 找到 {len(skeleton_files)} 个骨架数据文件")
        else:
            print("⚠️ 未找到.skeleton文件，可能需要解压数据")
    else:
        print(f"❌ 数据路径不存在: {data_path}")
        print("   请确保NTU RGB+D数据集已下载并放置在正确位置")
    
    # 9. 生成训练命令
    print("\n🚀 9. 推荐训练命令:")
    print("基本训练命令:")
    print("  python main.py --config cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml")
    print("\n带GPU指定:")
    print("  python main.py --config cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml --gpu 0")
    print("\n测试模式:")
    print("  python main.py --config cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml --test --ckpts path/to/checkpoint.pth")
    
    # 10. 总结
    print("\n" + "=" * 60)
    print("检查总结:")
    print(f"总检查项: {total_checks}")
    print(f"通过项: {success_count}")
    print(f"失败项: {total_checks - success_count}")
    print(f"通过率: {success_count/total_checks*100:.1f}%")
    
    if success_count == total_checks:
        print("\n🎉 所有检查通过！可以开始训练了！")
        return True
    elif success_count >= total_checks * 0.8:
        print("\n⚠️ 大部分检查通过，可以尝试训练，但可能遇到问题")
        return False
    else:
        print("\n❌ 多个关键文件缺失，请先解决这些问题")
        return False

def show_training_tips():
    """显示训练技巧"""
    print("\n" + "=" * 60)
    print("💡 训练技巧提示:")
    print("=" * 60)
    
    tips = [
        "1. 首次训练建议使用小批次大小(bs=4)避免显存不足",
        "2. 使用TensorBoard监控训练过程: tensorboard --logdir experiments/",
        "3. 训练日志保存在 experiments/gcn_skeleton_memory_optimized/logs/",
        "4. 模型检查点保存在 experiments/gcn_skeleton_memory_optimized/checkpoints/",
        "5. 如遇到数据加载慢，可设置 num_workers=0 进行调试",
        "6. VQ损失和重建损失的权重可在配置文件中调整",
        "7. 支持从检查点继续训练: --resume --ckpts path/to/checkpoint.pth",
    ]
    
    for tip in tips:
        print(f"  {tip}")
    
    print("\n📚 更多帮助文档:")
    print("  - docs/GCNSkeletonTokenizer_Training_Guide.md")
    print("  - docs/GCNSkeletonTokenizer_Usage_Guide.md")
    print("  - docs/GCNSkeletonTokenizer_Config_Examples.md")

def main():
    """主函数"""
    # 切换到项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"当前工作目录: {os.getcwd()}")
    
    # 执行环境检查
    success = check_training_environment()
    
    # 显示训练技巧
    show_training_tips()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)