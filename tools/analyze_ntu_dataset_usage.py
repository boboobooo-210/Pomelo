"""
分析NTU数据集在训练和测试中的实际使用情况
统计文件数量、数据分布、训练/验证/测试集划分等
"""

import os
import sys
import numpy as np
import re
from collections import defaultdict

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.NTUDatasetAugmented import NTUAugmented


def analyze_ntu_data_directory():
    """分析NTU数据目录的文件情况"""
    print("📁 分析NTU数据目录")
    print("=" * 50)
    
    data_path = '../data/NTU-RGB+D'
    
    if not os.path.exists(data_path):
        print(f"❌ 数据目录不存在: {data_path}")
        return None
    
    # 统计.skeleton文件
    skeleton_files = []
    for file in os.listdir(data_path):
        if file.endswith('.skeleton'):
            skeleton_files.append(file)
    
    print(f"📊 总.skeleton文件数: {len(skeleton_files)}")
    
    # 分析文件名模式
    action_counts = defaultdict(int)
    subject_counts = defaultdict(int)
    camera_counts = defaultdict(int)
    setup_counts = defaultdict(int)
    
    for file in skeleton_files:
        # NTU文件名格式: SsssCcccPpppRrrrAaaa.skeleton
        # S: setup, C: camera, P: performer, R: replication, A: action
        match = re.match(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})\.skeleton', file)
        if match:
            setup, camera, performer, replication, action = match.groups()
            action_counts[int(action)] += 1
            subject_counts[int(performer)] += 1
            camera_counts[int(camera)] += 1
            setup_counts[int(setup)] += 1
    
    print(f"📈 动作类别数: {len(action_counts)} (范围: {min(action_counts.keys())}-{max(action_counts.keys())})")
    print(f"📈 受试者数: {len(subject_counts)} (范围: {min(subject_counts.keys())}-{max(subject_counts.keys())})")
    print(f"📈 摄像头数: {len(camera_counts)} (范围: {min(camera_counts.keys())}-{max(camera_counts.keys())})")
    print(f"📈 设置数: {len(setup_counts)} (范围: {min(setup_counts.keys())}-{max(setup_counts.keys())})")
    
    return {
        'total_files': len(skeleton_files),
        'action_counts': dict(action_counts),
        'subject_counts': dict(subject_counts),
        'camera_counts': dict(camera_counts),
        'setup_counts': dict(setup_counts),
        'skeleton_files': skeleton_files
    }


def analyze_dataset_splits():
    """分析数据集的训练/验证/测试划分"""
    print(f"\n📊 分析数据集划分")
    print("=" * 50)
    
    splits = ['train', 'val', 'test']
    split_info = {}
    
    for split in splits:
        try:
            # 创建配置
            class Config:
                def __init__(self, subset):
                    self.DATA_PATH = '../data/NTU-RGB+D'
                    self.subset = subset
                    self.N_POINTS = 25
                    self.npoints = 720
                    self.density_uniform = True
                    self.min_points_per_bone = 3
                    self.action_filter = 'dvae'
                    self.augment = False
                    self.whole = False
                
                def get(self, key, default=None):
                    return getattr(self, key, default)
            
            config = Config(split)
            dataset = NTUAugmented(config)
            
            split_info[split] = {
                'size': len(dataset),
                'config': config
            }
            
            print(f"✅ {split.upper()}集: {len(dataset)} 个样本")
            
        except Exception as e:
            print(f"❌ {split.upper()}集加载失败: {e}")
            split_info[split] = {'size': 0, 'error': str(e)}
    
    return split_info


def analyze_action_filter():
    """分析动作过滤的效果"""
    print(f"\n🎯 分析动作过滤效果")
    print("=" * 50)
    
    # 分析不同action_filter的效果
    filters = ['dvae', 'daily', 'rehab', 'all']
    filter_results = {}
    
    for filter_type in filters:
        try:
            class Config:
                def __init__(self, action_filter):
                    self.DATA_PATH = '../data/NTU-RGB+D'
                    self.subset = 'train'
                    self.N_POINTS = 25
                    self.npoints = 720
                    self.density_uniform = True
                    self.min_points_per_bone = 3
                    self.action_filter = action_filter
                    self.augment = False
                    self.whole = False
                
                def get(self, key, default=None):
                    return getattr(self, key, default)
            
            config = Config(filter_type)
            dataset = NTUAugmented(config)
            
            filter_results[filter_type] = len(dataset)
            print(f"📊 {filter_type.upper()}过滤: {len(dataset)} 个样本")
            
        except Exception as e:
            print(f"❌ {filter_type.upper()}过滤失败: {e}")
            filter_results[filter_type] = 0
    
    return filter_results


def analyze_training_logs():
    """分析训练日志中的数据使用情况"""
    print(f"\n📋 分析训练日志")
    print("=" * 50)
    
    log_dir = '../experiments/skeleton_dvae_pretrain/NTU_models/ntu_skeleton_tokenizer_720pts'
    
    if not os.path.exists(log_dir):
        print(f"❌ 日志目录不存在: {log_dir}")
        return None
    
    # 查找最新的日志文件
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    if not log_files:
        print(f"❌ 未找到日志文件")
        return None
    
    latest_log = sorted(log_files)[-1]
    log_path = os.path.join(log_dir, latest_log)
    
    print(f"📄 分析日志文件: {latest_log}")
    
    training_info = {
        'train_batches': 0,
        'val_samples': 0,
        'epochs_completed': 0
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 查找训练批次信息
                if '[Batch' in line and '/31]' in line:
                    training_info['train_batches'] = 31  # 每个epoch有31个batch
                
                # 查找验证样本信息
                if 'human_skeleton' in line and 'Sample' in line:
                    match = re.search(r'human_skeleton\s+(\d+)', line)
                    if match:
                        training_info['val_samples'] = int(match.group(1))
                
                # 查找完成的epoch数
                if '[Training] EPOCH:' in line:
                    match = re.search(r'EPOCH:\s+(\d+)', line)
                    if match:
                        training_info['epochs_completed'] = max(training_info['epochs_completed'], int(match.group(1)) + 1)
    
    except Exception as e:
        print(f"❌ 读取日志失败: {e}")
        return None
    
    print(f"📊 训练信息:")
    print(f"  每个epoch的batch数: {training_info['train_batches']}")
    print(f"  验证集样本数: {training_info['val_samples']}")
    print(f"  已完成epoch数: {training_info['epochs_completed']}")
    
    # 计算训练集大小
    if training_info['train_batches'] > 0:
        batch_size = 32  # 从配置文件得知
        train_samples = training_info['train_batches'] * batch_size
        print(f"  训练集样本数: ~{train_samples} (31 batches × 32 batch_size)")
    
    return training_info


def calculate_data_usage_summary():
    """计算数据使用总结"""
    print(f"\n📊 数据使用总结")
    print("=" * 50)
    
    # 分析目录信息
    dir_info = analyze_ntu_data_directory()
    
    # 分析数据集划分
    split_info = analyze_dataset_splits()
    
    # 分析动作过滤
    filter_info = analyze_action_filter()
    
    # 分析训练日志
    log_info = analyze_training_logs()
    
    print(f"\n🎯 总结:")
    print(f"=" * 30)
    
    if dir_info:
        print(f"📁 数据目录:")
        print(f"  总.skeleton文件: {dir_info['total_files']}")
        print(f"  动作类别数: {len(dir_info['action_counts'])}")
        print(f"  受试者数: {len(dir_info['subject_counts'])}")
    
    if split_info:
        total_used = sum(info['size'] for info in split_info.values() if 'size' in info)
        print(f"\n📊 数据集使用:")
        for split, info in split_info.items():
            if 'size' in info:
                print(f"  {split.upper()}集: {info['size']} 样本")
        print(f"  总使用样本: {total_used}")
    
    if filter_info:
        print(f"\n🎯 动作过滤效果:")
        for filter_type, count in filter_info.items():
            print(f"  {filter_type.upper()}: {count} 样本")
    
    if log_info:
        print(f"\n🏃 实际训练使用:")
        print(f"  训练集: ~{log_info['train_batches'] * 32} 样本")
        print(f"  验证集: {log_info['val_samples']} 样本")
        print(f"  已训练: {log_info['epochs_completed']} epochs")
    
    # 计算使用率
    if dir_info and split_info:
        total_files = dir_info['total_files']
        total_used = sum(info['size'] for info in split_info.values() if 'size' in info)
        usage_rate = (total_used / total_files) * 100 if total_files > 0 else 0
        
        print(f"\n📈 数据使用率:")
        print(f"  总文件数: {total_files}")
        print(f"  实际使用: {total_used}")
        print(f"  使用率: {usage_rate:.1f}%")


def main():
    """主分析函数"""
    print("🔍 NTU数据集使用情况分析")
    print("=" * 60)
    
    # 执行完整分析
    calculate_data_usage_summary()
    
    print(f"\n🎉 分析完成！")


if __name__ == '__main__':
    main()
