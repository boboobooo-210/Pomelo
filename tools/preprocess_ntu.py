"""
NTU RGB+D 数据预处理脚本
根据动作类型进行数据划分和预处理
"""

import os
import sys
import numpy as np
import argparse
from collections import defaultdict, Counter
import yaml
import random

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 动作分类定义
SINGLE_DAILY_ACTIONS = list(range(1, 41)) + list(range(61, 103))  # A1-A40, A61-A102 (除康复动作)
REHABILITATION_ACTIONS = list(range(41, 50)) + list(range(103, 106))  # A41-A49, A103-A105
INTERACTION_ACTIONS = list(range(50, 61)) + list(range(107, 121))  # A50-A60, A107-A120

# 移除康复动作从单人日常动作中
for action in REHABILITATION_ACTIONS:
    if action in SINGLE_DAILY_ACTIONS:
        SINGLE_DAILY_ACTIONS.remove(action)

# DVAE训练使用的动作
DVAE_ACTIONS = SINGLE_DAILY_ACTIONS + REHABILITATION_ACTIONS


def parse_filename(filename):
    """解析NTU文件名获取信息"""
    # 文件名格式: S001C001P001R001A001.skeleton
    parts = filename.replace('.skeleton', '').split('C')[0].split('P')
    
    setup = int(filename[1:4])  # S001 -> 1
    camera = int(filename[5:8])  # C001 -> 1
    person = int(filename[9:12])  # P001 -> 1
    replication = int(filename[13:16])  # R001 -> 1
    action = int(filename[17:20])  # A001 -> 1
    
    return {
        'setup': setup,
        'camera': camera, 
        'person': person,
        'replication': replication,
        'action': action,
        'filename': filename
    }


def analyze_dataset(data_path):
    """分析数据集统计信息"""
    print("🔍 分析NTU RGB+D数据集...")
    
    skeleton_files = [f for f in os.listdir(data_path) if f.endswith('.skeleton')]
    print(f"总文件数: {len(skeleton_files)}")
    
    # 解析所有文件信息
    file_info = []
    for file in skeleton_files:
        try:
            info = parse_filename(file)
            file_info.append(info)
        except:
            print(f"⚠️ 无法解析文件名: {file}")
            continue
    
    # 统计信息
    actions = [info['action'] for info in file_info]
    action_counts = Counter(actions)
    
    # 按类别分类
    daily_files = [info for info in file_info if info['action'] in SINGLE_DAILY_ACTIONS]
    rehab_files = [info for info in file_info if info['action'] in REHABILITATION_ACTIONS]
    interaction_files = [info for info in file_info if info['action'] in INTERACTION_ACTIONS]
    dvae_files = [info for info in file_info if info['action'] in DVAE_ACTIONS]
    
    print(f"\n📊 动作类别统计:")
    print(f"单人日常动作: {len(daily_files)} 个文件")
    print(f"康复动作: {len(rehab_files)} 个文件")
    print(f"双人互动动作: {len(interaction_files)} 个文件")
    print(f"DVAE训练数据: {len(dvae_files)} 个文件")
    
    print(f"\n🎯 DVAE训练动作分布:")
    dvae_actions = [info['action'] for info in dvae_files]
    dvae_action_counts = Counter(dvae_actions)
    
    for action_id in sorted(dvae_action_counts.keys()):
        count = dvae_action_counts[action_id]
        category = "康复" if action_id in REHABILITATION_ACTIONS else "日常"
        print(f"A{action_id:03d}: {count} 个样本 ({category})")
    
    return {
        'total_files': len(skeleton_files),
        'valid_files': len(file_info),
        'daily_files': daily_files,
        'rehab_files': rehab_files,
        'interaction_files': interaction_files,
        'dvae_files': dvae_files,
        'action_counts': action_counts,
        'dvae_action_counts': dvae_action_counts
    }


def create_data_splits(dvae_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """创建训练/验证/测试数据划分"""
    print(f"\n📂 创建数据划分 (训练:{train_ratio}, 验证:{val_ratio}, 测试:{test_ratio})")
    
    # 按动作类别分组
    action_groups = defaultdict(list)
    for file_info in dvae_files:
        action_groups[file_info['action']].append(file_info)
    
    train_files = []
    val_files = []
    test_files = []
    
    # 对每个动作类别进行划分
    for action_id, files in action_groups.items():
        random.shuffle(files)
        
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train+n_val])
        test_files.extend(files[n_train+n_val:])
        
        print(f"A{action_id:03d}: {n_total} -> 训练:{n_train}, 验证:{n_val}, 测试:{n_test}")
    
    print(f"\n✅ 数据划分完成:")
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件") 
    print(f"测试集: {len(test_files)} 个文件")
    
    return train_files, val_files, test_files


def save_data_splits(train_files, val_files, test_files, output_dir):
    """保存数据划分信息"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存文件列表
    splits = {
        'train': [f['filename'] for f in train_files],
        'val': [f['filename'] for f in val_files],
        'test': [f['filename'] for f in test_files]
    }
    
    for split_name, filenames in splits.items():
        output_file = os.path.join(output_dir, f'{split_name}_files.txt')
        with open(output_file, 'w') as f:
            for filename in sorted(filenames):
                f.write(filename + '\n')
        print(f"💾 保存 {split_name} 文件列表: {output_file}")
    
    # 保存统计信息
    stats = {
        'total_files': len(train_files) + len(val_files) + len(test_files),
        'train_files': len(train_files),
        'val_files': len(val_files),
        'test_files': len(test_files),
        'action_categories': {
            'single_daily': SINGLE_DAILY_ACTIONS,
            'rehabilitation': REHABILITATION_ACTIONS,
            'interaction': INTERACTION_ACTIONS,
            'dvae_actions': DVAE_ACTIONS
        }
    }
    
    stats_file = os.path.join(output_dir, 'dataset_stats.yaml')
    with open(stats_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    print(f"📊 保存统计信息: {stats_file}")


def create_dataset_config(output_dir):
    """创建数据集配置文件"""
    config = {
        'dataset_name': 'NTU_RGB+D',
        'description': 'NTU RGB+D 骨架数据集，用于DVAE训练构建人体骨架点云码本',
        'data_path': './data/NTU-RGB+D',
        'splits': {
            'train': 'train_files.txt',
            'val': 'val_files.txt', 
            'test': 'test_files.txt'
        },
        'action_filter': 'dvae',
        'num_joints': 25,
        'num_points': 1024,
        'action_categories': {
            'single_daily': {
                'actions': SINGLE_DAILY_ACTIONS,
                'description': '单人日常动作',
                'count': len(SINGLE_DAILY_ACTIONS)
            },
            'rehabilitation': {
                'actions': REHABILITATION_ACTIONS,
                'description': '康复动作',
                'count': len(REHABILITATION_ACTIONS)
            },
            'interaction': {
                'actions': INTERACTION_ACTIONS,
                'description': '双人互动动作（不用于DVAE训练）',
                'count': len(INTERACTION_ACTIONS)
            }
        },
        'dvae_training': {
            'use_actions': DVAE_ACTIONS,
            'total_actions': len(DVAE_ACTIONS),
            'description': '用于DVAE训练的动作类别（单人日常 + 康复）'
        }
    }
    
    config_file = os.path.join(output_dir, 'ntu_dataset_config.yaml')
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"⚙️ 保存数据集配置: {config_file}")


def main():
    parser = argparse.ArgumentParser(description='NTU RGB+D 数据预处理')
    parser.add_argument('--data_path', type=str, default='./data/NTU-RGB+D',
                       help='NTU RGB+D 数据路径')
    parser.add_argument('--output_dir', type=str, default='./data/NTU-RGB+D/splits',
                       help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                       help='测试集比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("🎯 NTU RGB+D 数据预处理")
    print("=" * 50)
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据划分: 训练{args.train_ratio}, 验证{args.val_ratio}, 测试{args.test_ratio}")
    
    # 检查数据路径
    if not os.path.exists(args.data_path):
        print(f"❌ 数据路径不存在: {args.data_path}")
        return
    
    # 分析数据集
    stats = analyze_dataset(args.data_path)
    
    # 创建数据划分
    train_files, val_files, test_files = create_data_splits(
        stats['dvae_files'], 
        args.train_ratio, 
        args.val_ratio, 
        args.test_ratio
    )
    
    # 保存数据划分
    save_data_splits(train_files, val_files, test_files, args.output_dir)
    
    # 创建配置文件
    create_dataset_config(args.output_dir)
    
    print(f"\n🎉 NTU RGB+D 数据预处理完成！")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🎯 DVAE训练数据: {len(stats['dvae_files'])} 个文件")
    print(f"📊 动作类别: {len(DVAE_ACTIONS)} 个动作")


if __name__ == '__main__':
    main()
