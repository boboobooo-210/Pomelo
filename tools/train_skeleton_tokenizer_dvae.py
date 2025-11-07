#!/usr/bin/env python3
"""
DVAE风格SkeletonTokenizer训练脚本
使用改进的分组策略和损失函数，50轮训练
"""

import os
import sys
import argparse
import datetime
import torch
import torch.distributed as dist
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.runner import run_net
from utils import misc, dist_utils
import time
import json
from tensorboardX import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                       default='experiments/skeleton_dvae_pretrain/NTU_models/ntu_skeleton_tokenizer_dvae_50epochs/config.yaml',
                       help='yaml config file')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--exp_name', type=str, default='ntu_skeleton_tokenizer_dvae_50epochs', help='experiment name')
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=5, help='test freq')
    parser.add_argument('--resume', action='store_true', default=False, help='autoresume training (interrupted by accident)')
    parser.add_argument('--test', action='store_true', default=False, help='test mode for certain ckpt')
    parser.add_argument('--finetune_model', action='store_true', default=False, help='finetune modelnet with pretrained weight')
    parser.add_argument('--scratch_model', action='store_true', default=False, help='training modelnet from scratch')
    parser.add_argument('--mode', choices=['easy', 'median', 'hard'], default=None, help='difficulty mode for shapenet')
    parser.add_argument('--way', type=int, default=-1)
    parser.add_argument('--shot', type=int, default=-1)
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')

    args = parser.parse_args()
    return args


def setup_experiment_dir(args):
    """设置实验目录"""
    # 创建实验目录
    exp_dir = Path(f'experiments/skeleton_dvae_pretrain/NTU_models/{args.exp_name}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志目录
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # 创建检查点目录
    ckpt_dir = exp_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    
    return exp_dir


def log_training_info(exp_dir):
    """记录训练信息"""
    info = {
        'experiment_name': 'DVAE Style SkeletonTokenizer Training',
        'start_time': datetime.datetime.now().isoformat(),
        'modifications': {
            'grouping_strategy': 'DVAE style FPS random sampling (8 groups)',
            'codebook_design': 'Unified codebook size (512 per group)',
            'loss_function': 'Improved with bone ratio and global shape consistency',
            'training_epochs': 50,
            'expected_improvements': [
                'Solve joint clustering problem',
                'Improve codebook utilization rate',
                'Maintain skeleton structure flexibility'
            ]
        },
        'technical_details': {
            'num_groups': 8,
            'codebook_size': 512,
            'points_per_group': 90,
            'total_points': 720,
            'loss_weights': {
                'reconstruction': 10.0,
                'structure': 1.0,
                'codebook': 0.25,
                'commitment': 0.25
            }
        }
    }
    
    with open(exp_dir / 'training_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("🚀 DVAE风格SkeletonTokenizer训练开始")
    print("=" * 60)
    print(f"📁 实验目录: {exp_dir}")
    print(f"🎯 训练轮数: 50轮")
    print(f"🔧 分组策略: 8组DVAE风格FPS采样")
    print(f"📚 码本设计: 统一512码字/组")
    print(f"💡 预期效果: 解决团状集中问题")
    print("=" * 60)


def main():
    # 解析参数
    args = get_args()
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 设置实验目录
    exp_dir = setup_experiment_dir(args)
    
    # 记录训练信息
    log_training_info(exp_dir)
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        return
    
    print(f"✅ 使用配置文件: {args.config}")
    
    # 设置随机种子
    if args.fix_random_seed:
        misc.set_random_seed(args.seed)
        if args.deterministic:
            misc.set_deterministic()
    
    # 初始化分布式训练
    if args.launcher == 'none':
        dist_utils.setup_env(args.launcher, 0, 1, 1, args.local_rank)
    else:
        dist_utils.setup_env(args.launcher, args.local_rank, 1, 1, args.local_rank)
    
    # 开始训练
    try:
        print(f"🚀 开始DVAE风格SkeletonTokenizer训练...")
        run_net(args)
        print(f"🎉 训练完成！")
        
        # 记录完成信息
        completion_info = {
            'completion_time': datetime.datetime.now().isoformat(),
            'status': 'completed',
            'experiment_dir': str(exp_dir)
        }
        
        with open(exp_dir / 'completion_info.json', 'w') as f:
            json.dump(completion_info, f, indent=2)
            
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        
        # 记录错误信息
        error_info = {
            'error_time': datetime.datetime.now().isoformat(),
            'status': 'failed',
            'error_message': str(e),
            'experiment_dir': str(exp_dir)
        }
        
        with open(exp_dir / 'error_info.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        raise


if __name__ == '__main__':
    main()
