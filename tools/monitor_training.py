#!/usr/bin/env python3
"""
训练监控脚本
实时监控DVAE风格SkeletonTokenizer的训练进度
"""

import os
import time
import glob
import json
from datetime import datetime

def monitor_training_progress():
    """监控训练进度"""
    exp_dir = "./experiments/skeleton_dvae_pretrain/NTU_models/ntu_skeleton_dvae_50epochs"
    
    print("🔍 DVAE风格SkeletonTokenizer训练监控")
    print("=" * 60)
    print(f"📁 实验目录: {exp_dir}")
    print(f"⏰ 开始监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    while True:
        try:
            # 检查日志文件
            log_files = glob.glob(f"{exp_dir}/*.log")
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                print(f"\n📄 最新日志: {os.path.basename(latest_log)}")
                
                # 读取最后几行
                with open(latest_log, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        print(f"📊 日志行数: {len(lines)}")
                        # 显示最后5行
                        for line in lines[-5:]:
                            if 'epoch' in line.lower() or 'loss' in line.lower():
                                print(f"  {line.strip()}")
            
            # 检查检查点文件
            ckpt_files = glob.glob(f"{exp_dir}/ckpt-*.pth")
            if ckpt_files:
                print(f"💾 检查点文件数: {len(ckpt_files)}")
                latest_ckpt = max(ckpt_files, key=os.path.getctime)
                print(f"📦 最新检查点: {os.path.basename(latest_ckpt)}")
            
            # 检查TensorBoard日志
            tb_dir = f"./experiments/skeleton_dvae_pretrain/NTU_models/TFBoard/ntu_skeleton_dvae_50epochs"
            if os.path.exists(tb_dir):
                tb_files = glob.glob(f"{tb_dir}/**/*", recursive=True)
                print(f"📈 TensorBoard文件数: {len([f for f in tb_files if os.path.isfile(f)])}")
            
            print(f"⏰ 更新时间: {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ 监控错误: {e}")
        
        time.sleep(30)  # 每30秒更新一次

def check_training_status():
    """检查训练状态"""
    exp_dir = "./experiments/skeleton_dvae_pretrain/NTU_models/ntu_skeleton_dvae_50epochs"
    
    print("🔍 训练状态检查")
    print("=" * 40)
    
    # 检查实验目录
    if os.path.exists(exp_dir):
        print(f"✅ 实验目录存在: {exp_dir}")
        
        # 列出目录内容
        files = os.listdir(exp_dir)
        print(f"📁 目录内容 ({len(files)} 个文件):")
        for f in sorted(files):
            file_path = os.path.join(exp_dir, f)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"  📄 {f} ({size} bytes, {mtime.strftime('%H:%M:%S')})")
            else:
                print(f"  📁 {f}/")
    else:
        print(f"❌ 实验目录不存在: {exp_dir}")
    
    # 检查配置文件
    config_file = f"{exp_dir}/config.yaml"
    if os.path.exists(config_file):
        print(f"✅ 配置文件存在")
        with open(config_file, 'r') as f:
            lines = f.readlines()
            print(f"📄 配置文件行数: {len(lines)}")
    else:
        print(f"❌ 配置文件不存在")

def show_training_summary():
    """显示训练总结"""
    print("🎯 DVAE风格SkeletonTokenizer训练总结")
    print("=" * 50)
    print("🔧 关键改进:")
    print("  ✅ 分组策略: 6个身体部位 → 8组DVAE风格FPS分组")
    print("  ✅ 码本设计: 不均匀(256-1024) → 统一(512)")
    print("  ✅ 损失函数: 添加骨骼比例和全局形状一致性损失")
    print("  ✅ 训练轮数: 300轮 → 50轮")
    print()
    print("🎯 预期效果:")
    print("  📈 集中度比率: 0.047 → 0.8+")
    print("  📈 码本利用率: 4.5-17.4% → 30-50%")
    print("  📈 解决团状集中问题")
    print()
    print("📊 训练配置:")
    print("  🔢 批次大小: 32")
    print("  📚 训练样本: 71,250")
    print("  🎯 目标点数: 720")
    print("  🔧 分组数: 8")
    print("  📖 码本大小: 512/组")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'status':
            check_training_status()
        elif sys.argv[1] == 'summary':
            show_training_summary()
        else:
            print("用法: python monitor_training.py [status|summary]")
    else:
        monitor_training_progress()
