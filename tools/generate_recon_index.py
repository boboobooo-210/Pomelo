#!/usr/bin/env python3
"""
为重构数据生成 CSV 索引文件
用于快速查找和加载样本
"""

import os
import sys
import numpy as np
import json
from pathlib import Path

def generate_index_csv(npz_dir, output_csv='index.csv'):
    """从 npz 文件生成索引 CSV"""
    
    npz_dir = Path(npz_dir)
    splits = ['train', 'test', 'validate']
    
    # 收集所有样本信息
    all_entries = []
    
    for split in splits:
        npz_path = npz_dir / f'{split}_recon.npz'
        
        if not npz_path.exists():
            print(f"⚠️  {split}_recon.npz not found, skipping")
            continue
        
        print(f"Processing {npz_path}...")
        
        try:
            data = np.load(npz_path, allow_pickle=True)
            
            # 获取样本数量
            if 'reconstructed' in data:
                n_samples = len(data['reconstructed'])
            else:
                print(f"❌ No 'reconstructed' key in {npz_path}")
                continue
            
            # 加载 metadata（如果存在）
            metadata = []
            if 'metadata' in data:
                try:
                    metadata_str = str(data['metadata'])
                    metadata = json.loads(metadata_str)
                except:
                    print(f"⚠️  Could not parse metadata in {split}")
            
            # 加载 tokens（如果存在）
            tokens = None
            if 'token_sequences' in data:
                tokens = data['token_sequences']
            else:
                tokens_path = npz_dir / f'{split}_tokens.npy'
                if tokens_path.exists():
                    tokens = np.load(tokens_path, allow_pickle=True)
            
            # 加载 VQ losses
            vq_losses = data.get('vq_losses', np.zeros(n_samples))
            
            # 生成每个样本的条目
            for i in range(n_samples):
                entry = {
                    'split': split,
                    'index': i,
                    'npz_file': f'{split}_recon.npz',
                    'vq_loss': float(vq_losses[i]) if i < len(vq_losses) else 0.0
                }
                
                # 添加 token 信息
                if tokens is not None and i < len(tokens):
                    if tokens.ndim == 2:
                        # (N, num_groups) 格式
                        entry['tokens'] = ','.join(map(str, tokens[i]))
                    else:
                        entry['tokens'] = str(tokens[i])
                else:
                    entry['tokens'] = ''
                
                # 添加 metadata
                if i < len(metadata):
                    entry['src_file'] = metadata[i].get('src_file', '')
                    entry['original_index'] = metadata[i].get('index', i)
                else:
                    entry['src_file'] = ''
                    entry['original_index'] = i
                
                all_entries.append(entry)
        
        except Exception as e:
            print(f"❌ Error processing {split}: {e}")
            continue
    
    # 写入 CSV
    if not all_entries:
        print("❌ No entries to write")
        return
    
    csv_path = npz_dir / output_csv
    print(f"\nWriting {len(all_entries)} entries to {csv_path}...")
    
    with open(csv_path, 'w') as f:
        # 写入表头
        headers = ['split', 'index', 'original_index', 'npz_file', 'vq_loss', 'tokens', 'src_file']
        f.write(','.join(headers) + '\n')
        
        # 写入数据行
        for entry in all_entries:
            row = [
                entry['split'],
                str(entry['index']),
                str(entry.get('original_index', entry['index'])),
                entry['npz_file'],
                f"{entry['vq_loss']:.6f}",
                f'"{entry.get("tokens", "")}"',  # 用引号包裹以支持逗号分隔的token
                entry.get('src_file', '')
            ]
            f.write(','.join(row) + '\n')
    
    print(f"✅ Index CSV saved: {csv_path}")
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print("INDEX STATISTICS:")
    print(f"{'='*80}")
    for split in splits:
        count = sum(1 for e in all_entries if e['split'] == split)
        if count > 0:
            print(f"{split:10s}: {count:6d} samples")
    print(f"{'='*80}")
    print(f"Total:      {len(all_entries):6d} samples")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_recon_index.py <npz_directory>")
        print("\nExample:")
        print("  python tools/generate_recon_index.py data/MARS_recon_tokens_new")
        sys.exit(1)
    
    npz_dir = sys.argv[1]
    
    if not os.path.isdir(npz_dir):
        print(f"❌ Directory not found: {npz_dir}")
        sys.exit(1)
    
    generate_index_csv(npz_dir)


if __name__ == '__main__':
    main()
