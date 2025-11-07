#!/usr/bin/env python3
"""
快速检查已保存的 token 数据的多样性
用于诊断码本是否学到了有意义的离散表示
"""

import numpy as np
import json
import sys
from pathlib import Path

def analyze_token_diversity(tokens_array, num_groups=5, tokens_per_group=128):
    """分析 token 序列的多样性"""
    
    group_names = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
    
    analysis = {}
    
    print("\n" + "="*80)
    print("TOKEN DIVERSITY ANALYSIS")
    print("="*80)
    print(f"Total samples: {len(tokens_array)}")
    print(f"Token sequence shape: {tokens_array.shape}")
    
    for group_idx in range(min(num_groups, tokens_array.shape[1])):
        group_tokens = tokens_array[:, group_idx]
        
        # 计算该组token的offset（用于还原原始token ID）
        group_offset = group_idx * tokens_per_group
        
        # 还原为原始token ID（移除offset）
        original_tokens = group_tokens - group_offset
        
        # 统计唯一token
        unique_tokens, counts = np.unique(original_tokens, return_counts=True)
        
        # 计算熵（衡量分布均匀性）
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(tokens_per_group)  # 完全均匀分布的最大熵
        
        # Top-10 最常用的token
        top_indices = np.argsort(-counts)[:10]
        top_tokens = unique_tokens[top_indices]
        top_counts = counts[top_indices]
        
        group_name = group_names[group_idx] if group_idx < len(group_names) else f'group_{group_idx}'
        
        analysis[group_name] = {
            'unique_tokens': len(unique_tokens),
            'total_tokens': len(original_tokens),
            'coverage': len(unique_tokens) / tokens_per_group * 100,
            'entropy': entropy,
            'normalized_entropy': entropy / max_entropy if max_entropy > 0 else 0,
            'top_10_tokens': top_tokens.tolist(),
            'top_10_counts': top_counts.tolist(),
            'top_10_freq': (top_counts / counts.sum() * 100).tolist()
        }
        
        # 打印该组信息
        print(f"\n{group_name.upper()} (Group {group_idx}):")
        print(f"  Unique tokens: {len(unique_tokens)}/{tokens_per_group} ({len(unique_tokens)/tokens_per_group*100:.1f}% coverage)")
        print(f"  Entropy: {entropy:.3f} / {max_entropy:.3f} (normalized: {entropy/max_entropy:.3f})")
        
        # 检查是否所有样本都使用相同的token
        if len(unique_tokens) == 1:
            print(f"  ❌ CRITICAL: ALL samples use the SAME token ({unique_tokens[0]})!")
        
        print(f"  Top-10 most used tokens:")
        for i, (tok, cnt, freq) in enumerate(zip(top_tokens[:10], top_counts[:10], top_counts[:10]/counts.sum()*100)):
            marker = "❌" if freq > 90 else "⚠️" if freq > 50 else "✓"
            print(f"    {marker} #{i+1}: Token {tok:3d} - {cnt:5d} times ({freq:5.2f}%)")
    
    # 全局统计
    print(f"\n{'='*80}")
    print("GLOBAL SUMMARY:")
    print(f"{'='*80}")
    
    total_unique = sum(a['unique_tokens'] for a in analysis.values())
    total_possible = num_groups * tokens_per_group
    avg_coverage = np.mean([a['coverage'] for a in analysis.values()])
    avg_entropy = np.mean([a['normalized_entropy'] for a in analysis.values()])
    
    print(f"Total unique tokens used: {total_unique}/{total_possible} ({total_unique/total_possible*100:.1f}%)")
    print(f"Average coverage per group: {avg_coverage:.1f}%")
    print(f"Average normalized entropy: {avg_entropy:.3f}")
    
    # 诊断建议
    print(f"\n{'='*80}")
    print("DIAGNOSTIC:")
    print(f"{'='*80}")
    
    # 检查是否所有组都只用一个token（最严重的collapse）
    all_single_token = all(a['unique_tokens'] == 1 for a in analysis.values())
    if all_single_token:
        print("❌❌❌ CATASTROPHIC COLLAPSE!")
        print("   ALL groups use ONLY 1 token across all samples")
        print("   → Codebook is COMPLETELY UNUSED")
        print("   → Model is 100% relying on residual connections")
        print("   → VQ loss weight is TOO LOW or residual gate is TOO HIGH")
    elif avg_coverage < 5:
        print("❌ SEVERE COLLAPSE: Less than 5% of codebook used!")
        print("   → Model is NOT learning meaningful discrete representations")
        print("   → Increase VQ loss weight significantly (e.g., 2.0)")
        print("   → Reduce residual gate initialization (e.g., -10.0)")
    elif avg_coverage < 10:
        print("❌ SEVERE COLLAPSE: Less than 10% of codebook used!")
        print("   → Model is NOT learning meaningful discrete representations")
    elif avg_coverage < 30:
        print("⚠️  MODERATE COLLAPSE: 10-30% of codebook used")
        print("   → Codebook learning is weak, consider increasing VQ loss weight")
    elif avg_coverage < 60:
        print("✓  FAIR: 30-60% of codebook used")
        print("   → Codebook is learning, but still has room for improvement")
    else:
        print("✅ GOOD: >60% of codebook used")
        print("   → Codebook is learning diverse representations")
    
    if avg_entropy < 0.3:
        print("❌ HIGHLY SKEWED: Token distribution is extremely imbalanced")
    elif avg_entropy < 0.6:
        print("⚠️  SKEWED: Token distribution is somewhat imbalanced")
    else:
        print("✅ BALANCED: Token distribution is relatively uniform")
    
    return analysis


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_token_diversity_check.py <path_to_recon.npz>")
        print("\nExample:")
        print("  python tools/quick_token_diversity_check.py data/MARS_recon_tokens/test_recon.npz")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    
    print(f"Loading token data from: {npz_path}")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        if 'token_sequences' in data:
            tokens = data['token_sequences']
        else:
            print("❌ No 'token_sequences' found in the npz file")
            print(f"Available keys: {list(data.keys())}")
            
            # 尝试加载单独的 tokens.npy 文件
            tokens_path = npz_path.replace('_recon.npz', '_tokens.npy')
            if Path(tokens_path).exists():
                print(f"Found separate tokens file: {tokens_path}")
                tokens = np.load(tokens_path, allow_pickle=True)
            else:
                sys.exit(1)
        
        print(f"Token array shape: {tokens.shape}")
        print(f"Token array dtype: {tokens.dtype}")
        
        # 分析
        analysis = analyze_token_diversity(tokens)
        
        # 保存分析结果
        output_path = npz_path.replace('.npz', '_token_analysis.json')
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✅ Analysis saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
