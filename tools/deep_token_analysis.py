#!/usr/bin/env python3
"""
深度码本使用分析
分析为什么码本collapse但重建仍然成功
"""

import numpy as np
import json
import sys
from pathlib import Path
from collections import defaultdict

def deep_token_analysis(npz_path):
    """深度分析 token 使用模式"""
    
    print("="*80)
    print("DEEP CODEBOOK USAGE ANALYSIS")
    print("="*80)
    
    # 加载数据
    data = np.load(npz_path, allow_pickle=True)
    
    if 'token_sequences' not in data:
        print("❌ No token_sequences found")
        return
    
    tokens = data['token_sequences']
    extracted = data['extracted']
    reconstructed = data['reconstructed']
    
    print(f"\nDataset: {npz_path}")
    print(f"Samples: {len(tokens)}")
    print(f"Token shape: {tokens.shape}")
    
    # 分析每组的 token 使用模式
    group_names = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
    tokens_per_group = 128
    
    print(f"\n{'='*80}")
    print("PER-GROUP TOKEN USAGE PATTERNS")
    print(f"{'='*80}")
    
    group_token_to_samples = {}  # 记录每个token对应哪些样本
    
    for group_idx in range(tokens.shape[1]):
        group_tokens = tokens[:, group_idx]
        group_offset = group_idx * tokens_per_group
        original_tokens = group_tokens - group_offset
        
        group_name = group_names[group_idx]
        
        # 统计每个token被哪些样本使用
        token_to_samples = defaultdict(list)
        for sample_idx, tok in enumerate(original_tokens):
            token_to_samples[tok].append(sample_idx)
        
        group_token_to_samples[group_name] = token_to_samples
        
        unique_tokens = list(token_to_samples.keys())
        
        print(f"\n{group_name.upper()}:")
        print(f"  Unique tokens: {len(unique_tokens)}/{tokens_per_group}")
        
        # 找出最常用的3个token
        sorted_tokens = sorted(token_to_samples.items(), key=lambda x: len(x[1]), reverse=True)
        
        print(f"  Top-3 most used tokens:")
        for rank, (tok, sample_list) in enumerate(sorted_tokens[:3], 1):
            print(f"    #{rank}: Token {tok} - used by {len(sample_list)} samples ({len(sample_list)/len(tokens)*100:.1f}%)")
        
        # 分析这些高频token对应的骨架特征
        if len(sorted_tokens) > 0:
            top_token, top_samples = sorted_tokens[0]
            
            # 随机抽取该token对应的5个样本，检查它们的骨架是否相似
            sample_subset = np.random.choice(top_samples, min(5, len(top_samples)), replace=False)
            
            # 计算这些样本之间的骨架相似度（MSE）
            similarities = []
            for i in range(len(sample_subset)):
                for j in range(i+1, len(sample_subset)):
                    skel_i = extracted[sample_subset[i]]
                    skel_j = extracted[sample_subset[j]]
                    mse = np.mean((skel_i - skel_j) ** 2)
                    similarities.append(mse)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                print(f"  → Samples using Token {top_token} have avg pairwise MSE: {avg_similarity:.4f}")
                
                if avg_similarity < 0.1:
                    print(f"     ✓  These samples are SIMILAR (token represents a coherent pattern)")
                elif avg_similarity < 0.5:
                    print(f"     ?  Moderate similarity (token may represent a cluster)")
                else:
                    print(f"     ❌ These samples are DIVERSE (token is being overused!)")
    
    # 全局统计：分析 token 组合模式
    print(f"\n{'='*80}")
    print("TOKEN COMBINATION ANALYSIS")
    print(f"{'='*80}")
    
    # 计算有多少种不同的 token 组合
    token_combos = set()
    for i in range(len(tokens)):
        combo = tuple(tokens[i])
        token_combos.add(combo)
    
    print(f"Unique token combinations: {len(token_combos)}/{len(tokens)}")
    print(f"Combination diversity: {len(token_combos)/len(tokens)*100:.1f}%")
    
    if len(token_combos) < len(tokens) * 0.5:
        print("⚠️  Many samples share the same token combination!")
        print("   → This suggests limited expressiveness")
    
    # 找出最常见的组合
    combo_counts = defaultdict(int)
    for i in range(len(tokens)):
        combo = tuple(tokens[i])
        combo_counts[combo] += 1
    
    sorted_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop-5 most common token combinations:")
    for rank, (combo, count) in enumerate(sorted_combos[:5], 1):
        print(f"  #{rank}: {combo} - {count} samples ({count/len(tokens)*100:.1f}%)")
    
    # 分析：最常见组合的重建质量
    print(f"\n{'='*80}")
    print("RECONSTRUCTION QUALITY BY TOKEN FREQUENCY")
    print(f"{'='*80}")
    
    # 将样本分为三组：高频token组合、中频、低频
    top_combo = sorted_combos[0][0]
    top_combo_indices = [i for i in range(len(tokens)) if tuple(tokens[i]) == top_combo]
    
    # 随机选择非高频样本
    other_indices = [i for i in range(len(tokens)) if tuple(tokens[i]) != top_combo]
    
    if len(top_combo_indices) > 10 and len(other_indices) > 10:
        # 计算高频组合的重建误差
        top_errors = []
        for idx in top_combo_indices[:100]:  # 取前100个
            err = np.mean((extracted[idx] - reconstructed[idx]) ** 2)
            top_errors.append(err)
        
        # 计算其他组合的重建误差
        other_errors = []
        for idx in other_indices[:100]:
            err = np.mean((extracted[idx] - reconstructed[idx]) ** 2)
            other_errors.append(err)
        
        print(f"Most common combination (n={len(top_combo_indices)}):")
        print(f"  Average reconstruction MSE: {np.mean(top_errors):.6f}")
        
        print(f"\nOther combinations (n={len(other_indices)}):")
        print(f"  Average reconstruction MSE: {np.mean(other_errors):.6f}")
        
        if np.mean(top_errors) < np.mean(other_errors) * 0.8:
            print("\n✓  Most common combination has BETTER reconstruction")
            print("   → Frequent tokens are well-learned")
        elif np.mean(top_errors) > np.mean(other_errors) * 1.2:
            print("\n⚠️  Most common combination has WORSE reconstruction")
            print("   → Model may be over-fitting to specific patterns")
        else:
            print("\n✓  Reconstruction quality is similar across frequencies")
    
    # 总结
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")
    
    total_unique = sum(len(group_token_to_samples[g]) for g in group_names)
    avg_per_group = total_unique / len(group_names)
    
    print(f"Total unique tokens: {total_unique}/640 ({total_unique/640*100:.1f}%)")
    print(f"Average per group: {avg_per_group:.1f}/128")
    print(f"Unique combinations: {len(token_combos)}/{len(tokens)} ({len(token_combos)/len(tokens)*100:.1f}%)")
    
    if total_unique < 100:
        print("\n❌ SEVERE COLLAPSE detected:")
        print("   • Very few tokens are being used")
        
        if len(token_combos) < len(tokens) * 0.3:
            print("   • Many samples share the same token combination")
            print("   • Codebook is acting like a LOOKUP TABLE with <100 entries")
            print("   • NOT learning compositional discrete representations")
        else:
            print("   • But token combinations are relatively diverse")
            print("   • Limited tokens are being COMPOSED in different ways")
            print("   • Codebook may be learning a small set of 'primitives'")
    
    elif total_unique < 300:
        print("\n⚠️  MODERATE usage detected:")
        print("   • Using 100-300 tokens out of 640")
        print("   • Codebook has learned some diversity but still limited")
        
        if len(token_combos) > len(tokens) * 0.7:
            print("   • Token combinations are diverse")
            print("   • Model is using ~50% of codebook effectively")
    
    else:
        print("\n✅ GOOD usage detected:")
        print("   • Using >300 tokens out of 640")
        print("   • Codebook has learned meaningful diversity")


def main():
    if len(sys.argv) < 2:
        print("Usage: python deep_token_analysis.py <path_to_recon.npz>")
        print("\nExample:")
        print("  python tools/deep_token_analysis.py data/MARS_recon_tokens_new/test_recon.npz")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    
    if not Path(npz_path).exists():
        print(f"❌ File not found: {npz_path}")
        sys.exit(1)
    
    deep_token_analysis(npz_path)


if __name__ == '__main__':
    main()
