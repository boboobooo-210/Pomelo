#!/usr/bin/env python3
"""
NTU/MARS数据集GCN码本使用率统计与对比分析

本脚本用于:
1. 统计GCN模型在MARS数据集上的码本使用情况
2. 生成详细的统计报告
3. 可视化码本使用分布
4. 对比不同语义分组的使用模式
"""

import numpy as np
import torch
from collections import Counter
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_mars_statistics(data_path='data/MARS_recon_tokens/train_recon.npz'):
    """加载MARS数据集上的token序列"""
    data = np.load(data_path, allow_pickle=True)
    return data['token_sequences']  # (N, 5)

def analyze_codebook_usage(token_sequences):
    """分析码本使用情况"""
    group_names = ['head_spine', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
    group_names_cn = ['头脊柱', '左臂', '右臂', '左腿', '右腿']
    
    results = {
        'total_samples': len(token_sequences),
        'groups': {},
        'global_stats': {},
        'combinations': {}
    }
    
    # 1. 统计每组的token使用情况
    all_used_tokens = set()
    for i, (name, name_cn) in enumerate(zip(group_names, group_names_cn)):
        group_tokens = token_sequences[:, i]
        unique_tokens = np.unique(group_tokens)
        used_count = len(unique_tokens)
        usage_rate = (used_count / 128) * 100
        
        # Token频率统计
        token_counts = Counter(group_tokens)
        top_tokens = token_counts.most_common(10)
        
        results['groups'][name] = {
            'name_cn': name_cn,
            'used_tokens': int(used_count),
            'total_tokens': 128,
            'usage_rate': float(usage_rate),
            'unique_tokens': [int(t) for t in unique_tokens],
            'token_frequencies': {int(token): int(count) for token, count in top_tokens},
            'top_3_coverage': sum([count for _, count in top_tokens[:3]]) / len(group_tokens) * 100
        }
        
        all_used_tokens.update(unique_tokens)
    
    # 2. 全局统计
    results['global_stats'] = {
        'total_tokens': 640,
        'used_tokens': len(all_used_tokens),
        'usage_rate': len(all_used_tokens) / 640 * 100,
        'unused_tokens': 640 - len(all_used_tokens)
    }
    
    # 3. Token组合统计
    unique_combinations = set()
    combination_counts = Counter()
    for tokens in token_sequences:
        combo = tuple(tokens)
        unique_combinations.add(combo)
        combination_counts[combo] += 1
    
    results['combinations'] = {
        'unique_count': len(unique_combinations),
        'diversity_ratio': len(unique_combinations) / len(token_sequences) * 100,
        'top_combinations': [
            {'tokens': list(combo), 'count': int(count), 'percentage': count/len(token_sequences)*100}
            for combo, count in combination_counts.most_common(10)
        ]
    }
    
    return results

def print_statistics_report(results):
    """打印统计报告"""
    print('=' * 80)
    print('GCN码本使用率统计报告 - MARS数据集')
    print('=' * 80)
    
    print(f'\n数据集信息:')
    print(f'  总样本数: {results["total_samples"]:,}')
    
    print(f'\n全局码本使用统计:')
    gs = results['global_stats']
    print(f'  总Token数: {gs["total_tokens"]}')
    print(f'  已使用Token数: {gs["used_tokens"]}')
    print(f'  未使用Token数: {gs["unused_tokens"]}')
    print(f'  全局使用率: {gs["usage_rate"]:.2f}%')
    
    print(f'\n语义分组使用率:')
    print(f'{"分组":<20} {"使用数/总数":<15} {"使用率":<10} {"前3覆盖率":<12}')
    print('-' * 60)
    for name, info in results['groups'].items():
        print(f'{info["name_cn"]:<18} {info["used_tokens"]}/{info["total_tokens"]:<12} '
              f'{info["usage_rate"]:>6.1f}%    {info["top_3_coverage"]:>8.1f}%')
    
    print(f'\nToken组合多样性:')
    cs = results['combinations']
    print(f'  唯一组合数: {cs["unique_count"]}')
    print(f'  多样性比例: {cs["diversity_ratio"]:.2f}%')
    
    print(f'\n最常见的Token组合:')
    for i, combo_info in enumerate(cs['top_combinations'][:5], 1):
        tokens_str = str(combo_info['tokens'])
        print(f'  {i}. {tokens_str:<40} {combo_info["count"]:>6}次 ({combo_info["percentage"]:.2f}%)')
    
    print('\n' + '=' * 80)

def plot_codebook_usage(results, output_dir='data/visualizations'):
    """可视化码本使用情况"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 各组使用率对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1.1 使用率条形图
    ax = axes[0, 0]
    groups = list(results['groups'].keys())
    usage_rates = [results['groups'][g]['usage_rate'] for g in groups]
    names_cn = [results['groups'][g]['name_cn'] for g in groups]
    
    bars = ax.bar(range(len(groups)), usage_rates, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(names_cn, rotation=45, ha='right')
    ax.set_ylabel('Usage Rate (%)')
    ax.set_title('Codebook Usage Rate by Semantic Group')
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, rate) in enumerate(zip(bars, usage_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 1.2 使用数量堆叠图
    ax = axes[0, 1]
    used = [results['groups'][g]['used_tokens'] for g in groups]
    unused = [results['groups'][g]['total_tokens'] - results['groups'][g]['used_tokens'] for g in groups]
    
    x = np.arange(len(groups))
    ax.bar(x, used, label='Used', color='#2ecc71', alpha=0.8)
    ax.bar(x, unused, bottom=used, label='Unused', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names_cn, rotation=45, ha='right')
    ax.set_ylabel('Number of Tokens')
    ax.set_title('Used vs Unused Tokens by Group')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 1.3 Token频率分布（以head_spine为例）
    ax = axes[1, 0]
    head_spine_freqs = results['groups']['head_spine']['token_frequencies']
    tokens = list(head_spine_freqs.keys())
    freqs = list(head_spine_freqs.values())
    
    ax.bar(range(len(tokens)), freqs, color='coral', alpha=0.8)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([f'T{t}' for t in tokens], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Frequency')
    ax.set_title('Token Frequency Distribution (Head-Spine Group)')
    ax.grid(axis='y', alpha=0.3)
    
    # 1.4 全局使用率饼图
    ax = axes[1, 1]
    gs = results['global_stats']
    sizes = [gs['used_tokens'], gs['unused_tokens']]
    labels = [f'Used\n{gs["used_tokens"]} ({gs["usage_rate"]:.1f}%)', 
              f'Unused\n{gs["unused_tokens"]} ({100-gs["usage_rate"]:.1f}%)']
    colors = ['#3498db', '#ecf0f1']
    explode = (0.05, 0)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='',
           shadow=True, startangle=90)
    ax.set_title('Global Codebook Usage')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/codebook_usage_overview.png', dpi=300, bbox_inches='tight')
    print(f'✓ 可视化图表已保存: {output_dir}/codebook_usage_overview.png')
    plt.close()
    
    # 2. 热力图：每组中每个token的使用频率
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, (name, ax) in enumerate(zip(groups, axes)):
        info = results['groups'][name]
        token_freq = np.zeros(128)
        
        # Token值包含了组偏移，需要减去组的起始索引
        group_offset = idx * 128
        for token, freq in info['token_frequencies'].items():
            local_token = token - group_offset  # 转换为组内索引 (0-127)
            if 0 <= local_token < 128:
                token_freq[local_token] = freq
        
        # 归一化
        token_freq_norm = token_freq / results['total_samples'] * 100
        
        # 重塑为16x8以便显示
        token_freq_2d = token_freq_norm.reshape(16, 8)
        
        im = ax.imshow(token_freq_2d, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        ax.set_title(info['name_cn'], fontsize=12)
        ax.set_xlabel('Token (col)')
        ax.set_ylabel('Token (row×8)')
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, label='Usage %')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/token_frequency_heatmap.png', dpi=300, bbox_inches='tight')
    print(f'✓ 热力图已保存: {output_dir}/token_frequency_heatmap.png')
    plt.close()

def save_results_json(results, output_path='data/ntu_codebook_statistics.json'):
    """保存统计结果为JSON"""
    # 转换numpy类型为Python原生类型
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    results_native = convert_to_native(results)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_native, f, indent=2, ensure_ascii=False)
    print(f'✓ 统计结果已保存: {output_path}')

def main():
    print('开始分析GCN码本使用率...\n')
    
    # 1. 加载数据
    print('[1/4] 加载MARS token序列...')
    token_sequences = load_mars_statistics()
    print(f'✓ 加载完成: {len(token_sequences)} 个样本\n')
    
    # 2. 统计分析
    print('[2/4] 分析码本使用情况...')
    results = analyze_codebook_usage(token_sequences)
    print('✓ 分析完成\n')
    
    # 3. 打印报告
    print('[3/4] 生成统计报告...')
    print_statistics_report(results)
    
    # 4. 保存结果
    print('\n[4/4] 保存结果...')
    save_results_json(results)
    plot_codebook_usage(results)
    
    print('\n✓ 所有任务完成!')

if __name__ == '__main__':
    main()
