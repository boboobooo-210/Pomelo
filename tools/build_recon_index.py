#!/usr/bin/env python3
"""
扫描 recon 输出目录，生成 index.csv 用于人工标注调度

- 支持 per-sample 文件 (split_sample_000123.npz)
- 支持聚合文件 (split_recon.npz + optional split_tokens.npy)
- 输出 index.csv: columns = sample_index,split,file_path,tokens_str,token_first,vq_loss

用法：
  python tools/build_recon_index.py --recon_dir data/MARS_recon_tokens --out_csv data/MARS_recon_tokens/index.csv
"""

import os
import sys
import argparse
import json
import numpy as np
import csv
from pathlib import Path


def inspect_npz_file(path):
    """返回一个字典：{'tokens': list_or_array, 'vq_loss': float, 'extracted_shape':..., 'reconstructed_shape':...}"""
    try:
        data = np.load(path, allow_pickle=True)
        # per-sample uses 'tokens' or 'token_sequences'
        tokens = None
        if 'tokens' in data:
            tokens = data['tokens']
        elif 'token_sequences' in data:
            tokens = data['token_sequences']
        elif 'token_sequence' in data:
            tokens = data['token_sequence']
        # fallback: attempt metadata
        vq = None
        if 'vq_loss' in data:
            vq = float(data['vq_loss'])
        elif 'vq_losses' in data:
            vq_arr = data['vq_losses']
            # if single-value array
            try:
                vq = float(np.array(vq_arr).reshape(-1)[0])
            except Exception:
                vq = None
        # shapes
        reconstructed_shape = data['reconstructed'].shape if 'reconstructed' in data else None
        extracted_shape = data['extracted'].shape if 'extracted' in data else None

        # convert tokens to list string
        tokens_str = None
        token_first = None
        if tokens is not None:
            try:
                t = np.array(tokens)
                if t.dtype == object:
                    # pickled list
                    first = t[0]
                    tokens_str = str(first)
                    token_first = str(first[0]) if len(first) > 0 else ''
                else:
                    # regular ndarray
                    if t.ndim == 1:
                        tokens_str = str(t.tolist())
                        token_first = str(int(t[0])) if t.size>0 else ''
                    elif t.ndim == 2:
                        tokens_str = str(t.tolist())
                        token_first = str(int(t[0][0])) if t.shape[1]>0 else ''
                    else:
                        tokens_str = str(t.tolist())
                        token_first = ''
            except Exception:
                tokens_str = str(tokens)
        return {
            'tokens': tokens,
            'tokens_str': tokens_str,
            'token_first': token_first,
            'vq_loss': vq,
            'reconstructed_shape': reconstructed_shape,
            'extracted_shape': extracted_shape
        }
    except Exception as e:
        return {'error': str(e)}


def scan_per_sample_dir(recon_dir):
    """扫描 per-sample 文件名并返回条目列表"""
    entries = []
    p = Path(recon_dir)
    for path in sorted(p.glob('**/*sample_*.npz')):
        name = path.stem
        # attempt to parse index and split from filename
        # expected: {split}_sample_{idx:06d}.npz
        parts = name.split('_sample_')
        if len(parts) == 2:
            split = parts[0]
            try:
                idx = int(parts[1])
            except Exception:
                idx = None
        else:
            split = 'unknown'
            idx = None
        info = inspect_npz_file(str(path))
        entries.append({'index': idx, 'split': split, 'file_path': str(path), **info})
    return entries


def scan_aggregate_files(recon_dir):
    """扫描 <split>_recon.npz 并结合 <split>_tokens.npy（若存在）生成条目（index基于数组下标）"""
    entries = []
    p = Path(recon_dir)
    for npz_path in sorted(p.glob('*_recon.npz')):
        split = npz_path.name.split('_recon.npz')[0]
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print(f"Failed to load {npz_path}: {e}")
            continue
        reconstructed = data['reconstructed'] if 'reconstructed' in data else None
        vq = data['vq_losses'] if 'vq_losses' in data else None

        # tokens may be inside or in separate file
        tokens = None
        tokens_path = p / f'{split}_tokens.npy'
        if 'token_sequences' in data:
            tokens = data['token_sequences']
        elif tokens_path.exists():
            tokens = np.load(tokens_path, allow_pickle=True)

        N = reconstructed.shape[0] if reconstructed is not None else 0
        for i in range(N):
            tok = None
            tok_str = None
            tok_first = None
            if tokens is not None:
                try:
                    tok_i = tokens[i]
                    tok = tok_i
                    # stringify
                    if isinstance(tok_i, np.ndarray):
                        tok_str = str(tok_i.tolist())
                        tok_first = str(int(tok_i[0])) if tok_i.size>0 else ''
                    elif isinstance(tok_i, (list, tuple)):
                        tok_str = str(list(tok_i))
                        tok_first = str(int(tok_i[0])) if len(tok_i)>0 else ''
                    else:
                        tok_str = str(tok_i)
                except Exception:
                    tok_str = ''
            vq_val = None
            try:
                vq_val = float(vq[i]) if vq is not None else None
            except Exception:
                vq_val = None
            entries.append({'index': i, 'split': split, 'file_path': str(npz_path) + f':{i}', 'tokens': tok, 'tokens_str': tok_str, 'token_first': tok_first, 'vq_loss': vq_val})
    return entries


def build_index(recon_dir, out_csv):
    per_entries = scan_per_sample_dir(recon_dir)
    agg_entries = scan_aggregate_files(recon_dir)

    all_entries = per_entries + agg_entries
    # normalize and write csv
    fieldnames = ['index','split','file_path','tokens_str','token_first','vq_loss']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in all_entries:
            row = {k: e.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f'Wrote index {out_csv} with {len(all_entries)} entries (per_sample={len(per_entries)}, aggregate_indexed={len(agg_entries)})')

    # print summary by token_first top-10
    token_counts = {}
    for e in all_entries:
        t = e.get('token_first')
        if t is None or t == '':
            continue
        token_counts[t] = token_counts.get(t, 0) + 1
    print('Top token_first counts:')
    for k, v in sorted(token_counts.items(), key=lambda kv: kv[1], reverse=True)[:20]:
        print(f'  token {k}: {v}')

    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recon_dir', required=True)
    parser.add_argument('--out_csv', required=True)
    args = parser.parse_args()

    build_index(args.recon_dir, args.out_csv)

if __name__ == '__main__':
    main()
