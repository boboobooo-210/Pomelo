#!/usr/bin/env python3
"""
工具：可视化 token-only 解码的骨架并保存 GIF。

用法：在项目根目录运行（建议在含 torch 的 conda 环境中）：
  conda run -n pb_final bash -lc "PYTHONPATH=$(pwd) python tools/visualize_token_decodes.py"

脚本行为：
 - 加载 visualizer（GCNSkeletonGifVisualizer），复用其模型和预处理函数
 - 加载 data/MARS_recon_tokens/test_recon.npz 中的 token_sequences 和 reconstructed/extracted
 - 对前 N 个样本，用 model.decode(token) 得到 token-only 重建
 - 使用 visualizer.create_skeleton_gif 将 saved reconstructed（作为“原始”）与 token-only 重建（作为“重建”）保存为 GIF
"""
import os
import sys
import numpy as np
import torch

# 确保项目根在路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 由于目录名以数字开头（`visualizations/0_gcn`），无法直接用标准import导入模块。
# 使用 importlib 按文件路径动态加载 visualizer 类。
import importlib.util
vis_path = os.path.join(os.path.dirname(__file__), '..', 'visualizations', '0_gcn', 'gcn_skeleton_gif_visualizer.py')
spec = importlib.util.spec_from_file_location('gcn_skeleton_gif_visualizer', vis_path)
vis_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis_mod)
GCNSkeletonGifVisualizer = vis_mod.GCNSkeletonGifVisualizer

def main():
    model_path = 'experiments/gcn_skeleton_memory_optimized/NTU_models/default/ckpt-best.pth'
    config_path = 'cfgs/NTU_models/gcn_skeleton_memory_optimized.yaml'
    recon_npz = 'data/MARS_recon_tokens/test_recon.npz'
    out_dir = 'visualizations/1_gcn/token_decodes_gifs'
    os.makedirs(out_dir, exist_ok=True)

    print('Loading visualizer and model...')
    vis = GCNSkeletonGifVisualizer(model_path=model_path, config_path=config_path)

    # load aggregated recon file
    print('Loading token/recon file:', recon_npz)
    npz = np.load(recon_npz, allow_pickle=True)
    token_seqs = npz['token_sequences']
    recon_saved = npz['reconstructed']
    extracted = npz['extracted'] if 'extracted' in npz.files else None

    N = token_seqs.shape[0]
    num_samples = min(8, N)
    frames_per_gif = 8
    fps = 6

    device = vis.device

    for i in range(num_samples):
        try:
            tok = token_seqs[i:i+1]
            tok_t = torch.tensor(tok, dtype=torch.long).to(device)

            with torch.no_grad():
                decoded = vis.model.decode(tok_t)  # (1,25,3) likely in model coord
            dec_np = decoded.detach().cpu().numpy()[0]

            # convert decoded to visualization format: model seems to output (x,z,y) -> convert to (x,y,z)
            dec_vis = dec_np[:, [0, 2, 1]]

            # prepare original sequence: use saved reconstructed as reference (assumed vis-format)
            orig = recon_saved[i]

            # replicate per-frame to form a short sequence
            # normalize original the same way visualizer does in reconstruct_skeleton_sequence
            skeleton_for_model = orig[:, [0, 2, 1]]  # vis (x,y,z) -> model (x,z,y)
            aligned = vis.align_skeleton_orientation(skeleton_for_model)
            normalized = vis.normalize_skeleton(aligned)
            normalized_for_vis = normalized[:, [0, 2, 1]]

            original_sequence = [normalized_for_vis.copy() for _ in range(frames_per_gif)]
            reconstructed_sequence = [dec_vis.copy() for _ in range(frames_per_gif)]

            sample_name = f'token_decode_sample_{i:06d}'
            save_path = os.path.join(out_dir, f'{sample_name}.gif')

            print(f'Creating GIF for sample {i} -> {save_path}')
            ok = vis.create_skeleton_gif(original_sequence, reconstructed_sequence, sample_name, save_path, fps=fps)
            if not ok:
                print('Failed to create GIF for sample', i)
        except Exception as e:
            print('Error processing sample', i, e)

    print('Done. GIFs saved to', out_dir)

if __name__ == '__main__':
    main()
