#!/usr/bin/env python3
"""
Debug helper: inspect GCN checkpoint and run encode/forward on reconstructed samples.
Run this in your environment (where torch is installed).
Example:
  python tools/debug_gcn_checkpoint.py \
    --ckpt experiments/gcn_skeleton_memory_optimized/NTU_models/default/ckpt-best.pth \
    --recon data/MARS_recon_tokens/test_recon.npz \
    --indices 0 1

The script will:
 - list top-level checkpoint keys and search for embedding/codebook-related keys
 - try to load the checkpoint into GCNSkeletonTokenizer (strict=False)
 - run forward/encode on the selected reconstructed samples and print token sequences and vq_loss
"""

import argparse
import sys
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True, help='path to gcn checkpoint .pth')
    parser.add_argument('--recon', required=True, help='path to aggregated recon npz')
    parser.add_argument('--indices', type=int, nargs='+', default=[0,1], help='sample indices to test')
    args = parser.parse_args()

    try:
        import torch
    except Exception as e:
        print('ERROR: torch is required to run this script. Please run in an environment with PyTorch installed.')
        print(e)
        sys.exit(2)

    import numpy as np
    print('Loading checkpoint:', args.ckpt)
    ckpt = torch.load(args.ckpt, map_location='cpu')

    if isinstance(ckpt, dict):
        keys = list(ckpt.keys())
        print('Top-level keys count:', len(keys))
        print('Top 20 keys:', keys[:20])
    else:
        print('Checkpoint loaded, top-level type:', type(ckpt))
        keys = []

    # search for codebook/embedding keys
    found = [k for k in keys if any(x in k.lower() for x in ('group_codebook','semantic_codebook','group_codebooks','embedding','codebook'))]
    print('Found candidate keys count:', len(found))
    for k in found[:100]:
        print('  ', k)

    # instantiate model
    try:
        from models.GCNSkeletonTokenizer import create_gcn_skeleton_tokenizer
    except Exception as e:
        print('Failed to import GCNSkeletonTokenizer from models:', e)
        sys.exit(3)

    model = create_gcn_skeleton_tokenizer()
    model.cpu()
    model.eval()

    # pick state dict (support various checkpoint layouts)
    state_dict = ckpt.get('base_model', None) or ckpt.get('state_dict', None) or ckpt
    if not isinstance(state_dict, dict):
        print('Warning: state_dict is not a dict; aborting load check')
    else:
        # strip possible 'module.' prefix
        new_state = {}
        for k,v in state_dict.items():
            nk = k[7:] if k.startswith('module.') else k
            new_state[nk] = v

        try:
            res = model.load_state_dict(new_state, strict=False)
            print('\nLoaded checkpoint into model with strict=False')
            missing = res.missing_keys if hasattr(res,'missing_keys') else []
            unexpected = res.unexpected_keys if hasattr(res,'unexpected_keys') else []
            print(' missing keys:', len(missing))
            print(' unexpected keys:', len(unexpected))
            if len(missing) > 0:
                print('  Some missing keys (first 20):', missing[:20])
            if len(unexpected) > 0:
                print('  Some unexpected keys (first 20):', unexpected[:20])
        except Exception as e:
            print('Error loading state dict into model:', e)

    # load reconstructed file
    print('\nLoading recon file:', args.recon)
    npz = np.load(args.recon, allow_pickle=True)
    print('files in recon:', npz.files)
    if 'token_sequences' in npz.files:
        ts = npz['token_sequences']
        print('token_sequences shape:', ts.shape)
        uniq = np.unique(ts, axis=0)
        print('unique token rows:', uniq.shape[0])
        if uniq.shape[0] <= 10:
            print('some unique rows:', uniq[:10])
    else:
        print('token_sequences not present in recon (maybe saved per-sample)')

    if 'reconstructed' not in npz.files:
        print('No reconstructed key in recon file; aborting encode test')
        return

    R = npz['reconstructed']
    print('reconstructed shape', R.shape)

    import torch as _t
    indices = args.indices
    for idx in indices:
        if idx < 0 or idx >= len(R):
            print('index out of range:', idx)
            continue
        x = _t.tensor(R[idx:idx+1], dtype=_t.float32)
        try:
            out = model.forward(x, return_recon=True)
            tok = out.get('token_sequence', None)
            vq = out.get('vq_loss', None)
            print(f'\nSample {idx}: token_sequence ->', None if tok is None else tok.cpu().numpy(), ' vq_loss ->', float(vq) if isinstance(vq,(float,int)) else (vq.item() if hasattr(vq,'item') else vq))
        except Exception as e:
            print('Model forward failed for sample', idx, e)

    # also try encode helper
    try:
        print('\nTesting encode() helper on first two indices')
        enc = model.encode(_t.tensor(R[indices], dtype=_t.float32))
        print('encode() output shape/type:', type(enc), getattr(enc,'shape',None))
        try:
            print('encode() ->', enc.cpu().numpy())
        except Exception:
            pass
    except Exception as e:
        print('encode() failed:', e)

if __name__ == '__main__':
    main()
