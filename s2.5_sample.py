#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FALCON Stage-2.5 — Hi-Token Sequence Sampler (M-AR Version)
"""

from __future__ import annotations
import argparse, random, shutil
from pathlib import Path
from typing import Dict
import os

# ───── ① First parse GPU args and set environment ───── #
early_parser = argparse.ArgumentParser(add_help=False)
early_parser.add_argument(
    "--gpus",
    default="1",
    help="GPU IDs to use, e.g. '0,1'; '-1' for CPU only"
)
early_args, _ = early_parser.parse_known_args()
if early_args.gpus == "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = early_args.gpus

import numpy as np
import torch
import torch.nn.functional as F

from models.mar_generator import MDNGenerator
import config

# ────────────────────────── Helpers ──────────────────────────
def load_real_counts(dataset: str, cid: int) -> Dict[int, int]:
    """Load counts of real features per class for a client"""
    base = Path(config.PATHS['real_feature'](dataset, cid))
    return {cls: len(list((base / str(cls)).glob('*.npy')))
            for cls in range(config.DATASETS[dataset]['num_classes'])}

def clear_dirs(*dirs: Path):
    """Clear and recreate directories"""
    for d in dirs:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

def sample_truncated_normal_discrete(delta: int, mean_frac: float, std_frac: float) -> int:
    """Sample integer from truncated normal distribution (0 ≤ x ≤ delta)"""
    if delta <= 0:
        return 0
    mu, sigma = mean_frac * delta, max(std_frac * delta, 1e-6)
    for _ in range(10):
        x = np.random.normal(mu, sigma)
        if 0 <= x <= delta:
            return int(round(x))
    return random.randint(0, delta)

# ───────────────────── Generator ─────────────────────
def prepare_generator(dataset: str, cid: int, device: torch.device) -> MDNGenerator:
    """Initialize and load MDN generator model"""
    num_classes = config.DATASETS[dataset]['num_classes']
    feature_dim = config.MODEL_CONFIG['feature_dim']
    split_n = getattr(config, 'MODEL_PATCHING', {}).get(
        'patch_split_n', config.MODEL_CONFIG.get('proxy_grid', 0))
    
    gen = MDNGenerator(
        cond_dim=num_classes,
        feature_dim=feature_dim,
        num_mixtures=config.MODEL_CONFIG['stage2'].get('num_mixtures', 32),
        depth=config.MODEL_CONFIG['stage2'].get('depth', 4),
        hidden_dim=config.MODEL_CONFIG['stage2'].get('hidden_dim', 768),
        num_heads=config.MODEL_CONFIG['stage2'].get('num_heads', 8),
    ).to(device)
    
    # Load pre-trained weights
    wpath = Path(config.PATHS['local_fgenerator_weights'](dataset, cid)) / 'best_fgenerator.pth'
    gen.load_state_dict(torch.load(wpath, map_location=device), strict=True)
    gen.eval()
    gen.proxy_n = split_n + 1
    return gen

@torch.no_grad()
def generate_proxies(
    gen: MDNGenerator,
    dataset: str,
    cls: int,
    count: int,
    sample_T: float,
    device: torch.device,
    real_feats: np.ndarray = None,
    use_nearest: bool = False,
    candidate_mult: int = 1.5
) -> np.ndarray:
    """
    Generate synthetic proxy samples
    
    Parameters:
    -----------
    use_nearest: If True and real_feats is not empty:
      1. Build candidate pool of size count * candidate_mult
      2. Generate candidate proxies and compute average Euclidean distance to real samples
      3. Select top count samples with smallest distances
    Otherwise: Directly sample count proxies randomly
    """
    if count == 0:
        return np.empty((0, gen.proxy_n**2 + 1, gen.D))

    # Increase synthetic sample count for better coverage
    count = round(count * 1.5)    
    # Prepare one-hot condition
    num_classes = config.DATASETS[dataset]['num_classes']
    label_1hot = F.one_hot(torch.tensor([cls], device=device), num_classes).float()

    # Nearest neighbor filtering logic
    if use_nearest and real_feats is not None and real_feats.size > 0:
        M = round(count * max(candidate_mult, 1))
        candidates = torch.cat([
            gen.generate_full_proxy(label_1hot, gen.proxy_n, sample_T)
            for _ in range(M)
        ], dim=0).cpu().numpy()

        # Flatten to (M, F)
        cand_flat = candidates.reshape(M, -1)
        real_flat = real_feats.reshape(real_feats.shape[0], -1)

        # Compute average Euclidean distance
        diffs = cand_flat[:, None, :] - real_flat[None, :, :]
        
        mean_dists = np.linalg.norm(diffs, axis=2).mean(axis=1)
        top_idx = np.argsort(mean_dists)[:count] # Select top count with smallest distances
        
        selected = candidates[top_idx]
        return selected

    # Random sampling logic
    proxies = torch.cat([
        gen.generate_full_proxy(label_1hot, gen.proxy_n, sample_T)
        for _ in range(count)
    ], dim=0)
    return proxies.cpu().numpy()

# ───────────────────── Client Routine ─────────────────────
def process_client(
    dataset: str,
    cid: int,
    mode: str,
    rr: float,
    label_blur: int,
    bdry: int,
    bdry_T: float,
    sample_T: float,
    device: torch.device,
    mean_frac: float,
    std_frac: float,
    bdry_metric: str = 'mdn',
    cls_metric_weight: float = 0.7,
    use_nearest: bool = False
):
    """Process a single client's data"""
    real_counts = load_real_counts(dataset, cid)
    merged_dir, synth_dir = (Path(config.PATHS[k](dataset, cid))
                            for k in ('merged_feature', 'synthetic_feature'))
    clear_dirs(merged_dir, synth_dir)

    gen = prepare_generator(dataset, cid, device)
    m = max(real_counts.values())
    num_classes = config.DATASETS[dataset]['num_classes']

    # Only 'cls' mode requires local classifier
    if bdry and bdry_metric == 'cls':
        from models.global_classifier import GlobalClassifier
        feat_dim = config.MODEL_CONFIG['feature_dim']
        clf = GlobalClassifier(feature_dim=feat_dim,
                              num_classes=num_classes,
                              use_moe=False, num_experts=0).to(device)
        weight_path = Path(config.PATHS['local_classifier_weights'](dataset, cid)) / 'linear_head.pth'
        clf.load_state_dict(torch.load(weight_path, map_location=device))
        clf.eval()

    for cls, n_real in real_counts.items():
        # Calculate retain/generate quantities
        if mode == 'replica':
            delta = max(m - n_real, 0)
            a_c = sample_truncated_normal_discrete(delta, mean_frac, std_frac) if label_blur else 0
            n_synth, n_keep = n_real + a_c, 0
        else:  # experiment
            rep = int(n_real * rr)
            n_keep = n_real - rep
            delta = max(m - n_real, 0)
            a_c = sample_truncated_normal_discrete(delta, mean_frac, std_frac) if label_blur else 0
            n_final = n_real + a_c
            n_synth = max(n_final - n_keep, 0) if label_blur else max(m - n_keep, 0)

        cls_real_dir = Path(config.PATHS['real_feature'](dataset, cid)) / str(cls)
        real_files = list(cls_real_dir.glob('*.npy'))

        # Boundary sampling variants
        if bdry and rr > 0 and n_keep > 0:
            if bdry_metric == 'mdn':
                label = F.one_hot(torch.tensor([cls], device=device), num_classes).float()
                metric_arr, real_feats = [], []
                for p in real_files:
                    feat = torch.tensor(np.load(p), device=device).unsqueeze(0)
                    L = feat.size(1)
                    upsample = feat[:, :1].repeat(1, L - 1, 1)
                    pi, mu, logv = gen(label, upsample)
                    nll = gen.nll_loss(feat, pi, mu, logv).item()
                    metric_arr.append(nll)
                    real_feats.append(p)
                metric_arr = np.asarray(metric_arr)

            elif bdry_metric == 'cls':
                loss_list, ent_list, real_feats = [], [], []
                for p in real_files:
                    feat = torch.tensor(np.load(p), device=device).unsqueeze(0)
                    probs = torch.softmax(clf(feat), 1).detach()[0].cpu().numpy()
                    loss_list.append(-np.log(probs[cls] + 1e-9))
                    ent_list.append(-np.sum(probs * np.log(probs + 1e-9)))
                    real_feats.append(p)
                loss_arr = np.asarray(loss_list)
                ent_arr = np.asarray(ent_list)
                loss_n = (loss_arr-loss_arr.min())/(loss_arr.max()-loss_arr.min()+1e-12)
                ent_n = (ent_arr -ent_arr.min())/(ent_arr.max() -ent_arr.min() +1e-12)
                metric_arr = cls_metric_weight * loss_n + (1-cls_metric_weight) * ent_n

            elif bdry_metric == 'maha':
                feats, real_feats = [], []
                for p in real_files:
                    feats.append(np.load(p).reshape(-1))  # flatten (L*D,)
                    real_feats.append(p)
                X = np.stack(feats, 0)                 # shape (N, F)
                mu = X.mean(0)
                var = X.var(0) + 1e-6                 # Diagonal covariance
                diff = X - mu
                metric_arr = (diff**2 / var).sum(1)    # Diagonal Mahalanobis^2

            else:
                raise ValueError("bdry_metric must be 'mdn' | 'cls' | 'maha'")

            # Softmax temperature sampling
            logits = (metric_arr - metric_arr.min()) / max(bdry_T, 1e-6)
            probs = np.exp(logits - logits.max())
            nz = probs > 0
            if probs.sum() == 0 or np.count_nonzero(nz) < n_keep:
                keep_idx = np.where(nz)[0].tolist()
                fill_idx = list(set(range(len(real_feats))) - set(keep_idx))
                if fill_idx:
                    keep_idx += random.sample(fill_idx, min(n_keep - len(keep_idx), len(fill_idx)))
                keep_files = [real_feats[i] for i in keep_idx]
            else:
                probs /= probs.sum()
                idxs = np.random.choice(len(real_feats), n_keep, replace=False, p=probs)
                keep_files = [real_feats[i] for i in idxs]
        else:
            keep_files = random.sample(real_files, n_keep) if n_keep else []

        # Load real features
        real_feat_list = []
        for p in real_files:
            arr = np.load(p)
            real_feat_list.append(arr.reshape(-1))
        real_feats_arr = np.stack(real_feat_list) if real_feat_list else np.empty((0,0))
        
        # Generate proxies
        synth_feats = generate_proxies(
            gen, dataset, cls, n_synth, sample_T, device,
            use_nearest=use_nearest,
            real_feats=real_feats_arr
        )

        # Save results
        cls_synth = synth_dir / str(cls); cls_synth.mkdir(parents=True, exist_ok=True)
        cls_merged = merged_dir / str(cls); cls_merged.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(synth_feats): 
            np.save(cls_synth / f"{i}.npy", f)

        for p in keep_files: 
            shutil.copy(p, cls_merged / p.name)

        next_idx = (max([int(p.stem) for p in cls_merged.glob('*.npy') 
                        if p.stem.isdigit()], default=-1) + 1)
        for f in synth_feats: 
            np.save(cls_merged / f"{next_idx}.npy", f)
            next_idx += 1

        print(f"{dataset}|Client{cid}|Class{cls}: kept={len(keep_files)} synth={n_synth} merged={len(list(cls_merged.glob('*.npy')))}")

# ────────────────────────── Control ──────────────────────────
def run_dataset(ds: str, args):
    """Run processing for a single dataset"""
    device = config.DEVICE
    for cid in range(len(config.DATASETS[ds]['clients'])):
        print(f"\n[{ds}|Client{cid}] mode={args.mode}, replace={args.replace_ratio}, "
              f"blur={args.label_blur}, bdry={args.bdry_sampling}({args.bdry_metric}), "
              f"T={args.temperature}, sampleT={args.sample_temperature}")
        process_client(
            ds, cid,
            args.mode, args.replace_ratio,
            args.label_blur, args.bdry_sampling,
            args.temperature, args.sample_temperature,
            device, args.blur_mean, args.blur_std,
            args.bdry_metric, args.cls_metric_weight,
            args.use_nearest,
        )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser("FALCON Stage-2.5 Sampler")
    parser.add_argument('--dataset', type=str, help='Single dataset name')
    parser.add_argument('--all_datasets', action='store_true')
    
    parser.add_argument('--mode', choices=['replica', 'experiment'], default='replica')
    
    parser.add_argument('--replace_ratio', type=float, default=0.95) # Replacement ratio
    parser.add_argument('--bdry_sampling', type=int, choices=[0, 1], default=0) # Whether to perform boundary sampling
    parser.add_argument('--bdry_metric', choices=['mdn', 'cls', 'maha'], default='maha',
                       help="mdn=NLL, cls=loss/entropy, maha=Mahalanobis distance")
    parser.add_argument('--cls_metric_weight', type=float, default=0.5) # Only valid in cls mode, weight between loss and entropy
    parser.add_argument('--temperature', type=float, default=0.8) # Boundary sampling temperature
    
    # Synthetic sample generation settings
    parser.add_argument('--sample_temperature', type=float, default=1.0)
    parser.add_argument('--use_nearest', type=int, choices=[0, 1], default=0) # Whether to use nearest neighbor filtering
    
    # Label blurring settings
    parser.add_argument('--label_blur', type=int, choices=[0, 1], default=0)
    parser.add_argument('--blur_mean', type=float, default=0.5)
    parser.add_argument('--blur_std', type=float, default=0.25)
    args = parser.parse_args()

    if not args.dataset and not args.all_datasets:
        args.all_datasets = True
    if args.dataset and args.dataset not in config.DATASETS:
        parser.error(f"Dataset '{args.dataset}' not registered in config.DATASETS")

    for ds in (config.DATASETS.keys() if args.all_datasets else [args.dataset]):
        print(f"\n================ Stage-2.5 on {ds} ================")
        run_dataset(ds, args)
        
    # Visualization functions omitted for brevity...

if __name__ == '__main__':
    main()