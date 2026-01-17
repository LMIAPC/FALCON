#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FALCON Stage-2 - M-AR Generator Training Script
"""

"""
chmod +x run_all_datasets_s2.0.sh
./run_all_datasets_s2.0.sh
"""

# ───── ① First parse GPU args and set environment ───── #
import os
import argparse

parser = argparse.ArgumentParser(description="Stage2 MDN Generator (GPU-aware)", add_help=False)
parser.add_argument(
    "--gpus",
    default="3",
    help="GPU IDs, e.g. '0,1,2'; '-1' for CPU only"
)  # GPU selection logic only applies here
args_early, remaining_argv = parser.parse_known_args()

if args_early.gpus != "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = args_early.gpus
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ───── ② Suppress all Python warnings ───── #
import warnings
warnings.filterwarnings("ignore")

# ───── ③ Regular imports (torch now sees only available GPUs) ───── #
import sys
import numpy as np
from pathlib import Path
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from typing import List, Tuple

import config
from models.mar_generator import MDNGenerator


# ───────────────── Utility Functions ────────────────── #
def parse_gpu_ids(gpu_str: str) -> List[int]:
    return [int(x) for x in gpu_str.split(",") if x.strip().isdigit()]


def load_features(dataset: str, cid: int):
    """
    Load all features and labels from <REAL_FEATURE_DIR>/<cid>/<cls>/*.npy,
    return (X, Y) where X is (N, feat_dim) Tensor and Y is (N,) LongTensor.
    """
    feat_dir = config.PATHS["real_feature"](dataset, cid)
    X_list, Y_list = [], []
    for cls in range(config.DATASETS[dataset]["num_classes"]):
        cls_dir = os.path.join(feat_dir, str(cls))
        if not os.path.isdir(cls_dir):
            continue
        for fn in os.listdir(cls_dir):
            if fn.endswith(".npy"):
                X_list.append(np.load(os.path.join(cls_dir, fn)))
                Y_list.append(cls)
    if len(X_list) == 0:
        raise RuntimeError(f"No features found under {feat_dir}")
    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    Y = torch.tensor(Y_list, dtype=torch.long)
    return X, Y


# ────────────────── Train Single Client ────────────────── #
def train_client(
    dataset: str,
    cid: int,
    cname: str,
    device,
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    patience: int,
    delta: float,
    window_size: int,
):
    """
    Train MDN generator for single client cid and save best model at lowest loss.
    """
    # ───── Load all real features for this client ─────
    X, Y = load_features(dataset, cid)
    num_classes = config.DATASETS[dataset]['num_classes']
    feature_dim = config.MODEL_CONFIG['feature_dim']

    # ───── Calculate number of proxies N2 ─────
    n = (config.MODEL_PATCHING.get('patch_split_n')
         if hasattr(config, 'MODEL_PATCHING') and 'patch_split_n' in config.MODEL_PATCHING
         else config.MODEL_CONFIG.get('proxy_grid', 0))
    N2 = (n + 1) * (n + 1)

    # ───── Construct global proxy / structured proxy ─────
    global_proxy = X[:, 0, :].unsqueeze(1)
    structured_proxy = X
    upsampled_global = global_proxy.repeat(1, N2, 1)

    # ───── Create training dataset ─────
    Y_onehot = torch.nn.functional.one_hot(Y, num_classes).float()
    ds = TensorDataset(Y_onehot, upsampled_global, structured_proxy)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    # ───── Initialize MDN generator model ─────
    model = MDNGenerator(
        cond_dim=num_classes,
        feature_dim=feature_dim,
        num_mixtures=config.MODEL_CONFIG['stage2'].get('num_mixtures', 32),
        depth=config.MODEL_CONFIG['stage2'].get('depth', 4),
        hidden_dim=config.MODEL_CONFIG['stage2'].get('hidden_dim', 768),
        num_heads=config.MODEL_CONFIG['stage2'].get('num_heads', 8),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()

    best_loss = float('inf')
    no_improve = 0

    # ───── Prepare save path ─────
    save_dir = Path(config.PATHS['local_fgenerator_weights'](dataset, cid))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_fgenerator.pth"

    # ───── Start training loop ─────
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for cond, up_g, target in loader:
            cond = cond.to(device, non_blocking=True)
            up_g = up_g.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                # For diagonal covariance version, use these two lines
                pi, mu, logv = model(cond, up_g)                   # MDN output
                loss = model.nll_loss(target, pi, mu, logv)       # NLL Loss
                # loss = model.nll_dro_loss(target, pi, mu, logv) # Introduce loss variance
                
                # pi, mu, L, log_diag = model(cond, upsampled_global=up_g)
                # # loss = model.nll_loss(target, pi, mu, L, log_diag) # Fallback to mean loss only
                # loss = model.nll_dro_loss(target, pi, mu, L, log_diag) # Introduce loss variance
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[GPU {device.index}] {cname}(Client{cid}) Epoch {epoch}/{epochs} Loss: {avg_loss:.4f}")

        # ───── Save model if loss improves ─────
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"[GPU {device.index}] {cname}(Client{cid}) improved Loss: {best_loss:.4f}, saved → {best_model_path}")
            no_improve = 0
        else:
            no_improve += 1

        # ───── Early-stopping check ─────
        if no_improve > patience:
            print(f"[GPU {device.index}] {cname}(Client{cid}) Early-stop ({patience} epochs no gain)")
            break

    print(f"[GPU {device.index}] {cname}(Client{cid}) best Loss: {best_loss:.4f}")


# ────────────────── Worker Process Entry ────────────────── #
def worker_process(
    gpu_local_idx: int,
    assigned: List[Tuple[int, str]],
    dataset: str,
    epochs: int,
    lr: float,
    batch_size: int,
    num_workers: int,
    patience: int,
    delta: float,
    window_size: int,
):
    """
    Single subprocess trains assigned clients on specified GPU (local idx).
    """
    torch.cuda.set_device(gpu_local_idx)  # local idx: index in visible GPU list
    device = torch.device(f"cuda:{gpu_local_idx}" if torch.cuda.is_available() else "cpu")
    print(f"[Worker] cuda:{gpu_local_idx} ← Clients {assigned}")

    for cid, cname in assigned:
        train_client(
            dataset,
            cid,
            cname,
            device,
            epochs,
            lr,
            batch_size,
            num_workers,
            patience,
            delta,
            window_size
        )


"""
chmod +x run_all_datasets_s2.0.sh
./run_all_datasets_s2.0.sh
"""
# ────────────────────── Main Program ────────────────────── #
def main():
    parser = argparse.ArgumentParser(description="Stage2 MDN Generator (multiprocess)")
    # —— Dataset Selection —— 
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default=None,
        help="Single dataset name (ignored if --all_datasets is used)"
    )
    parser.add_argument(
        "--all_datasets",
        action="store_true",
        default=False,
        help="Iterate through all datasets in config.DATASETS"
    ) # Now replaced by script (parallel training all clients), consider enabling only when GPU memory is scarce
    # —— Training Parameters —— 
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size per DataLoader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience")
    parser.add_argument("--delta", type=float, default=0.2, help="Threshold for smooth_loss vs best_loss difference")
    parser.add_argument("--window", type=int, default=10, help="Sliding window size for smoothing loss")
    args = parser.parse_args(remaining_argv)  # Parse remaining argv

    # Prepare visible GPU list
    visible_gpu_cnt = torch.cuda.device_count()
    if args_early.gpus == "-1" or visible_gpu_cnt == 0:
        print("Using CPU only")
        return

    # local GPU idx list: [0, 1, ..., visible_gpu_cnt-1]
    gpu_ids = list(range(visible_gpu_cnt))
    print(f"[Main] Visible GPUs: {gpu_ids}  ← ({args_early.gpus})")

    # Prepare dataset list to run
    if args.all_datasets:
        datasets_to_run = list(config.DATASETS.keys())
    else:
        ds0 = args.dataset or config.DATASET
        if ds0 not in config.DATASETS:
            raise ValueError(f"Unknown dataset '{ds0}'. Available: {list(config.DATASETS.keys())}")
        datasets_to_run = [ds0]

    # Launch multiprocess for each dataset
    for dataset in datasets_to_run:
        epochs = config.MODEL_CONFIG['stage2']['epochs']
        lr = config.MODEL_CONFIG['stage2']['learning_rate']
        client_list = config.DATASETS[dataset]['clients']

        # Evenly distribute clients across visible GPUs
        assigns = [[] for _ in gpu_ids]
        for idx, cname in enumerate(client_list):
            assigns[idx % visible_gpu_cnt].append((idx, cname))

        mp.set_start_method("spawn", force=True)
        procs = []
        for gpu_idx, task in zip(gpu_ids, assigns):
            p = mp.Process(
                target=worker_process,
                args=(
                    gpu_idx,
                    task,
                    dataset,
                    epochs,
                    lr,
                    args.batch_size,
                    args.num_workers,
                    args.patience,
                    args.delta,
                    args.window
                )
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


if __name__ == "__main__":
    main()