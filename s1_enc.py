#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FALCON Stage-1 HSE Script
"""

import os
import sys
import argparse

# ───── Early GPU argument parsing and environment setup ───── #
parser = argparse.ArgumentParser(description="Stage1 Feature Encoding (GPU-aware)", add_help=False)
parser.add_argument("--gpus", default="2", help="GPU IDs, e.g. '0,1,2'; '-1' for CPU")
args, remaining_args = parser.parse_known_args()

if args.gpus != "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ───── Import remaining libraries (torch affected by CUDA_VISIBLE_DEVICES) ───── #
import shutil
import glob
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.encoders import build_encoder
from dataset import Dataset
import config

# ───────────────────── Utility Functions ────────────────────── #
def parse_gpu_ids(gpu_str: str) -> List[int]:
    return [int(x) for x in gpu_str.split(',') if x.strip().isdigit()]

def clear_and_mkdir(path: str):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except Exception as e:
            tqdm.write(f"[WARN] rmtree failed ({e}), cleaning files individually...")
            for f in glob.glob(os.path.join(path, '**', '*'), recursive=True):
                if os.path.isfile(f):
                    try:
                        os.remove(f)
                    except Exception as ee:
                        tqdm.write(f"  ↳ Failed to delete file {f}: {ee}")
            for root, dirs, _ in os.walk(path, topdown=False):
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
    os.makedirs(path, exist_ok=True)

# ───────────────────── Core Logic ────────────────────── #
def extract_and_save_features(
    extractor: nn.Module,
    loader: DataLoader,
    save_dir: str,
    device: torch.device,
    num_classes: int,
    keep_filename: bool = False,
):
    if os.path.exists(save_dir):
        count = sum(len(files) for _, _, files in os.walk(save_dir))
        if count > 0:
            ans = input(f"[WARN] Directory {save_dir} contains {count} files, continue and overwrite? (y/N): ")
            if ans.strip().lower() not in ('y', 'yes', '1'):
                tqdm.write(f"[INFO] Skipping feature extraction: {save_dir}")
                return

    for lbl in range(num_classes):
        os.makedirs(os.path.join(save_dir, str(lbl)), exist_ok=True)

    counters = {i: 0 for i in range(num_classes)}
    printed_shape = False

    extractor.eval()
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Extracting → {os.path.basename(save_dir)}", dynamic_ncols=True)
        for batch in pbar:
            if keep_filename:
                inputs, labels, fnames = batch
            else:
                inputs, labels = batch
                fnames = [None] * len(labels)

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            feats = extractor(inputs)

            if not printed_shape:
                shape_str = (
                    f"{tuple(feats.shape)} (B, N, D)" if feats.ndim == 3 else f"{tuple(feats.shape)} (B, D)"
                )
                tqdm.write(f"[Info] Extracted features: {shape_str}")
                printed_shape = True

            for feat, lbl, fname in zip(feats, labels, fnames):
                out_dir = os.path.join(save_dir, str(lbl.item()))
                if keep_filename and fname is not None:
                    base = os.path.splitext(fname)[0]
                    out_path = os.path.join(out_dir, f"{base}.npy")
                else:
                    out_path = os.path.join(out_dir, f"{counters[lbl.item()]}.npy")
                    counters[lbl.item()] += 1
                np.save(out_path, feat.cpu().numpy())

# ───────────────────── Main Entry ────────────────────── #
def main():
    parser = argparse.ArgumentParser(description="Stage1 Feature Encoding")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--gpus", default="2", help="GPU IDs, e.g. '0,1,2'; '-1' for CPU")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size override")
    parser.add_argument("--keep-filename", action="store_true", default=True, help="Save features using original image filenames")
    args = parser.parse_args()

    gpu_ids = [] if args.gpus == "-1" else parse_gpu_ids(args.gpus)
    if torch.cuda.is_available() and gpu_ids:
        device = torch.device("cuda:0")
        print(f"Using GPUs: {args.gpus} (mapped as cuda:0…{len(gpu_ids)-1})")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    dataset_name = config.DATASET if args.dataset is None else args.dataset
    ds_conf = config.DATASETS[dataset_name]
    num_classes, resolution = ds_conf["num_classes"], ds_conf["resolution"]
    client_ids = range(len(ds_conf["clients"]))

    batch_size = args.batch_size
    print(f"Batch size: {batch_size}")

    extractor = build_encoder(config)
    if device.type == "cuda" and len(gpu_ids) > 1:
        extractor = nn.DataParallel(extractor, device_ids=list(range(len(gpu_ids))))
        print(f"DataParallel on {len(gpu_ids)} GPUs")
    extractor = extractor.to(device)

    preproc_dir = config.PATHS["preprocessed"](dataset_name)
    for cid in client_ids:
        cname = ds_conf["clients"][cid]
        tqdm.write(f"\n>>> [{dataset_name}] Client{cid} ({cname}) …")

        for split, key in zip(["train", "val", "test"], ["real_feature", "val_feature", "test_feature"]):
            data_dir = os.path.join(preproc_dir, f"{cname}_{split}")
            save_dir = config.PATHS[key](dataset_name, cid)
            loader = DataLoader(
                Dataset(data_dir, num_classes, resolution, use_augment=False, return_filename=args.keep_filename),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=device.type == "cuda",
            )
            extract_and_save_features(
                extractor, loader, save_dir, device, num_classes, keep_filename=args.keep_filename
            )

if __name__ == "__main__":
    main()