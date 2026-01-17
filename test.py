#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage-3 Test
- Supports single dataset (--dataset) or all datasets (--all_datasets)
- In --all_datasets mode, disables progress bars and shows summary table
"""

import os
import argparse
import logging
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

import config

# Suppress all warnings
warnings.filterwarnings("ignore")

# ───────────────────────── Dataset ───────────────────────── #
class FeatureDataset(TorchDataset):
    def __init__(self, feature_dir: str, num_classes: int):
        """Load features from directory structure:
        feature_dir/
            ├── 0/  # class 0
            │   ├── 0.npy
            │   └── ...
            ├── 1/  # class 1
            │   └── ...
        """
        if not os.path.isdir(feature_dir):
            raise FileNotFoundError(f"Test feature dir not found: {feature_dir}")
        self.samples = []
        for label in range(num_classes):
            class_dir = os.path.join(feature_dir, str(label))
            if os.path.isdir(class_dir):
                for fn in os.listdir(class_dir):
                    if fn.endswith(".npy"):
                        self.samples.append((os.path.join(class_dir, fn), label))
        if not self.samples:
            raise RuntimeError(f"No test features under {feature_dir}")

    def __len__(self):  
        return len(self.samples)
        
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return torch.from_numpy(np.load(path)).float(), label


# ───────────────────────── Logger ───────────────────────── #
def setup_logger(path: str):
    """Configure logger with file and console handlers"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh, ch = logging.FileHandler(path), logging.StreamHandler()
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ───────────────────────── Evaluation ───────────────────────── #
@torch.no_grad()
def evaluate(loader, classifier, device, num_classes, silent=False):
    """Evaluate model and return metrics dict and sample count"""
    classifier.eval()
    preds, gts = [], []
    for feats, labels in tqdm(loader, desc="Testing", dynamic_ncols=True, disable=silent):
        feats, labels = feats.to(device), labels.to(device)
        logits = classifier(feats)
        preds += logits.argmax(1).cpu().tolist()
        gts += labels.cpu().tolist()

    metrics = {
        "acc": accuracy_score(gts, preds),
        "prec": precision_score(gts, preds, average="weighted", zero_division=0)
    }
    
    if num_classes == 2:
        metrics.update({
            "rec": recall_score(gts, preds, pos_label=1, average="binary", zero_division=0),
            "f1": f1_score(gts, preds, pos_label=1, average="binary", zero_division=0)
        })
    else:
        metrics.update({
            "rec": recall_score(gts, preds, average="weighted", zero_division=0),
            "f1": f1_score(gts, preds, average="weighted", zero_division=0)
        })
        
    return metrics, len(gts)


# ───────────────────────── Test Single Dataset ───────────────────────── #
def run_on_dataset(dataset: str, args, silent=False):
    """Test single dataset and return aggregated (globally weighted) metrics"""
    num_classes = config.DATASETS[dataset]['num_classes']
    log_path = config.PATHS['logs'](dataset, stage='test_clients_strict')
    logger = setup_logger(log_path)
    logger.info(f"Dataset={dataset} | STRICT TEST | GPU={args.device} | FeatureType={args.feature_type}")

    # Load global classifier
    from models.global_classifier import GlobalClassifier
    feat_dim = config.MODEL_CONFIG['feature_dim']
    ckpt = os.path.join(
        config.PATHS['global_classifier_weights'](dataset), 
        args.feature_type, 
        "model_weights.pth"
    )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Classifier checkpoint not found: {ckpt}")
        
    clf = GlobalClassifier(
        feat_dim, 
        num_classes,
        use_moe=config.MODEL_CONFIG['stage3']['use_moe'],
        num_experts=config.MODEL_CONFIG['stage3']['num_experts']
    ).to(args.device)
    
    clf.load_state_dict(torch.load(ckpt, map_location=args.device))
    clf.eval()
    logger.info(f"Loaded classifier from {ckpt}")

    # Test each client
    client_metrics, sample_counts = [], []
    for cid, cname in enumerate(config.DATASETS[dataset]['clients']):
        logger.info(f"-- Client{cid} ({cname}) --")
        feat_dir = config.PATHS['test_feature'](dataset, cid)
        test_set = FeatureDataset(feat_dir, num_classes)
        loader = DataLoader(
            test_set,
            batch_size=config.MODEL_CONFIG['stage3']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=(args.device.type == 'cuda')
        )
        metrics, count = evaluate(loader, clf, args.device, num_classes, silent=silent)
        logger.info(
            f"Acc={metrics['acc']*100:.2f}%, Prec={metrics['prec']*100:.2f}%, "
            f"Rec={metrics['rec']*100:.2f}%, F1={metrics['f1']*100:.2f}% | N={count}"
        )
        client_metrics.append(metrics)
        sample_counts.append(count)

    # Aggregate results
    total_samples = sum(sample_counts)
    weighted_metrics = {
        k: sum(m[k]*c for m, c in zip(client_metrics, sample_counts))/total_samples
        for k in ['acc', 'prec', 'rec', 'f1']
    }
    
    logger.info("== Aggregated (Global Weighted) ==")
    logger.info(", ".join([f"{k.upper()}={v*100:.2f}%" for k,v in weighted_metrics.items()]))
    return weighted_metrics


# ───────────────────────── Main Program ───────────────────────── #
def main():
    parser = argparse.ArgumentParser("Stage-3 Strict Test (Feature-Only Evaluation)")

    # Dataset options
    parser.add_argument('--dataset', type=str, help="Single dataset name")
    parser.add_argument('--all_datasets', action='store_true', 
                        help="Run on all available datasets")

    # Feature type & reproducibility
    parser.add_argument('--feature_type', type=str, 
                        choices=['real', 'synthetic', 'merged'], 
                        default='merged',
                        help="Type of features to test on")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducibility")

    # Device setting
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU index to use")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(args.seed)
        
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Determine datasets to run
    if args.all_datasets:
        datasets_to_run = list(config.DATASETS.keys())
    else:
        ds = args.dataset or config.DATASET
        if ds not in config.DATASETS:
            raise ValueError(f"Unknown dataset '{ds}'. Available: {list(config.DATASETS.keys())}")
        datasets_to_run = [ds]

    # Run evaluation on each dataset
    summary = {}
    for ds in datasets_to_run:
        metrics = run_on_dataset(ds, args, silent=args.all_datasets)
        summary[ds] = metrics

    # Print summary table in --all_datasets mode
    if args.all_datasets:
        print("\n=== Stage-3 Strict Test Summary ===")
        header = f"{'Dataset':<15}{'ACC%':>8}{'PREC%':>8}{'REC%':>8}{'F1%':>8}"
        print(header)
        print("-" * len(header))
        for ds, metrics in summary.items():
            print(f"{ds:<15}"
                  f"{metrics['acc']*100:8.2f}"
                  f"{metrics['prec']*100:8.2f}"
                  f"{metrics['rec']*100:8.2f}"
                  f"{metrics['f1']*100:8.2f}")
        print("-" * len(header))

if __name__ == "__main__":
    main()