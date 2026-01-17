#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FALCON Stage-1.5 – Local Classifier Training
"""

# ───────────── GPU Selection (Must be before torch import) ─────────────
import os, sys, argparse
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--gpu', type=str, default='3')
_pre_args, _ = _pre.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.gpu

# ───────────── Imports ─────────────
import glob, logging, numpy as np, warnings, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config                                        # Using the same config
from models.global_classifier import GlobalClassifier  # Reusing existing classifier module

warnings.filterwarnings("ignore")

# ───────────────────── Logger ─────────────────────
def get_logger(log_path):
    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not logger.handlers:
        fh = logging.FileHandler(log_path)
        ch = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(fmt);  ch.setFormatter(fmt)
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

# ───────────────────── Dataset ─────────────────────
class FeatureDataset(Dataset):
    def __init__(self, feat_dirs: list, num_classes: int):
        """
        feat_dirs: list of directories, each containing .npy feature files organized by class subfolders
        """
        self.records = []
        for cid, base in enumerate(feat_dirs):
            for cls in range(num_classes):
                cls_dir = os.path.join(base, str(cls))
                if not os.path.isdir(cls_dir):
                    continue
                for path in glob.glob(f"{cls_dir}/*.npy"):
                    self.records.append((path, cls))
        # Final self.records is a list of [(filepath, label), ...]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        feat = np.load(path).astype(np.float32)
        feat = torch.from_numpy(feat)  # Return as tensor directly
        return feat, torch.tensor(label, dtype=torch.long)

# ───────────────────── Train & Validate (Only called in non-test mode) ─────────────────────
def train_and_validate_local_head(ds_name: str, cid: int, args, device: torch.device):
    """
    Train classifier head on single client's real_feature and validate on val_feature.
    """
    num_classes = config.DATASETS[ds_name]['num_classes']
    feat_dim    = config.MODEL_CONFIG['feature_dim']

    # Local training/validation directories
    train_dir = config.PATHS['real_feature'](ds_name, cid)
    val_dir   = config.PATHS['val_feature'] (ds_name, cid)

    # Create Dataset/DataLoader
    tr_ds = FeatureDataset([train_dir], num_classes)
    if len(tr_ds) == 0:
        # logger.warning(f"[Stage‑1.5] Dataset={ds_name} | Client={cid} "
        #                f"has **no training samples**, skip training.")
        return None
    va_ds = FeatureDataset([val_dir],   num_classes)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Create lightweight classifier head: using GlobalClassifier but disabling MoE
    model = GlobalClassifier(feature_dim=feat_dim, num_classes=num_classes,
                             use_moe=False, num_experts=0).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Logging & weight saving directory
    logger = get_logger(config.PATHS['logs'](ds_name, stage='1.5', cid=cid))
    save_dir = config.PATHS['local_classifier_weights'](ds_name, cid)
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"[Stage-1.5 TRAIN] Dataset={ds_name} | Client={cid} | "
                f"TrainSamples={len(tr_ds)} | ValSamples={len(va_ds)}")

    best_acc = 0.0
    for ep in range(args.epochs):
        # ------- Training -------
        model.train()
        total = correct = 0
        loss_sum = 0.0
        for features, labels in tqdm(tr_ld, desc=f"Client{cid}-Train Ep{ep+1}", leave=False):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss = loss_sum / len(tr_ld)
        train_acc  = 100.0 * correct / total

        # ------- Validation -------
        model.eval()
        total = correct = 0
        loss_val = 0.0
        with torch.no_grad():
            for features, labels in tqdm(va_ld, desc="Validation", leave=False):
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss_val += loss.item()
                total += labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_loss = loss_val / len(va_ld)
        val_acc  = 100.0 * correct / total

        logger.info(f"Epoch {ep+1}/{args.epochs} | "
                    f"Train L={train_loss:.4f} Acc={train_acc:.2f}% | "
                    f" Val   L={val_loss:.4f} Acc={val_acc:.2f}%")

        # Save best weights if validation accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, "linear_head.pth"))
            logger.info(f"  ↳ Saved best weights (ValAcc={best_acc:.2f}%)")

    return best_acc

# ───────────────────── Test-Only Evaluation ─────────────────────
def test_only_local_head(ds_name: str, cid: int, args, device: torch.device):
    """
    Evaluate this client's classifier head on ALL clients' test sets (no training).
    """
    num_classes = config.DATASETS[ds_name]['num_classes']
    feat_dim    = config.MODEL_CONFIG['feature_dim']

    # Load saved classifier weights
    save_dir = config.PATHS['local_classifier_weights'](ds_name, cid)
    weight_path = os.path.join(save_dir, "linear_head.pth")
    logger = get_logger(config.PATHS['logs'](ds_name, stage='1.5', cid=cid))

    if not os.path.isfile(weight_path):
        logger.warning(f"No pre-trained weights found at {weight_path}. Skipping test for Client{cid}.")
        return None

    # Create model and load weights
    model = GlobalClassifier(feature_dim=feat_dim, num_classes=num_classes,
                             use_moe=False, num_experts=0).to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Collect all clients' test_feature directories
    all_test_dirs = [
        config.PATHS['test_feature'](ds_name, other_cid)
        for other_cid in range(len(config.DATASETS[ds_name]['clients']))
    ]
    test_ds = FeatureDataset(all_test_dirs, num_classes)
    test_ld = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    total = correct = 0
    with torch.no_grad():
        for features, labels in tqdm(test_ld, desc=f"Client{cid}-TestAll", leave=False):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    test_acc = 100.0 * correct / total if total > 0 else 0.0
    logger.info(f"[Test-Only] Client{cid} on ALL test samples: Acc={test_acc:.2f}%")
    return test_acc

# ───────────────────── Main Workflow ─────────────────────
def run_on_client(ds_name: str, cid: int, args, device: torch.device):
    """
    If args.test=True, go directly to test_only; otherwise train_and_validate first.
    """
    if args.test:
        return test_only_local_head(ds_name, cid, args, device)
    else:
        return train_and_validate_local_head(ds_name, cid, args, device)



"""
chmod +x run_all_datasets_s1.5.sh
./run_all_datasets_s1.5.sh
"""

# ─────────────────────── Main ───────────────────────
def main():
    parser = argparse.ArgumentParser("Stage-1.5 Local Classifier (Includes Test-Only Mode)")
    parser.add_argument('--dataset', default=None,
                        help='Single dataset name (overrides config.DATASET)')
    parser.add_argument('--all_datasets', action='store_true', default=False, # Default False, True functionality implemented by .sh script
                        help='Run all datasets at once')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--test', action='store_true', 
                        # default=True,
                        help='Enable Test-Only mode: Skip training, directly load classifier weights and evaluate test set')

    args = parser.parse_args()

    # Set random seeds
    SEED = 1
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Determine dataset list to run
    if args.all_datasets:
        ds_list = list(config.DATASETS.keys())
    else:
        ds_list = [args.dataset or config.DATASET]

    # Record test accuracy for each dataset and client
    # results[dataset] = [ acc_client0, acc_client1, ... ]
    results = {}

    for ds_name in ds_list:
        n_clients = len(config.DATASETS[ds_name]['clients'])
        results[ds_name] = []
        print(f"\n================ Dataset={ds_name} Test-Only Mode ================\n")
        for cid in range(n_clients):
            acc = run_on_client(ds_name, cid, args, device)
            # Returns None in train mode; returns float in test mode
            results[ds_name].append(acc)
            
    # === Check if all results for this dataset are None ===
    if all(r is None for r in results[ds_name]):
         raise RuntimeError(
            f"[Stage‑1.5] Dataset '{ds_name}' has no training samples "
            f"for any of its {n_clients} clients. Please check data paths or preprocessing.")

    # If in Test-Only mode, print a "dataset × client" table with 2 decimal places
    if args.test:
        # Print header: Dataset name + Client columns + "Mean"
        all_clients = max(len(config.DATASETS[ds]['clients']) for ds in ds_list)
        header = ["Dataset"] + [f"Client{i}" for i in range(all_clients)] + ["Mean"]
        # First print a tabular row
        print("\n" + " | ".join(f"{h:^10}" for h in header))
        print("-" * (13 * (len(header))))

        # Output each dataset row by row
        for ds_name in ds_list:
            accs = results[ds_name]
            # Pad with "--" for missing clients
            padded = [f"{acc:.2f}%" if acc is not None else "--" for acc in accs]
            padded += ["--"] * (all_clients - len(accs))
            # Calculate row mean (only for actual numbers), output "--" if all empty
            valid = [a for a in accs if a is not None]
            mean_val = f"{(np.mean(valid)):.2f}%" if valid else "--"
            row = [f"{ds_name:^10}"] + [f"{v:^10}" for v in padded] + [f"{mean_val:^10}"]
            print(" | ".join(row))

        print("\n")

if __name__ == "__main__":
    main()