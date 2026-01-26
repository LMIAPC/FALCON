#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FALCON Stage 3 – Global Classifier Training
"""

# ─────────────── GPU Selection (before importing torch) ───────────────
import os
import sys
import argparse

pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument('--gpu', type=str, default='3')
pre_args, _ = pre_parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = pre_args.gpu

# ─────────────── Imports ───────────────
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from models.global_classifier import GlobalClassifier

import warnings
warnings.filterwarnings("ignore")

# ───────────────────── Logger Helper ─────────────────────

def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # 防止重复 handler
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path) for h in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        console_handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        for h in (file_handler, console_handler):
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger

# ───────────────────── EMA Helper ─────────────────────

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow, self.backup = {}, {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1. - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# ───────────────────── Dataset ─────────────────────

class FeatureDataset(Dataset):
    def __init__(self, feature_dirs, num_classes, use_dp=False, dp_noise_std=1.0):
        self.records = []
        for cid, fdir in enumerate(feature_dirs):
            for cls in range(num_classes):
                cls_dir = os.path.join(fdir, str(cls))
                if not os.path.exists(cls_dir):
                    continue
                files = glob.glob(os.path.join(cls_dir, '*.npy'))
                self.records.extend([(p, cls, cid) for p in files])
        self.use_dp = use_dp
        self.dp_noise_std = dp_noise_std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label, cid = self.records[idx]
        feat = np.load(path)
        if feat.ndim == 1:
            feat = feat[np.newaxis, :]
        if self.use_dp:
            feat = feat + np.random.normal(0., self.dp_noise_std, size=feat.shape)
        feat = torch.tensor(feat, dtype=torch.float32)
        return feat, torch.tensor(label, dtype=torch.long), torch.tensor(cid, dtype=torch.long)

# ───────────────────── KD Utilities ─────────────────────

def load_teacher_models(ds_name, cids, feat_dim, num_classes, device):
    """Load all local classifiers and return an evaluation‑mode ensemble list."""
    teacher_models = []
    for cid in cids:
        wdir = config.PATHS['local_classifier_weights'](ds_name, cid)
        weight_file = os.path.join(wdir, 'linear_head.pth')
        if not os.path.isfile(weight_file):
            continue
        model_t = GlobalClassifier(feature_dim=feat_dim, num_classes=num_classes, use_moe=False).to(device)
        model_t.load_state_dict(torch.load(weight_file, map_location=device))
        model_t.eval()
        teacher_models.append(model_t)
    return teacher_models

# @torch.no_grad()
# def evaluate_ensemble(models, loader, num_classes, device):
#     if len(models) == 0:
#         return 0.0
#     total, correct = 0, 0
#     for feats, labels, _ in loader:
#         feats, labels = feats.to(device), labels.to(device)
#         probs = torch.zeros(feats.size(0), num_classes, device=device)
#         for m in models:
#             probs += torch.softmax(m(feats), dim=1)
#         probs /= len(models)
#         pred = probs.argmax(1)
#         correct += pred.eq(labels).sum().item()
#         total += labels.size(0)
#     return 100. * correct / total

@torch.no_grad()
def evaluate_ensemble(models, loader, num_classes, device, weights=None):
    if len(models) == 0:
        return 0.0
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    total, correct = 0, 0
    for feats, labels, _ in loader:
        feats, labels = feats.to(device), labels.to(device)
        probs = torch.zeros(feats.size(0), num_classes, device=device)
        for w, m in zip(weights, models):
            probs += w * torch.softmax(m(feats), dim=1)
        pred = probs.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100. * correct / total


# ───────────────────── Train & Validate ─────────────────────

# def train_epoch(model, loader, optimizer, device, epoch, logger,
#                 teacher_models=None, kd_alpha=0.2, kd_temp=2.0):
#     model.train()
#     total, correct, loss_sum = 0, 0, 0.0
#     for feats, labels, _ in tqdm(loader, desc=f"Epoch {epoch+1} Training"):
#         feats, labels = feats.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(feats)
#         ce_loss = F.cross_entropy(outputs, labels)
#         loss = ce_loss
#         # KD branch
#         if teacher_models:
#             with torch.no_grad():
#                 teacher_probs = torch.zeros_like(outputs)
#                 for m in teacher_models:
#                     teacher_probs += torch.softmax(m(feats) / kd_temp, dim=1)
#                 teacher_probs /= len(teacher_models)
#             kd_loss = F.kl_div(torch.log_softmax(outputs / kd_temp, dim=1),
#                                teacher_probs, reduction='batchmean') * (kd_temp ** 2)
#             loss = (1 - kd_alpha) * ce_loss + kd_alpha * kd_loss
#         loss.backward()
#         optimizer.step()
#         loss_sum += loss.item()
#         correct += outputs.argmax(1).eq(labels).sum().item()
#         total += labels.size(0)
#     return loss_sum / len(loader), 100. * correct / total

def train_epoch(model, loader, optimizer, device, epoch, logger,
                teacher_models=None, kd_weights=None,
                kd_alpha=0.2, kd_temp=2.0):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for feats, labels, _ in tqdm(loader, desc=f"Epoch {epoch+1} Training"):
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(feats)
        ce_loss = F.cross_entropy(outputs, labels)
        loss = ce_loss
        if teacher_models:
            if kd_weights is None:
                kd_weights = [1.0 / len(teacher_models)] * len(teacher_models)
            with torch.no_grad():
                teacher_probs = torch.zeros_like(outputs)
                for w, m in zip(kd_weights, teacher_models):
                    teacher_probs += w * torch.softmax(m(feats) / kd_temp, dim=1)
            kd_loss = F.kl_div(torch.log_softmax(outputs / kd_temp, dim=1),
                               teacher_probs, reduction='batchmean') * (kd_temp ** 2)
            loss = (1 - kd_alpha) * ce_loss + kd_alpha * kd_loss
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return loss_sum / len(loader), 100. * correct / total

@torch.no_grad()
def validate(model, loader, device, logger):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for feats, labels, _ in tqdm(loader, desc="Validation"):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)
        loss_sum += F.cross_entropy(outputs, labels).item()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    acc = 100. * correct / total
    logger.info(f"Validation Loss: {loss_sum / len(loader):.6f}, Acc: {acc:.2f}%")
    return loss_sum / len(loader), acc

# ───────────────────── Helper to run one dataset ─────────────────────

def run_on_dataset(ds_name, args, device, seed):
    """Train and validate on a single dataset with optional KD."""
    num_classes = config.DATASETS[ds_name]['num_classes']
    feat_dim = config.MODEL_CONFIG['feature_dim']
    stage3_cfg = config.MODEL_CONFIG['stage3']

    batch_size = stage3_cfg['batch_size']
    lr = stage3_cfg['learning_rate']
    epochs = args.epochs or stage3_cfg['epochs']
    use_moe = stage3_cfg['use_moe']
    num_experts = stage3_cfg['num_experts']
    
    use_dp = args.use_dp
    dp_noise_std = args.dp_noise_std
    kd_alpha = args.kd_alpha
    kd_temp = args.kd_temp

    ema_cfg = config.MODEL_CONFIG.get('stage3_ema', {})
    ema_enabled = ema_cfg.get('enabled', False)
    ema_decay = ema_cfg.get('decay', 0.999)

    log_path = config.PATHS['logs'](ds_name, stage=3)
    logger = setup_logger(log_path)
    logger.info("\n" + "=" * 80)
    patch_window = tuple(config.MODEL_CONFIG.get('patch_window', (224, 224)))
    patch_stride = tuple(config.MODEL_CONFIG.get('patch_stride', (224, 224)))
    logger.info(f"Stage3 | Dataset={ds_name} | KD={'ON' if args.distill else 'OFF'} | AttentionPooling={config.MODEL_CONFIG['aggregation']['method']} | Patch_window={patch_window} | Patch_stride={patch_stride}")

    cids = list(range(len(config.DATASETS[ds_name]['clients'])))
    logger.info(f"Using all clients: {cids}")

    feat_key = {'real': 'real_feature', 'synthetic': 'synthetic_feature', 'merged': 'merged_feature'}[args.data_type]
    train_dirs = [config.PATHS[feat_key](ds_name, cid) for cid in cids]
    val_dirs = [config.PATHS['val_feature'](ds_name, cid) for cid in cids]

    train_ds = FeatureDataset(train_dirs, num_classes, use_dp=use_dp, dp_noise_std=dp_noise_std)
    val_ds = FeatureDataset(val_dirs, num_classes, use_dp=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ───────────── Load teacher ensemble when KD is enabled ─────────────
    kd_weights = None
    if args.kd_weight_by_size:
        client_sizes = [len(glob.glob(os.path.join(fdir, '*', '*.npy')))
                        for fdir in train_dirs]
        sizes = np.array(client_sizes, dtype=float) + 1e-8       # 防零
        # tau   = max(args.kd_weight_temp, 1e-6)                   # 防 NaN
        # sizes = sizes ** tau                                     # n_i^τ
        kd_weights = (sizes / sizes.sum()).tolist()

    
    teacher_models = []

    if args.distill:
        teacher_models = load_teacher_models(ds_name, cids, feat_dim, num_classes, device)
        if teacher_models:
            teacher_acc = evaluate_ensemble(
                teacher_models, val_loader, num_classes, device,
                weights=kd_weights)
            logger.info(f"Teacher Ensemble Acc on validation set: {teacher_acc:.2f}% "
                        f"(models={len(teacher_models)}, "
                        f"KD weights = {'size‑based' if kd_weights else 'uniform'})")
        else:
            logger.warning("KD enabled but no local classifier weights found – proceeding without KD.")
            args.distill = False

    # ───────────── Student / Global classifier ─────────────
    model = GlobalClassifier(feature_dim=feat_dim, num_classes=num_classes, use_moe=use_moe, num_experts=num_experts).to(device)
    
    if args.init_from_local_mean:
        local_models = load_teacher_models(ds_name, cids, feat_dim, num_classes, device)
        if local_models:
            logger.info(f"Initializing global model from {len(local_models)} local classifiers (mean weights).")
            avg_state_dict = {}
            model_keys = model.state_dict().keys()
            for key in model_keys:
                params = [m.state_dict()[key] for m in local_models if key in m.state_dict()]
                if len(params) > 0:
                    avg_state_dict[key] = torch.mean(torch.stack(params, dim=0), dim=0)
            model.load_state_dict(avg_state_dict, strict=False)
        else:
            logger.warning("`--init_from_local_mean` is set, but no local models found. Using random init.")
    
    ema = EMA(model, decay=ema_decay) if ema_enabled else None
    optimizer = optim.Adam(model.parameters(), lr=lr)

    save_dir = os.path.join(config.PATHS['global_classifier_weights'](ds_name), args.data_type)
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))

    best_acc = 0.
    try:
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, epoch, logger,
                teacher_models=teacher_models if args.distill else None,
                kd_weights=kd_weights if args.distill else None,
                kd_alpha=kd_alpha, kd_temp=kd_temp)

            if ema:
                ema.apply_shadow()
            val_loss, val_acc = validate(model, val_loader, device, logger)
            if ema:
                ema.restore()

            writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch)
            writer.flush()

            logger.info(f"Epoch {epoch + 1}/{epochs} | TrainAcc={train_acc:.2f}% | ValAcc={val_acc:.2f}%")
            if val_acc > best_acc and epoch + 1 >= args.min_save_epoch:
                best_acc = val_acc
                if ema:
                    ema.apply_shadow()
                torch.save(model.state_dict(), os.path.join(save_dir, 'model_weights.pth'))
                if ema:
                    ema.restore()
                logger.info(f"Saved best model @ Epoch {epoch + 1} (Acc={best_acc:.2f}%)")

    except KeyboardInterrupt:
        ckpt = os.path.join(save_dir, f'interrupt_epoch{epoch + 1}.pth')
        torch.save(model.state_dict(), ckpt)
        logger.info(f"Training interrupted. Weights saved to {ckpt}")

    finally:
        writer.close()
        logger.info(f"Training done for dataset {ds_name}. Best ValAcc={best_acc:.2f}%\n")

# ───────────────────── Main Entry ─────────────────────
"""
chmod +x run_all_datasets_s3.sh
./run_all_datasets_s3.sh
"""
def main():
    parser = argparse.ArgumentParser()
    # Dataset options
    parser.add_argument('--dataset', type=str, required=True,)
    parser.add_argument('--all_datasets', action='store_true', default=False,)
    parser.add_argument('--data_type', type=str, default='merged', choices=['real', 'synthetic', 'merged'],)

    # Training schedule
    parser.add_argument('--epochs', type=int, default=50,)
    parser.add_argument('--min_save_epoch', type=int, default=10,)

    # Device & Reproducibility
    parser.add_argument('--gpu', type=str, default='0',)
    parser.add_argument('--seed', type=int, default=926,)

    # Distillation options
    kd_group = parser.add_mutually_exclusive_group()
    kd_group.add_argument('--distill', action='store_true',)
    kd_group.add_argument('--no-distill', dest='distill', action='store_false',)
    parser.set_defaults(distill=True)
    # parser.set_defaults(distill=False)

    parser.add_argument('--kd_alpha', type=float, default=0.5,
                        help='Weight for KD loss (1-alpha for CE loss)')
    parser.add_argument('--kd_temp', type=float, default=4.0,
                        help='Distillation temperature')
    parser.add_argument('--kd_weight_by_size', action='store_true', 
                        default=True,
                        help='Weight teacher logits by client dataset size')

    # Initialization
    parser.add_argument('--init_from_local_mean', action='store_true',
                        default=True,
                        help='Initialize global model as average of local classifiers')
    
    # Noise injection
    parser.add_argument('--use_dp', action='store_true', default=True,)
    parser.add_argument('--dp_noise_std', type=float, default=1,)

    # Parse arguments
    args = parser.parse_args()  

    ####### 以上是命令行参数设置 #######
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets_to_run = list(config.DATASETS.keys()) if args.all_datasets else [args.dataset or config.DATASET]

    for ds in datasets_to_run:
        run_on_dataset(ds, args, device, seed)

if __name__ == '__main__':
    main()
