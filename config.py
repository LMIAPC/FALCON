import os
import torch

# ───────────────────────────────────────────────────────────────────
# Basic paths and device configuration
# ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ───────────────────────────────────────────────────────────────────
# Dataset configuration
# ───────────────────────────────────────────────────────────────────
DATASET = 'OfficeHome'

DATASETS = {
    'TB_dataset':      {'num_classes': 2,   'resolution': 448, 'clients': ['China', 'India', 'Montgomery']},
    'PACS':            {'num_classes': 7,   'resolution': 448, 'clients': ['art_painting', 'cartoon', 'photo', 'sketch']},
    'OfficeHome':      {'num_classes': 65,  'resolution': 224, 'clients': ['Art', 'Clipart', 'Product', 'Real_World']},
    'MRI-0.1':         {'num_classes': 4,   'resolution': 448, 'clients': ['Client0', 'Client1', 'Client2', 'Client3', 'Client4']},
    'MRI-0.3':         {'num_classes': 4,   'resolution': 448, 'clients': ['Client0', 'Client1', 'Client2', 'Client3', 'Client4']},
    'MRI-0.5':         {'num_classes': 4,   'resolution': 224, 'clients': ['Client0', 'Client1', 'Client2', 'Client3', 'Client4']},
    'Pneumonia-0.1':   {'num_classes': 2,   'resolution': 448, 'clients': ['Client0', 'Client1', 'Client2', 'Client3', 'Client4']},
    'Pneumonia-0.3':   {'num_classes': 2,   'resolution': 448, 'clients': ['Client0', 'Client1', 'Client2', 'Client3', 'Client4']},
    'Pneumonia-0.5':   {'num_classes': 2,   'resolution': 448, 'clients': ['Client0', 'Client1', 'Client2', 'Client3', 'Client4']},
    'tiny-0.1':        {'num_classes': 200, 'resolution': 448, 'clients': ['Client0', 'Client1', 'Client2', 'Client3', 'Client4']},
    'tiny-0.3':        {'num_classes': 200, 'resolution': 448, 'clients': ['Client0', 'Client1', 'Client2', 'Client3', 'Client4']},
    'tiny-0.5':        {'num_classes': 200, 'resolution': 448, 'clients': ['Client0', 'Client1', 'Client2', 'Client3', 'Client4']},
}

# ───────────────────────────────────────────────────────────────────
# Pretrained encoder configuration
# ───────────────────────────────────────────────────────────────────
ENCODER = {
    'type': 'dino_base',  # Options: 'dino_base', 'dino_large', 'dino_giant', 'clip_b16', 'clip_l14', 'bmclip_b16'
    'dims': {
        'res18'      : 512,
        'res50'      : 2048,
        'dino_base'  : 768,
        # 'dino_large' : 1024,
        # 'dino_giant' : 1536,
        'clip_b16'   : 512,
        # 'clip_l14'   : 768,
        # 'bmclip_b16' : 512,   # BioMedCLIP ViT-B/16 also defaults to 512 dims
    }
}

# ───────────────────────────────────────────────────────────────────
# Patch Splitting & Aggregation
# ───────────────────────────────────────────────────────────────────
# patch_split_n = 0 disables patching; otherwise (1+n)^2 patches are extracted
# ───────────────────────────────────────────────────────────────────
MODEL_PATCHING = {
    'patch_split_n': 1,
    'aggregation': {
        'method': 'query',           # Options: 'query' | 'cross' | 'mean'
        'use_residual': True,        # ✅ Only effective for 'query' mode; 'cross' always uses residual add-back
        'use_pos_encoding': False,   # ✅ Can be enabled for all aggregation methods
    }
}

# ───────────────────────────────────────────────────────────────────
# Model hyperparameters
# ───────────────────────────────────────────────────────────────────
MODEL_CONFIG = {
    'feature_dim': ENCODER['dims'][ENCODER['type']],
    **MODEL_PATCHING,
    'stage1': {'learning_rate': 1e-4, 'prior_weight': 1.0},
    'stage1_ema': {'enabled': True, 'decay': 0.999},
    'stage2': {'epochs': 500, 'learning_rate': 1e-4, 'num_mixtures': 20, 'depth': 4, 'hidden_dim': 768, 'num_heads': 16},
    'stage3': {'epochs': 60, 'batch_size': 512, 'learning_rate': 5e-4, 'kd_temp': 2.0, 'kd_alpha': 0.2, 'use_kd': False, 'use_moe': False, 'num_experts': 0},
    'stage3_ema': {'enabled': False, 'decay': 0.99},
}

# ───────────────────────────────────────────────────────────────────
# Path configuration
# ───────────────────────────────────────────────────────────────────
# Feature tag ensures uniqueness of directory by encoder type and patch_n
FEATURE_TAG = f"{ENCODER['type']}_n{MODEL_PATCHING['patch_split_n']}"

PATHS = {
    # Raw and preprocessed data
    'dataset':      lambda ds, cid=None: os.path.join(BASE_DIR, 'dataset', ds),
    'preprocessed': lambda ds, cid=None: os.path.join(BASE_DIR, 'dataset_preprocessed', ds),

    # Feature saving paths (differentiated by encoder type and patch_n)
    'real_feature':      lambda ds, cid: os.path.join(BASE_DIR, 'feature', FEATURE_TAG, 'real', ds, f'Client{cid}'),
    'val_feature':       lambda ds, cid: os.path.join(BASE_DIR, 'feature', FEATURE_TAG, 'val', ds, f'Client{cid}'),
    'test_feature':      lambda ds, cid: os.path.join(BASE_DIR, 'feature', FEATURE_TAG, 'test', ds, f'Client{cid}'),
    'synthetic_feature': lambda ds, cid: os.path.join(BASE_DIR, 'feature', FEATURE_TAG, 'synthetic', ds, f'Client{cid}'),
    'merged_feature':    lambda ds, cid: os.path.join(BASE_DIR, 'feature', FEATURE_TAG, 'merged', ds, f'Client{cid}'),

    # Intermediate files and weights
    # 'kd_logits':               lambda ds, cid: os.path.join(BASE_DIR, 'KD_logits', FEATURE_TAG, ds, f'Client{cid}'),
    'local_classifier_weights':lambda ds, cid: os.path.join(BASE_DIR, 'weights', 'local_classifier', FEATURE_TAG, ds, f'Client{cid}'),
    'local_fgenerator_weights':lambda ds, cid: os.path.join(BASE_DIR, 'weights', 'local_fgenerator', FEATURE_TAG, ds, f'Client{cid}'),
    'global_classifier_weights':lambda ds: os.path.join(BASE_DIR, 'weights', 'global_classifier', FEATURE_TAG, ds),
    'aggregator_weights':       lambda ds: os.path.join(BASE_DIR, 'weights', 'query_aggregator', FEATURE_TAG, ds),

    # Pretrained model weights path
    'pretrained_model': lambda m: os.path.join(BASE_DIR, 'weights', 'fextractor',
        f'dinov2-{m.split("_")[1]}' if m.startswith('dino') else
        f'clip-b16' if m == 'clip_b16' else
        f'clip-l14' if m == 'clip_l14' else
        f'bmclip-b16' if m == 'bmclip_b16' else
        'unknown_encoder'),

    # Logging
    'logs': lambda ds, stage=None, cid=None: os.path.join(
        BASE_DIR, 'logs', FEATURE_TAG, ds,
        f'stage{stage}_client_{cid}.log' if stage is not None and cid is not None else
        f'stage{stage}.log' if stage is not None else 'general.log'
    ),
}

# ───────────────────────────────────────────────────────────────────
# Create all necessary directories
# ───────────────────────────────────────────────────────────────────
def create_directories():
    for ds, cfg in DATASETS.items():
        os.makedirs(PATHS['dataset'](ds), exist_ok=True)
        os.makedirs(PATHS['preprocessed'](ds), exist_ok=True)
        n_clients = len(cfg['clients'])
        for cid in range(n_clients):
            for key in ['real_feature','val_feature','test_feature','synthetic_feature','merged_feature',
                        'kd_logits','local_classifier_weights','local_fgenerator_weights']:
                os.makedirs(PATHS[key](ds, cid), exist_ok=True)
        os.makedirs(PATHS['global_classifier_weights'](ds), exist_ok=True)
        os.makedirs(PATHS['aggregator_weights'](ds), exist_ok=True)
        os.makedirs(os.path.dirname(PATHS['logs'](ds)), exist_ok=True)

create_directories()
