# models/encoders/dino.py
# -*- coding: utf-8 -*-
import os
import torch, torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from .base import BaseEncoder, register_encoder
from .utils import split_uniform_patches

# Mapping between model names and HuggingFace hub IDs
DINO_MAP = {
    "dino_base": "facebook/dinov2-base",
    "dino_large": "facebook/dinov2-large",
    "dino_giant": "facebook/dinov2-giant",
}

@register_encoder("dino_base")
@register_encoder("dino_large")
@register_encoder("dino_giant")
class FrozenDINO(BaseEncoder):
    def __init__(self,
                 hub_id: str,
                 local_path: str,
                 patch_split_n: int,
                 feature_dim: int):
        super().__init__()
        # Use local files if available, otherwise download from hub
        local_kwargs = {"local_files_only": True} if os.path.isdir(local_path) else {}
        src = local_path if local_kwargs else hub_id

        # Initialize processor and model
        self.proc = AutoImageProcessor.from_pretrained(
            src, trust_remote_code=True, **local_kwargs
        )
        self.backbone = AutoModel.from_pretrained(
            src, trust_remote_code=True, **local_kwargs
        )
        self.backbone.eval().requires_grad_(False)

        self.patch_split_n = patch_split_n
        self.feature_dim = feature_dim

    @classmethod
    def build(cls,
              *,
              encoder_cfg: dict,
              model_cfg: dict,
              paths: dict):
        # 1. Get encoder type and feature dimension from encoder_cfg
        enc_type = encoder_cfg['type']
        feat_dim = encoder_cfg['dims'][enc_type]

        # 2. Get local pretrained weights path from paths
        # PATHS['pretrained_model'] accepts parameter m (e.g. 'dino_base')
        local_path = paths['pretrained_model'](enc_type)

        # 3. Get patch_split_n from MODEL_CONFIG
        patch_n = model_cfg.get('patch_split_n', 0)

        # 4. Get hub_id from DINO_MAP
        hub_id = DINO_MAP[enc_type]

        return cls(
            hub_id=hub_id,
            local_path=local_path,
            patch_split_n=patch_n,
            feature_dim=feat_dim
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input tensor
        x = (x * 255.0 + 128.0).clamp(0, 255) / 255.0
        n, B, _, H, W = self.patch_split_n, *x.shape
        
        # Split into patches
        patches = split_uniform_patches(x, n, (H, W))
        flat = torch.cat(patches, 0)
        
        # Process input
        inp = self.proc(images=list(flat), return_tensors="pt", do_rescale=False)
        inp = {k: v.to(x.device) for k, v in inp.items()}
        
        # Get features
        cls_flat = self.backbone(**inp).last_hidden_state[:, 0]  # (B*(1+(1+n)^2), D)
        N = 1 if n == 0 else 1 + (n + 1) ** 2
        
        # Reshape output
        return cls_flat.view(N, B, self.feature_dim).permute(1, 0, 2)  # (B, N, D)