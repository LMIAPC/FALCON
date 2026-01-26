# models/encoders/dino.py
# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoImageProcessor, AutoModel
from .base import BaseEncoder, register_encoder
from .utils import split_sliding_patches

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
                 patch_window: tuple[int, int],
                 patch_stride: tuple[int, int],
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

        self.patch_window = patch_window
        self.patch_stride = patch_stride
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

        patch_window = tuple(model_cfg.get('patch_window', (224, 224)))
        patch_stride = tuple(model_cfg.get('patch_stride', (224, 224)))

        # 4. Get hub_id from DINO_MAP
        hub_id = DINO_MAP[enc_type]

        return cls(
            hub_id=hub_id,
            local_path=local_path,
            patch_window=patch_window,
            patch_stride=patch_stride,
            feature_dim=feat_dim
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input tensor
        x = (x * 255.0 + 128.0).clamp(0, 255) / 255.0
        B, _, H, W = x.shape
        patches = split_sliding_patches(x, self.patch_window, self.patch_stride)
        flat = torch.cat(patches, 0)
        
        # Process input
        inp = self.proc(images=list(flat), return_tensors="pt", do_rescale=False)
        inp = {k: v.to(x.device) for k, v in inp.items()}
        
        # Get features
        cls_flat = self.backbone(**inp).last_hidden_state[:, 0]
        N = len(patches)
        
        # Reshape output
        return cls_flat.view(N, B, self.feature_dim).permute(1, 0, 2)
