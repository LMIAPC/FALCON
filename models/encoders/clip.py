# -*- coding: utf-8 -*-
from __future__ import annotations
import os, torch, torch.nn as nn
from transformers import CLIPModel, CLIPImageProcessor
from .base import BaseEncoder, register_encoder
from .utils import split_uniform_patches

CLIP_MAP = {
    "clip_b16": "openai/clip-vit-base-patch16",
    "clip_l14": "openai/clip-vit-large-patch14",
}

@register_encoder("clip_b16")
@register_encoder("clip_l14")
class FrozenCLIP(BaseEncoder):
    def __init__(self, hub_id, local_path, patch_n, feat_dim):
        super().__init__()
        local_kwargs = {"local_files_only": True} if os.path.isdir(local_path) else {}
        src = local_path if local_kwargs else hub_id

        # Load the full CLIP model to access its components
        full_clip_model = CLIPModel.from_pretrained(src, **local_kwargs)

        self.proc = CLIPImageProcessor.from_pretrained(src, **local_kwargs)
        self.backbone = full_clip_model.vision_model  # This is the CLIPVisionTransformer
        self.visual_projection = full_clip_model.visual_projection  # This is the projection layer

        self.backbone.eval().requires_grad_(False)
        self.visual_projection.eval().requires_grad_(False)

        self.patch_split_n = patch_n
        self.feature_dim   = feat_dim

    # factory
    @classmethod
    def build(cls, *, encoder_cfg, model_cfg, paths):
        typ      = encoder_cfg["type"]
        hub_id   = CLIP_MAP[typ]
        loc_path = paths["pretrained_model"](typ)
        dim      = encoder_cfg["dims"][typ]
        patch_n  = model_cfg.get("patch_split_n", 0)
        return cls(hub_id, loc_path, patch_n, dim)

    # forward
    @torch.no_grad()
    def forward(self, x):
        x = (x * 255.0 + 128.0).clamp(0, 255) / 255.0
        n, B, C, H, W = self.patch_split_n, *x.shape
        patches = split_uniform_patches(x, n, (H, W))
        # # resize to 224x224
        # patches = [nn.functional.interpolate(p, size=224) for p in patches]
        flat    = torch.cat(patches, 0)
        inp     = self.proc(images=list(flat), return_tensors="pt", do_rescale=False)
        inp     = {k: v.to(x.device) for k, v in inp.items()}
        
        # Get outputs from the vision transformer
        vision_outputs = self.backbone(**inp)
        # The pooler_output is typically the CLS token's representation after LayerNorm
        pooled_output = vision_outputs.pooler_output # Shape: (effective_batch_size, 768 for ViT-B/16)
        
        # Apply the visual projection layer to get the final embeddings
        cls_flat = self.visual_projection(pooled_output) # Shape: (effective_batch_size, 512 for ViT-B/16)
        
        N = 1 if n == 0 else 1 + (n + 1) ** 2
        # Now cls_flat has num_elements = effective_batch_size * 512
        # And N * B * self.feature_dim = effective_batch_size * 512, so view will work
        return cls_flat.view(N, B, self.feature_dim).permute(1, 0, 2)
