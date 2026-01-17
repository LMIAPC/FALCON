# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Type, Callable
import torch.nn as nn

# ───────────────────────────────────────────────────────────────────
# Registry
# ───────────────────────────────────────────────────────────────────
_ENCODER_REGISTRY: Dict[str, Type["BaseEncoder"]] = {}

def register_encoder(name: str) -> Callable[[Type["BaseEncoder"]], Type["BaseEncoder"]]:
    """Class decorator: @register_encoder("dino_base")"""
    def _decorator(cls):
        _ENCODER_REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return _decorator

# ───────────────────────────────────────────────────────────────────
# ⛓️ Abstract base
# ───────────────────────────────────────────────────────────────────
class BaseEncoder(nn.Module):
    feature_dim: int  # Must be defined by subclass

    @classmethod
    def build(cls, cfg: dict) -> "BaseEncoder":
        """Return instance based on global config"""
        raise NotImplementedError

    def forward(self, x):  # -> (B, N, D)
        raise NotImplementedError