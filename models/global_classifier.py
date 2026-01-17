# -*- coding: utf-8 -*-
"""
FALCON – Global Classifier & Aggregation Module
==============================================

Provides three switchable aggregation methods via `config.MODEL_CONFIG['aggregation']['method']`:

| Method String | Description                                 | Module Implementation        |
|---------------|---------------------------------------------|------------------------------|
| "query"       | A single learnable query attends to all     | `AttentionPooling`           |
| "cross"       | Global proxy does cross-attention           | `GlobalProxyCrossAttention`  |
| "mean"        | Simple arithmetic mean                      | `AttentionPooling`           |

All modules take `(B, N, D)` input and return `(B, num_classes)`.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# --- Rotary Positional Encoding (RoPE) ---
def _rotate_every_two(x):
    """Rotate pairwise elements of feature dimension."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def rope_cache(seq_len, dim, device):
    """Precompute cosine/sine for rotary embeddings."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device) / dim))
    pos = torch.arange(seq_len, device=device).unsqueeze(1)
    theta = pos * inv_freq
    cos = torch.stack((theta.cos(), theta.cos()), dim=-1).flatten(-2)
    sin = torch.stack((theta.sin(), theta.sin()), dim=-1).flatten(-2)
    return cos, sin

def apply_rope(x, cos, sin):
    """Apply rotary positional encoding to input."""
    return (x * cos) + (_rotate_every_two(x) * sin)

# --- Attention Pooling Module ---
class AttentionPooling(nn.Module):
    """Pooling via attention mechanism with optional rotary encoding."""
    def __init__(self, feature_dim, method="query", use_pos_encoding=False, use_residual=False):
        super().__init__()
        self.method = method
        self.D = feature_dim
        self.use_pos_encoding = use_pos_encoding
        self.use_residual = use_residual
        
        if method == "query":
            # Initialize learnable query and projection layers
            self.q = nn.Parameter(torch.randn(1, 1, feature_dim) / math.sqrt(feature_dim))
            self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
            self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)

    def forward(self, x):
        """Forward pass handling both 1D and 2D inputs."""
        if x.dim() == 2 or x.size(1) == 1:
            return x if x.dim() == 2 else x.squeeze(1)
            
        B, N, D = x.shape
        if self.method == "mean":
            return x.mean(dim=1)
            
        q = self.q.expand(B, -1, -1)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        if self.use_pos_encoding:
            cos, sin = rope_cache(N, D, x.device)
            K = apply_rope(K, cos, sin)
            
        attn = (q @ K.transpose(-2, -1)).squeeze(1) / math.sqrt(D)
        attn = F.softmax(attn, dim=-1)
        pooled = (attn.unsqueeze(-1) * V).sum(dim=1)
        return pooled + q.squeeze(1) if self.use_residual else pooled

# --- Global Proxy Cross Attention ---
class GlobalProxyCrossAttention(nn.Module):
    """Cross-attention between global proxy and token features."""
    def __init__(self, feature_dim, use_pos_encoding=False):
        super().__init__()
        self.D = feature_dim
        self.use_pos_encoding = use_pos_encoding
        
        # Initialize projection layers
        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        for p in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.xavier_uniform_(p.weight)

    def forward(self, x):
        """Forward pass handling both 1D and 2D inputs."""
        if x.dim() == 2 or x.size(1) == 1:
            return x if x.dim() == 2 else x.squeeze(1)
            
        B, N, D = x.shape
        proxy, tokens = x[:, :1, :], x[:, 1:, :]
        
        Q = self.q_proj(proxy)
        K = self.k_proj(tokens)
        V = self.v_proj(tokens)
        
        if self.use_pos_encoding:
            cos, sin = rope_cache(N - 1, D, x.device)
            K = apply_rope(K, cos, sin)
            
        attn = F.softmax((Q @ K.transpose(-2, -1)) / math.sqrt(D), dim=-1)
        return (attn @ V).squeeze(1) + proxy.squeeze(1)

# --- Global Classifier ---
class GlobalClassifier(nn.Module):
    """Main classification head with configurable aggregation and optional MoE."""
    def __init__(self, feature_dim=2048, num_classes=10, dropout_rate=0.3,
                 use_moe=False, num_experts=3):
        super().__init__()
        method = config.MODEL_CONFIG["aggregation"].get("method", "query")
        agg_cfg = config.MODEL_CONFIG["aggregation"]
        
        # Configure aggregation method
        if method == "cross":
            self.pool = GlobalProxyCrossAttention(
                feature_dim, use_pos_encoding=agg_cfg.get("use_pos_encoding", False))
        elif method in {"query", "mean"}:
            self.pool = AttentionPooling(
                feature_dim, method=method,
                use_pos_encoding=agg_cfg.get("use_pos_encoding", False),
                use_residual=agg_cfg.get("use_residual", False))
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

        # Configure expert network
        self.use_moe = use_moe
        if use_moe:
            self.gate = nn.Sequential(
                nn.Linear(feature_dim, 256), nn.ReLU(),
                nn.Linear(256, num_experts), nn.Softmax(dim=1))
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, 4 * feature_dim), nn.ReLU(),
                    nn.Linear(4 * feature_dim, feature_dim), nn.Dropout(dropout_rate))
                for _ in range(num_experts)])
        else:
            self.ffn = nn.Sequential(
                nn.Linear(feature_dim, 4 * feature_dim), nn.ReLU(),
                nn.Linear(4 * feature_dim, feature_dim), nn.Dropout(dropout_rate))
                
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """Forward pass through pooling and classification head."""
        x = self.pool(x)
        
        if self.use_moe:
            w = self.gate(x)
            expert_out = torch.stack([e(x) for e in self.experts], dim=1)
            x = (expert_out * w.unsqueeze(-1)).sum(dim=1)
        else:
            x = self.ffn(x)
            
        return self.head(x)

# --- Knowledge Distillation Loss ---
class KnowledgeDistillationLoss(nn.Module):
    """Combination of hard target and soft distillation losses."""
    def __init__(self, alpha=0.5, temperature=2.0):
        super().__init__()
        self.alpha, self.T = alpha, temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, out, tgt, teacher):
        """Compute combined loss between predictions and teacher outputs."""
        hard = self.ce(out, tgt)
        soft = F.kl_div(
            F.log_softmax(out / self.T, dim=1),
            F.softmax(teacher / self.T, dim=1),
            reduction="batchmean") * (self.T ** 2)
        return (1 - self.alpha) * hard + self.alpha * soft