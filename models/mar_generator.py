import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────── Transformer Submodule ──────────────────────
class _Block(nn.Module):
    """Pre-norm Transformer block (self-attention → FFN)."""
    def __init__(self, dim: int, heads: int, ff_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.attn(
            self.ln1(x), self.ln1(x), self.ln1(x),
            attn_mask=attn_mask, need_weights=False
        )[0]
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


# ───────────────────── MDN Generator Main Class ──────────────────────
class MDNGenerator(nn.Module):
    """
    Mixture Density Network Generator with Transformer backbone
    + Sinusoidal positional encoding (absolute)
    + Switch: cond_pos_enc controls whether to add positional encoding to condition token
    """
    def __init__(
        self,
        cond_dim: int,
        feature_dim: int,
        num_mixtures: int = 32,
        depth: int = 4,
        hidden_dim: int = 512,
        num_heads: int = 8,
        cond_pos_enc: bool = True,   # ← New switch
    ):
        super().__init__()
        self.D = feature_dim
        self.M = num_mixtures
        self.cond_pos_enc = cond_pos_enc

        # Projection & Transformer
        self.cond_proj = nn.Linear(cond_dim, feature_dim)
        self.blocks = nn.ModuleList([
            _Block(feature_dim, num_heads, hidden_dim) for _ in range(depth)
        ])

        # MDN Head
        self.pi_out   = nn.Linear(feature_dim, num_mixtures)
        self.mu_out   = nn.Linear(feature_dim, num_mixtures * feature_dim)
        self.logv_out = nn.Linear(feature_dim, num_mixtures * feature_dim)

        # Cache
        self._mask_cache: dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._pe_cache:   dict[Tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    # ------------------------------ Forward Pass ------------------------------
    def forward(
        self,
        cond: torch.Tensor,
        upsampled_global: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cond:              (B, cond_dim)
            upsampled_global:  (B, N², D) or None
        Returns:
            Tuple of (pi, mu, logv)
        """
        B = cond.size(0)
        c_tok = self.cond_proj(cond).unsqueeze(1)                 # (B,1,D)
        x = c_tok if upsampled_global is None else torch.cat([c_tok, upsampled_global], dim=1)
        L = x.size(1)

        # Positional encoding
        pe = self._get_sinusoidal_encoding(L, x.device, x.dtype)  # (1,L,D)
        if not self.cond_pos_enc:
            pe = pe.clone()
            pe[:, 0, :] = 0.0                                      # No PE for condition token
        x = x + pe

        # Transformer
        mask = self._get_mask(L, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, mask)

        # MDN Head
        pi   = F.softmax(self.pi_out(x), dim=-1)                   # (B,L,M)
        mu   = self.mu_out(x).view(B, L, self.M, self.D)           # (B,L,M,D)
        logv = self.logv_out(x).view(B, L, self.M, self.D)         # (B,L,M,D)
        return pi, mu, logv

    # --------------------------- Proxy Sampling ----------------------------
    @torch.no_grad()
    def generate_full_proxy(
        self,
        cond: torch.Tensor,
        n: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Two-step generation: (1) Global token → (2) Structured proxy sequence"""
        # Step 1: Global token
        global_vec = self._sample_from_mixture(*self.forward(cond, None), temperature)  # (B,1,D)

        if n == 1:
            return global_vec

        # Step 2: Structured proxy
        g_up = global_vec.repeat(1, n * n, 1)                         # (B, n², D)
        pi, mu, logv = self.forward(cond, g_up)
        return self._sample_from_mixture(pi, mu, logv, temperature)   # (B, 1+n², D)

    # --------------------------- Negative Log Likelihood ---------------------------
    @staticmethod
    def nll_loss(
        target: torch.Tensor,
        pi: torch.Tensor,
        mu: torch.Tensor,
        logv: torch.Tensor,
    ) -> torch.Tensor:
        B, L, M, D = mu.shape
        x = target.unsqueeze(2).expand(-1, -1, M, -1)
        var = torch.exp(logv)
        const = D * math.log(2.0 * math.pi)
        log_prob = -0.5 * (torch.sum((x - mu) ** 2 / var + logv, dim=-1) + const)  # (B,L,M)
        weighted = torch.log(pi + 1e-9) + log_prob
        log_mix = torch.logsumexp(weighted, dim=-1)                                # (B,L)
        return -log_mix.mean()

    # --------------------------- Utility Functions ----------------------------
    def _sample_from_mixture(
        self,
        pi: torch.Tensor,
        mu: torch.Tensor,
        logv: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        B, L, M = pi.shape
        pi_adj = F.softmax(torch.log(pi + 1e-9) / temperature, dim=-1)
        comp = torch.multinomial(pi_adj.view(-1, M), 1).squeeze(-1)   # (B*L,)
        mu   = mu.view(B * L, M, self.D)
        logv = logv.view(B * L, M, self.D)

        idx = torch.arange(B * L, device=pi.device)
        chosen_mu   = mu[idx, comp]                                   # (B*L,D)
        chosen_logv = logv[idx, comp]
        std = torch.exp(0.5 * chosen_logv)
        eps = torch.randn_like(std)
        sample = chosen_mu + eps * std
        return sample.view(B, L, self.D)

    def _get_mask(self, L: int, device, dtype) -> torch.Tensor:
        key = (L, device)
        if key not in self._mask_cache:
            m = torch.zeros(L, L, dtype=dtype, device=device)
            m[0, 1:] = float("-inf")          # Only mask condition token
            self._mask_cache[key] = m
        return self._mask_cache[key]

    def _get_sinusoidal_encoding(self, L: int, device, dtype) -> torch.Tensor:
        key = (L, device, dtype)
        if key in self._pe_cache:
            return self._pe_cache[key]

        position = torch.arange(L, device=device, dtype=dtype).unsqueeze(1)          # (L,1)
        div_term = torch.exp(
            torch.arange(0, self.D, 2, device=device, dtype=dtype) *
            -(math.log(10000.0) / self.D)
        )                                                                            # (D/2,)

        pe = torch.zeros(L, self.D, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)            # (1,L,D)
        pe.requires_grad_(False)
        self._pe_cache[key] = pe
        return pe