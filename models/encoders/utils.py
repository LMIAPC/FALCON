# -*- coding: utf-8 -*-
"""
Helper functions for encoders (private implementation)
"""
from typing import List, Tuple
import torch
import torch.nn.functional as F

# ───────────────────────────────────────────────────────────────────
# Uniform patch splitting (1 + (1+n)^2 patches) (Copied from old version, unmodified)
# ───────────────────────────────────────────────────────────────────
def split_uniform_patches(
    x: torch.Tensor, 
    n: int, 
    target_size: Tuple[int, int]
) -> List[torch.Tensor]:
    """
    Split input tensor into uniform patches with optional interpolation.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        n: Number of splits per dimension (0 means no splitting)
        target_size: Target size (height, width) for all patches
        
    Returns:
        List of patches where:
        - First element is the full image (possibly resized)
        - Subsequent elements are n×n uniform patches (total (n+1)^2 patches)
    """
    if n == 0:
        # No splitting - just return resized full image if needed
        return [x if x.shape[-2:] == target_size
                else F.interpolate(x, size=target_size,
                                 mode="bilinear", align_corners=False)]
    
    B, C, H, W = x.shape
    
    # Calculate split sizes (equal division with remainder added to last split)
    h_sizes = [H // (n + 1)] * (n + 1)
    w_sizes = [W // (n + 1)] * (n + 1)
    h_sizes[-1] += H - sum(h_sizes)
    w_sizes[-1] += W - sum(w_sizes)
    
    # Calculate split starting positions
    h_starts = [sum(h_sizes[:i]) for i in range(n + 1)]
    w_starts = [sum(w_sizes[:j]) for j in range(n + 1)]

    patches: List[torch.Tensor] = []
    
    # Extract and resize each patch
    for hi, h0 in enumerate(h_starts):
        h1 = h0 + h_sizes[hi]
        for wi, w0 in enumerate(w_starts):
            w1 = w0 + w_sizes[wi]
            patch = x[:, :, h0:h1, w0:w1]
            if patch.shape[-2:] != target_size:
                patch = F.interpolate(patch, size=target_size,
                                    mode="bilinear", align_corners=False)
            patches.append(patch)

    # Ensure full image is first element and properly sized
    if x.shape[-2:] != target_size:
        x = F.interpolate(x, size=target_size,
                        mode="bilinear", align_corners=False)
    patches.insert(0, x)
    
    return patches