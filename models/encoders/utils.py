# -*- coding: utf-8 -*-
"""
Helper functions for encoders (private implementation)
"""
from typing import List, Tuple
import torch
import torch.nn.functional as F

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
    
    _, _, H, W = x.shape
    
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

def split_sliding_patches(
    x: torch.Tensor,
    window_size: Tuple[int, int],
    stride: Tuple[int, int],
    include_full: bool = True
) -> List[torch.Tensor]:
    win_h, win_w = window_size
    stride_h, stride_w = stride
    _, _, H, W = x.shape

    if H < win_h or W < win_w:
        resized = x if (H, W) == (win_h, win_w) else F.interpolate(
            x, size=(win_h, win_w), mode="bilinear", align_corners=False
        )
        return [resized]

    if H == win_h and W == win_w:
        resized = x if (H, W) == (win_h, win_w) else F.interpolate(
            x, size=(win_h, win_w), mode="bilinear", align_corners=False
        )
        return [resized]

    h_starts = [0] if H <= win_h else list(range(0, H - win_h + 1, stride_h))
    w_starts = [0] if W <= win_w else list(range(0, W - win_w + 1, stride_w))
    if h_starts[-1] != H - win_h:
        h_starts.append(H - win_h)
    if w_starts[-1] != W - win_w:
        w_starts.append(W - win_w)

    patches: List[torch.Tensor] = []

    if include_full:
        full = x if (H, W) == (win_h, win_w) else F.interpolate(
            x, size=(win_h, win_w), mode="bilinear", align_corners=False
        )
        patches.append(full)

    for h0 in h_starts:
        h1 = h0 + win_h
        for w0 in w_starts:
            w1 = w0 + win_w
            patch = x[:, :, h0:h1, w0:w1]
            if patch.shape[-2:] != (win_h, win_w):
                patch = F.interpolate(patch, size=(win_h, win_w),
                                      mode="bilinear", align_corners=False)
            patches.append(patch)

    return patches

def sliding_window_count(
    height: int,
    width: int,
    window_size: Tuple[int, int],
    stride: Tuple[int, int]
) -> int:
    win_h, win_w = window_size
    stride_h, stride_w = stride

    if height < win_h or width < win_w:
        return 0
    if height == win_h and width == win_w:
        return 0

    h_starts = [0] if height <= win_h else list(range(0, height - win_h + 1, stride_h))
    w_starts = [0] if width <= win_w else list(range(0, width - win_w + 1, stride_w))
    if h_starts[-1] != height - win_h:
        h_starts.append(height - win_h)
    if w_starts[-1] != width - win_w:
        w_starts.append(width - win_w)
    return len(h_starts) * len(w_starts)
