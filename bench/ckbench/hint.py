"""GPU chroma-key alpha hint (research D8): difference key, smoothstep, 5x5 box blur."""
from __future__ import annotations

import torch
import torch.nn.functional as F

T0 = 0.05
T1 = 0.15
BLUR = 5


def chroma_hint(rgb: torch.Tensor, t0: float = T0, t1: float = T1, blur: int = BLUR) -> torch.Tensor:
    """rgb: [3,H,W] float in [0,1] (sRGB) -> mask [H,W] in [0,1]; 1 = subject, 0 = green screen."""
    r, g, b = rgb[0], rgb[1], rgb[2]
    d = g - torch.maximum(r, b)
    t = ((d - t0) / (t1 - t0)).clamp(0.0, 1.0)
    mask = 1.0 - t * t * (3.0 - 2.0 * t)
    if blur > 1:
        mask = F.avg_pool2d(mask[None, None], kernel_size=blur, stride=1, padding=blur // 2,
                            count_include_pad=False)[0, 0]
    return mask.clamp(0.0, 1.0)
