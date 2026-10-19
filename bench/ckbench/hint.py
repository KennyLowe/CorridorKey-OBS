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


# --- Calibrated hint (spec Session 2026-10-17) -------------------------------------------------
# Used only for clips whose clips.toml entry says hint = "calibrated". The screen colour is the
# per-channel median of an outer border band across the clip's frames (like picking the key colour
# in OBS). The key is the same difference key g - max(r, b), normalised by the screen's own
# difference, so 1.0 = exactly the screen colour and <= 0 = no green excess. The smoothstep then
# runs on that normalised value, followed by the same 5x5 box blur.
BORDER_PX = 32
CAL_T0 = 0.30  # normalised difference below which a pixel is fully subject
CAL_T1 = 0.70  # normalised difference above which a pixel is fully screen
MIN_KEY_DIFF = 0.02


def _border(rgb: torch.Tensor, px: int = BORDER_PX) -> torch.Tensor:
    return torch.cat([rgb[:, :px].reshape(3, -1), rgb[:, -px:].reshape(3, -1),
                      rgb[:, :, :px].reshape(3, -1), rgb[:, :, -px:].reshape(3, -1)], dim=1)


def estimate_key_colour(frames, px: int = BORDER_PX) -> tuple[float, float, float]:
    """frames: iterable of [3,H,W] float tensors in [0,1]. Median RGB of the outer band."""
    band = torch.cat([_border(f.float().cpu(), px) for f in frames], dim=1)
    key = tuple(float(v) for v in band.median(dim=1).values)
    if key[1] - max(key[0], key[2]) < MIN_KEY_DIFF:
        raise ValueError(f"border colour {key} has no green excess; cannot calibrate a green-screen key")
    return key


def calibrated_hint(rgb: torch.Tensor, key, t0: float = CAL_T0, t1: float = CAL_T1,
                    blur: int = BLUR) -> torch.Tensor:
    """rgb: [3,H,W] in [0,1]; key: screen (r,g,b) -> mask [H,W], 1 = subject, 0 = screen."""
    kd = key[1] - max(key[0], key[2])
    r, g, b = rgb[0], rgb[1], rgb[2]
    d = (g - torch.maximum(r, b)) / kd
    t = ((d - t0) / (t1 - t0)).clamp(0.0, 1.0)
    mask = 1.0 - t * t * (3.0 - 2.0 * t)
    if blur > 1:
        mask = F.avg_pool2d(mask[None, None], kernel_size=blur, stride=1, padding=blur // 2,
                            count_include_pad=False)[0, 0]
    return mask.clamp(0.0, 1.0).to(rgb.dtype)
