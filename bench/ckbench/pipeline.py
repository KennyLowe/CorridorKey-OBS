"""End-to-end per-frame pipeline, timed per stage with CUDA events (research D7, FR-008).

upload (pinned host uint8 -> GPU) -> hint (uint8->fp16 + chroma key at camera res)
-> resize_in (to internal size, + normalisation) -> infer (network only)
-> resize_out (alpha/fg back to camera res, pack RGBA8) -> download (GPU -> pinned host).

Resizing follows upstream CorridorKey: torchvision bilinear (antialiased) down, bilinear up.
The keyer is pluggable: anything with ``size_wh``, ``prepare(rgb, hint)`` and ``run(inp)``.
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from torchvision.transforms import InterpolationMode

from .config import CAMERA_RES
from .hint import chroma_hint

STAGES = ("upload", "hint", "resize_in", "infer", "resize_out", "download")


def _down(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return TF.resize(x, [h, w], interpolation=InterpolationMode.BILINEAR, antialias=True)


def _up(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return TF.resize(x.float(), [h, w], interpolation=InterpolationMode.BILINEAR, antialias=False)


def key_tensor(keyer, rgb: torch.Tensor, hint_fn=chroma_hint) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """rgb [3,H,W] fp16/fp32 CUDA in [0,1] -> (alpha [H,W], fg [3,H,W], hint [H,W]) at camera res, fp32."""
    H, W = rgb.shape[-2:]
    w, h = keyer.size_wh
    hint = hint_fn(rgb)
    rgb_s = _down(rgb[None], h, w)
    hint_s = _down(hint[None, None], h, w)
    alpha, fg = keyer.run(keyer.prepare(rgb_s, hint_s))
    return _up(alpha, H, W)[0, 0], _up(fg, H, W)[0], hint.float()


def fit_frame(frame_rgb_u8: np.ndarray, camera_res: str) -> np.ndarray:
    """Resize a host RGB uint8 frame to the camera resolution (only if needed)."""
    import cv2

    W, H = CAMERA_RES[camera_res]
    if frame_rgb_u8.shape[:2] != (H, W):
        frame_rgb_u8 = cv2.resize(frame_rgb_u8, (W, H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame_rgb_u8)


class Pipeline:
    def __init__(self, keyer, camera_res: str, frame_rgb_u8: np.ndarray):
        self.keyer = keyer
        self.W, self.H = CAMERA_RES[camera_res]
        frame = fit_frame(frame_rgb_u8, camera_res)
        self.host_in = torch.from_numpy(frame).pin_memory()
        self.host_out = torch.empty((self.H, self.W, 4), dtype=torch.uint8).pin_memory()
        self.ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(STAGES) + 1)]

    @torch.inference_mode()
    def __call__(self) -> dict:
        ev = self.ev
        w, h = self.keyer.size_wh
        ev[0].record()
        dev = self.host_in.to("cuda", non_blocking=True)
        ev[1].record()
        rgb = dev.permute(2, 0, 1).to(torch.float16).div_(255.0)
        hint = chroma_hint(rgb)
        ev[2].record()
        inp = self.keyer.prepare(_down(rgb[None], h, w), _down(hint[None, None], h, w))
        ev[3].record()
        alpha, fg = self.keyer.run(inp)
        ev[4].record()
        rgba = torch.cat([_up(fg, self.H, self.W), _up(alpha, self.H, self.W)], dim=1)[0]
        rgba = rgba.permute(1, 2, 0).mul(255.0).round_().clamp_(0, 255).to(torch.uint8)
        ev[5].record()
        self.host_out.copy_(rgba, non_blocking=True)
        ev[6].record()
        torch.cuda.synchronize()
        return {name: ev[i].elapsed_time(ev[i + 1]) for i, name in enumerate(STAGES)}
