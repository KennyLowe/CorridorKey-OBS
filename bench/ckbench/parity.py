"""Reference fixtures for the OBS plugin's GPU tests (feature 002, T013/T014).

    python -m ckbench.parity kernels   # hint / preprocess / postprocess references
    python -m ckbench.parity runner    # 512_r0 model input + PyTorch outputs + engine path
    python -m ckbench.parity all

Everything is computed with the spike's own code (``hint.py``, ``pipeline._down``/``_up``,
``TorchKeyer``) so the plugin's CUDA kernels and TensorRT runner are checked against what was
benchmarked and quality-reviewed in feature 001, not against a re-implementation. Inputs are
deterministic synthetic frames (``frames.synthetic_frame``); files go to the git-ignored
``outputs/parity/``. Kernel references are float32 on the CPU (the spike ran fp16 on the GPU;
the difference is fp16 rounding, not algorithm).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .results import OUTPUTS_DIR

PARITY_DIR = OUTPUTS_DIR / "parity"
# (camera W, H, internal w, h): a real 1080p -> 1024x576 case and an odd 4:3 -> square case.
KERNEL_CASES = [(1920, 1080, 1024, 576), (643, 487, 256, 256)]
CAL_KEY = (0.20, 0.70, 0.25)  # synthetic_frame's screen colour
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _rgba(frame_rgb: np.ndarray) -> np.ndarray:
    a = np.full(frame_rgb.shape[:2] + (1,), 255, np.uint8)
    return np.ascontiguousarray(np.concatenate([frame_rgb, a], axis=2))


def kernels(out_dir: Path = PARITY_DIR / "kernels") -> Path:
    import torch

    from .frames import synthetic_frame
    from .hint import calibrated_hint, chroma_hint
    from .pipeline import _down, _up

    out_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    meta = {"cases": [], "cal_key": CAL_KEY}
    for i, (W, H, w, h) in enumerate(KERNEL_CASES):
        frame = synthetic_frame(w=W, h=H, seed=100 + i)
        rgba = _rgba(frame)
        rgb = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        hint_fixed = chroma_hint(rgb)
        hint_cal = calibrated_hint(rgb, CAL_KEY)
        # the model input exactly as the spike built it: hint at camera res, both downscaled
        # (antialiased bilinear), RGB normalised with ImageNet mean/std, hint not normalised
        x = torch.cat([(_down(rgb[None], h, w) - mean) / std, _down(hint_fixed[None, None], h, w)], dim=1)
        # postprocess: smooth synthetic network outputs at internal res -> camera-res RGBA8
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        alpha = (0.5 + 0.6 * torch.sin(6.0 * xx) * torch.cos(4.0 * yy)).clamp(-0.1, 1.1)[None, None]
        fg = torch.stack([xx, yy, 0.5 + 0.5 * torch.sin(9.0 * xx * yy)])[None] * 1.05 - 0.02
        rgba_out = torch.cat([_up(fg, H, W), _up(alpha, H, W)], dim=1)[0]
        rgba_out = rgba_out.permute(1, 2, 0).mul(255.0).round_().clamp_(0, 255).to(torch.uint8)
        name = f"case{i}"
        np.save(out_dir / f"{name}_rgba.npy", rgba)
        np.save(out_dir / f"{name}_hint_fixed.npy", hint_fixed.numpy().astype(np.float32))
        np.save(out_dir / f"{name}_hint_cal.npy", hint_cal.numpy().astype(np.float32))
        np.save(out_dir / f"{name}_input.npy", x.numpy().astype(np.float32))
        np.save(out_dir / f"{name}_alpha.npy", alpha.numpy().astype(np.float32))
        np.save(out_dir / f"{name}_fg.npy", fg.numpy().astype(np.float32))
        np.save(out_dir / f"{name}_out.npy", rgba_out.numpy())
        meta["cases"].append({"name": name, "W": W, "H": H, "w": w, "h": h})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def runner(out_dir: Path = PARITY_DIR / "runner", size=(512, 512), refiner=False) -> Path:
    """Model input from a synthetic 1080p frame (spike pipeline) + PyTorch fp16-autocast outputs."""
    import torch

    from . import prepare as P
    from .env import WEIGHTS_PATH
    from .frames import synthetic_frame
    from .hint import chroma_hint
    from .pipeline import _down
    from .runtime_torch import TorchKeyer

    w, h = size
    out_dir.mkdir(parents=True, exist_ok=True)
    events = []
    rc = P.prepare(str(WEIGHTS_PATH), w, h, refiner, str(P.DEFAULT_CACHE), events.append)
    res = events[-1]
    if rc != 0:
        raise RuntimeError(f"ckbench prepare failed: {res}")
    frame = synthetic_frame(seed=7)
    keyer = TorchKeyer((w, h), refiner)
    with torch.inference_mode():
        rgb = torch.from_numpy(frame).cuda().permute(2, 0, 1).float() / 255.0
        hint = chroma_hint(rgb)
        x = keyer.prepare(_down(rgb[None], h, w), _down(hint[None, None], h, w))
        alpha, fg = keyer.run(x)
    np.save(out_dir / "input.npy", x.float().cpu().numpy())
    np.save(out_dir / "alpha.npy", alpha.float().cpu().numpy())
    np.save(out_dir / "fg.npy", fg.float().cpu().numpy())
    (out_dir / "meta.json").write_text(json.dumps({"engine_path": res["engine_path"], "w": w, "h": h,
                                                   "refiner": refiner, "trt_version": res["trt_version"]}, indent=2))
    return out_dir


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("kernels", "all"):
        print(f"kernel fixtures: {kernels()}")
    if what in ("runner", "all"):
        print(f"runner fixtures: {runner()}")
