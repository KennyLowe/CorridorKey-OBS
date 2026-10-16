"""PyTorch eager fp16-autocast keyer wrapping CorridorKey's own engine (research D1, D3).

The model is built through ``CorridorKeyEngine`` exactly as upstream does (fp32 weights,
``torch.autocast(float16)``, ``torch.compile`` skipped). Square sizes use the engine unchanged:
it already bicubic-resizes the checkpoint's 2048 position embedding to the requested grid.
Non-square sizes (16:9 variant, research D2) use a small subclass whose loader resizes the
embedding to the rectangular token grid instead of assuming a square one.

``prepare()`` reproduces the engine's input normalisation (ImageNet mean/std on sRGB, hint
channel un-normalised) and ``run()`` is the network call alone, so timing isolates the model.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

from .env import VENDOR_DIR, WEIGHTS_PATH

CKPT_SIZE = 2048  # the v1.0 checkpoint's native square size (pos_embed grid 512x512)
ENCODER = "hiera_base_plus_224.mae_in1k_ft_in1k"


def _vendor_import():
    os.environ["CORRIDORKEY_SKIP_COMPILE"] = "1"  # cl.exe not installed; eager mode (research D3)
    if str(VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_DIR))
    from CorridorKeyModule.core.model_transformer import GreenFormer
    from CorridorKeyModule.inference_engine import CorridorKeyEngine
    return CorridorKeyEngine, GreenFormer


def _rect_engine_class():
    CorridorKeyEngine, GreenFormer = _vendor_import()

    class RectCorridorKeyEngine(CorridorKeyEngine):
        """img_size is (h, w); pos_embed interpolated to the (h/4, w/4) token grid."""

        def _load_model(self):
            from safetensors.torch import load_file

            model = GreenFormer(encoder_name=ENCODER, img_size=self.img_size, use_refiner=self.use_refiner)
            model = model.to(self.device).eval()
            sd = load_file(self.checkpoint_path, device=str(self.device))
            target = model.state_dict()
            new = {}
            for k, v in sd.items():
                if k.startswith("_orig_mod."):
                    k = k[10:]
                if "pos_embed" in k and k in target and v.shape != target[k].shape:
                    n_src, c = v.shape[1], v.shape[2]
                    g = int(round(n_src ** 0.5))
                    gh, gw = self.img_size[0] // 4, self.img_size[1] // 4
                    assert gh * gw == target[k].shape[1], (gh, gw, target[k].shape)
                    img = v.permute(0, 2, 1).reshape(1, c, g, g)
                    v = F.interpolate(img, size=(gh, gw), mode="bicubic", align_corners=False)
                    v = v.flatten(2).transpose(1, 2)
                new[k] = v
            model.load_state_dict(new, strict=False)
            return model.to(self.model_precision)

    return RectCorridorKeyEngine


class TorchKeyer:
    runtime = "torch"

    def __init__(self, size_wh: tuple[int, int], refiner: bool, checkpoint=WEIGHTS_PATH, device: str = "cuda"):
        w, h = size_wh
        self.size_wh = (w, h)
        self.refiner = refiner
        CorridorKeyEngine, _ = _vendor_import()
        if w == h:
            self.engine = CorridorKeyEngine(str(checkpoint), device=device, img_size=w,
                                            use_refiner=refiner, mixed_precision=True)
        else:
            self.engine = _rect_engine_class()(str(checkpoint), device=device, img_size=(h, w),
                                               use_refiner=refiner, mixed_precision=True)
        self.model = self.engine.model
        self.mean = self.engine.mean.view(1, 3, 1, 1)
        self.std = self.engine.std.view(1, 3, 1, 1)
        self.posembed_interpolated = (w, h) != (CKPT_SIZE, CKPT_SIZE)

    def prepare(self, rgb: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        """rgb [1,3,h,w] sRGB in [0,1], hint [1,1,h,w] in [0,1] -> model input [1,4,h,w] fp32."""
        x = (rgb.float() - self.mean) / self.std
        return torch.cat([x, hint.float()], dim=1)

    @torch.inference_mode()
    def run(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.engine.mixed_precision):
            out = self.model(inp)
        return out["alpha"], out["fg"]

    def infer(self, rgb: torch.Tensor, hint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.run(self.prepare(rgb, hint))

    def close(self) -> None:
        del self.model, self.engine
