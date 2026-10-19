"""TensorRT engine keyer with the same interface as TorchKeyer (US2)."""
from __future__ import annotations

from pathlib import Path

import tensorrt as trt
import torch

from .runtime_torch import CKPT_SIZE

_TRT_TO_TORCH = {trt.float32: torch.float32, trt.float16: torch.float16}
_LOGGER = trt.Logger(trt.Logger.WARNING)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TrtKeyer:
    runtime = "trt"

    def __init__(self, engine_file: Path, size_wh: tuple[int, int]):
        w, h = size_wh
        self.size_wh = (w, h)
        self.posembed_interpolated = (w, h) != (CKPT_SIZE, CKPT_SIZE)
        self._rt = trt.Runtime(_LOGGER)
        with open(engine_file, "rb") as f:
            self.engine = self._rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_file}")
        self.ctx = self.engine.create_execution_context()
        self.outputs: dict[str, torch.Tensor] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = _TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name, self.in_dtype = name, dtype
                assert shape == (1, 4, h, w), f"engine input {shape} != (1,4,{h},{w})"
            else:
                t = torch.empty(shape, dtype=dtype, device="cuda")
                self.outputs[name] = t
                self.ctx.set_tensor_address(name, t.data_ptr())
        self.mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device="cuda").view(1, 3, 1, 1)
        # TensorRT warns (and adds syncs) on the legacy default stream, so run on our own stream.
        self.stream = torch.cuda.Stream()

    def prepare(self, rgb: torch.Tensor, hint: torch.Tensor) -> torch.Tensor:
        x = (rgb.float() - self.mean) / self.std
        return torch.cat([x, hint.float()], dim=1).to(self.in_dtype).contiguous()

    def run(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns views of reusable output buffers: consume them before the next call."""
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        inp.record_stream(self.stream)
        self.ctx.set_tensor_address(self.in_name, inp.data_ptr())
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        cur.wait_stream(self.stream)
        return self.outputs["alpha"], self.outputs["fg"]

    def infer(self, rgb: torch.Tensor, hint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.run(self.prepare(rgb, hint))

    def close(self) -> None:
        del self.ctx, self.engine, self.outputs
