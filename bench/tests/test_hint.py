import pytest
import torch

from ckbench.hint import chroma_hint

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _frame(rgb, h=32, w=48, device="cpu", dtype=torch.float32):
    return torch.tensor(rgb, dtype=dtype, device=device).view(3, 1, 1).expand(3, h, w).contiguous()


@pytest.mark.parametrize("device", DEVICES)
def test_pure_green_is_background(device):
    m = chroma_hint(_frame([0.1, 0.8, 0.1], device=device))
    assert m.shape == (32, 48)
    assert float(m.max()) < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_grey_is_foreground(device):
    m = chroma_hint(_frame([0.5, 0.5, 0.5], device=device))
    assert float(m.min()) > 1 - 1e-3


def test_range_and_soft_edge():
    f = _frame([0.1, 0.8, 0.1])
    f[:, :, :24] = 0.5  # left half grey subject
    m = chroma_hint(f)
    assert m.shape == (32, 48)
    assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0
    row = m[16]
    assert row[0] == pytest.approx(1.0, abs=1e-4) and row[-1] == pytest.approx(0.0, abs=1e-4)
    assert 0.0 < float(row[23]) < 1.0  # blur softens the edge


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp16_cuda_dtype_preserved():
    m = chroma_hint(_frame([0.5, 0.5, 0.5], device="cuda", dtype=torch.float16))
    assert m.dtype == torch.float16 and m.device.type == "cuda"
