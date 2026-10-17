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


# --- calibrated hint (spec Session 2026-10-17: per-clip screen colour from the frame border) ---
from ckbench.hint import calibrated_hint, estimate_key_colour  # noqa: E402


def _scene(screen, subject=(0.55, 0.50, 0.40), h=96, w=160, noise=0.01, seed=0):
    g = torch.Generator().manual_seed(seed)
    f = torch.tensor(screen).view(3, 1, 1).expand(3, h, w).clone()
    f[:, 30:96, 60:100] = torch.tensor(subject).view(3, 1, 1)          # body touching the bottom edge
    f[:, 10:30, 70:90] = torch.tensor((0.70, 0.52, 0.45)).view(3, 1, 1)  # skin-toned head
    return (f + noise * torch.randn(f.shape, generator=g)).clamp(0, 1)


@pytest.mark.parametrize("screen", [(0.32, 0.38, 0.20),   # ungraded flat/olive screen (Pexels-like)
                                    (0.25, 0.55, 0.35)])  # off-shade green
def test_calibrated_hint_keys_off_shade_screens(screen):
    f = _scene(screen)
    key = estimate_key_colour([f])
    assert key == pytest.approx(screen, abs=0.02)
    m = calibrated_hint(f, key)
    assert m.shape == f.shape[1:]
    assert float(m[:, :40].mean()) < 0.05            # screen
    assert float(m[45:90, 65:95].mean()) > 0.95       # body
    assert float(m[14:26, 74:86].mean()) > 0.95       # head


def test_fixed_hint_fails_on_flat_screen():
    # documents why calibration is needed: the fixed thresholds treat the flat screen as subject
    m = chroma_hint(_scene((0.32, 0.38, 0.20)))
    assert float(m[:, :40].mean()) > 0.5


def test_estimate_key_colour_rejects_non_green():
    with pytest.raises(ValueError):
        estimate_key_colour([_scene((0.5, 0.5, 0.5))])
