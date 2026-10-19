"""'Use source colour' in real OBS 32.2.2: the model's matte with the live frame's colour.

On a paused clip (the same frame for every screenshot) it checks that, with the setting on,
- the matte is unchanged (alpha matches the setting-off screenshot),
- solid foreground pixels are the live frame's own pixels with the green clamp applied,
- that foreground is sharper than the model's enlarged low-resolution colour,
- and switching it off again restores the old look.
"""
import time

import cv2
import numpy as np
import pytest

import media
import obs_harness as H
from conftest import record

SRC = "srccol"
Q = "512_r0"  # the lowest-resolution model colour, where the difference is largest


def keying(s):
    return s.get("found") and s["state"] in ("keying", "behind")


def sharpness(rgb: np.ndarray, mask: np.ndarray) -> float:
    grey = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F)[mask].var())


@pytest.fixture(scope="module")
def paused(obs_session):
    o = obs_session
    o.add_media(SRC, media.clip_mp4("hcw-godiva-close"))
    o.add_filter(SRC, H.default_filter_settings(quality=Q))
    o.wait_status(SRC, lambda s: keying(s) and s["quality"] == Q and s["mattes"] > 10, 120)
    o.send("TriggerMediaInputAction", {"inputName": SRC, "mediaAction": "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_PAUSE"})
    time.sleep(1.5)
    yield o
    o.remove_filter(SRC)
    o.send("RemoveInput", {"inputName": SRC})


def shot(o, settle=1.0):
    time.sleep(settle)
    return o.screenshot(SRC, 1920, 1080).astype(np.float32)


def test_source_colour_keeps_matte_and_uses_live_pixels(paused):
    o = paused
    off = shot(o)
    o.enable_filter(SRC, False)
    raw = shot(o)
    o.enable_filter(SRC, True)
    o.wait_status(SRC, keying, 30)
    o.set_filter(SRC, {"source_colour": True})
    on = shot(o)
    s = o.status(SRC)
    assert keying(s), s
    assert not any("effect" in line.lower() and "error" in line.lower() for line in o.ck_log())

    a_off, a_on = off[..., 3] / 255.0, on[..., 3] / 255.0
    alpha_diff = float(np.abs(a_off - a_on).mean())
    assert alpha_diff < 0.01, alpha_diff  # same matte

    solid = (a_on > 0.99) & (a_off > 0.99)
    assert solid.mean() > 0.05, solid.mean()  # enough subject to compare
    want = raw[..., :3].copy()
    want[..., 1] = np.minimum(want[..., 1], np.maximum(want[..., 0], want[..., 2]))
    live_err = float(np.abs(on[..., :3] - want)[solid].mean())
    model_err = float(np.abs(off[..., :3] - want)[solid].mean())
    assert live_err < 1.5, live_err  # the live frame's own pixels (8-bit rounding only)

    sh_off, sh_on = sharpness(off[..., :3], solid), sharpness(on[..., :3], solid)
    assert sh_on > 1.5 * sh_off, (sh_off, sh_on)

    o.set_filter(SRC, {"source_colour": False})
    back = shot(o)
    restored_err = float(np.abs(back - off).mean())
    assert restored_err < 1.0, restored_err

    out = H.ARTIFACTS / "stills"
    out.mkdir(parents=True, exist_ok=True)
    for name, img in (("off", off), ("on", on)):
        a = img[..., 3:4] / 255.0
        comp = img[..., :3] * a + 128.0 * (1 - a)
        cv2.imwrite(str(out / f"source-colour-{name}-{Q}.png"), cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2BGR))
    record("source_colour", {"quality": Q, "alpha_mean_abs_diff": alpha_diff, "solid_fraction": float(solid.mean()),
                             "live_pixel_err": live_err, "model_colour_err": model_err,
                             "sharpness_off": sh_off, "sharpness_on": sh_on, "restored_err": restored_err})
