"""US6 (T033): screen-colour calibration on pexels-athlete, the clip where feature 001's fixed hint
failed. With "Show hint", the rough mask's mean over a 32 px border (where the screen is) must drop
below 0.25 after calibration (the 001 diagnostic)."""
import time

import numpy as np

import media
import obs_harness as H
from conftest import record

SRC = "athlete"


def border_mean(o) -> float:
    img = o.screenshot(SRC, 1920, 1080)[..., 0].astype(np.float64) / 255.0
    b = 32
    band = np.concatenate([img[:b].ravel(), img[-b:].ravel(), img[:, :b].ravel(), img[:, -b:].ravel()])
    return float(band.mean())


def test_calibration_separates_screen_from_subject(obs_session):
    o = obs_session
    o.add_color_background("bg", (128, 128, 128))
    o.add_media(SRC, media.clip_mp4("pexels-athlete"))
    o.add_filter(SRC, H.default_filter_settings(quality="512_r0", show_hint=True, hint_mode="fixed"))
    o.wait_status(SRC, lambda s: s["state"] in ("keying", "behind"), 60)
    time.sleep(1)
    fixed = border_mean(o)

    o.set_filter(SRC, {"calibrate_requested": 1})
    s = o.wait_status(SRC, lambda s: s["calibration"].startswith("Calibrated"), 20)
    s = o.wait_status(SRC, lambda s: s["hintMode"] == "calibrated" and s["hintG"] != 1.0, 10)
    time.sleep(1)
    calibrated = border_mean(o)
    key = [round(s[k] * 255) for k in ("hintR", "hintG", "hintB")]
    record("us6_calibration", {"clip": "pexels-athlete", "hint_border_mean_fixed": fixed,
                               "hint_border_mean_calibrated": calibrated, "key_rgb": key,
                               "status": s["calibration"]})
    assert fixed > 0.5  # the fixed key doesn't recognise this screen (001: 0.97)
    assert calibrated < 0.25
    # the key itself still runs on the calibrated hint
    o.set_filter(SRC, {"show_hint": False})
    s = o.wait_status(SRC, lambda s: s["state"] in ("keying", "behind") and s["output"] == "composite", 10)
