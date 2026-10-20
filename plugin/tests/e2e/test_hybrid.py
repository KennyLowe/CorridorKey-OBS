"""Hybrid key in real OBS 32.2.2 on a hard case with a known matte (media.hard_case).

Static: compares the matte of OBS's own Chroma Key, CorridorKey with "Use source colour", and
CorridorKey's Hybrid against the true alpha, region by region: solid subject, clean screen, edges, a
sage-green patch of clothing, a dark crease on the screen, thin strands and a translucent panel.
Hybrid should keep the colour key's clean, full-resolution interior and fix what it gets wrong.

Moving: the subject slides sideways at up to ~20 px per frame. The model's matte is about a frame old,
so a key that trusted it against the current frame would cut a hole at the leading edge and leave a
ghost behind the trailing one; Hybrid only uses the model where the current frame's key is unsure.
"""
import time

import cv2
import numpy as np
import pytest

import media
import obs_harness as H
from conftest import record

SRC = "hard"
MOVING = "hard-moving"
Q = "768x448_r0"  # what the RTX A2000 runs at 1080p30


def keying(s):
    return s.get("found") and s["state"] in ("keying", "behind")


def calibrated(s):
    """Hybrid has calibrated the screen colour (it switches the model's hint to it too)."""
    return keying(s) and s.get("hintMode") == "calibrated" and (s["hintR"], s["hintG"], s["hintB"]) != (0, 1, 0)


def shot(o, src, settle=1.5):
    time.sleep(settle)
    return o.screenshot(src, media.HARD_W, media.HARD_H).astype(np.float32) / 255.0


def save_still(name: str, img: np.ndarray) -> None:
    out = H.ARTIFACTS / "stills"
    out.mkdir(parents=True, exist_ok=True)
    comp = img[..., :3] * img[..., 3:4] + 0.5 * (1 - img[..., 3:4])
    cv2.imwrite(str(out / f"{name}.png"), cv2.cvtColor((comp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def regions(truth) -> dict:
    """media.hard_case's regions, with the screen and the crease measured at least 20 px from the
    subject (nearer, the edge band legitimately uses the model's soft edge)."""
    near = cv2.dilate((truth["alpha"] > 0.02).astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
    out = {r: truth[r] for r in media.HARD_REGIONS}
    out["screen"] = out["screen"] & ~near
    out["shadow"] = out["shadow"] & ~near
    return out


def region_mae(alpha: np.ndarray, truth) -> dict:
    out = {"all": float(np.abs(alpha - truth["alpha"]).mean())}
    for r, m in regions(truth).items():
        out[r] = float(np.abs(alpha - truth["alpha"])[m].mean())
    return out


def render_ms(o, seconds: float = 4.0) -> float:
    """OBS's mean frame render time over a few seconds."""
    vals = []
    for _ in range(int(seconds / 0.25)):
        vals.append(o.stats()["averageFrameRenderTime"])
        time.sleep(0.25)
    return float(np.median(vals))


@pytest.fixture(scope="module")
def hard(obs_session):
    o = obs_session
    mp4, truth = media.hard_case()
    o.add_media(SRC, mp4)
    o.add_filter(SRC, H.default_filter_settings(quality=Q, source_colour=True))
    o.wait_status(SRC, lambda s: keying(s) and s["quality"] == Q and s["mattes"] > 10, 600)
    yield o, np.load(truth)
    o.send("RemoveInput", {"inputName": SRC})


def test_hybrid_static_hard_case(hard):
    o, truth = hard
    shots = {"source_colour": shot(o, SRC)}
    o.set_filter(SRC, {"hybrid": True})
    o.wait_status(SRC, calibrated, 30)
    shots["hybrid"] = shot(o, SRC)
    assert keying(o.status(SRC))
    assert not any("effect" in line.lower() and "error" in line.lower() for line in o.ck_log())
    o.enable_filter(SRC, False)
    o.send("CreateSourceFilter", {"sourceName": SRC, "filterName": "obs-chroma", "filterKind": "chroma_key_filter_v2",
                                  "filterSettings": {}})
    shots["obs_chroma_key"] = shot(o, SRC)
    o.remove_filter(SRC, "obs-chroma")
    o.enable_filter(SRC, True)

    alpha = {k: v[..., 3] for k, v in shots.items()}
    mae = {k: region_mae(a, truth) for k, a in alpha.items()}
    for k, v in shots.items():
        save_still(f"hard-case-{k}", v)
    np.savez_compressed(H.ARTIFACTS / "hard-case-mattes.npz", **alpha)
    hy, reg = alpha["hybrid"], regions(truth)
    checks = {
        "subject_holes": float((hy[reg["subject"]] < 0.5).mean()),
        "screen_spots": float((hy[reg["screen"]] > 0.05).mean()),
        "shadow_mean": float(hy[reg["shadow"]].mean()),
        "green_cloth_mean": float(hy[reg["green_cloth"]].mean()),
        "glass_mean": float(hy[reg["glass"]].mean()),
    }
    record("hybrid_static", {"quality": Q, "alpha_mae": mae, "hybrid_checks": checks})
    for k, v in mae.items():
        print(k, {r: round(x, 4) for r, x in v.items()})
    print("hybrid", checks)

    ck, cs, h = mae["obs_chroma_key"], mae["source_colour"], mae["hybrid"]
    assert h["all"] < min(ck["all"], cs["all"]), mae  # closest to the truth overall
    assert checks["subject_holes"] < 0.001, checks  # the interior stays solid
    assert checks["screen_spots"] < 0.001, checks  # the clean screen stays clear
    assert checks["shadow_mean"] < 0.05, checks  # the crease keys out (OBS's chroma key keeps it)
    assert checks["green_cloth_mean"] > 0.95, checks  # green-ish clothing stays (both others punch holes)
    assert h["edge"] <= cs["edge"] + 0.02, mae
    assert h["strands"] <= cs["strands"] + 0.02, mae


@pytest.fixture(scope="module")
def moving(obs_session):
    o = obs_session
    mp4, layers = media.hard_case_moving()
    o.add_media(MOVING, mp4)
    o.add_filter(MOVING, H.default_filter_settings(quality=Q, source_colour=True))
    o.wait_status(MOVING, lambda s: keying(s) and s["quality"] == Q and s["mattes"] > 10, 600)
    yield o, layers
    o.send("RemoveInput", {"inputName": MOVING})


def lag_errors(o, layers, n: int = 24) -> dict:
    """Leading holes and trailing ghosts near the true edge, over n screenshots of the moving clip."""
    holes = ghosts = near = 0
    for _ in range(n):
        img = o.screenshot(MOVING, media.HARD_W, media.HARD_H)
        i = media.decode_strip(img[..., :3])
        if i is None or i >= media.MOVE_FRAMES:
            continue
        _, truth = media.moving_frame(i, layers)
        edge = ((truth > 0.02) & (truth < 0.98)).astype(np.uint8)
        zone = cv2.dilate(edge, np.ones((51, 51), np.uint8)).astype(bool)
        a = img[..., 3].astype(np.float32) / 255.0
        holes += int(((truth > 0.98) & (a < 0.5) & zone).sum())
        ghosts += int(((truth < 0.02) & (a > 0.3) & zone).sum())
        near += int(zone.sum())
        time.sleep(0.07)
    assert near > 0, "no screenshot could be matched to a frame"
    return {"hole_frac": holes / near, "ghost_frac": ghosts / near}


def test_hybrid_moving_no_lag_artefacts(moving):
    o, layers = moving
    time.sleep(1)
    res = {"source_colour": lag_errors(o, layers)}
    ms_src = render_ms(o)
    o.set_filter(MOVING, {"hybrid": True})
    o.wait_status(MOVING, calibrated, 30)
    time.sleep(1)
    res["hybrid"] = lag_errors(o, layers)
    ms_hy = render_ms(o)
    record("hybrid_moving", {"quality": Q, "amp_px": media.MOVE_AMP_PX, **res,
                             "render_ms_source_colour": ms_src, "render_ms_hybrid": ms_hy})
    print(res, "render ms", ms_src, ms_hy)
    lag = {k: v["hole_frac"] + v["ghost_frac"] for k, v in res.items()}
    # What remains is the edge band, where the frame-old matte is used by design.
    assert res["hybrid"]["hole_frac"] < 0.01, res
    assert res["hybrid"]["ghost_frac"] < 0.01, res
    assert lag["hybrid"] < lag["source_colour"] / 3, res
    assert ms_hy - ms_src < 1.0, (ms_src, ms_hy)  # the 25-tap key costs well under a millisecond


def paused_source(o, name: str, mp4) -> None:
    o.add_media(name, mp4)
    o.add_filter(name, H.default_filter_settings(quality=Q, source_colour=True))
    o.wait_status(name, lambda s: keying(s) and s["quality"] == Q and s["mattes"] > 10, 600)
    o.send("TriggerMediaInputAction", {"inputName": name, "mediaAction": "OBS_WEBSOCKET_MEDIA_INPUT_ACTION_PAUSE"})


def test_hybrid_matches_chroma_key_on_a_clean_screen(obs_session):
    """The field report's case: on a clean, even screen OBS's Chroma Key already looks right, so away
    from the edge Hybrid's matte must agree with it (and the edge may use the model)."""
    o, name = obs_session, "clean"
    paused_source(o, name, media.clip_mp4("hcw-locked-backdrop"))
    time.sleep(1.5)
    o.set_filter(name, {"hybrid": True})
    o.wait_status(name, calibrated, 30)
    hy = shot(o, name)[..., 3]
    o.enable_filter(name, False)
    o.send("CreateSourceFilter", {"sourceName": name, "filterName": "obs-chroma", "filterKind": "chroma_key_filter_v2",
                                  "filterSettings": {}})
    ck = shot(o, name)[..., 3]
    o.send("RemoveInput", {"inputName": name})
    edge = ((ck > 0.02) & (ck < 0.98)).astype(np.uint8)
    far = ~cv2.dilate(edge, np.ones((21, 21), np.uint8)).astype(bool)
    off = float((np.abs(hy - ck)[far] > 0.1).mean())
    record("hybrid_clean_screen", {"quality": Q, "disagree_frac_away_from_edge": off,
                                   "subject_frac": float((ck[far] > 0.5).mean())})
    assert off < 0.005, off


def test_hybrid_without_a_green_screen_shows_source_colour(obs_session):
    """No green screen to calibrate on: Hybrid must not key with a wrong screen colour; it shows the
    'Use source colour' output and says why."""
    o, name = obs_session, "noscreen"
    paused_source(o, name, media.no_screen_mp4())
    time.sleep(1.5)
    before = shot(o, name)
    o.set_filter(name, {"hybrid": True})
    st = o.wait_status(name, lambda s: "Hybrid needs" in s.get("status", ""), 30)
    after = shot(o, name)
    o.send("RemoveInput", {"inputName": name})
    diff = float(np.abs(after - before).mean())
    record("hybrid_no_screen", {"status": st["status"], "mean_abs_diff": diff})
    assert diff < 0.002, diff
