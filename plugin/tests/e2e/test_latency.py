"""SC-005 (T031): the added latency the filter displays agrees with a test-pattern measurement within
±1 frame.

Ground truth is the program output, i.e. what viewers get: an NVENC recording of the scene. The
filter is enabled part-way through the recording; every recorded frame's counter is decoded and
(counter - recorded frame index) is compared before and after. The step is the added latency.
(An earlier version compared GetSourceScreenshot images with the media cursor; that measured
~4.5 frames where the plugin's own frame-exact dump showed the matte on screen 1 frame after its
input, so the screenshot path is not a valid reference.)

A media source plays a 30 fps frame-counter pattern (media.counter_mp4: each frame's index as 16
black/white blocks on a green screen). For each sample the screenshot of the source (filters
applied) is decoded and compared with the media cursor read just before and after it. The same is
done with the filter disabled, which calibrates out OBS's own source/screenshot offset; the
difference between the two medians is the latency the filter adds.
"""
import statistics
import time

import pytest

import media
import obs_harness as H
from conftest import record

SRC = "counter"
N = 100
FRAME_MS = 1000.0 / 30


def sample_delays(o, n=N):
    delays, why = [], {"cursor": 0, "unreadable": 0, "out_of_range": 0}
    for _ in range(n * 5):
        if len(delays) >= n:
            return delays
        c1 = o.send("GetMediaInputStatus", {"inputName": SRC})["mediaCursor"]
        img = o.screenshot(SRC, 960, 540)
        c2 = o.send("GetMediaInputStatus", {"inputName": SRC})["mediaCursor"]
        if c1 is None or c2 is None or c2 < c1:  # skip samples across the loop point
            why["cursor"] += 1
            continue
        shown = media.decode_counter(img)
        if shown is None:
            why["unreadable"] += 1
            continue
        d = ((c1 + c2) / 2.0) / FRAME_MS - shown
        if -5 < d < 30:
            delays.append(d)
        else:
            why["out_of_range"] += 1
        time.sleep(0.05)
    raise AssertionError(f"only {len(delays)} of {n} samples: {why}")


def recorded_counters(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
        out.append(media.decode_counter(rgba))
    return out


@pytest.mark.parametrize("quality", ["512_r0", "1024x576_r0"])
def test_displayed_latency_matches_the_recording(obs_session, quality):
    o = obs_session
    if not o.send("GetInputList")["inputs"]:
        o.add_color_background("bg", (128, 128, 128))
        o.add_media(SRC, media.counter_mp4())
        o.add_filter(SRC, H.default_filter_settings(quality=quality, fallback_mode="passthrough"))
    else:
        o.set_filter(SRC, {"quality": quality})
        o.enable_filter(SRC, True)
    o.wait_status(SRC, lambda s: s["state"] == "keying" and s["quality"] == quality and s["mattes"] > 60, 60)
    o.enable_filter(SRC, False)
    time.sleep(2)
    o.start_record()
    time.sleep(5)
    o.enable_filter(SRC, True)
    o.wait_status(SRC, lambda s: s["state"] == "keying", 20)
    time.sleep(5)
    st = o.status(SRC)
    path = o.send("StopRecord")["outputPath"]
    time.sleep(2)
    vals = recorded_counters(path)
    offs = [(c - k) % media.COUNTER_FRAMES for k, c in enumerate(vals) if c is not None]
    n = len(offs)
    before = statistics.median(offs[: n // 3])
    after = statistics.median(offs[-n // 3:])
    measured = (before - after) % media.COUNTER_FRAMES
    shown = st["addedLatencyFrames"]
    record(f"sc005_latency_{quality}", {"method": "NVENC recording, filter enabled mid-recording",
                             "recorded_frames": len(vals), "decoded": n,
                             "offset_before_frames": before, "offset_after_frames": after,
                             "measured_added_frames": measured, "displayed_added_frames": shown,
                             "displayed_added_ms": st["addedLatencyMs"],
                             "pairing_delay_frames": st["pairingDelayFrames"]})
    assert n > 0.8 * len(vals)
    assert abs(measured - shown) <= 1.0, (measured, shown)
