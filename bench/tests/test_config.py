import pytest

from ckbench.config import LIVE_MODES, Configuration, LiveMode, live_mode


def test_config_id_slug():
    c = Configuration(runtime="torch", internal_w=768, internal_h=768, refiner=True, load="idle")
    assert c.config_id == "torch-fp16-768x768-ref1-idle"
    c2 = Configuration(runtime="trt", internal_w=1024, internal_h=576, refiner=False, load="nvenc1080p30")
    assert c2.config_id == "trt-fp16-1024x576-ref0-nvenc1080p30"


@pytest.mark.parametrize("w,h", [(500, 512), (512, 500), (0, 512)])
def test_rejects_non_multiple_of_32(w, h):
    with pytest.raises(ValueError):
        Configuration(runtime="torch", internal_w=w, internal_h=h, refiner=True, load="idle")


def test_rejects_bad_runtime_and_load():
    with pytest.raises(ValueError):
        Configuration(runtime="onnx", internal_w=512, internal_h=512, refiner=True, load="idle")
    with pytest.raises(ValueError):
        Configuration(runtime="torch", internal_w=512, internal_h=512, refiner=True, load="busy")
    with pytest.raises(ValueError):
        Configuration(runtime="torch", internal_w=512, internal_h=512, refiner=True, load="idle", precision="fp32")


def test_config_dict_roundtrip():
    c = Configuration(runtime="trt", internal_w=512, internal_h=512, refiner=False, load="idle",
                      posembed_interpolated=True)
    d = c.to_dict()
    assert d["config_id"] == c.config_id and d["precision"] == "fp16"
    assert Configuration.from_dict(d) == c


def test_with_load():
    c = Configuration(runtime="torch", internal_w=512, internal_h=512, refiner=False, load="idle")
    assert c.with_load("nvenc1080p30").config_id == "torch-fp16-512x512-ref0-nvenc1080p30"


def test_live_modes():
    ids = [m.mode_id for m in LIVE_MODES]
    assert ids == ["720p20", "720p30", "1080p30", "1080p60", "2160p30", "2160p60"]
    m = live_mode("1080p30")
    assert m.camera_res == "1080p" and m.fps == 30
    assert m.budget_ms == pytest.approx(1000 / 30)
    assert live_mode("720p20").budget_ms == pytest.approx(50.0)
    with pytest.raises(ValueError):
        LiveMode(mode_id="480p30", camera_res="480p", fps=30)
