"""Configuration and LiveMode definitions (data-model.md)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

RUNTIMES = ("torch", "trt")
LOADS = ("idle", "nvenc1080p30")
PRECISIONS = ("fp16",)
SQUARE_SIZES = (512, 768, 1024, 2048)

# camera resolution id -> (width, height)
CAMERA_RES = {"720p": (1280, 720), "1080p": (1920, 1080), "2160p": (3840, 2160)}


@dataclass(frozen=True)
class Configuration:
    runtime: str
    internal_w: int
    internal_h: int
    refiner: bool
    load: str = "idle"
    precision: str = "fp16"
    posembed_interpolated: bool = False

    def __post_init__(self) -> None:
        if self.runtime not in RUNTIMES:
            raise ValueError(f"runtime must be one of {RUNTIMES}, got {self.runtime!r}")
        if self.load not in LOADS:
            raise ValueError(f"load must be one of {LOADS}, got {self.load!r}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {self.precision!r}")
        for v in (self.internal_w, self.internal_h):
            if not isinstance(v, int) or v <= 0 or v % 32:
                raise ValueError(f"internal size must be a positive multiple of 32, got {v!r}")

    @property
    def config_id(self) -> str:
        return (f"{self.runtime}-{self.precision}-{self.internal_w}x{self.internal_h}"
                f"-ref{int(self.refiner)}-{self.load}")

    @property
    def size(self) -> tuple[int, int]:
        return (self.internal_w, self.internal_h)

    def with_load(self, load: str) -> "Configuration":
        return replace(self, load=load)

    def to_dict(self) -> dict:
        return {"config_id": self.config_id, **asdict(self)}

    @classmethod
    def from_dict(cls, d: dict) -> "Configuration":
        return cls(runtime=d["runtime"], internal_w=d["internal_w"], internal_h=d["internal_h"],
                   refiner=d["refiner"], load=d["load"], precision=d.get("precision", "fp16"),
                   posembed_interpolated=d.get("posembed_interpolated", False))


MODE_IDS = ("720p20", "720p30", "1080p30", "1080p60", "2160p30", "2160p60")


@dataclass(frozen=True)
class LiveMode:
    mode_id: str
    camera_res: str
    fps: int

    def __post_init__(self) -> None:
        if self.mode_id not in MODE_IDS:
            raise ValueError(f"mode_id must be one of {MODE_IDS}, got {self.mode_id!r}")
        if self.camera_res not in CAMERA_RES:
            raise ValueError(f"camera_res must be one of {tuple(CAMERA_RES)}, got {self.camera_res!r}")
        if self.mode_id != f"{self.camera_res}{self.fps}":
            raise ValueError(f"mode_id {self.mode_id!r} does not match {self.camera_res}@{self.fps}")

    @property
    def budget_ms(self) -> float:
        return 1000.0 / self.fps


LIVE_MODES = tuple(LiveMode(m, m[: m.index("p") + 1], int(m[m.index("p") + 1:])) for m in MODE_IDS)


def live_mode(mode_id: str) -> LiveMode:
    for m in LIVE_MODES:
        if m.mode_id == mode_id:
            return m
    raise ValueError(f"unknown live mode {mode_id!r}")
