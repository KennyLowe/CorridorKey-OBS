"""Quality pass (US3; research D11): every idle `ok` configuration over every clip vs the reference.

Reference = PyTorch fp16, 2048, refiner on, run through the same pipeline (chroma hint at 1080p,
resize to the internal size, model, resize back). Reference outputs are cached as fp16 .npy in
outputs/ref/<clip>/ and held in RAM during the pass (~22 GB for all clips at 1080p).
"""
from __future__ import annotations

import gc
import time

import numpy as np
import torch

from . import results as R
from .bench_speed import build_keyer
from .compare import CompareWriter, write_index
from .config import Configuration
from .frames import clip_frames, read_rgb
from .pipeline import key_tensor
from .quality import ClipMetrics
from .results import OUTPUTS_DIR

REF_CONFIG = Configuration(runtime="torch", internal_w=2048, internal_h=2048, refiner=True, load="idle",
                           posembed_interpolated=False)
REF_DIR = OUTPUTS_DIR / "ref"


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def _to_cuda(path) -> torch.Tensor:
    return torch.from_numpy(read_rgb(path)).cuda().permute(2, 0, 1).to(torch.float16).div_(255.0)


@torch.inference_mode()
def _key_np(keyer, path) -> tuple[np.ndarray, np.ndarray]:
    alpha, fg, _ = key_tensor(keyer, _to_cuda(path))
    return alpha.cpu().numpy(), fg.cpu().numpy()


def reference_outputs(clip_ids: list[str], log=print) -> dict:
    """{clip_id: [(alpha fp16 [H,W], fg fp16 [3,H,W]), ...]} computing and caching as needed."""
    refs: dict = {}
    keyer = None
    for cid in clip_ids:
        frames = clip_frames(cid)
        d = REF_DIR / cid
        cached = [d / f"{i:05d}.npy" for i in range(len(frames))]
        out = []
        if all(p.exists() for p in cached) and (d / "frames.txt").read_text() == str(len(frames)):
            for p in cached:
                arr = np.load(p)
                out.append((arr[3], arr[:3]))
            log(f"  reference {cid}: {len(out)} frames from cache")
        else:
            if keyer is None:
                keyer = build_keyer(REF_CONFIG, {})
            d.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            for f, p in zip(frames, cached):
                a, fg = _key_np(keyer, f)
                arr = np.concatenate([fg, a[None]], axis=0).astype(np.float16)  # [4,H,W]: fg rgb + alpha
                np.save(p, arr)
                out.append((arr[3], arr[:3]))
            (d / "frames.txt").write_text(str(len(frames)))
            log(f"  reference {cid}: {len(out)} frames computed in {time.perf_counter() - t0:.0f}s")
        refs[cid] = out
    if keyer is not None:
        keyer.close()
        del keyer
        _free()
    return refs


def candidate_configs(doc: dict, only: list[str] | None = None) -> list[Configuration]:
    seen, out = set(), []
    for b in doc.get("benchmarks", []):
        c = b["config"]
        if b["status"] != "ok" or c["load"] != "idle" or c["config_id"] in seen:
            continue
        if only and c["config_id"] not in only:
            continue
        seen.add(c["config_id"])
        out.append(Configuration.from_dict(c))
    return out


def run_quality(doc: dict, only: list[str] | None = None, log=print) -> int:
    clips = [c for c in doc.get("clips", []) if c.get("frames", 0) > 1]
    if not clips:
        raise RuntimeError("no clips in results; run `ckbench clips` first")
    clip_ids = [c["clip_id"] for c in clips]
    configs = candidate_configs(doc, only)
    log(f"[quality] {len(configs)} configurations x {len(clip_ids)} clips")
    refs = reference_outputs(clip_ids, log)
    doc.setdefault("quality", [])
    key = lambda q: (q["config_id"], q["clip_id"])  # noqa: E731
    failures = 0
    for cfg in configs:
        is_ref = (cfg.runtime, cfg.size, cfg.refiner) == ("torch", (2048, 2048), True)
        try:
            keyer = build_keyer(cfg, doc)
        except Exception as e:
            log(f"  {cfg.config_id}: keyer build failed ({type(e).__name__}: {e}); skipped")
            failures += 1
            continue
        for cid in clip_ids:
            t0 = time.perf_counter()
            frames = clip_frames(cid)
            m = ClipMetrics()
            w = CompareWriter(cfg.config_id, cid, len(frames))
            try:
                for f, (ra, rf) in zip(frames, refs[cid]):
                    ra32, rf32 = ra.astype(np.float32), rf.astype(np.float32)
                    if is_ref:
                        a, fg = ra32, rf32  # the reference config *is* the reference
                    else:
                        a, fg = _key_np(keyer, f)
                    m.update(a, fg, ra32, rf32)
                    w.add(a, fg, ra32, rf32)
            finally:
                w.close()
            res = m.result()
            entry = {"config_id": cfg.config_id, "clip_id": cid, **res, "owner_verdict": "pending",
                     "frames": len(frames)}
            if is_ref:
                entry["note"] = "this configuration is the 2048 reference; metrics are zero by definition"
            R.upsert(doc["quality"], entry, key)
            R.save(doc)
            log(f"  {cfg.config_id} x {cid}: edge_alpha {res['edge_alpha_mae']:.4f} edge_fg "
                f"{res['edge_fg_mae']:.4f} flicker+ {res['flicker_excess']:.4f} flags {res['flags']} "
                f"({time.perf_counter() - t0:.0f}s)")
        keyer.close()
        del keyer
        _free()
    write_index(doc["quality"])
    return failures
