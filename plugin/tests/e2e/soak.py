"""SC-001 soak helper (T036, run by the owner on the RTX A2000).

Connects to a running OBS (obs-websocket enabled: Tools -> WebSocket Server Settings), samples
OBS's render statistics and the CorridorKey filter's status every 10 s for --minutes, and writes a
JSON summary. Start your NVENC stream or recording first, with the filter on Auto.

    uv run --project bench python plugin/tests/e2e/soak.py --source "Camera" --minutes 60
    (password: --password or the OBS_WS_PASSWORD environment variable)

SC-001 passes when OBS skipped no render frames over the run (none attributable to the filter is
inferred from zero in total) and the filter skipped at most 1% of its input frames.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", required=True, help="the source the CorridorKey filter is on")
    ap.add_argument("--filter", default="CorridorKey", help="the filter's name in OBS")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=4455)
    ap.add_argument("--password", default=os.environ.get("OBS_WS_PASSWORD", ""))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import obsws_python as obs

    cl = obs.ReqClient(host=a.host, port=a.port, password=a.password, timeout=10)

    def stats():
        return cl.send("GetStats", {}, raw=True)

    def status():
        r = cl.send("CallVendorRequest", {"vendorName": "corridorkey", "requestType": "GetFilterStatus",
                                          "requestData": {"sourceName": a.source, "filterName": a.filter}}, raw=True)
        return r["responseData"]

    st0 = status()
    if not st0.get("found"):
        print(f"filter {a.filter!r} not found on source {a.source!r}", file=sys.stderr)
        return 1
    outputs = {k: cl.send(k, {}, raw=True).get("outputActive") for k in ("GetStreamStatus", "GetRecordStatus")}
    s0, t0 = stats(), time.time()
    samples = []
    print(f"soak: {a.minutes:g} min on {a.source!r}; stream {outputs['GetStreamStatus']}, "
          f"record {outputs['GetRecordStatus']}; filter {st0['state']} {st0['quality']}", flush=True)
    while time.time() - t0 < a.minutes * 60:
        time.sleep(a.interval)
        s, f = stats(), status()
        samples.append({"t": round(time.time() - t0, 1), "active_fps": s["activeFps"],
                        "render_skipped": s["renderSkippedFrames"] - s0["renderSkippedFrames"],
                        "output_skipped": s["outputSkippedFrames"] - s0["outputSkippedFrames"],
                        "state": f["state"], "worker_ms_median": f["workerMsMedian"],
                        "render_us_p99": f["renderUsP99"], "render_us_max": f["renderUsMax"]})
        x = samples[-1]
        print(f"  {x['t']:7.0f}s fps {x['active_fps']:.2f} render skipped {x['render_skipped']} "
              f"state {x['state']} keying {x['worker_ms_median']:.1f} ms", flush=True)
    s1, st1 = stats(), status()
    submitted = st1["submittedTotal"] - st0["submittedTotal"]
    skipped = st1["skippedTotal"] - st0["skippedTotal"]
    summary = {
        "started": dt.datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
        "minutes": round((time.time() - t0) / 60, 2),
        "source": a.source, "gpu": st1["gpuName"], "quality": st1["quality"], "mode": st1["modeId"],
        "outputs_active_at_start": outputs,
        "obs_render_skipped": s1["renderSkippedFrames"] - s0["renderSkippedFrames"],
        "obs_render_total": s1["renderTotalFrames"] - s0["renderTotalFrames"],
        "obs_output_skipped": s1["outputSkippedFrames"] - s0["outputSkippedFrames"],
        "active_fps_min": min((x["active_fps"] for x in samples), default=None),
        "filter_submitted": submitted, "filter_skipped": skipped,
        "filter_skipped_pct": 100.0 * skipped / submitted if submitted else None,
        "filter_fallbacks": st1["fallbacks"] - st0["fallbacks"],
        "filter_worker_errors": st1["workerErrors"] - st0["workerErrors"],
        "keying_ms_median_end": st1["workerMsMedian"], "keying_ms_p95_end": st1["workerMsP95"],
        "added_latency_ms_end": st1["addedLatencyMs"],
        "samples": samples,
    }
    summary["sc001_pass"] = bool(summary["obs_render_skipped"] == 0 and summary["filter_skipped_pct"] is not None
                                 and summary["filter_skipped_pct"] <= 1.0)
    out = Path(a.out) if a.out else Path(__file__).resolve().parent / ".artifacts" / (
        "soak-" + dt.datetime.now().strftime("%Y%m%dT%H%M%S") + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "samples"}, indent=2))
    print(f"summary: {out}")
    return 0 if summary["sc001_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
