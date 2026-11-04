#!/usr/bin/env python
"""Film the Licon VBVR adapter on top of the house base route.

VBVR is a QUALITY adapter for MiniMax-H3 (object consistency, no unintended
duplication, no clipping/interpenetration), not a speed adapter, so it is
CHAINED after the engine's own turbo LoRA rather than replacing it. The
result is a third column in the same matched experiment: the same reference,
the same prompt, the same seed, the same dimensions and the same frame count
as the base and Pika clips, with one extra adapter in the graph.

The GPU lease gate is IMPORTED from film_pika_batch.py, not copied, so there
is one implementation of "the other lane has no job in flight".

usage:
    python tools/film_vbvr_batch.py [--only shake,gallop,...] [--strength 1.0]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import film_pika_batch as fpb  # noqa: E402  (the lease gate lives there)

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
FILM = str(ROOT / "tools" / "film_h3.py")
VBVR_LORA = "H3-Licon-VBVR-I2V.comfy.safetensors"
MOVES = ("shake", "gallop", "howl", "pounce")


def build_jobs(strength: float) -> list:
    jobs = []
    for move in MOVES:
        script = ROOT / "prompts" / "wolf" / "dual" / "short" / f"{move}.txt"
        jobs.append({
            "key": move,
            "out": ROOT / "runs" / "wolf" / "videos" / move / "s1" / f"h3-vbvr-{move}.mp4",
            "argv": [
                PY, FILM, "--creature", "wolf", "--move", move,
                "--script", str(script),
                "--seconds", "1.6", "--seed", "42",
                "--engine", "base",
                "--extra-lora", VBVR_LORA,
                "--extra-lora-strength", str(strength),
                "--out-name", f"h3-vbvr-{move}",
            ],
        })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="film the VBVR adapter on the base route")
    ap.add_argument("--only", default="", help="comma-separated move names")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--force", action="store_true", help="re-film even if the clip exists")
    ap.add_argument("--lease-wait-sec", type=int, default=1800)
    args = ap.parse_args()

    wanted = {k.strip() for k in args.only.split(",") if k.strip()}
    jobs = [j for j in build_jobs(args.strength) if not wanted or j["key"] in wanted]
    if not jobs:
        print("no jobs matched", file=sys.stderr)
        return 1

    fpb.log("=" * 70)
    fpb.log(f"START vbvr batch (base route + {VBVR_LORA} @ {args.strength}): "
            f"{', '.join(j['key'] for j in jobs)}")

    results = {}
    for job in jobs:
        out = Path(job["out"])
        if out.is_file() and not args.force:
            fpb.log(f"SKIP {job['key']}: {out} already exists "
                    f"({out.stat().st_size / 2**20:.2f} MB)")
            results[job["key"]] = {"status": "skipped", "path": str(out)}
            continue

        fpb.wait_lease(args.strength and 3800 or 3800, args.lease_wait_sec)
        fpb.log(f"---- filming {job['key']} ----")
        t0 = time.time()
        proc = subprocess.run(job["argv"], cwd=str(ROOT), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=7200)
        wall = round(time.time() - t0, 1)
        for ln in [l for l in ((proc.stdout or "") + (proc.stderr or "")).splitlines()
                   if l.strip()][-12:]:
            fpb.log(f"    {ln}")
        if proc.returncode != 0:
            fpb.log(f"    FAILED rc={proc.returncode}")
            results[job["key"]] = {"status": "failed", "rc": proc.returncode}
            continue
        fpb.log(f"    OK {job['key']} in {wall}s -> {out}")
        results[job["key"]] = {"status": "ok", "wall_s": wall, "path": str(out),
                               "bytes": out.stat().st_size if out.is_file() else None}

    fpb.log(f"DONE {json.dumps(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
