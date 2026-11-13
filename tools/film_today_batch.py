#!/usr/bin/env python
"""Film several engines back-to-back in ONE session, with matched settings.

Why this exists
---------------
Wall clock on this box is not a stable measurement. The SAME base-route clip
with the SAME prompt, seed, steps and dimensions took 390.3 s on 2026-10-31
and 510.5 s on 2026-11-05 - a 31 % swing from machine state alone (what else
is resident, whether the other ComfyUI lane is up, how loaded the CPU is).

So a timing table built from clips filmed on different days is not a
comparison. This script films every engine CONSECUTIVELY in one session, with
the identical reference, prompt file, script, seed, seconds and dimensions, so
the only difference between the runs is the engine (and the adapter chain).

The output names are NEW (`<prefix>-<move>`): the published clips are the
control and must not be overwritten.

The GPU lease gate is IMPORTED from film_pika_batch.py, not copied, so there
is one implementation of "the other lane has no job in flight".

usage:
    python tools/film_today_batch.py \
        --set base:h3-base-today \
        --set pika-flex:h3-pika-today \
        --set base+vbvr:h3-vbvr-today
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
MOVES = ("shake", "gallop", "howl", "pounce")
VBVR_LORA = "H3-Licon-VBVR-I2V.comfy.safetensors"


def parse_set(spec: str) -> tuple[str, str]:
    """"engine:prefix" -> (engine, prefix)."""
    engine, _, prefix = spec.partition(":")
    if not engine or not prefix:
        raise argparse.ArgumentTypeError(f"--set needs engine:prefix, got {spec!r}")
    return engine.strip(), prefix.strip()


def build_jobs(engine: str, prefix: str) -> list:
    jobs = []
    for move in MOVES:
        script = ROOT / "prompts" / "wolf" / "dual" / "short" / f"{move}.txt"
        argv = [
            PY, FILM, "--creature", "wolf", "--move", move,
            "--script", str(script),
            "--seconds", "1.6", "--seed", "42",
            "--out-name", f"{prefix}-{move}",
        ]
        if engine == "base+vbvr":
            argv += ["--engine", "base",
                     "--extra-lora", VBVR_LORA, "--extra-lora-strength", "1.0"]
        else:
            argv += ["--engine", engine]
        jobs.append({
            "key": f"{prefix}-{move}",
            "out": ROOT / "runs" / "wolf" / "videos" / move / "s1" / f"{prefix}-{move}.mp4",
            "argv": argv,
        })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="film several engines back-to-back, matched")
    ap.add_argument("--set", action="append", default=[], metavar="engine:prefix",
                    help="repeatable: e.g. base:h3-base-today, pika-flex:h3-pika-today")
    ap.add_argument("--only", default="", help="comma-separated move names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lease-wait-sec", type=int, default=1800)
    args = ap.parse_args()

    if not args.set:
        print("no --set given", file=sys.stderr)
        return 1
    wanted_moves = {k.strip() for k in args.only.split(",") if k.strip()}

    results = {}
    for spec in args.set:
        engine, prefix = parse_set(spec)
        jobs = build_jobs(engine, prefix)
        if wanted_moves:
            jobs = [j for j in jobs if j["key"].rsplit("-", 1)[-1] in wanted_moves]
        fpb.log("=" * 70)
        fpb.log(f"START {engine} -> {prefix}: {', '.join(j['key'] for j in jobs)}")

        for job in jobs:
            out = Path(job["out"])
            if out.is_file() and not args.force:
                fpb.log(f"SKIP {job['key']}: already exists "
                        f"({out.stat().st_size / 2**20:.2f} MB)")
                results[job["key"]] = {"status": "skipped", "path": str(out)}
                continue

            fpb.wait_lease(3800, args.lease_wait_sec)
            fpb.log(f"---- filming {job['key']} ----")
            t0 = time.time()
            proc = subprocess.run(job["argv"], cwd=str(ROOT), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace",
                                  timeout=7200)
            wall = round(time.time() - t0, 1)
            for ln in [l for l in ((proc.stdout or "") + (proc.stderr or "")).splitlines()
                       if l.strip()][-10:]:
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
