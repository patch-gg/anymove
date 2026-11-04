#!/usr/bin/env python
"""Film the BASE route again, today, on the same four matched moves.

Why this exists
---------------
The matched comparison between the base route, the Pika route and the VBVR
adapter is matched in PROMPT, reference, seed and dimensions - but not in
CLOCK. The base clips were filmed on 2026-10-31; the Pika and VBVR clips were
filmed on 2026-11-04, on a machine whose ComfyUI lane state differed (the
VBVR run found the other lane down and the base run did not). A cross-day
wall clock is indicative, not an A/B.

This re-films the base route today, with exactly the same reference, prompt
file, script, seed, seconds and dimensions as the base clips it is compared
against - so the wall clock becomes a same-day comparison. The output names
are NEW (`h3-base-today-<move>`): the 2026-10-31 clips are the published
control and must not be overwritten.

The GPU lease gate is IMPORTED from film_pika_batch.py, not copied, so there
is one implementation of "the other lane has no job in flight".

usage:
    python tools/film_base_batch.py [--only shake,gallop,...] [--force]
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
PREFIX = "h3-base-today-"


def build_jobs() -> list:
    jobs = []
    for move in MOVES:
        script = ROOT / "prompts" / "wolf" / "dual" / "short" / f"{move}.txt"
        jobs.append({
            "key": move,
            "out": ROOT / "runs" / "wolf" / "videos" / move / "s1" / f"{PREFIX}{move}.mp4",
            "argv": [
                PY, FILM, "--creature", "wolf", "--move", move,
                "--script", str(script),
                "--seconds", "1.6", "--seed", "42",
                "--engine", "base",
                "--out-name", f"{PREFIX}{move}",
            ],
        })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="re-film the base route today, matched")
    ap.add_argument("--only", default="", help="comma-separated move names")
    ap.add_argument("--force", action="store_true", help="re-film even if the clip exists")
    ap.add_argument("--lease-wait-sec", type=int, default=1800)
    args = ap.parse_args()

    wanted = {k.strip() for k in args.only.split(",") if k.strip()}
    jobs = [j for j in build_jobs() if not wanted or j["key"] in wanted]
    if not jobs:
        print("no jobs matched", file=sys.stderr)
        return 1

    fpb.log("=" * 70)
    fpb.log(f"START base re-film (same-day control): {', '.join(j['key'] for j in jobs)}")

    results = {}
    for job in jobs:
        out = Path(job["out"])
        if out.is_file() and not args.force:
            fpb.log(f"SKIP {job['key']}: {out} already exists "
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
