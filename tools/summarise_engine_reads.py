#!/usr/bin/env python
"""Summarise a read set: per-engine means and the per-move head-to-head.

The read is the decision, not the wall clock. `read_pika_batch.py` reads a set
of clips through the house chain (`mocap.detect` -> `mocap.solve`) and writes
a `motion.json` per clip; the numbers that matter are the reprojection error
against the split-screen reference (front px, side px) and the sync gap
between the two panels (sync px). LOWER IS BETTER on all three - they are
pixel errors of the rig fit against the reference composite.

This summariser groups a set by ENGINE and reports:

  * the per-engine mean on each of the three metrics;
  * the per-move head-to-head, so a mean cannot hide a move where an engine
    is much worse - the number of moves each engine wins.

It reads the set definitions from `read_pika_batch.py` rather than repeating
them, so a set is defined once.

usage:
    python tools/summarise_engine_reads.py --set hf11
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(r"D:\anymove")
CLIPS = ROOT / "runs" / "wolf" / "clips"
sys.path.insert(0, str(ROOT / "tools"))
import read_pika_batch as rpb  # noqa: E402  (the set definitions live there)

METRICS = (("front", "front px"), ("side", "side px"), ("sync", "sync px"))


def read_motion(name: str) -> dict | None:
    path = CLIPS / name / "motion.json"
    if not path.is_file():
        return None
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    r = m.get("reproj_px", {}) or {}
    s = m.get("sync", {}) or {}
    return {
        "front": r.get("front"),
        "side": r.get("side"),
        "sync": s.get("mean_height_gap_px"),
        "worst": s.get("worst_frame_gap_px"),
        "calib": m.get("calib_converged", True),
    }


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def fmt(v) -> str:
    return f"{v:8.2f}" if isinstance(v, (int, float)) else f"{'-':>8}"


def main() -> int:
    ap = argparse.ArgumentParser(description="summarise a read set by engine")
    ap.add_argument("--set", default="hf11", choices=sorted(rpb.SETS),
                    help="the read set to summarise")
    args = ap.parse_args()

    jobs = rpb.SETS[args.set]()
    rows = []
    for key, _clip, name, _single, engine in jobs:
        m = read_motion(name)
        rows.append((key, engine, name, m))

    missing = [n for _k, _e, n, m in rows if m is None]
    engines = []
    for _k, engine, _n, _m in rows:
        if engine not in engines:
            engines.append(engine)

    print("=" * 96)
    print(f"READ SET {args.set} - per clip")
    print("=" * 96)
    print(f"{'clip':<28}{'engine':<12}{'front':>9}{'side':>9}{'sync':>9}{'worst':>9}{'calib':>11}")
    for key, engine, name, m in rows:
        if m is None:
            print(f"{name:<28}{engine:<12}{'(not read)':>37}")
            continue
        calib = "converged" if m["calib"] else "FALLBACK"
        print(f"{name:<28}{engine:<12}{fmt(m['front']):>9}{fmt(m['side']):>9}"
              f"{fmt(m['sync']):>9}{fmt(m['worst']):>9}{calib:>11}")

    print()
    print("=" * 96)
    print(f"READ SET {args.set} - per engine MEAN (lower is better)")
    print("=" * 96)
    print(f"{'engine':<14}{'front':>10}{'side':>10}{'sync':>10}{'moves read':>13}")
    for engine in engines:
        sel = [m for _k, e, _n, m in rows if e == engine and m is not None]
        print(f"{engine:<14}{fmt(mean([m['front'] for m in sel])):>10}"
              f"{fmt(mean([m['side'] for m in sel])):>10}"
              f"{fmt(mean([m['sync'] for m in sel])):>10}{len(sel):>13}")

    print()
    print("=" * 96)
    print(f"READ SET {args.set} - per-move HEAD TO HEAD (lower wins)")
    print("=" * 96)
    moves = []
    for key, _c, _n, _s, _e in jobs:
        if key not in moves:
            moves.append(key)
    tally = {engine: 0 for engine in engines}
    for key in moves:
        per = {engine: None for engine in engines}
        for k, engine, _n, m in rows:
            if k != key or m is None:
                continue
            per[engine] = m
        if any(v is None for v in per.values()):
            print(f"{key:<12} incomplete: " + ", ".join(
                f"{e}={'-' if v is None else 'read'}" for e, v in per.items()))
            continue
        scored = []
        for metric, _label in METRICS:
            best = min(per[e][metric] for e in engines
                       if isinstance(per[e][metric], (int, float)))
            for e in engines:
                if per[e][metric] == best:
                    scored.append((metric, e))
        # one win per metric, to the engine that is best on it
        for metric, e in scored:
            tally[e] += 1
        detail = "  ".join(
            f"{label} " + " ".join(f"{e}={per[e][metric]:.2f}" for e in engines)
            for metric, label in METRICS)
        print(f"{key:<12} {detail}")
    print()
    print(f"metric wins across {len(moves)} moves x {len(METRICS)} metrics "
          f"= {len(moves) * len(METRICS)}:")
    for engine in engines:
        print(f"   {engine:<14}{tally[engine]}")

    if missing:
        print()
        print("not read yet: " + ", ".join(missing))
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
