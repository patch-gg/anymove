#!/usr/bin/env python
"""Read a batch of filmed clips through the house chain and compare them.

Two-camera clips (base H3 and Pika H3 Flex FL2V) go through the PLAIN upstream
read: mocap.detect -> mocap.solve, against the split-screen reference.

The reference-to-video clip is a SINGLE shot, so it goes through the new
single-view path: mocap.detect --single-view against the side-view reference,
and mocap.solve's 2.5D fit (Y and Z from the side view, X held at rest).

The comparison table says which numbers are comparable and which are not.

usage:
    python tools/read_pika_batch.py [--only shake,gallop,...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"D:\anymove")
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
CLIPS = ROOT / "runs" / "wolf" / "clips"
VIDEOS = ROOT / "runs" / "wolf" / "videos"
REF_SPLIT = ROOT / "runs" / "wolf" / "ref" / "ref.png"
#: The single-camera reference. It is the house side camera ON THE STUDIO
#: BACKDROP - the side panel of ref.png, in the side panel's own columns.
#: It is NOT ref_side.png: that is the raw alpha render (the wolf on
#: transparent white, tight-cropped, no floor band), and a reference-to-video
#: clip is a grey-studio shot with a floor band. Measured on the R2V shake
#: clip, ECC against ref_side.png does not converge at all (residual 148 px);
#: against this studio panel it converges (cc 0.86, residual 84 px).
REF_SIDE = ROOT / "runs" / "wolf" / "ref" / "ref_side_studio.png"
RIG = ROOT / "assets" / "wolf" / "rigged" / "rig_info.json"
READER = ROOT / "runs" / "wolf" / "reader.pt"

#: (key, clip file, capture name, single view?, engine label)
#:
#: The BASE route was filmed on all four moves too (the clips are on disk from
#: 2026-10-31, with the SAME prompt_file and script per move - only the engine
#: differs), so the comparison is matched move for move. Reading only the base
#: shake clip was what made the earlier "Pika's front panel is better" claim
#: look general when it was one clip.
#:
#: The VBVR route is a THIRD engine on the same base route: base H3 with the
#: house turbo LoRA plus the VBVR quality adapter stacked after it. Same
#: reference, same prompt, same seed, same dimensions - only the adapter chain
#: differs. The engine label is explicit rather than derived from the file
#: name, because three engines now share one naming scheme.
PUBLISHED_JOBS = [
    ("shake", VIDEOS / "shake" / "s1" / "h3-local-shake.mp4", "h3-local-shake", False, "base H3"),
    ("shake", VIDEOS / "shake" / "s1" / "h3-pika-shake.mp4", "h3-pika-shake", False, "Pika FL2V"),
    ("shake", VIDEOS / "shake" / "s1" / "h3-vbvr-shake.mp4", "h3-vbvr-shake", False, "base+VBVR"),
    ("gallop", VIDEOS / "gallop" / "s1" / "h3-local-gallop.mp4", "h3-local-gallop", False, "base H3"),
    ("gallop", VIDEOS / "gallop" / "s1" / "h3-pika-gallop.mp4", "h3-pika-gallop", False, "Pika FL2V"),
    ("gallop", VIDEOS / "gallop" / "s1" / "h3-vbvr-gallop.mp4", "h3-vbvr-gallop", False, "base+VBVR"),
    ("howl", VIDEOS / "howl" / "s1" / "h3-local-howl.mp4", "h3-local-howl", False, "base H3"),
    ("howl", VIDEOS / "howl" / "s1" / "h3-pika-howl.mp4", "h3-pika-howl", False, "Pika FL2V"),
    ("howl", VIDEOS / "howl" / "s1" / "h3-vbvr-howl.mp4", "h3-vbvr-howl", False, "base+VBVR"),
    ("pounce", VIDEOS / "pounce" / "s1" / "h3-local-pounce.mp4", "h3-local-pounce", False, "base H3"),
    ("pounce", VIDEOS / "pounce" / "s1" / "h3-pika-pounce.mp4", "h3-pika-pounce", False, "Pika FL2V"),
    ("pounce", VIDEOS / "pounce" / "s1" / "h3-vbvr-pounce.mp4", "h3-vbvr-pounce", False, "base+VBVR"),
    ("r2v-side", VIDEOS / "shake" / "s1" / "h3-pika-r2v-shake-side.mp4",
     "h3-pika-r2v-shake-side", True, "Pika R2V"),
]

#: The SAME-SESSION re-films (2026-11-05): every move, every engine, one day,
#: one machine state. The published set was filmed across three days
#: (2026-10-31, 11-03/04) on a machine whose ComfyUI lane state changed between
#: days, so its wall clock is NOT a controlled comparison - this set is. The
#: prompts, seeds, samplers, schedule, VAE and reference are the same per move.
TODAY_JOBS = [
    ("shake", VIDEOS / "shake" / "s1" / "h3-base-today-shake.mp4", "h3-base-today-shake", False, "base H3"),
    ("shake", VIDEOS / "shake" / "s1" / "h3-pika-today-shake.mp4", "h3-pika-today-shake", False, "Pika FL2V"),
    ("shake", VIDEOS / "shake" / "s1" / "h3-vbvr-today-shake.mp4", "h3-vbvr-today-shake", False, "base+VBVR"),
    ("gallop", VIDEOS / "gallop" / "s1" / "h3-base-today-gallop.mp4", "h3-base-today-gallop", False, "base H3"),
    ("gallop", VIDEOS / "gallop" / "s1" / "h3-pika-today-gallop.mp4", "h3-pika-today-gallop", False, "Pika FL2V"),
    ("gallop", VIDEOS / "gallop" / "s1" / "h3-vbvr-today-gallop.mp4", "h3-vbvr-today-gallop", False, "base+VBVR"),
    ("howl", VIDEOS / "howl" / "s1" / "h3-base-today-howl.mp4", "h3-base-today-howl", False, "base H3"),
    ("howl", VIDEOS / "howl" / "s1" / "h3-pika-today-howl.mp4", "h3-pika-today-howl", False, "Pika FL2V"),
    ("howl", VIDEOS / "howl" / "s1" / "h3-vbvr-today-howl.mp4", "h3-vbvr-today-howl", False, "base+VBVR"),
    ("pounce", VIDEOS / "pounce" / "s1" / "h3-base-today-pounce.mp4", "h3-base-today-pounce", False, "base H3"),
    ("pounce", VIDEOS / "pounce" / "s1" / "h3-pika-today-pounce.mp4", "h3-pika-today-pounce", False, "Pika FL2V"),
    ("pounce", VIDEOS / "pounce" / "s1" / "h3-vbvr-today-pounce.mp4", "h3-vbvr-today-pounce", False, "base+VBVR"),
]

#: The HyperFlow set (2026-11-11): the base route and the HyperFlow route
#: filmed BACK TO BACK in one session on the four matched moves - same
#: reference, same prompt file, same beat script, same seed, same seconds,
#: same dimensions, same text encoder, same video VAE. The only variable is
#: the denoiser: the stock pruned int8 checkpoint with the turbo LoRA against
#: the HyperFlow H3 Flash checkpoint with its own loader and its baked-in
#: 6-step schedule.
#:
#: It is a separate set from "today" because "today" was filmed on 2026-11-05,
#: and a timing or read table built across days is not a controlled
#: comparison - the machine state changes between days.
HF11_JOBS = [
    ("shake", VIDEOS / "shake" / "s1" / "h3-base-hf11-shake.mp4", "h3-base-hf11-shake", False, "base H3"),
    ("shake", VIDEOS / "shake" / "s1" / "h3-hyperflow-hf11-shake.mp4", "h3-hyperflow-hf11-shake", False, "HyperFlow"),
    ("gallop", VIDEOS / "gallop" / "s1" / "h3-base-hf11-gallop.mp4", "h3-base-hf11-gallop", False, "base H3"),
    ("gallop", VIDEOS / "gallop" / "s1" / "h3-hyperflow-hf11-gallop.mp4", "h3-hyperflow-hf11-gallop", False, "HyperFlow"),
    ("howl", VIDEOS / "howl" / "s1" / "h3-base-hf11-howl.mp4", "h3-base-hf11-howl", False, "base H3"),
    ("howl", VIDEOS / "howl" / "s1" / "h3-hyperflow-hf11-howl.mp4", "h3-hyperflow-hf11-howl", False, "HyperFlow"),
    ("pounce", VIDEOS / "pounce" / "s1" / "h3-base-hf11-pounce.mp4", "h3-base-hf11-pounce", False, "base H3"),
    ("pounce", VIDEOS / "pounce" / "s1" / "h3-hyperflow-hf11-pounce.mp4", "h3-hyperflow-hf11-pounce", False, "HyperFlow"),
]

#: Which set to read. "published" is the three-day set the docs quote;
#: "today" is the one-day controlled re-film of the same twelve clips.
SETS = {"published": lambda: PUBLISHED_JOBS, "today": lambda: TODAY_JOBS,
        "hf11": lambda: HF11_JOBS}


def run(cmd: list[str], label: str) -> int:
    print(f"--- {label}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    for line in [l for l in ((proc.stdout or "") + (proc.stderr or "")).splitlines()
                 if l.strip()][-8:]:
        print("   ", line)
    return proc.returncode


def read_one(clip: Path, name: str, single: bool, device: str = "cpu") -> dict:
    out = CLIPS / name
    (out / "render").mkdir(parents=True, exist_ok=True)
    ref = REF_SIDE if single else REF_SPLIT
    detect = [PY, "-m", "mocap.detect", str(clip), str(READER), str(ref),
              str(out / "det.npz"), str(out / "overlay.mp4")]
    if single:
        detect.append("--single-view")
    detect += ["--device", device]
    if run(detect, f"detect {name}{' (single view)' if single else ''}") != 0:
        return {"name": name, "status": "detect failed"}
    solve = [PY, "-m", "mocap.solve", str(out / "det.npz"), str(RIG),
             str(out / "motion.json"), "--device", device]
    if run(solve, f"solve {name}") != 0:
        return {"name": name, "status": "solve failed"}
    return {"name": name, "status": "ok"}


def main() -> int:
    ap = argparse.ArgumentParser(description="read the Pika clips through the house chain")
    ap.add_argument("--only", default="", help="comma-separated job keys")
    ap.add_argument("--set", default="published", choices=tuple(SETS),
                    help="'published' is the three-day set the docs quote; 'today' "
                         "is the one-day controlled re-film of the same twelve "
                         "clips, so the read and the wall clock are comparable.")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                    help="where the joint reader runs. The reader needs ~2.4 GB of "
                         "VRAM; on this box the other ComfyUI lane holds the card, "
                         "so cpu is the contention-free default.")
    ap.add_argument("--engine", default="", help="only read the clips of this engine")
    args = ap.parse_args()
    wanted = {k.strip() for k in args.only.split(",") if k.strip()}
    jobs = SETS[args.set]()
    if args.engine.strip():
        jobs = [j for j in jobs if j[4] == args.engine.strip()]

    results = []
    for key, clip, name, single, _engine in jobs:
        if wanted and key not in wanted:
            continue
        if not clip.is_file():
            print(f"SKIP {name}: no clip at {clip}")
            continue
        print(f"\n=== reading {name} ===", flush=True)
        t0 = time.time()
        results.append(read_one(clip, name, single, args.device))
        print(f"    ({time.time() - t0:.0f}s)")

    print("\n" + "=" * 104)
    print("READ ACCURACY - the plain upstream read, except the R2V clip (single view, 2.5D)")
    print(f"reader device: {args.device}")
    print(f"clip set: {args.set}")
    print("=" * 104)
    print(f"{'clip':<24}{'engine':<12}{'front px':>10}{'side px':>10}{'sync px':>10}"
          f"{'conf f/s':>12}{'calib':>10}{'read':>8}")
    for (key, clip, name, single, engine) in jobs:
        mpath = CLIPS / name / "motion.json"
        if not mpath.is_file():
            continue
        m = json.loads(mpath.read_text(encoding="utf-8"))
        r = m.get("reproj_px", {}) or {}
        s = m.get("sync", {}) or {}
        f = lambda v: f"{v:10.1f}" if isinstance(v, (int, float)) else f"{'-':>10}"
        # confidence lives in the detection, not in motion.json
        conf = "-"
        dnpz = CLIPS / name / "det.npz"
        if dnpz.is_file():
            try:
                import numpy as np
                d = np.load(dnpz)
                cf = float(d["front_conf"].mean()) if d["front_conf"].size else 0.0
                cs = float(d["side_conf"].mean()) if d["side_conf"].size else 0.0
                conf = f"{cf:.2f}/{cs:.2f}"
            except Exception:
                conf = "-"
        calib = "converged" if m.get("calib_converged", True) else "FALLBACK"
        print(f"{name:<24}{engine:<12}{f(r.get('front'))}"
              f"{f(r.get('side'))}{f(s.get('mean_height_gap_px'))}"
              f"{conf:>12}{calib:>10}{('1-view' if single else '2-view'):>8}")
    print("=" * 104)
    if any(single for (_k, _c, _n, single, _e) in jobs):
        print("the R2V row is NOT comparable on front or sync: one camera cannot see")
        print("left-right motion, so its rig fit is 2.5D (X held at the rest pose).")
    print("a FALLBACK calibration means the clip is not framed like the reference:")
    print("the reader still finds the joints, but the rig fit is indicative only.")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    sys.exit(main())
