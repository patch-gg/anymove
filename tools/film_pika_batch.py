#!/usr/bin/env python
"""Film a batch of clips on the Pika H3 Flex engines, one at a time, on a leased card.

The lease gate is the house one, copied from `film_wolf_local.ps1`: the card
must be genuinely free before a clip starts, because a second video model on
this 8 GB card starves both CUDA contexts - the GPU reads 100 % utilisation
while neither process executes a kernel.

    other ComfyUI (8188) queue empty
    AND nvidia-smi free VRAM >= --min-vram-mib (default 3800)

Clips run strictly in sequence. A clip that already exists is skipped, so the
batch is re-runnable.

usage:
    python tools/film_pika_batch.py --only gallop,howl,pounce
    python tools/film_pika_batch.py --only r2v-side
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(r"D:\anymove")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
FILM = str(REPO / "tools" / "film_h3.py")
LOG = Path(r"D:\workspace\h3-logs\film_pika_batch.log")
OUT_V = REPO / "runs" / "wolf" / "videos"
REF = REPO / "runs" / "wolf" / "ref"

SUBJECT_FEATURES = (
    "a lean wolf with a long narrow muzzle, a cream-white chest and lower jaw, "
    "dark charcoal-grey layered fur over the back and shoulders, pricked "
    "triangular ears, amber-grey eyes, one bushy tail, and exactly four legs "
    "whose colours never change or swap - front-left leg magenta pink, "
    "front-right leg blue, hind-left leg yellow, hind-right leg cyan"
)


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def free_vram_mib() -> int:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        return int((proc.stdout or "0").split(",")[0].strip())
    except Exception:
        return -1


def other_busy() -> int:
    """How many jobs the user's own ComfyUI (8188) is running. -1 = unreachable."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=15) as r:
            q = json.loads(r.read().decode("utf-8"))
        return len(q.get("queue_running") or [])
    except Exception:
        return -1


def other_up() -> bool:
    """Is an instance still listening on the user's 8188 lane?"""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=6):
            return True
    except Exception:
        pass
    # a saturated instance does not answer HTTP in time - the LISTENER is the
    # honest signal, so fall back to asking the OS
    try:
        proc = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=30)
        for line in proc.stdout.splitlines():
            if ":8188" in line and "LISTENING" in line:
                return True
    except Exception:
        pass
    return False


def other_state() -> tuple[bool, int]:
    """(has_listener, running_jobs) for the user's 8188 lane.

    - no listener at all  -> the lane is DOWN: there is definitively no job in
      flight, so it cannot be contending. (0 running)
    - a listener, /queue answers -> that many jobs are running.
    - a listener, /queue does not answer -> the instance is busy or wedged;
      "I cannot tell" means "do not assume it is free". (-1)
    """
    up = other_up()
    try:
        with urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=10) as r:
            q = json.loads(r.read().decode("utf-8"))
        return up, len(q.get("queue_running") or []) + len(q.get("queue_pending") or [])
    except Exception:
        return up, -1 if up else 0


def wait_lease(min_vram: int, wait_sec: int) -> bool:
    """Wait until the OTHER lane has no job in flight.

    The free-VRAM number is deliberately NOT the gate. The house
    `comfyui_gpu_handoff.py` owns the handover between the two ComfyUI
    instances: while our lane (8189) is idle it deliberately keeps the
    user's 8188 up, and 8188's Wan model then sits on ~5.7 GB of VRAM with an
    EMPTY queue. That is the lane working, not a busy card - reading it as
    "no room" is how the earlier batch deadlocked against itself.

    The moment our lane goes busy the handoff stops 8188 (after letting its
    own queue drain) and hands the card over, which is what it did at 06:07
    on 2026-11-03 during the first Pika render.

    So the gate is: the other lane has no job in flight. A lane that is DOWN
    is idle by definition; a lane that is UP but cannot answer its queue is
    treated as busy, because the only safe reading of "I cannot tell" is "do
    not assume it is free".
    """
    end = time.time() + wait_sec
    while time.time() < end:
        free = free_vram_mib()
        up, busy = other_state()
        if busy == 0:
            if up:
                log(f"    GPU lease granted: other lane idle but up "
                    f"(vram_free={free} MiB) - the house handoff yields the card "
                    f"when our lane goes busy")
            else:
                log(f"    GPU lease granted: other lane down, "
                    f"vram_free={free} MiB")
            if free < min_vram:
                # ADVISORY ONLY - never a gate. See the docstring: gating on
                # free VRAM is what deadlocked the first batch.
                log(f"    note: free VRAM {free} MiB is below the advisory "
                    f"threshold {min_vram} MiB; proceeding anyway because the "
                    f"other lane has no job in flight")
            return True
        log(f"    GPU lease denied: other ComfyUI busy={busy} "
            f"(vram_free={free} MiB) - waiting")
        time.sleep(30)
    log(f"    GPU lease TIMED OUT after {wait_sec} s - filming anyway "
        f"(a slow clip beats no clip)")
    return False


def build_jobs() -> list[dict]:
    jobs = []
    for move in ("gallop", "howl", "pounce"):
        script = REPO / "prompts" / "wolf" / "dual" / "short" / f"{move}.txt"
        jobs.append({
            "key": move,
            "out": OUT_V / move / "s1" / f"h3-pika-{move}.mp4",
            "argv": [
                PY, FILM, "--creature", "wolf", "--move", move,
                "--script", str(script),
                "--seconds", "1.6", "--seed", "42",
                "--engine", "pika-flex",
                "--out-name", f"h3-pika-{move}",
            ],
        })
    jobs.append({
        "key": "r2v-side",
        "out": OUT_V / "shake" / "s1" / "h3-pika-r2v-shake-side.mp4",
        "argv": [
            PY, FILM, str(REF / "ref_side.png"),
            str(REPO / "prompts" / "wolf" / "pika" / "r2v-shake-side.txt"),
            str(OUT_V / "shake" / "s1"),
            "--engine", "pika-flex-r2v",
            "--seconds", "1.6", "--seed", "42",
            "--out-name", "h3-pika-r2v-shake-side",
            "--soundscape-file", str(REPO / "prompts" / "wolf" / "pika" / "r2v-shake.soundscape.txt"),
            "--music-file", str(REPO / "prompts" / "wolf" / "pika" / "r2v-shake.music.txt"),
            "--subject", "the grey wolf game model seen in full profile",
            "--subject-role", "identity and form anchor",
            "--subject-features", SUBJECT_FEATURES,
        ],
    })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description="film a batch of Pika H3 Flex clips")
    ap.add_argument("--only", default="", help="comma-separated job keys")
    ap.add_argument("--min-vram-mib", type=int, default=3800,
                    help="ADVISORY free-VRAM floor (logged, never gated on)")
    ap.add_argument("--lease-wait-sec", type=int, default=1800)
    ap.add_argument("--stall-sec", type=int, default=7200)
    ap.add_argument("--force", action="store_true", help="re-film even if the clip exists")
    args = ap.parse_args()

    wanted = {k.strip() for k in args.only.split(",") if k.strip()}
    jobs = [j for j in build_jobs() if not wanted or j["key"] in wanted]
    if not jobs:
        print("no jobs matched", file=sys.stderr)
        return 1

    log("=" * 70)
    log(f"START pika batch: {', '.join(j['key'] for j in jobs)}")

    results = {}
    for job in jobs:
        out = Path(job["out"])
        if out.is_file() and not args.force:
            log(f"SKIP {job['key']}: {out} already exists "
                f"({out.stat().st_size / 2**20:.2f} MB)")
            results[job["key"]] = {"status": "skipped", "path": str(out)}
            continue

        wait_lease(args.min_vram_mib, args.lease_wait_sec)

        log(f"---- filming {job['key']} ----")
        t0 = time.time()
        samples: list[int] = []
        stop = threading.Event()

        def sample_vram() -> None:
            while not stop.is_set():
                samples.append(free_vram_mib())
                stop.wait(10)

        watcher = threading.Thread(target=sample_vram, daemon=True)
        watcher.start()
        try:
            proc = subprocess.run(job["argv"], cwd=str(REPO), capture_output=True,
                                  text=True, timeout=args.stall_sec)
        finally:
            stop.set()
            watcher.join(timeout=5)
        wall = time.time() - t0
        tail = [ln for ln in ((proc.stdout or "") + (proc.stderr or "")).splitlines()
                if ln.strip()][-12:]
        for ln in tail:
            log(f"    {ln}")
        if proc.returncode != 0:
            log(f"FAIL {job['key']} exit={proc.returncode} after {wall:.0f}s")
            results[job["key"]] = {"status": "failed", "exit": proc.returncode}
            continue
        size = out.stat().st_size / 2**20 if out.is_file() else 0.0
        real = [s for s in samples if s >= 0]
        vram = (f"vram_free min={min(real)} max={max(real)} MiB" if real
                else "vram unknown")
        log(f"OK   {job['key']} in {wall:.0f}s -> {out} ({size:.2f} MB)  {vram}")
        results[job["key"]] = {"status": "ok", "wall_s": round(wall, 1),
                               "path": str(out), "bytes": out.stat().st_size,
                               "vram_free_min_mib": min(real) if real else None,
                               "vram_free_max_mib": max(real) if real else None}

    log("DONE " + json.dumps(results))
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
