"""Keep the audio track of a filmed clip as its own ambience stem.

WHY THIS EXISTS
---------------
MiniMax-H3 writes a soundtrack with every clip: all four locally filmed wolf
clips carry `h264 video + aac audio` at 32 kHz stereo. The capture stage
(`mocap.detect` -> `mocap.solve` -> Blender -> `mocap.showcase`) reads the
VIDEO stream only, so that soundtrack is dropped on the floor. Nothing
downstream can lay the clip's own sound under anything, and re-generating it
later would not be the same take.

This extracts that track as a separate stem inside the capture bundle:

    <out>/ambience.m4a    the AAC track exactly as H3 wrote it (stream copy,
                          no re-encode, so the bytes are the model's own)
    <out>/ambience.wav    the same track as 48 kHz stereo PCM, for mixing
    <out>/ambience.json   the probe facts: codec, rate, channels, duration,
                          mean and peak level, integrated LUFS

A clip with no audio stream is NOT an error - it simply produces no stem, and
the caller is told so.

Usage:
    python tools/ambience_stem.py <clip.mp4> <out-dir> [name]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: ffmpeg/ffprobe, in the order we look for them
FFMPEG_CANDIDATES = (
    "ffmpeg",
    r"C:\ProgramData\chocolatey\lib\ffmpeg-full\tools\ffmpeg\bin\ffmpeg.exe",
    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
)
FFPROBE_CANDIDATES = (
    "ffprobe",
    r"C:\ProgramData\chocolatey\lib\ffmpeg-full\tools\ffmpeg\bin\ffprobe.exe",
    r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
)

#: the mix rate every drama pipeline expects
MIX_RATE = 48000


def _tool(candidates: tuple[str, ...], name: str) -> str | None:
    for cand in candidates:
        if os.path.isabs(cand):
            if os.path.isfile(cand):
                return cand
        else:
            found = shutil.which(cand)
            if found:
                return found
    return None


def ffmpeg() -> str | None:
    return _tool(FFMPEG_CANDIDATES, "ffmpeg")


def ffprobe() -> str | None:
    return _tool(FFPROBE_CANDIDATES, "ffprobe")


def probe_audio(clip: Path) -> dict | None:
    """The audio stream's facts, or None when the clip has no audio at all."""
    probe = ffprobe()
    if not probe:
        return {"error": "ffprobe was not found on this machine"}
    proc = subprocess.run(
        [probe, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate,channels,channel_layout,"
         "bit_rate,duration:format=duration",
         "-of", "json", str(clip)],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return {"error": (proc.stderr or "").strip()[:400]}
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {"error": "ffprobe returned something that is not JSON"}
    streams = data.get("streams") or []
    if not streams:
        return None
    stream = streams[0]
    duration = stream.get("duration")
    if not duration:
        try:
            duration = float((data.get("format") or {}).get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
    return {
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "channels": int(stream.get("channels") or 0),
        "channel_layout": stream.get("channel_layout"),
        "bit_rate": int(stream.get("bit_rate") or 0),
        "duration_s": float(duration or 0.0),
    }


def measure_levels(clip: Path) -> dict:
    """Mean level, peak level and integrated LUFS of the track.

    The numbers are what a mixing pipeline needs to know whether it can lay
    music under this stem without swamping it.
    """
    out: dict = {}
    ff = ffmpeg()
    if not ff:
        return out
    proc = subprocess.run(
        [ff, "-hide_banner", "-nostats", "-i", str(clip),
         "-map", "0:a:0", "-af", "volumedetect,ebur128=peak=true",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=300)
    err = proc.stderr or ""

    mean = re.search(r"mean_volume:\s*(-?[\d.]+) dB", err)
    peak = re.search(r"max_volume:\s*(-?[\d.]+) dB", err)
    if mean:
        out["mean_volume_db"] = float(mean.group(1))
    if peak:
        out["max_volume_db"] = float(peak.group(1))

    # ebur128 prints a line PER FRAME, and that line carries an "I:" field
    # too - it starts at -70.0 (the initial state) and is NOT the integrated
    # loudness. Only the trailing Summary block is authoritative, so parse
    # from there and report nothing if the summary is missing.
    if "Summary:" in err:
        summary = err.rsplit("Summary:", 1)[1]
        lufs = re.search(r"^\s*I:\s*(-?[\d.]+)\s*LUFS", summary, re.M)
        if lufs:
            out["integrated_lufs"] = float(lufs.group(1))
        lra = re.search(r"^\s*LRA:\s*(-?[\d.]+)\s*LU", summary, re.M)
        if lra:
            out["loudness_range_lu"] = float(lra.group(1))
        true_peak = re.search(r"^\s*Peak:\s*(-?[\d.]+)\s*dBFS", summary, re.M)
        if true_peak:
            out["true_peak_dbfs"] = float(true_peak.group(1))
    return out


def extract(clip: Path, out_dir: Path, name: str = "ambience") -> dict | None:
    """Write the clip's audio track out as a stem. Returns the receipt.

    None means the clip has no audio stream - which is not a failure.
    """
    clip = Path(clip)
    out_dir = Path(out_dir)
    if not clip.is_file():
        raise FileNotFoundError(f"no clip at {clip}")

    ff = ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg was not found on this machine")

    facts = probe_audio(clip)
    if facts is None:
        print(f"[ambience] {clip.name} has no audio stream - no stem written")
        return None
    if facts.get("error"):
        print(f"[ambience] could not probe {clip.name}: {facts['error']}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    m4a = out_dir / f"{name}.m4a"
    wav = out_dir / f"{name}.wav"

    # The model's own bytes: a stream copy, no re-encode.
    copy_proc = subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(clip),
         "-map", "0:a:0", "-c:a", "copy", str(m4a)],
        capture_output=True, text=True, timeout=300)
    if copy_proc.returncode != 0 or not m4a.is_file():
        print(f"[ambience] the stream copy failed: "
              f"{(copy_proc.stderr or '').strip()[:300]}")
        m4a = None  # type: ignore[assignment]

    # The mixable form: linear PCM at the pipeline's rate.
    wav_proc = subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", str(clip),
         "-map", "0:a:0", "-ac", "2", "-ar", str(MIX_RATE),
         "-c:a", "pcm_s16le", str(wav)],
        capture_output=True, text=True, timeout=300)
    if wav_proc.returncode != 0 or not wav.is_file():
        print(f"[ambience] the wav decode failed: "
              f"{(wav_proc.stderr or '').strip()[:300]}")
        wav = None  # type: ignore[assignment]

    receipt = dict(facts)
    receipt.update(measure_levels(clip))
    receipt["source_clip"] = str(clip)
    receipt["stem_m4a"] = m4a.name if m4a else None
    receipt["stem_wav"] = wav.name if wav else None
    receipt["stem_m4a_bytes"] = m4a.stat().st_size if m4a else None
    receipt["stem_wav_bytes"] = wav.stat().st_size if wav else None
    receipt["mix_rate"] = MIX_RATE

    receipt_path = out_dir / f"{name}.json"
    receipt_path.write_text(json.dumps(receipt, indent=1), encoding="utf-8")

    print(f"[ambience] {clip.name} -> {m4a.name if m4a else '(no m4a)'} "
          f"+ {wav.name if wav else '(no wav)'}  "
          f"({facts.get('codec')} {facts.get('sample_rate')} Hz "
          f"{facts.get('channels')}ch, {facts.get('duration_s'):.2f}s)")
    if "integrated_lufs" in receipt:
        print(f"[ambience]   {receipt['integrated_lufs']:.1f} LUFS, "
              f"peak {receipt.get('max_volume_db')} dB")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(
        description="extract a filmed clip's audio track as a separate ambience stem")
    ap.add_argument("clip", type=Path, help="the clip to read")
    ap.add_argument("out_dir", type=Path, help="where to write the stem")
    ap.add_argument("name", nargs="?", default="ambience",
                    help="stem base name (default: ambience)")
    args = ap.parse_args()

    try:
        receipt = extract(args.clip, args.out_dir, args.name)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0 if receipt else 0


if __name__ == "__main__":
    sys.exit(main())
