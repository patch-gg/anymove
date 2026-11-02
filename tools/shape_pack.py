"""Shape every pick of a creature for its pack. tools/pack.sh runs this, then Blender.

python tools/shape_pack.py CREATURE PICKS.json OUT_DIR
PICKS.json names each animation and the take (runs/CREATURE/clips/<take>/) it comes from, for example:
  {"Idle": {"take": "idle_s2", "loop": true, "skip": 24, "cycle": [48, 96]},
   "Walk": {"take": "walk_s1", "loop": true, "skip": 12, "cycle": [24, 96], "locomotion": true},
   "Attack": {"take": "attack_s3"},
   "Kick": {"take": "kick_s1", "start": 4, "end": 118},
   "Hit": {"take": "hit_s2", "flagged": true}}
  loop        find a seamless cycle: skip the first frames settling out of the reference pose, cycle = [shortest, longest]
  locomotion  also write <Name>_Root, with root motion; the plain one stays on the spot
  start, end  trim a one-shot by hand; without them it's trimmed to the action automatically
  flagged     a weak clip or near miss: goes in <creature>_flagged.glb, not the main pack
Writes OUT_DIR/clips/<Name>.json, shape_report.json (each clip's loop, seam and speed) and pack_args.txt (one line per
GLB to build: the file's name, then Name=motion.json for each clip).
"""
import json, pathlib, subprocess, sys

ORDER = ["Idle", "Walk", "Run", "Trot", "Gallop", "Attack", "Hit", "Jump"]
ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    c, picks_path, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    picks = json.load(open(picks_path))
    (out / "clips").mkdir(parents=True, exist_ok=True)
    names = sorted(picks, key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))
    report, main_clips, flagged_clips = {}, [], []
    for name in names:
        p = picks[name]
        src = f"runs/{c}/clips/{p['take']}/motion.json"
        versions = [(name, [])]
        if p.get("locomotion"):
            versions.append((name + "_Root", ["--root"]))
        for vname, extra in versions:
            dst = out / "clips" / f"{vname}.json"
            cmd = [sys.executable, str(ROOT / "tools/shape_clip.py"), src, str(dst.resolve()), *extra]
            if p.get("loop"):
                cmd += ["--loop", "--skip", str(p.get("skip", 12)), "--cycle", ",".join(map(str, p.get("cycle", [20, 96])))]
            elif "start" in p or "end" in p:
                cmd += ["--start", str(p.get("start", 0)), "--end", str(p.get("end", 10 ** 6))]
            else:
                cmd += ["--auto"]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            if r.returncode:
                raise SystemExit(f"{vname}: {r.stderr[-800:]}")
            report[vname] = {"take": p["take"], **json.loads(r.stdout.strip().splitlines()[-1])}
            print(vname, report[vname])
            (flagged_clips if p.get("flagged") else main_clips).append(f"{vname}={dst.as_posix()}")
    json.dump(report, open(out / "shape_report.json", "w"), indent=1)
    lines = [f"{glb} {' '.join(clips)}\n" for glb, clips in ((f"{c}.glb", main_clips), (f"{c}_flagged.glb", flagged_clips)) if clips]
    (out / "pack_args.txt").write_text("".join(lines), newline="\n")


if __name__ == "__main__":
    main()
