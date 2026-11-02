"""One row per take of a creature with the numbers the gates use, to choose what goes in a pack.

python tools/score_takes.py CREATURE   -> table, and runs/CREATURE/takes.csv
PASS means: the feet's heights agree between the views (0.8 or more) and the rig lands on the readings (front x 3 px or
less, side 8.5 px or less). Both use only confident readings (above 0.3): a limb hidden behind the body shouldn't count
against a fit that rightly ignored it, and a foot moving under 8 px in both views counts as planted, not as a score. A
pass isn't a yes, it's a shortlist: look at each take's showcase.mp4, the numbers don't see a run that looks like a walk.
"""
import csv, json, os, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
c = sys.argv[1]
os.environ["MOCAP_CREATURE"] = f"assets/{c}"
os.chdir(ROOT)
from mocap import layout  # noqa: E402

rows = []
for d in sorted(pathlib.Path(f"runs/{c}/clips").glob("*")):
    f = d / "motion.json"
    if not f.exists() or not (d / "det.npz").exists():
        continue
    m = json.load(open(f))
    pa = dict(m["sync"]["paw_agreement"])
    det = np.load(d / "det.npz")
    names = [str(x) for x in det["names"]]
    for k in [k for k in pa if k != "score"]:
        i = names.index(k)
        a, b = det["front_uv"][:, i, 1], det["side_uv"][:, i, 1]
        ok = (det["front_conf"][:, i] > 0.3) & (det["side_conf"][:, i] > 0.3)
        if ok.sum() < 30 or (np.ptp(a[ok]) < 8 and np.ptp(b[ok]) < 8):
            pa[k] = None
        else:
            pa[k] = float(np.corrcoef(a[ok], b[ok])[0, 1])
    r = m.get("reproj_confident_px") or m["reproj_px"]
    raw = m["reproj_px"]
    sl = m.get("contacts", {}).get("slide_cm", {})
    sus = sum(m.get("swap_suspect_frames", {}).values())
    J = np.array(m["joints"])
    acc = np.linalg.norm(J[2:] - 2 * J[1:-1] + J[:-2], axis=-1) * 100  # cm/frame^2 per joint
    pop = float(acc.max() / max(np.percentile(acc, 95), 1e-6))  # one frame far outside the usual: a pop
    _, sp = layout.project(J)
    near = [i for i, n in enumerate(m["joint_names"]) if not n.endswith("_L")]  # the side camera sees the right flank
    side_near = float(np.linalg.norm(sp[:, near] - det["side_uv"][:, near], axis=-1).mean())
    arms = [v for k, v in pa.items() if k != "score" and k.startswith(("wrist", "handtip")) and v is not None]
    legs = [v for k, v in pa.items() if k != "score" and not k.startswith(("wrist", "handtip")) and v is not None]
    paw = sum(legs) / len(legs) if legs else None  # the gate: feet (and wings), not hanging arms
    arm = min(arms) if arms else None
    ok = (paw is None or paw >= 0.8) and r["front_x_only"] <= 3 and r["side"] <= 8.5
    rows.append({"take": d.name, "paw": None if paw is None else round(paw, 2), "arm_min": None if arm is None else round(arm, 2),
                 "feet": " ".join(f"{k}:{'-' if v is None else round(v, 2)}" for k, v in pa.items() if k != "score"),
                 "front_x": round(r["front_x_only"], 2), "side": round(r["side"], 2), "side_near": round(side_near, 2),
                 "front": round(r["front"], 2), "front_x_raw": round(raw["front_x_only"], 2), "side_raw": round(raw["side"], 2),
                 "fronts_fixed": (m.get("swaps") or {}).get("front", {}).get("swapped", 0),
                 "gap_px": round(m["sync"]["mean_height_gap_px"], 1), "pop": round(pop, 1), "swap_suspect": sus,
                 "slide_cm": sl.get("mean"), "plants": sum(len(x) for x in m.get("contacts", {}).get("runs", [])),
                 "speed_m_s": round((sum(v * v for v in m.get("contacts", {}).get("ground_velocity", [0, 0])) ** 0.5) * m["fps"], 2),
                 "numbers": "PASS" if ok else "fail"})
if not rows:
    raise SystemExit(f"no captured takes in runs/{c}/clips")
with open(f"runs/{c}/takes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(f"{r['take']:16s} paw {str(r['paw']):5s} arm {str(r['arm_min']):5s} fx {r['front_x']:5.2f} ({r['front_x_raw']:5.2f}) side {r['side']:5.2f} "
          f"near {r['side_near']:5.2f} gap {r['gap_px']:5.1f} pop {r['pop']:4.1f} sus {r['swap_suspect']:3d} plants {r['plants']:3d} "
          f"v {r['speed_m_s']:4.2f} {r['numbers']}  [{r['feet']}]")
print(f"{sum(r['numbers'] == 'PASS' for r in rows)} of {len(rows)} takes pass the numbers")
