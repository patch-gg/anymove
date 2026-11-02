"""Foot sliding, stride and cadence for every locomotion loop in a pack, or for takes before picking one.

python tools/locomotion.py CREATURE [picks=picks/CREATURE.json] [pack=runs/CREATURE/pack]   -> table, pack/locomotion.json
python tools/locomotion.py --candidates CREATURE MIN,MAX TAKE [TAKE ...]   shapes each take as a root-motion loop to compare

Measured on the shipped root-motion loop (<Name>_Root), independently of the solver's own plant detection: a contact
joint counts as down while it is within 2 cm (or 1% of the creature's height) of its rest height. While it is down,
its horizontal drift relative to the ground (the treadmill velocity of the on-the-spot clip) is the slide; the worst
drift of any one footfall, in cm, is reported. Stride length = ground speed x gait cycle; cadence = footfalls of one
foot per minute x 2. Flight = frames in which no contact joint is down.
"""
import json, pathlib, subprocess, sys, tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def analyse(c, pick, root, rep):
    """pick: its take; root, rep: the shaped root-motion loop and its entry in the shape report."""
    m = json.load(open(ROOT / f"runs/{c}/clips/{pick['take']}/motion.json"))
    P = rep["loop_frames"]
    J = np.array(root["joints"])
    fps = m["fps"]
    names = m["joint_names"]
    con = m["contacts"]
    idx = [names.index(n) for n in con["joints"]]
    h0 = np.array(con["rest_height"])
    g = np.zeros(3)  # root-motion clip: the ground is still, a planted foot must not move
    tall = float(np.array(m["joints"])[..., 2].max())
    tol = max(0.02, 0.01 * tall)
    t = np.arange(len(J))[:, None]
    P_ = J[:, idx] - (g[None] * t)[:, None]              # contact joints in the ground's frame
    v = np.linalg.norm(np.gradient(P_[..., :2], axis=0), axis=-1)  # speed relative to the ground, m/frame
    v1 = max(0.01, 0.005 * tall) * 24 / fps        # 1 cm/frame for a 2 m creature: slower than this on the ground = planted
    low = (J[:, idx, 2] - h0[None]) < tol
    down = low & (v < v1)
    # skating: a contact point at ground height moving faster than a plant but slower than a swing, away from lift-off
    near = (J[:, idx, 2] - h0[None]) < 0.5 * tol
    sk = near & (v >= v1) & (v < 4 * v1)
    edge = np.zeros_like(down)
    for dt in (-2, -1, 1, 2):
        edge |= np.roll(down, dt, axis=0)
    sk &= ~edge
    skate_cm = float((v * sk).sum() * 100)
    # group the contact joints by foot (the paint group, i.e. the limb, they belong to)
    cj = json.load(open(ROOT / f"assets/{c}/creature.json"))
    info = json.load(open(ROOT / f"assets/{c}/rigged/rig_info.json"))
    ends = {bb["name"]: (bb["head_joint"], bb["tail_joint"]) for bb in info["bones"]}
    groups = []
    for grp in cj.get("paint", {}).values():
        js = {j for bn in grp["bones"] for j in ends.get(bn, ())}
        ks = [k for k, n in enumerate(con["joints"]) if n in js]
        if ks:
            groups.append(ks)
    # per foot: while any of its contact points is down, follow the stillest one and add up how far it travels
    worst, drifts = 0.0, []
    for ks in groups:
        anyd = down[:, ks].any(1)
        vel = np.gradient(P_[:, ks, :2], axis=0)                       # (T, k, 2) relative to the ground
        still = np.where(down[:, ks], v[:, ks], np.inf).argmin(1)      # the stillest down point each frame
        sv = vel[np.arange(len(J)), still]
        f = 0
        while f < len(J):
            if anyd[f]:
                e = f
                while e < len(J) and anyd[e]:
                    e += 1
                if e - f >= 3:
                    d = float(np.linalg.norm(sv[f:e].sum(0)) * 100)  # net travel of the anchor: jitter cancels
                    drifts.append(d)
                    worst = max(worst, d)
                f = e
            else:
                f += 1
    speed = float(np.linalg.norm(np.array(root["trans"][-1][:2]) - np.array(root["trans"][0][:2])) / P * fps)
    period = P / fps
    # gait cycle (one left and one right step) from the autocorrelation of the first foot's height over the loop
    feet = json.load(open(ROOT / f"assets/{c}/creature.json")).get("feet", [])
    foot = [n for n in feet if n in con["joints"]] or con["joints"]
    z = J[:-1, idx[con["joints"].index(foot[0])], 2]
    z = z - z.mean()
    ac = np.array([np.dot(z[:len(z) - k], z[k:]) / max(len(z) - k, 1) for k in range(len(z))])
    lo = int(0.35 * fps)
    peaks = [k for k in range(lo, len(ac) - 1) if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.3 * ac[0]]
    cycle = (peaks[0] if peaks else P) / fps
    flight = float(np.mean(~down.any(1)))
    return {"take": pick["take"], "loop_s": round(period, 2), "ground_speed_m_s": round(speed, 2),
            "cycle_s": round(cycle, 2), "stride_m": round(speed * cycle, 2), "cadence_steps_min": round(120 / cycle),
            "flight_frac": round(flight, 2), "worst_slide_cm": round(worst, 2),
            "median_slide_cm": round(float(np.median(drifts)) if drifts else 0.0, 2), "footfalls": len(drifts),
            "skate_cm": round(skate_cm, 1)}


def candidate(c, take, cycle, skip=12):
    """Shape one take as a root-motion loop in a temp file and analyse it (to compare seeds before picking)."""
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "root.json"
        r = subprocess.run([sys.executable, str(ROOT / "tools/shape_clip.py"), str(ROOT / f"runs/{c}/clips/{take}/motion.json"),
                            str(out), "--root", "--loop", "--skip", str(skip), "--cycle", cycle], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"{take}: {r.stderr[-800:]}")
        return analyse(c, {"take": take}, json.load(open(out)), json.loads(r.stdout.strip().splitlines()[-1]))


def main():
    if sys.argv[1:2] == ["--candidates"]:  # --candidates CREATURE CYCLE TAKE...
        c, cycle = sys.argv[2], sys.argv[3]
        for t in sys.argv[4:]:
            r = candidate(c, t, cycle)
            print(f"{c:8s} {t:12s} speed {r['ground_speed_m_s']:4.2f} m/s stride {r['stride_m']:4.2f} m cadence {r['cadence_steps_min']:4d}/min "
                  f"flight {r['flight_frac']:.2f} slide worst {r['worst_slide_cm']:5.2f} median {r['median_slide_cm']:4.2f} cm skate {r['skate_cm']:5.1f} cm  loop {r['loop_s']} s")
        return
    c = sys.argv[1]
    picks = json.load(open(sys.argv[2] if len(sys.argv) > 2 else ROOT / f"picks/{c}.json"))
    pack = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else ROOT / f"runs/{c}/pack")
    reports = json.load(open(pack / "shape_report.json"))
    out = {}
    for name, p in picks.items():
        if p.get("locomotion"):
            root = json.load(open(pack / "clips" / f"{name}_Root.json"))
            r = out[name] = analyse(c, p, root, reports[name])
            print(f"{c:8s} {name:7s} {r['take']:14s} speed {r['ground_speed_m_s']:4.2f} m/s  stride {r['stride_m']:4.2f} m  "
                  f"cadence {r['cadence_steps_min']:4d}/min  flight {r['flight_frac']:.2f}  slide worst {r['worst_slide_cm']:5.2f} cm "
                  f"median {r['median_slide_cm']:4.2f} ({r['footfalls']} footfalls)  skate {r['skate_cm']:5.1f} cm")
    json.dump(out, open(pack / "locomotion.json", "w"), indent=1)


if __name__ == "__main__":
    main()
