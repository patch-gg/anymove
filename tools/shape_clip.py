"""Shape a solved clip for a game pack: trim, loop, in place / root motion.

python tools/shape_clip.py MOTION.json OUT.json [--loop] [--root] [--skip N] [--start N] [--end N] [--cycle MIN,MAX]
  --loop        find the start and length (one or more whole cycles) whose ends match best, then spread the leftover
                difference across the loop so the last frame runs into the first with no pop
  --root        root motion: take away the ground's own velocity (feet planted on a treadmill clip ride backwards
                together), so planted feet stand still and the body travels; without it the clip stays in place
  --skip N      ignore the first N frames when looking for a loop (the move settling out of the reference pose)
  --start/--end trim a one-shot clip to these frames
Fingers (motion.json "fingers", from mocap/hands.py) are trimmed and looped with the body.
Prints the loop it chose and the seam check (pose jump at the wrap vs a typical frame-to-frame change).
"""
import json, sys

import numpy as np


def qslerp(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th)


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def features(m):
    """Pose per frame that a viewer would see pop: joints relative to the root joint, plus root height."""
    J = np.array(m["joints"])
    tr = np.array(m["trans"])
    rel = J - J[:, :1]
    return np.concatenate([rel.reshape(len(J), -1), tr[:, 2:3] * 3], 1)


def find_loop(m, skip, pmin, pmax):
    F = features(m)
    V = np.gradient(F, axis=0)
    T = len(F)
    scale = np.abs(V).mean() + 1e-9
    cands = []
    for s in range(skip, T - pmin):
        for P in range(pmin, min(pmax, T - 1 - s) + 1):
            e = np.abs(F[s] - F[s + P]).mean() / scale + 0.5 * np.abs(V[s] - V[s + P]).mean() / scale
            cands.append((e, s, P))
    if not cands:
        raise SystemExit(f"no loop: clip has {T} frames, skip {skip}, cycle {pmin}-{pmax}")
    best = min(cands)[0]
    # among near-best loops prefer the longest (more cycles average the video's frame-to-frame noise)
    ok = [c for c in cands if c[0] <= best * 1.15 + 0.05]
    e, s, P = max(ok, key=lambda c: (c[2], -c[0]))
    return s, P, e, scale


def ground_track(m, J):
    """Per-frame ground velocity (x, y, m/frame) under an on-the-spot clip, from its feet: each frame, the planted
    contact point that moves least (relative to the clip's average treadmill speed) gives the ground's motion, so a
    root-motion version made from it keeps that foot still. Flight frames carry the speed over from their neighbours."""
    con = m.get("contacts", {})
    names = m["joint_names"]
    if not con.get("joints"):
        return np.zeros((len(J), 2))
    idx = [names.index(n) for n in con["joints"]]
    h0 = np.array(con["rest_height"])
    g0 = np.array(con["ground_velocity"])
    tall = float(J[..., 2].max())
    tol = max(0.02, 0.01 * tall)
    vmax = 3 * max(0.015, 0.0025 * tall) * 24 / m["fps"]
    vel = np.gradient(J[:, idx, :2], axis=0)
    rel = np.linalg.norm(vel - g0[None, None], axis=-1)
    down = ((J[:, idx, 2] - h0[None]) < tol) & (rel < vmax)
    G = np.full((len(J), 2), np.nan)
    for t in range(len(J)):
        if down[t].any():
            k = np.where(down[t], rel[t], np.inf).argmin()
            G[t] = vel[t, k]
    ok = ~np.isnan(G[:, 0])
    if ok.sum() < 2:
        return np.tile(g0, (len(J), 1))
    tt = np.arange(len(J))
    G = np.stack([np.interp(tt, tt[ok], G[ok, i]) for i in range(2)], 1)
    k = np.ones(3) / 3
    return np.stack([np.convolve(np.pad(G[:, i], 1, mode="edge"), k, mode="valid") for i in range(2)], 1)


def main():
    a = sys.argv[1:]
    src, out = a[0], a[1]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    m = json.load(open(src))
    T = len(m["euler"])
    E = np.array(m["euler"])
    Q = np.array(m["root_quat"])
    tr = np.array(m["trans"])
    J = np.array(m["joints"])
    Fg = np.array(m["fingers"]["euler"]) if m.get("fingers") else None
    g = np.array(m.get("contacts", {}).get("ground_velocity", [0, 0]) + [0.0])
    info = {}
    if "--root" in a:  # foot-locked root motion: the root travels exactly as the planted feet slip under it
        G = ground_track(m, J)
        off = np.concatenate([[[0.0, 0.0]], np.cumsum((G[1:] + G[:-1]) / 2, axis=0)])
        off = np.concatenate([off, np.zeros((T, 1))], 1)
        tr = tr - off
        J = J - off[:, None]
        info["root_speed_m_s"] = float(np.linalg.norm(off[-1]) / max(T - 1, 1) * m["fps"])
    if "--loop" in a:
        pmin, pmax = (int(x) for x in opt("--cycle", "20,90").split(","))
        s, P, e, scale = find_loop({"joints": J.tolist(), "trans": tr.tolist()}, int(opt("--skip", 12)), pmin, pmax)
        idx = np.arange(s, s + P + 1)
        E, Q, tr, J = E[idx].copy(), Q[idx].copy(), tr[idx].copy(), J[idx].copy()
        w = np.linspace(0, 1, P + 1)
        # spread the mismatch between the last frame and the first across the loop
        dE = E[-1] - E[0]
        E -= w[:, None, None] * dE[None]
        if Fg is not None:
            Fg = Fg[idx].copy()
            Fg -= w[:, None, None] * (Fg[-1] - Fg[0])[None]
        disp = tr[-1] - tr[0]
        travel = np.zeros(3)
        if "--root" in a:  # keep the travel per cycle, but make the up-down wrap
            travel[:2] = disp[:2]
            disp = disp - travel
            info["root_speed_m_s"] = float(np.linalg.norm(travel) / P * m["fps"])
        tr -= w[:, None] * disp[None]
        dq = qmul(qinv(Q[0]), Q[-1])
        for i in range(P + 1):
            Q[i] = qmul(Q[i], qinv(qslerp(np.array([1.0, 0, 0, 0]), dq, w[i])))
        J -= w[:, None, None] * (J[-1] - J[0] - travel)[None]
        # the last frame equals the first (plus one cycle of travel with --root), so an engine looping over the
        # clip's duration wraps with no hitch; the seam check compares the step into the wrap with the others
        F = features({"joints": J.tolist(), "trans": tr.tolist()})
        step = np.abs(np.diff(F, axis=0)).mean(1)
        wrap = np.abs(F[1] - F[-2]).mean() / 2  # frame P-1 -> P (= 0) -> 1 spans two steps
        info.update({"loop_start": int(s), "loop_frames": int(P), "loop_seconds": round(P / m["fps"], 3),
                     "match_error": round(float(e), 3), "seam_jump_vs_median_step": round(float(wrap / np.median(step)), 2),
                     "seam_jump_vs_max_step": round(float(wrap / step.max()), 2)})
    else:
        s0, s1 = int(opt("--start", 0)), int(opt("--end", T))
        if "--auto" in a:  # keep the action: from just before the body starts moving to a short settle after it stops
            F = features({"joints": J.tolist(), "trans": tr.tolist()})
            sp = np.abs(np.diff(F, axis=0)).mean(1)
            sp = np.convolve(sp, np.ones(5) / 5, mode="same")
            active = np.nonzero(sp > 0.2 * sp.max())[0]
            s0 = max(0, int(active[0]) - 6)
            s1 = min(T, int(active[-1]) + 18)
        E, Q, tr, J = E[s0:s1], Q[s0:s1], tr[s0:s1], J[s0:s1]
        Fg = None if Fg is None else Fg[s0:s1]
        info.update({"start": s0, "end": s1})
    m2 = dict(m)
    m2.update({"euler": E.round(5).tolist(), "root_quat": Q.round(6).tolist(), "trans": tr.round(5).tolist(),
               "joints": J.round(4).tolist(), "shape": info})
    if Fg is not None:
        m2["fingers"] = {"bones": m["fingers"]["bones"], "euler": Fg.round(4).tolist()}
    json.dump(m2, open(out, "w"))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
