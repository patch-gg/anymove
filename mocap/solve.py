"""Fit the rig to the joints read from both views: one pose per frame, smooth over time, inside the joint limits.

python -m mocap.solve DETECTIONS.npz rig_info.json OUT.json
Before fitting, the side view's left and right legs are checked against the front view: a joint has the same height
in both halves, so if swapping a pair of legs in the side view makes the heights agree better, the side reader
mixed up the near and far leg and they are swapped back. The same height test gives the sync score: how far apart
the two halves disagree, and the time lag between them that agrees best.

The fit runs Blender's forward kinematics in torch (pose = parent pose x rest offset x Euler XYZ), with the pelvis
free to turn and move, every other bone limited to its range, and minimises the confidence-weighted distance between
the rig's projected joints and the readings in both views, plus smoothness and a small pull toward the rest pose.
"""
import json, sys

import numpy as np
import torch

from mocap import layout

LEGS = {"F": ["scap", "shoulder", "elbow", "wrist", "fpaw", "ftoe"], "H": ["hip", "knee", "hock", "hpaw", "htoe"]}


def creature():
    """creature.json of MOCAP_CREATURE, or {} (the wolf's defaults apply)."""
    import os, pathlib
    cfg = pathlib.Path(os.environ.get("MOCAP_CREATURE", "")) / "creature.json"
    return json.load(open(cfg)) if os.environ.get("MOCAP_CREATURE") and cfg.exists() else {}


def leg_pairs(names, info=None):
    """Index pairs (left, right) of every limb joint, grouped per left/right pair of limbs. The limbs are creature.json
    "paint" groups whose bones mirror each other (_L/_R); without paint, the wolf's front and hind legs."""
    paint = creature().get("paint", {})
    out = {}
    if paint and info is not None:
        ends = {b["name"]: (b["head_joint"], b["tail_joint"]) for b in info["bones"]}
        for k, g in paint.items():
            left = [b for b in g["bones"] if b.endswith("_L")]
            if not left:
                continue
            pairs = []
            for b in left:
                for j in ends.get(b, ()):
                    if j.endswith("_L") and j in names and j[:-2] + "_R" in names:
                        p = (names.index(j), names.index(j[:-2] + "_R"))
                        if p not in pairs:
                            pairs.append(p)
            if pairs:
                out[k] = pairs
        return out
    for k, stems in LEGS.items():
        pairs = []
        for s in stems:
            if f"{s}_L" in names:
                pairs.append((names.index(f"{s}_L"), names.index(f"{s}_R")))
        out[k] = pairs
    return out


def fix_swaps(d, names, info=None, margin=4.0, apply=True):
    """Swap near/far limbs in the side view where the heights say the reader mixed them up. With apply=False, only
    count the frames where a swap would agree better (a warning sign for the clip)."""
    fv, sv = d["front_uv"][..., 1], d["side_uv"][..., 1]
    c = np.minimum(d["front_conf"], d["side_conf"]).clip(0, 1)
    swapped = {}
    for k, pairs in leg_pairs(names, info).items():
        L, R = np.array(pairs).T
        same = (c[:, L] * np.abs(sv[:, L] - fv[:, L]) + c[:, R] * np.abs(sv[:, R] - fv[:, R])).sum(1)
        swap = (c[:, L] * np.abs(sv[:, R] - fv[:, L]) + c[:, R] * np.abs(sv[:, L] - fv[:, R])).sum(1)
        gain = (same - swap) / len(pairs)
        gain = np.array([np.median(gain[max(0, t - 2):t + 3]) for t in range(len(gain))])
        do = gain > margin
        if apply:
            for t in np.where(do)[0]:
                d["side_uv"][t, L], d["side_uv"][t, R] = d["side_uv"][t, R].copy(), d["side_uv"][t, L].copy()
                d["side_conf"][t, L], d["side_conf"][t, R] = d["side_conf"][t, R].copy(), d["side_conf"][t, L].copy()
        swapped[k] = int(do.sum())
    return swapped


def fix_front_swaps(d, names, info, margin=6.0):
    """Leg swaps in the front view. The creature's left is the image's right there, so a leg joint read left of its
    right twin was swapped (unpainted feet look alike); swap it back. Then a reading that sits further from its
    limb's parent joint than the bone could reach is dropped (confidence near 0): two labels on one foot."""
    cidx, _ = contact_joints(info, names)
    contacts = {names[i] for i in cidx}
    groups = [g for g in leg_pairs(names, info).values() if any(names[a] in contacts or names[b] in contacts for a, b in g)]
    fu = d["front_uv"][..., 0]
    swapped = 0
    for g in groups:
        for L, R in g:
            m = fu[:, L] < fu[:, R] - margin
            for k in ("front_uv", "front_conf"):
                a = d[k][m, L].copy(); d[k][m, L] = d[k][m, R]; d[k][m, R] = a
            swapped += int(m.sum())
    fu = d["front_uv"][..., 0]
    dropped = 0
    leg = {i for g in groups for pair in g for i in pair}
    for b in info["bones"]:
        h, t = b["head_joint"], b["tail_joint"]
        if h in names and t in names and names.index(t) in leg:
            hi, ti = names.index(h), names.index(t)
            far = np.abs(fu[:, ti] - fu[:, hi]) > 1.3 * b["length"] / layout.M_PER_PX + margin
            d["front_conf"][far, ti] *= 0.05
            dropped += int(far.sum())
    return {"swapped": swapped, "dropped": dropped}


def contact_joints(info, names):
    """The joints that touch the ground when the creature stands: creature.json "contacts" if given, else every joint
    of a painted limb (or a "feet" joint) that sits near the ground in the rest pose (hooves, toes, knuckles, claw
    tips). Returns their indices and rest heights."""
    cfg = creature()
    rest = {k["name"]: k["rest"] for k in info["keypoints"]}
    if "contacts" in cfg:
        cand = [n for n in cfg["contacts"] if n in names]
    else:
        tall = max(r[2] for r in rest.values())
        near = max(0.06, 0.05 * tall)
        limb = set(cfg.get("feet", []))
        for g in cfg.get("paint", {}).values():
            for b in info["bones"]:
                if b["name"] in g["bones"]:
                    limb |= {b["head_joint"], b["tail_joint"]}
        if not cfg.get("paint"):
            limb |= {n for n in names if n.startswith(("fpaw", "ftoe", "hpaw", "htoe"))}
        cand = [n for n in names if n in limb and rest[n][2] < near]
    idx = [names.index(n) for n in cand]
    return idx, np.array([rest[n][2] for n in cand], float)


def runs(mask, fill=2, shortest=4):
    """Clean a boolean contact track: close gaps of up to `fill` frames, drop contacts shorter than `shortest`."""
    m = mask.copy()
    T = len(m)
    t = 0
    while t < T:  # close short gaps between contacts
        if not m[t]:
            e = t
            while e < T and not m[e]:
                e += 1
            if 0 < t and e < T and e - t <= fill:
                m[t:e] = True
            t = e
        else:
            t += 1
    out, t = [], 0
    while t < T:
        if m[t]:
            e = t
            while e < T and m[e]:
                e += 1
            if e - t >= shortest:
                out.append((t, e))
            t = e
        else:
            t += 1
    return out


def find_contacts(J, cidx, h0, fps=24, hysteresis=1.0):
    """Planted frames per contact joint from a first fit, and the ground's own velocity: on a clip filmed "on the spot"
    the planted feet ride backwards together like on a treadmill, so a foot is planted when it is near its rest
    height and moving with the others. Returns per-joint lists of (start, end) frame runs and the ground velocity in
    metres per frame (x, y)."""
    P = J[:, cidx]
    T = len(P)
    tall = float(np.abs(J[..., 2]).max())
    tol_z = max(0.02, 0.015 * tall)
    tol_v = max(0.008, 0.006 * tall) * 24 / fps
    v = np.gradient(P[..., :2], axis=0)
    near = (P[..., 2] - h0[None]) < tol_z
    g = np.median(v[near], axis=0) if near.sum() >= 10 else np.zeros(2)
    if np.linalg.norm(g) < 0.25 * tol_v:
        g = np.zeros(2)
    rel = np.linalg.norm(v - g, axis=-1)
    planted = near & (rel < tol_v)
    if hysteresis > 1:  # grow each plant while the foot stays down and only a little faster (the whole stance, not its still core)
        loose = (P[..., 2] - h0[None]) < tol_z * hysteresis
        loose &= rel < tol_v * hysteresis
        for k in range(len(cidx)):
            grown = np.zeros(T, bool)
            for a, b in runs(loose[:, k]):
                if planted[a:b, k].any():
                    grown[a:b] = True
            planted[:, k] = grown
    return [runs(planted[:, k]) for k in range(len(cidx))], g


def height_features(uv, conf, names):
    """Per-frame heights that do not care about left/right labels: middle joints as they are, each leg pair as its
    mean height and its height difference. Returns (T, K) features and (T, K) weights."""
    v = uv[..., 1]
    feats, w = [], []
    for i, n in enumerate(names):
        if not n.endswith(("_L", "_R")):
            feats.append(v[:, i]); w.append(conf[:, i])
        elif n.endswith("_L"):
            a, b = i, names.index(n[:-2] + "_R")
            c = np.minimum(conf[:, a], conf[:, b])
            feats += [(v[:, a] + v[:, b]) / 2, np.abs(v[:, a] - v[:, b])]
            w += [c, c]
    return np.stack(feats, 1), np.stack(w, 1).clip(0, 1)


def align_views(d, names, band=48, step_penalty=1.0):
    """Video models often play the two halves out of step (one view runs ahead). Match the halves frame to frame
    with dynamic time warping on joint heights, which both views share, and resample the front view onto the side
    view's timeline. Returns how far each side frame's match was moved, in frames."""
    F, Fw = height_features(d["front_uv"], d["front_conf"], names)
    S, Sw = height_features(d["side_uv"], d["side_conf"], names)
    T = len(F)
    C = np.full((T, T), np.inf)
    for i in range(T):
        lo, hi = max(0, i - band), min(T, i + band + 1)
        w = np.minimum(Fw[i][None], Sw[lo:hi])
        C[i, lo:hi] = (w * np.abs(F[i][None] - S[lo:hi])).sum(1) / w.sum(1).clip(1e-6)
    D = np.full((T, T), np.inf)
    D[0, 0] = C[0, 0]
    for i in range(T):
        for j in range(max(0, i - band), min(T, i + band + 1)):
            if i == j == 0:
                continue
            best = min(D[i - 1, j - 1] if i and j else np.inf, D[i - 1, j] + step_penalty if i else np.inf,
                       D[i, j - 1] + step_penalty if j else np.inf)
            D[i, j] = C[i, j] + best
    i, j, path = T - 1, T - 1, [(T - 1, T - 1)]
    while i or j:
        cands = [(D[i - 1, j - 1], i - 1, j - 1) if i and j else (np.inf, 0, 0),
                 (D[i - 1, j], i - 1, j) if i else (np.inf, 0, 0), (D[i, j - 1], i, j - 1) if j else (np.inf, 0, 0)]
        _, i, j = min(cands)
        path.append((i, j))
    path = np.array(path[::-1])
    match = np.array([path[path[:, 1] == j, 0].mean() for j in range(T)])  # front frame for each side frame
    match = np.convolve(np.pad(match, 3, mode="edge"), np.ones(7) / 7, mode="valid")  # no stair steps
    t = np.arange(T)
    for k in ("front_uv", "front_conf"):
        a = d[k].reshape(T, -1)
        d[k] = np.stack([np.interp(match, t, a[:, c]) for c in range(a.shape[1])], 1).reshape(d[k].shape)
    return match - t


def feet():
    """The joints whose heights make the sync score: creature.json "feet" when MOCAP_CREATURE is set, else the wolf's."""
    import os, pathlib
    cfg = pathlib.Path(os.environ.get("MOCAP_CREATURE", "")) / "creature.json"
    if os.environ.get("MOCAP_CREATURE") and cfg.exists():
        return json.load(open(cfg)).get("feet", [])
    return ["fpaw_L", "fpaw_R", "hpaw_L", "hpaw_R"]


def paw_agreement(d, names):
    """Correlation of each paw's height between the two views over the clip: +1 when both halves show the same
    footfalls, near 0 or negative when the video model animated the views differently."""
    out = {}
    for k in feet():
        if k in names:
            j = names.index(k)
            a, b = d["front_uv"][:, j, 1], d["side_uv"][:, j, 1]
            out[k] = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0.5 and b.std() > 0.5 else None
    vals = [v for v in out.values() if v is not None]
    out["score"] = float(np.mean(vals)) if vals else None
    return out


def sync_report(d):
    """Mean height disagreement between the halves (layout px) and the lag (frames) that minimises it."""
    fv, sv = d["front_uv"][..., 1], d["side_uv"][..., 1]
    c = np.minimum(d["front_conf"], d["side_conf"]).clip(0, 1)
    err = (c * np.abs(fv - sv)).sum(1) / c.sum(1).clip(1e-6)
    best = None
    T = len(fv)
    for lag in range(-12, 13):
        a = fv[max(0, lag):T + min(0, lag)]
        b = sv[max(0, -lag):T + min(0, -lag)]
        e = np.abs(a - b).mean()
        if best is None or e < best[1]:
            best = (lag, e)
    return {"mean_height_gap_px": float(err.mean()), "worst_frame_gap_px": float(err.max()),
            "per_frame_gap_px": err.round(2).tolist(), "best_lag_frames": best[0], "gap_at_best_lag_px": float(best[1])}


def euler_xyz(a):
    """Blender Euler XYZ (applied X, then Y, then Z) to rotation matrices: Rz @ Ry @ Rx."""
    x, y, z = a.unbind(-1)
    cx, sx, cy, sy, cz, sz = x.cos(), x.sin(), y.cos(), y.sin(), z.cos(), z.sin()
    o, l = torch.zeros_like(x), torch.ones_like(x)
    Rx = torch.stack([l, o, o, o, cx, -sx, o, sx, cx], -1).view(*x.shape, 3, 3)
    Ry = torch.stack([cy, o, sy, o, l, o, -sy, o, cy], -1).view(*x.shape, 3, 3)
    Rz = torch.stack([cz, -sz, o, sz, cz, o, o, o, l], -1).view(*x.shape, 3, 3)
    return Rz @ Ry @ Rx


def axis_angle(v):
    th = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = v / th
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]
    return torch.eye(3, dtype=v.dtype, device=v.device) + th.sin() * K + (1 - th.cos()) * K @ K


class Rig:
    def __init__(self, info, dev="cpu"):
        self.names = [b["name"] for b in info["bones"]]
        self.parent = [b["parent"] for b in info["bones"]]
        B = torch.tensor([b["matrix_local"] for b in info["bones"]], dtype=torch.float64, device=dev)
        self.B = B
        self.L = torch.tensor([b["length"] for b in info["bones"]], dtype=torch.float64, device=dev)
        # rest offset of each bone in its parent's frame
        self.off = torch.stack([B[i] if p < 0 else torch.linalg.inv(B[p]) @ B[i] for i, p in enumerate(self.parent)])
        lim = np.radians(np.array([b["limits_deg"] for b in info["bones"]], float))
        self.lo = torch.tensor(lim[..., 0], device=dev)
        self.hi = torch.tensor(lim[..., 1], device=dev)
        self.kp = [(k["bone"], k["end"]) for k in info["keypoints"]]
        self.kp_names = [k["name"] for k in info["keypoints"]]
        self.rest_kp = torch.tensor([k["rest"] for k in info["keypoints"]], dtype=torch.float64, device=dev)

    def forward(self, trans, root_rot, eul):
        """trans (T,3) world offset, root_rot (T,3,3) pelvis basis, eul (T,nb,3) Euler for every bone (root ignored)."""
        T, nb = eul.shape[:2]
        R = euler_xyz(eul)
        P = [None] * nb
        for i, p in enumerate(self.parent):
            basis = torch.zeros(T, 4, 4, dtype=eul.dtype, device=eul.device)
            basis[:, :3, :3] = root_rot if p < 0 else R[:, i]
            basis[:, 3, 3] = 1
            if p < 0:
                M = self.off[i].expand(T, 4, 4) @ basis
                M = M.clone()
                M[:, :3, 3] += trans
            else:
                M = P[p] @ self.off[i] @ basis
            P[i] = M
        P = torch.stack(P, 1)
        joints = []
        for b, end in self.kp:
            if end == "head":
                joints.append(P[:, b, :3, 3])
            else:
                joints.append(P[:, b, :3, 1] * self.L[b] + P[:, b, :3, 3])
        return torch.stack(joints, 1), P


def project(J):
    v = layout.H / 2 - (J[..., 2] - layout.Z_CENTER) / layout.M_PER_PX
    f = torch.stack([layout.W_FRONT / 2 + J[..., 0] / layout.M_PER_PX, v], -1)
    s = torch.stack([layout.W_SIDE / 2 - J[..., 1] / layout.M_PER_PX, v], -1)
    return f, s


def triangulate(d, front_height_weight=0.25):
    """Joints in metres: x from the front view, y from the side view, height mostly from the side view."""
    fx = (d["front_uv"][..., 0] - layout.W_FRONT / 2) * layout.M_PER_PX
    sy = -(d["side_uv"][..., 0] - layout.W_SIDE / 2) * layout.M_PER_PX
    fz = layout.Z_CENTER + (layout.H / 2 - d["front_uv"][..., 1]) * layout.M_PER_PX
    sz = layout.Z_CENTER + (layout.H / 2 - d["side_uv"][..., 1]) * layout.M_PER_PX
    w1, w2 = front_height_weight * d["front_conf"].clip(1e-3), d["side_conf"].clip(1e-3)
    return np.stack([fx, sy, (fz * w1 + sz * w2) / (w1 + w2)], -1)


def root_from_triangulation(tri, conf, rig, names, info):
    """Per-frame root rotation (as the basis of the root bone, axis-angle) from the points that ride rigidly on it
    (its two ends and the heads of the bones hung straight off it), by weighted Kabsch against the rest pose;
    smoothed over time."""
    bones = info["bones"]
    ride = {bones[0]["head_joint"], bones[0]["tail_joint"]} | {b["head_joint"] for b in bones if b["parent"] == 0}
    body = sorted(names.index(n) for n in ride if n in names)
    h = names.index(bones[0]["head_joint"])
    rest = rig.rest_kp.cpu().numpy()
    a = rest[body] - rest[h]
    Brot = rig.B[0, :3, :3].cpu().numpy()
    out = []
    for t in range(len(tri)):
        b = tri[t, body] - tri[t, h]
        w = conf[t, body]
        U, _, Vt = np.linalg.svd((a * w[:, None]).T @ b)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        W = Vt.T @ D @ U.T
        R = Brot.T @ W @ Brot
        ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        out.append(axis / max(np.linalg.norm(axis), 1e-8) * ang if ang > 1e-6 else np.zeros(3))
    out = np.array(out)
    k = np.ones(5) / 5
    return np.stack([np.convolve(np.pad(out[:, i], 2, mode="edge"), k, mode="valid") for i in range(3)], 1)


def euler_from_matrix(R):
    """Inverse of euler_xyz for one 3x3 matrix (Blender XYZ: R = Rz Ry Rx)."""
    b = -np.arcsin(np.clip(R[2, 0], -1, 1))
    return np.array([np.arctan2(R[2, 1], R[2, 2]), b, np.arctan2(R[1, 0], R[0, 0])])


def rot_between(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def ik_init(tri, root_aa, info, rig):
    """Point every bone at its triangulated child joint, down the hierarchy, with no twist and inside its limits:
    a starting pose close to the answer, so the fit does not have to find a sit or a rear from a standing pose."""
    names = rig.kp_names
    bones = info["bones"]
    off = rig.off.cpu().numpy()
    Brot = rig.B[0, :3, :3].cpu().numpy()
    lo, hi = rig.lo.cpu().numpy(), rig.hi.cpu().numpy()
    T, nb = len(tri), len(bones)
    eul = np.zeros((T, nb, 3))
    y = np.array([0.0, 1.0, 0.0])
    for t in range(T):
        aa = root_aa[t]
        ang = np.linalg.norm(aa)
        k = aa / ang if ang > 1e-9 else np.array([1.0, 0, 0])
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        W = [None] * nb
        W[0] = Brot @ (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K)
        for i in range(1, nb):
            p = bones[i]["parent"]
            base = W[p] @ off[i][:3, :3]
            if bones[i]["head_joint"] not in names or bones[i]["tail_joint"] not in names:
                W[i] = base  # e.g. fur bones, whose root is not a tracked joint: start at rest
                continue
            d = tri[t, names.index(bones[i]["tail_joint"])] - tri[t, names.index(bones[i]["head_joint"])]
            if np.linalg.norm(d) < 1e-6:
                W[i] = base
                continue
            R = rot_between(y, base.T @ (d / np.linalg.norm(d)))
            e = np.clip(euler_from_matrix(R), lo[i], hi[i])
            eul[t, i] = e
            W[i] = base @ euler_xyz(torch.tensor(e)).numpy()
    return eul


def fit(d, info, iters=1500, dev="cpu", log=True, front_height_weight=0.25, fps=24, plant_iters=600, plant=True,
        far_side_weight=1.0, joint_weight=None, joint_sigma=None, plant_hysteresis=1.0):
    rig = Rig(info, dev)
    names = rig.kp_names
    T = len(d["front_uv"])
    f_obs = torch.tensor(d["front_uv"], dtype=torch.float64, device=dev)
    s_obs = torch.tensor(d["side_uv"], dtype=torch.float64, device=dev)
    f_c = torch.tensor(d["front_conf"], dtype=torch.float64, device=dev).clamp(0, 1) ** 2
    s_c = torch.tensor(d["side_conf"], dtype=torch.float64, device=dev).clamp(0, 1) ** 2
    jw = torch.ones(len(names), dtype=torch.float64, device=dev)  # per-joint data weight and robust-scale multiplier
    js = torch.ones(len(names), dtype=torch.float64, device=dev)
    for spec, vec in ((joint_weight, jw), (joint_sigma, js)):
        for pre, val in (spec or {}).items():
            for i, n in enumerate(names):
                if n.startswith(pre):
                    vec[i] = val
    f_c = f_c * jw
    s_c = s_c * jw
    if far_side_weight != 1.0:  # the side camera sees the right flank: _L joints are behind the body there
        far = [i for i, n in enumerate(names) if n.endswith("_L")]
        s_c[:, far] *= far_side_weight
    tri_np = triangulate(d, front_height_weight)
    tri = torch.tensor(tri_np, dtype=torch.float64, device=dev)
    hips = names.index(info["bones"][0]["head_joint"])
    trans = (tri[:, hips] - rig.rest_kp[hips]).clone().requires_grad_(True)
    conf = np.minimum(d["front_conf"], d["side_conf"]).clip(1e-3, 1)
    root0 = root_from_triangulation(tri_np, conf, rig, names, info) + 1e-4
    root = torch.tensor(root0, dtype=torch.float64, device=dev).requires_grad_(True)
    nb = len(rig.names)
    # Euler = lo + (hi - lo) * sigmoid(phi), starting from the analytic IK pose
    e0 = torch.tensor(ik_init(tri_np, root0, info, rig), dtype=torch.float64, device=dev)
    z = ((e0 - rig.lo) / (rig.hi - rig.lo)).clamp(1e-3, 1 - 1e-3)
    phi = torch.logit(z).clone().requires_grad_(True)
    # ground contact from creature.json (contact_joints): never below their rest height, and planted in stage 2
    cidx, h0 = contact_joints(info, names)
    h0_t = torch.tensor(h0, dtype=torch.float64, device=dev)
    plant_target, plant_w = None, None
    sigma = 6.0  # px, robust scale (annealed down from 40 px so far-off joints still pull early on)
    # The side view sees legs swing and the hips clearly; the front view foreshortens leg lifts and hides the hips
    # when the creature sits, and video models animate it less faithfully. So heights come mostly from the side
    # view and the front view mainly supplies left-right position.
    f_axis = torch.tensor([1.0, front_height_weight], dtype=torch.float64, device=dev)

    def loss_fn(stage):
        nonlocal sigma
        eul = rig.lo + (rig.hi - rig.lo) * torch.sigmoid(phi)
        J, _ = rig.forward(trans, axis_angle(root), eul)
        pf, ps = project(J)
        rf = ((pf - f_obs) ** 2 * f_axis).sum(-1)
        rs = ((ps - s_obs) ** 2).sum(-1)
        sj = (sigma * js) ** 2  # a joint with a wider robust scale is not written off as an outlier when it is far off
        data = (f_c * torch.log1p(rf / sj)).sum() + (s_c * torch.log1p(rs / sj)).sum()
        acc = ((J[2:] - 2 * J[1:-1] + J[:-2]) ** 2).sum() / layout.M_PER_PX ** 2 * 0.02
        vel = ((eul[1:] - eul[:-1]) ** 2).sum() * 2.0 + ((root[1:] - root[:-1]) ** 2).sum() * 2.0
        prior = (eul ** 2).sum() * 0.05
        ground = (torch.relu(h0_t - 0.01 - J[:, cidx, 2]) ** 2).sum() / layout.M_PER_PX ** 2 * 0.5 if cidx else J.sum() * 0
        planted = J.sum() * 0
        if plant_target is not None:  # planted feet stay where they landed (moving only with the ground)
            planted = (plant_w * ((J[:, cidx] - plant_target) ** 2).sum(-1)).sum() / layout.M_PER_PX ** 2 * 0.5
        total = data + acc + vel + prior + ground + planted
        return total, {"data": data.item(), "acc": acc.item(), "vel": vel.item(), "prior": prior.item(), "ground": ground.item(),
                       "planted": planted.item()}, J, pf, ps, eul

    opt = torch.optim.Adam([{"params": [trans], "lr": 0.01}, {"params": [root], "lr": 0.03}, {"params": [phi], "lr": 0.05}])
    for it in range(iters):
        sigma = 6.0 + 34.0 * max(0.0, 1 - it / (0.6 * iters))
        if it == int(iters * 0.7):
            for g in opt.param_groups:
                g["lr"] *= 0.3
        total, parts, J, pf, ps, eul = loss_fn(it)
        opt.zero_grad()
        total.backward()
        opt.step()
        if log and it % 250 == 0:
            print(f"  iter {it} loss {total.item():.1f}", {k: round(v, 1) for k, v in parts.items()}, flush=True)
    sigma = 6.0
    runs_, g = [[] for _ in cidx], np.zeros(2)
    if plant and cidx:
        # stage 2: find where each contact joint is planted in the first fit, then hold it there while refitting
        runs_, g = find_contacts(J.detach().cpu().numpy(), cidx, h0, fps, plant_hysteresis)
        tgt = np.zeros((T, len(cidx), 3))
        w = np.zeros((T, len(cidx)))
        Jn = J.detach().cpu().numpy()
        for k, rr in enumerate(runs_):
            for a, b in rr:
                t = np.arange(a, b)
                xy = Jn[a:b, cidx[k], :2]
                p0 = (xy - g[None] * (t - a)[:, None]).mean(0)
                tgt[a:b, k, :2] = p0 + g[None] * (t - a)[:, None]
                tgt[a:b, k, 2] = h0[k]
                w[a:b, k] = 1.0
                w[a, k] = w[b - 1, k] = 0.5  # ease in and out of the plant
        plant_target = torch.tensor(tgt, dtype=torch.float64, device=dev)
        plant_w = torch.tensor(w, dtype=torch.float64, device=dev)
        for it in range(plant_iters):
            total, parts, J, pf, ps, eul = loss_fn(iters + it)
            opt.zero_grad()
            total.backward()
            opt.step()
            if log and it % 200 == 0:
                print(f"  plant iter {it} loss {total.item():.1f}", {k: round(v, 1) for k, v in parts.items()}, flush=True)
        if log:
            print(f"planted contacts: {sum(len(r) for r in runs_)} plants over {len(cidx)} contact joints, "
                  f"ground velocity {np.round(g * fps, 3).tolist()} m/s")
    with torch.no_grad():
        total, parts, J, pf, ps, eul = loss_fn(iters)
        slide = foot_slide(J.cpu().numpy(), cidx, runs_, g)
        err_f = (pf - f_obs).norm(dim=-1)
        err_fx = (pf - f_obs)[..., 0].abs()
        err_s = (ps - s_obs).norm(dim=-1)
        okf, oks = f_c > 0.3 ** 2, s_c > 0.3 ** 2  # readings the reader was sure of (conf > 0.3)
        confident = {"front": float(err_f[okf].mean()), "front_x_only": float(err_fx[okf].mean()),
                     "side": float(err_s[oks].mean())}
    return {"reproj_confident_px": confident, "trans": trans.detach().cpu().numpy(), "root_rotmat": axis_angle(root).detach().cpu().numpy(),
            "euler": eul.detach().cpu().numpy(), "joints": J.cpu().numpy(),
            "reproj_px": {"front": float(err_f.mean()), "front_x_only": float(err_fx.mean()), "side": float(err_s.mean())},
            "bones": rig.names, "joint_names": names,
            "contacts": {"joints": [names[i] for i in cidx], "rest_height": h0.round(4).tolist(),
                         "runs": [[list(map(int, r)) for r in rr] for rr in runs_], "ground_velocity": g.round(6).tolist(),
                         "slide_cm": slide}}


def foot_slide(J, cidx, runs_, g):
    """How far planted contact joints still slide per frame (relative to the ground), in cm: mean and worst."""
    d = []
    for k, rr in enumerate(runs_):
        for a, b in rr:
            v = np.diff(J[a:b, cidx[k], :2], axis=0) - g[None]
            d.append(np.linalg.norm(v, axis=-1))
    if not d:
        return {"mean": None, "max": None}
    d = np.concatenate(d) * 100
    return {"mean": round(float(d.mean()), 3), "max": round(float(d.max()), 3)}


def to_quat(Rm):
    q = []
    for R in Rm:
        w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        x = np.copysign(np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
        y = np.copysign(np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
        z = np.copysign(np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
        q.append([w, x, y, z])
    return np.array(q)


def kv(flag):
    """--joint-weight tail=4,jaw=2 -> {"tail": 4.0, "jaw": 2.0} (joint-name prefixes)"""
    if flag not in sys.argv:
        return None
    return {k: float(v) for k, v in (x.split("=") for x in sys.argv[sys.argv.index(flag) + 1].split(","))}


def main():
    det_path, info_path, out = sys.argv[1:4]
    raw = np.load(det_path)
    d = {k: raw[k].copy() for k in ("front_uv", "side_uv", "front_conf", "side_conf")}
    info = json.load(open(info_path))
    names = [k["name"] for k in info["keypoints"]]
    before = sync_report(d)
    paws = paw_agreement(d, names)
    print("paw height agreement between views:", {k: (None if v is None else round(v, 2)) for k, v in paws.items()})
    print(f"sync as generated: mean height gap {before['mean_height_gap_px']:.1f} px, worst {before['worst_frame_gap_px']:.1f} px, "
          f"best constant lag {before['best_lag_frames']} frames")
    shift = None
    if "--no-align" not in sys.argv:
        trial = {k: v.copy() for k, v in d.items()}
        s = align_views(trial, names)
        gain = 1 - sync_report(trial)["mean_height_gap_px"] / max(before["mean_height_gap_px"], 1e-6)
        if gain > 0.15:  # only re-time when the halves really were out of step
            d.update(trial)
            shift = s
            print(f"time-aligned the front view onto the side view: mean shift {np.abs(shift).mean():.1f} frames, "
                  f"largest {np.abs(shift).max():.1f} frames (height gap down {gain:.0%})")
        else:
            print(f"views already in step (re-timing would only cut the height gap by {gain:.0%}); left as generated")
    # painted legs make the reader's left/right labels trustworthy, so swaps are only fixed on request
    suspect = fix_swaps({k: v.copy() for k, v in d.items()}, names, info, apply=False)
    print("frames where swapping near/far limbs in the side view would agree better:", suspect)
    swapped = fix_swaps(d, names, info) if "--fix-swaps" in sys.argv else {}
    if "--fix-front-swaps" in sys.argv or creature().get("fix_front_swaps"):
        swapped["front"] = fix_front_swaps(d, names, info)
    sync = sync_report(d)
    print("side-view leg swaps fixed (frames):", swapped)
    print(f"sync after: mean height gap {sync['mean_height_gap_px']:.1f} px, worst {sync['worst_frame_gap_px']:.1f} px")
    sync["before"] = {k: v for k, v in before.items() if k != "per_frame_gap_px"}
    sync["paw_agreement"] = paws
    sync["front_shift_frames"] = None if shift is None else shift.round(2).tolist()
    fhw = float(sys.argv[sys.argv.index("--front-height-weight") + 1]) if "--front-height-weight" in sys.argv else 0.25
    res = fit(d, info, dev="cuda" if torch.cuda.is_available() else "cpu", fps=float(raw["fps"]),
              plant="--no-plant" not in sys.argv, front_height_weight=fhw,
              far_side_weight=float(sys.argv[sys.argv.index("--far-side-weight") + 1]) if "--far-side-weight" in sys.argv else 1.0,
              joint_weight=kv("--joint-weight"), joint_sigma=kv("--joint-sigma"),
              plant_hysteresis=float(sys.argv[sys.argv.index("--plant-hysteresis") + 1]) if "--plant-hysteresis" in sys.argv else 1.0)
    print("reprojection error px:", {k: round(v, 2) for k, v in res["reproj_px"].items()})
    print("reprojection error px, confident readings only:", {k: round(v, 2) for k, v in res["reproj_confident_px"].items()})
    print("planted foot slide cm/frame:", res["contacts"]["slide_cm"])
    motion = {"fps": float(raw["fps"]), "bones": res["bones"], "trans": res["trans"].round(5).tolist(),
              "root_quat": to_quat(res["root_rotmat"]).round(6).tolist(), "euler": res["euler"].round(5).tolist(),
              "joints": res["joints"].round(4).tolist(), "joint_names": res["joint_names"],
              "reproj_px": res["reproj_px"], "reproj_confident_px": res["reproj_confident_px"], "sync": sync, "swaps": swapped, "swap_suspect_frames": suspect,
              "contacts": res["contacts"]}
    json.dump(motion, open(out, "w"))
    print("saved", out)


if __name__ == "__main__":
    main()
