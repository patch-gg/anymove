"""Fit the rig to the joints read from both views: one pose per frame, smooth over time, inside the joint limits.

python -m mocap.solve DETECTIONS.npz rig_info.json OUT.json
Before fitting, the side view's left and right legs are checked against the front view: a joint has the same height
in both halves, so if swapping a pair of legs in the side view makes the heights agree better, the side reader
mixed up the near and far leg and they are swapped back. The same height test gives the sync score: how far apart
the two halves disagree, and the time lag between them that agrees best.

The fit runs Blender's forward kinematics in torch (pose = parent pose x rest offset x Euler XYZ), with the pelvis
free to turn and move, every other bone limited to its range, and minimises the confidence-weighted distance between
the rig's projected joints and the readings in both views, plus smoothness and a small pull toward the rest pose.

A detection made with `mocap.detect --single-view` (one camera, not two) is fitted in 2.5D: the side view gives Y
and Z, and X - which one camera cannot see - is held at the rest pose. The swap, sync and paw-agreement tests need
two views and are skipped, and the front number is the rig's deviation from rest, NOT a reading accuracy.
"""
import json, sys

import numpy as np
import torch

from mocap import layout

LEGS = {"F": ["scap", "shoulder", "elbow", "wrist", "fpaw", "ftoe"], "H": ["hip", "knee", "hock", "hpaw", "htoe"]}


def leg_pairs(names):
    """Index pairs (left, right) of every leg joint, grouped per front/hind pair."""
    out = {}
    for k, stems in LEGS.items():
        pairs = []
        for s in stems:
            if f"{s}_L" in names:
                pairs.append((names.index(f"{s}_L"), names.index(f"{s}_R")))
        out[k] = pairs
    return out


def fix_swaps(d, names, margin=4.0):
    """Swap near/far legs in the side view where the heights say the reader mixed them up."""
    fv, sv = d["front_uv"][..., 1], d["side_uv"][..., 1]
    c = np.minimum(d["front_conf"], d["side_conf"]).clip(0, 1)
    swapped = {}
    for k, pairs in leg_pairs(names).items():
        L, R = np.array(pairs).T
        same = (c[:, L] * np.abs(sv[:, L] - fv[:, L]) + c[:, R] * np.abs(sv[:, R] - fv[:, R])).sum(1)
        swap = (c[:, L] * np.abs(sv[:, R] - fv[:, L]) + c[:, R] * np.abs(sv[:, L] - fv[:, R])).sum(1)
        gain = (same - swap) / len(pairs)
        gain = np.array([np.median(gain[max(0, t - 2):t + 3]) for t in range(len(gain))])
        do = gain > margin
        for t in np.where(do)[0]:
            d["side_uv"][t, L], d["side_uv"][t, R] = d["side_uv"][t, R].copy(), d["side_uv"][t, L].copy()
            d["side_conf"][t, L], d["side_conf"][t, R] = d["side_conf"][t, R].copy(), d["side_conf"][t, L].copy()
        swapped[k] = int(do.sum())
    return swapped


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


def fit(d, info, iters=1500, dev="cpu", log=True, front_height_weight=0.25):
    rig = Rig(info, dev)
    names = rig.kp_names
    T = len(d["front_uv"])
    f_obs = torch.tensor(d["front_uv"], dtype=torch.float64, device=dev)
    s_obs = torch.tensor(d["side_uv"], dtype=torch.float64, device=dev)
    f_c = torch.tensor(d["front_conf"], dtype=torch.float64, device=dev).clamp(0, 1) ** 2
    s_c = torch.tensor(d["side_conf"], dtype=torch.float64, device=dev).clamp(0, 1) ** 2
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
    paws = [names.index(n) for n in names if any(n.startswith(p) for p in ("fpaw", "ftoe", "hpaw", "htoe"))]
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
        data = (f_c * torch.log1p(rf / sigma ** 2)).sum() + (s_c * torch.log1p(rs / sigma ** 2)).sum()
        acc = ((J[2:] - 2 * J[1:-1] + J[:-2]) ** 2).sum() / layout.M_PER_PX ** 2 * 0.02
        vel = ((eul[1:] - eul[:-1]) ** 2).sum() * 2.0 + ((root[1:] - root[:-1]) ** 2).sum() * 2.0
        prior = (eul ** 2).sum() * 0.05
        ground = (torch.relu(-J[:, paws, 2]) ** 2).sum() / layout.M_PER_PX ** 2 * 0.5
        total = data + acc + vel + prior + ground
        return total, {"data": data.item(), "acc": acc.item(), "vel": vel.item(), "prior": prior.item(), "ground": ground.item()}, J, pf, ps, eul

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
    with torch.no_grad():
        total, parts, J, pf, ps, eul = loss_fn(iters)
        err_f = (pf - f_obs).norm(dim=-1)
        err_fx = (pf - f_obs)[..., 0].abs()
        err_s = (ps - s_obs).norm(dim=-1)
    return {"trans": trans.detach().cpu().numpy(), "root_rotmat": axis_angle(root).detach().cpu().numpy(),
            "euler": eul.detach().cpu().numpy(), "joints": J.cpu().numpy(),
            "reproj_px": {"front": float(err_f.mean()), "front_x_only": float(err_fx.mean()), "side": float(err_s.mean())},
            "bones": rig.names, "joint_names": names}


def to_quat(Rm):
    q = []
    for R in Rm:
        w = np.sqrt(max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        x = np.copysign(np.sqrt(max(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
        y = np.copysign(np.sqrt(max(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
        z = np.copysign(np.sqrt(max(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
        q.append([w, x, y, z])
    return np.array(q)


def main():
    # A flag's VALUE must not be mistaken for a positional argument, so the
    # flag and its value are lifted out before the positionals are read.
    argv = list(sys.argv[1:])
    # The rig fit can run on the GPU, but the GPU is a shared resource on this
    # box and the other ComfyUI lane holds most of the card. `--device cpu`
    # keeps the whole read off the VRAM entirely.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "--device" in argv:
        i = argv.index("--device")
        if i + 1 < len(argv):
            dev = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
    if dev not in ("cuda", "cpu"):
        raise SystemExit(f"--device must be cuda or cpu, not {dev!r}")
    # The iteration count is the fit's budget. It does not change WHAT is
    # reported - the contract, the keys, the flags - only how well the rig is
    # fitted. A guard that checks the contract does not need the full budget.
    iters = 1500
    if "--iters" in argv:
        i = argv.index("--iters")
        if i + 1 < len(argv):
            iters = int(argv[i + 1])
            argv = argv[:i] + argv[i + 2:]
    det_path, info_path, out = argv[:3]
    print(f"fit device: {dev}")
    print(f"fit iterations: {iters}")
    raw = np.load(det_path)
    single = bool(raw["single_view"]) if "single_view" in raw.files else False
    # Whether the read's calibration actually locked onto the reference. A False
    # here means the clip is not framed like the reference and the rig fit below
    # is INDICATIVE, not contract-grade - it must travel with the result.
    calib_converged = bool(raw["calib_converged"]) if "calib_converged" in raw.files else True
    if not calib_converged:
        print("WARNING: the read's calibration fell back to a plain stretch - "
              "this fit is indicative, not contract-grade")
    d = {k: raw[k].copy() for k in ("front_uv", "side_uv", "front_conf", "side_conf")}
    info = json.load(open(info_path))
    names = [k["name"] for k in info["keypoints"]]
    if single:
        # ONE camera, not two. A side view gives Y and Z; the X axis (left-right)
        # is not observable from a single camera, so it is HELD AT THE REST POSE
        # and the front observation is the rest projection. That is an ASSUMPTION,
        # not a measurement - so the front number below is the rig's deviation
        # from rest, not a reading accuracy against a real front camera.
        rest = Rig(info).rest_kp.numpy()
        v = layout.H / 2 - (rest[:, 2] - layout.Z_CENTER) / layout.M_PER_PX
        front_rest = np.stack([layout.W_FRONT / 2 + rest[:, 0] / layout.M_PER_PX, v], -1)
        T = len(d["side_uv"])
        d["front_uv"] = np.tile(front_rest, (T, 1, 1))
        d["front_conf"] = np.ones_like(d["side_conf"])
        print("SINGLE VIEW: fitting the rig in 2.5D - Y and Z from the side view, "
              "X held at the rest pose (one camera cannot see left-right motion)")
        res = fit(d, info, iters=iters, dev=dev)
        rp = res["reproj_px"]
        print("reprojection error px: side", round(rp["side"], 2),
              "| front rest-deviation", round(rp["front"], 2))
        motion = {"fps": float(raw["fps"]), "bones": res["bones"], "trans": res["trans"].round(5).tolist(),
                  "root_quat": to_quat(res["root_rotmat"]).round(6).tolist(), "euler": res["euler"].round(5).tolist(),
                  "joints": res["joints"].round(4).tolist(), "joint_names": res["joint_names"],
                  "reproj_px": {"side": rp["side"], "front_rest_deviation_px": rp["front"],
                                "front_x_only": rp["front_x_only"]},
                  "sync": None, "swaps": {}, "single_view": True,
                  "calib_converged": calib_converged}
        json.dump(motion, open(out, "w"))
        print("saved", out)
        return
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
    swapped = fix_swaps(d, names) if "--fix-swaps" in sys.argv else {}
    sync = sync_report(d)
    print("side-view leg swaps fixed (frames):", swapped)
    print(f"sync after: mean height gap {sync['mean_height_gap_px']:.1f} px, worst {sync['worst_frame_gap_px']:.1f} px")
    sync["before"] = {k: v for k, v in before.items() if k != "per_frame_gap_px"}
    sync["paw_agreement"] = paws
    sync["front_shift_frames"] = None if shift is None else shift.round(2).tolist()
    res = fit(d, info, iters=iters, dev=dev)
    print("reprojection error px:", {k: round(v, 2) for k, v in res["reproj_px"].items()})
    motion = {"fps": float(raw["fps"]), "bones": res["bones"], "trans": res["trans"].round(5).tolist(),
              "root_quat": to_quat(res["root_rotmat"]).round(6).tolist(), "euler": res["euler"].round(5).tolist(),
              "joints": res["joints"].round(4).tolist(), "joint_names": res["joint_names"],
              "reproj_px": res["reproj_px"], "sync": sync, "swaps": swapped,
              "calib_converged": calib_converged}
    json.dump(motion, open(out, "w"))
    print("saved", out)


if __name__ == "__main__":
    main()
