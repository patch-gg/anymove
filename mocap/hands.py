"""Wrists and fingers for a captured clip, from a hand model. The joint reader cannot see fingers: a hand is forty
pixels of a 720-pixel frame. A hand model can, once it is shown the hand and told which one it is.

HANDS_PY -m mocap.hands CLIP.mp4 DET.npz rig_info.json MOTION.json [--method wilor|mediapipe] [--wrist 0.8]
                        [--sheet SHEET.jpg] [--out OTHER.json]

This is the finger tracking of prompt2rig (github.com/blendi-remade/prompt2rig, hq/hands.py) fitted to two cameras.
prompt2rig finds hands with a detector and tells them apart by the body's wrists. Here the reader has already found
each wrist and knuckle in both views, and the rig says which hand is which, so every hand is cut out where it is
(both views), mirrored if it is a left hand, and only the pose model runs: WiLoR (CC-BY-NC-ND, and it needs MANO,
which is non-commercial too) or MediaPipe Hands (Apache-2.0; it rarely finds a hand this small). A view counts only
while the hand's paint colour is what the crop shows, so a hand hidden behind the body is not read off whatever
covers it. The two views' answers are turned into world axes (the cameras are fixed and known), blended by how
squarely each sees the palm, gap-filled and smoothed.

From the 21 keypoints: each finger bone is aimed along its measured segment, in the palm's own frame, so neither the
hand's size nor an error in its overall turn matters; folds are kept inside what a finger can do. The wrist is turned
toward the measured palm by `--wrist` (0 keeps the solver's). A hand that holds something (rig.json "hold") keeps its
grip and its solved wrist: the prop's two joints already pin that hand.

Writes into MOTION.json: "fingers" {"bones", "euler" [T][bones][3]} for blender/apply_motion.py, the adjusted hand
Eulers with the joints they move, and "hands" (method, how often each hand was seen, by which view, and the hand
Eulers the solver had found, so the step can be run again with other settings).
Runs under the hand model's Python, not anymove's: PROMPT2RIG_HQ is the hq folder of a prompt2rig checkout, whose
.venv has the models and whose weights/ holds their files (its setup_hands.py fetches them). See docs/hands.md.

How the two models are set up, and the gap and smoothing settings, follow prompt2rig's hq/hands.py (Copyright 2026
blendi-remade, Apache-2.0). The models themselves are only called, through their own packages.
"""
import argparse, json, math, os, pathlib

import cv2
import numpy as np
import torch

from mocap import layout
from mocap.solve import Rig, euler_from_matrix, euler_xyz

HQ = pathlib.Path(os.environ["PROMPT2RIG_HQ"]) if os.environ.get("PROMPT2RIG_HQ") else None  # prompt2rig's hq folder
# camera axes (x right, y down, z into the picture) -> world. The front camera looks along +Y, the side one along +X.
CAM_TO_WORLD = {"front": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float), "side": np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)}
VIEW_DIR = {"front": np.array([0, 1.0, 0]), "side": np.array([1.0, 0, 0])}
FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
# what a finger joint can do, degrees: fold range per bone (base to tip) and sideways swing of the first bone
FOLD_LIMIT = {"thumb": ((-20, 50), (-10, 65), (-10, 85)), "other": ((-12, 95), (-5, 112), (-5, 85))}
SIDE_LIMIT = {"thumb": 30.0, "other": 14.0}
MAX_GAP_S, SMOOTH_SIGMA = 0.4, 1.3


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def weights():
    if HQ is None or not (HQ / "weights").is_dir():
        raise SystemExit("hands: set PROMPT2RIG_HQ to the hq folder of a prompt2rig checkout with its weights (docs/hands.md)")
    return HQ / "weights"


# ---- hand models: crops (RGB, any size) + handedness -> 21 keypoints, camera axes, wrist-relative ------------------
class Wilor:
    name = "wilor"

    def __init__(self):
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

        root = weights() / "wilor"
        if not (root / "pretrained_models" / "MANO_RIGHT.pkl").exists():
            raise FileNotFoundError(f"WiLoR needs MANO_RIGHT.pkl in {root / 'pretrained_models'}: see prompt2rig's hq/setup_hands.py")
        gpu = torch.cuda.is_available()      # half precision on a GPU, as prompt2rig runs it; a CPU works, slowly
        self.pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda" if gpu else "cpu"),
                                                      dtype=torch.float16 if gpu else torch.float32, verbose=False,
                                                      wilor_pretrained_dir=str(root), WILOR_MINI_REPO_ID="__offline__")

    def __call__(self, frames, items):
        """items: (frame index, centre xy, size, right) in frame pixels. -> kp3d (N, 21, 3) camera axes, wrist-relative;
        kp2d (N, 21, 2) frame px; found (N,). Each frame's hands go through WiLoR-mini's own predict_with_bboxes, given
        the box and the handedness, so its detector never runs."""
        k3, k2 = np.zeros((len(items), 21, 3)), np.zeros((len(items), 21, 2))
        by_frame = {}
        for i, it in enumerate(items):
            by_frame.setdefault(int(it[0]), []).append(i)
        for t, idx in by_frame.items():
            boxes = np.array([[*(items[i][1] - items[i][2] / 2), *(items[i][1] + items[i][2] / 2)] for i in idx], np.float32)
            found = self.pipe.predict_with_bboxes(frames[t], boxes, [int(items[i][3]) for i in idx], rescale_factor=1.0)
            for i, f in zip(idx, found):
                k3[i], k2[i] = f["wilor_preds"]["pred_keypoints_3d"][0], f["wilor_preds"]["pred_keypoints_2d"][0]
        return k3 - k3[:, :1], k2, np.ones(len(items), bool)


class MediaPipe:
    name = "mediapipe"

    def __init__(self):
        import mediapipe as mp

        self.mp = mp
        model = os.environ.get("HAND_LANDMARKER") or str(weights() / "hand_landmarker.task")
        self.det = mp.tasks.vision.HandLandmarker.create_from_options(mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model), running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1, min_hand_detection_confidence=0.2, min_hand_presence_confidence=0.2))

    def __call__(self, frames, items, side=384):
        k3 = np.zeros((len(items), 21, 3))
        k2 = np.zeros((len(items), 21, 2))
        ok = np.zeros(len(items), bool)
        for i, (t, c, s, right) in enumerate(items):
            M = np.array([[side / s, 0, -(c[0] - s / 2) * side / s], [0, side / s, -(c[1] - s / 2) * side / s]], np.float32)
            crop = cv2.warpAffine(frames[t], M, (side, side), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            res = self.det.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
            if not res.hand_world_landmarks:
                continue
            w = np.array([(p.x, p.y, p.z) for p in res.hand_world_landmarks[0]])
            k3[i] = w - w[:1]
            k2[i] = np.array([(p.x * s + c[0] - s / 2, p.y * s + c[1] - s / 2) for p in res.hand_landmarks[0]])
            ok[i] = True
        return k3, k2, ok


# ---- where the hands are, and whether a view shows them ----------------------------------------------------------
def hand_crops(det, hand_px, bone, view, scale):
    """Crop centres (T, 2) and sizes (T,) in VIDEO pixels for a hand in `view`, from the reader's joints at the two
    ends of its bone: the wrist and the middle knuckle."""
    names = [str(n) for n in det["names"]]
    inv = np.linalg.inv(np.vstack([det["calib"], [0, 0, 1]]))
    uv = det[f"{view}_uv"].astype(float)
    w, far = uv[:, names.index(bone["head_joint"])], uv[:, names.index(bone["tail_joint"])]
    c = w + 1.15 * (far - w)                  # a little past the knuckle: the middle of an open hand, the end of a fist
    c = c + [layout.W_FRONT if view == "side" else 0.0, 0.0]
    c = np.concatenate([c, np.ones((len(c), 1))], 1) @ inv.T
    return c[:, :2], np.full(len(c), scale * hand_px * inv[0, 0])


def paint_hue(creature, hand):
    """OpenCV hue (0-180) of the paint on a hand bone, or None when the creature is not painted there."""
    for g in (creature or {}).get("paint", {}).values():
        if hand in g["bones"]:
            c = g["color"].lstrip("#")
            rgb = np.uint8([[[int(c[i:i + 2], 16) for i in (0, 2, 4)]]])
            return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[0, 0, 0])
    return None


def shows(frame, c, size, hue, tol=13.0):
    """Share of the crop's middle that is the hand's paint colour."""
    r = max(3, int(size * 0.2))
    x0, y0 = int(round(c[0])) - r, int(round(c[1])) - r
    patch = frame[max(0, y0):y0 + 2 * r, max(0, x0):x0 + 2 * r]
    if patch.size == 0:
        return 0.0
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).astype(float)
    dh = np.abs(hsv[..., 0] - hue)
    dh = np.minimum(dh, 180 - dh)
    return float(((dh < tol) & (hsv[..., 1] > 70) & (hsv[..., 2] > 50)).mean())


# ---- time: gaps and smoothing -------------------------------------------------------------------------------------
def fill_gaps(x, ok, max_gap):
    """Linear interpolation across runs of missing frames up to max_gap long. x (T, ...). Returns x, ok."""
    x, ok = x.copy(), ok.copy()
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return x, ok
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for t in range(a + 1, b):
                u = (t - a) / (b - a)
                x[t] = x[a] * (1 - u) + x[b] * u
                ok[t] = True
    return x, ok


def smooth(x, ok, sigma):
    """Gaussian over time inside each seen run."""
    if sigma <= 0:
        return x
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    out = x.copy()
    T = len(x)
    for t in range(T):
        if not ok[t]:
            continue
        lo, hi = max(0, t - r), min(T, t + r + 1)
        w = k[lo - t + r:hi - t + r] * ok[lo:hi]
        out[t] = np.tensordot(w, x[lo:hi], 1) / w.sum()
    return out


# ---- geometry -----------------------------------------------------------------------------------------------------
def palm_frame(wrist, index, middle, pinky, right):
    """Right-handed frame of a palm, columns (across, along the fingers, out of the palm). Arrays (..., 3)."""
    b = unit(middle - wrist)
    a = index - pinky
    n = unit(np.cross(a, b) if right else np.cross(b, a))
    return np.stack([np.cross(b, n), b, n], -1)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def slerp_mat(A, B, w):
    """Rotation matrix a share w of the way from A to B."""
    R = A.T @ B
    ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if ang < 1e-6 or w <= 0:
        return A
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.linalg.norm(ax) < 1e-9:
        return A if w < 0.5 else B
    ax /= np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    a = ang * w
    return A @ (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K)


class Hand:
    """One hand of the rig at rest: its finger bones and its palm frame."""

    def __init__(self, info, hand):
        f = info["fingers"]
        self.name, self.right = hand, hand.endswith("_R")
        self.bones = {}
        for b in f["bones"]:
            if b["hand"] == hand:
                self.bones[(b["finger"], b["k"])] = b
        R = {k: np.array(b["matrix_local"])[:3, :3] for k, b in self.bones.items()}
        head = {k: np.array(b["matrix_local"])[:3, 3] for k, b in self.bones.items()}
        self.R, self.head = R, head
        hb = next(b for b in info["bones"] if b["name"] == hand)
        self.bone = hb
        self.B = np.array(hb["matrix_local"])[:3, :3]
        self.wrist = np.array(hb["matrix_local"])[:3, 3]
        self.rest_frame = palm_frame(self.wrist, head[("index", 1)], head[("middle", 1)], head[("pinky", 1)], self.right)
        tip = head[("middle", 3)] + R[("middle", 3)][:, 1] * self.bones[("middle", 3)]["length"]
        self.length = float(np.linalg.norm(head[("middle", 1)] - self.wrist) + np.linalg.norm(head[("middle", 2)] - head[("middle", 1)])
                            + np.linalg.norm(head[("middle", 3)] - head[("middle", 2)]) + np.linalg.norm(tip - head[("middle", 3)]))

    def fingers_from(self, kp):
        """21 keypoints (any frame, any scale) -> {bone name: Euler (fold, 0, sideways)} with every joint inside its range."""
        F = palm_frame(kp[0], kp[5], kp[9], kp[17], self.right)
        out = {}
        for f, ids in FINGERS.items():
            lim = FOLD_LIMIT["thumb" if f == "thumb" else "other"]
            side_max = math.radians(SIDE_LIMIT["thumb" if f == "thumb" else "other"])
            posed_parent = None
            for k in (1, 2, 3):
                D = self.rest_frame @ (F.T @ unit(kp[ids[k]] - kp[ids[k - 1]]))    # where this bone should point, hand at rest
                pre = self.R[(f, k)] if k == 1 else posed_parent @ self.R[(f, k - 1)].T @ self.R[(f, k)]
                v = pre.T @ D                                                      # in the bone's own axes before it turns
                fold = math.atan2(v[2], v[1])
                fold = min(max(fold, math.radians(lim[k - 1][0])), math.radians(lim[k - 1][1]))
                side = 0.0
                if k == 1:   # only the first bone swings sideways, and less the more it is folded
                    side = -math.asin(max(-1.0, min(1.0, v[0]))) * max(0.0, math.cos(fold))
                    side = min(max(side, -side_max), side_max)
                out[self.bones[(f, k)]["name"]] = (fold, 0.0, side)
                posed_parent = pre @ rot_z(side) @ rot_x(fold)
        return out

    def tips(self, pose):
        """Fingertip positions (5, 3) in the hand's rest space for a finger pose {bone: Euler}."""
        tips = []
        for f in FINGERS:
            M = None
            for k in (1, 2, 3):
                b = self.bones[(f, k)]
                e = pose.get(b["name"], (0.0, 0.0, 0.0))
                rest = np.array(b["matrix_local"])
                loc = np.eye(4)
                loc[:3, :3] = rot_z(e[2]) @ rot_x(e[0])
                M = rest @ loc if k == 1 else M @ np.linalg.inv(np.array(self.bones[(f, k - 1)]["matrix_local"])) @ rest @ loc
            tips.append(M[:3, 3] + M[:3, 1] * self.bones[(f, 3)]["length"])
        return np.array(tips)


def track(frames, det, info, creature, method="wilor", scale=2.3, log=print):
    """-> {hand: {"kp": (T, 21, 3) world axes wrist-relative, "ok": (T,), "seen": {view: (T,)}, "kp2d": ...}}"""
    model = {"wilor": Wilor, "mediapipe": MediaPipe}[method]()
    T = min(len(frames), len(det["front_uv"]))
    out = {}
    for hand in info["fingers"]["hands"]:
        if hand in info["fingers"]["hold"]:
            continue
        H = Hand(info, hand)
        hue = paint_hue(creature, hand)
        hand_px = H.length / layout.M_PER_PX
        per = {}
        for view in ("front", "side"):
            c, size = hand_crops(det, hand_px, H.bone, view, scale)
            vis = np.array([shows(frames[t], c[t], size[t], hue) if hue is not None else 1.0 for t in range(T)])
            seen = vis > 0.15
            k3 = np.zeros((T, 21, 3))
            k2 = np.zeros((T, 21, 2))
            got = np.zeros(T, bool)
            for _ in range(2):          # second pass: the crop re-centred and sized on the hand the first pass found
                items = [(t, c[t], size[t], int(H.right)) for t in np.flatnonzero(seen)]
                if not items:
                    break
                a, b, ok = model(frames, items)
                ts = np.array([it[0] for it in items])
                k3[ts], k2[ts], got[ts] = a, b, ok
                box = k2[ts].max(1) - k2[ts].min(1)
                nc, ns = (k2[ts].max(1) + k2[ts].min(1)) / 2, np.clip(2.4 * box.max(1), 1.1 * hand_px, 2.8 * hand_px)
                keep = ok & (np.linalg.norm(nc - c[ts], axis=1) < 0.6 * size[ts])     # never wander off the hand
                c, size = c.copy(), size.copy()
                c[ts[keep]], size[ts[keep]] = nc[keep], ns[keep]
            per[view] = {"kp": k3 @ CAM_TO_WORLD[view].T, "ok": seen & got, "kp2d": k2, "c": c, "size": size, "vis": vis}
        out[hand] = per
        log(f"{hand}: front view shows it in {per['front']['ok'].mean():.0%} of frames, side view in {per['side']['ok'].mean():.0%}")
    if hasattr(model, "det"):
        model.det.close()        # MediaPipe: close before the interpreter does, or it complains on the way out
    return out


def fuse(per, right, shift, fps):
    """Blend the two views' keypoints on the motion's timeline (the side view's), fill short gaps, smooth."""
    T = len(per["side"]["kp"])
    t = np.arange(T)
    fi = t if shift is None else np.clip(np.round(t + np.asarray(shift)[:T]).astype(int), 0, T - 1)
    kp = {"front": per["front"]["kp"][fi], "side": per["side"]["kp"]}
    ok = {"front": per["front"]["ok"][fi], "side": per["side"]["ok"]}
    w, nrm = {}, {}
    for v in ("front", "side"):
        F = palm_frame(kp[v][:, 0], kp[v][:, 5], kp[v][:, 9], kp[v][:, 17], right)
        nrm[v] = F[..., 2]
        w[v] = ok[v] * (0.3 + np.abs(nrm[v] @ VIEW_DIR[v]))       # a palm seen squarely is read best
    # views that disagree about which way the palm faces: trust the one that sees it better
    apart = (nrm["front"] * nrm["side"]).sum(-1) < 0.5
    both = ok["front"] & ok["side"] & apart
    lose_f = both & (w["front"] < w["side"])
    w["front"][lose_f] = 0
    w["side"][both & ~lose_f] = 0
    tot = w["front"] + w["side"]
    good = tot > 0
    scale = lambda k: k / (np.linalg.norm(k[:, 9] - k[:, 0], axis=-1)[:, None, None] + 1e-9)  # noqa: E731  one size for both views
    fused = (scale(kp["front"]) * w["front"][:, None, None] + scale(kp["side"]) * w["side"][:, None, None]) / np.maximum(tot, 1e-9)[:, None, None]
    fused, good2 = fill_gaps(fused, good, int(round(MAX_GAP_S * fps)))
    fused = smooth(fused, good2, SMOOTH_SIGMA * fps / 30.0)
    return fused, good2, {"front_share": float((w["front"] > w["side"]).mean()), "seen": float(good.mean()), "filled": float(good2.mean())}


def apply(motion, info, tracked, wrist_weight=0.8, log=print):
    """Write fingers (and wrists) into the motion dict. Returns the report per hand and the hand Eulers the solver
    had found, which a second run starts from."""
    T = len(motion["euler"])
    fps = motion.get("fps", 24)
    rig = Rig(info)
    eul = torch.tensor(motion["euler"], dtype=torch.float64)
    q = np.array(motion["root_quat"])
    w_, x_, y_, z_ = q.T
    Rr = np.stack([1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - z_ * w_), 2 * (x_ * z_ + y_ * w_),
                   2 * (x_ * y_ + z_ * w_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - x_ * w_),
                   2 * (x_ * z_ - y_ * w_), 2 * (y_ * z_ + x_ * w_), 1 - 2 * (x_ * x_ + y_ * y_)], -1).reshape(-1, 3, 3)
    trans, Rr = torch.tensor(motion["trans"], dtype=torch.float64), torch.tensor(Rr, dtype=torch.float64)
    _, P = rig.forward(trans, Rr, eul)
    P = P.numpy()
    off = rig.off.numpy()
    lo, hi = rig.lo.numpy(), rig.hi.numpy()
    f = info["fingers"]
    names = [b["name"] for b in f["bones"]]
    default = {}
    for hand in f["hands"]:
        default.update(f["hold"].get(hand) or f["relaxed"][hand])
    E = np.array([[default[n] for n in names]] * T, float)
    report = {}
    shift = (motion.get("sync") or {}).get("front_shift_frames")
    new_euler = np.array(motion["euler"], float)
    first = dict((motion.get("hands") or {}).get("solved_euler", {}))
    for hand, e in first.items():                                     # run again: start from the solver's wrists, not ours
        new_euler[:, rig.names.index(hand)] = e
    planted = set((motion.get("contacts") or {}).get("joints", []))   # joints the solver holds on the ground
    for hand, per in tracked.items():
        H = Hand(info, hand)
        kp, ok, rep = fuse(per, H.right, shift, fps)
        hi_ = rig.names.index(hand)
        first[hand] = new_euler[:, hi_].round(5).tolist()
        rides = {k["name"] for k in info["keypoints"] if k["bone"] == hi_ and k["end"] != "head"}
        turn = wrist_weight if not (rides & planted) else 0.0         # a hand that walks keeps the wrist that plants it
        T2 = min(T, len(kp))
        # how much to trust the tracking at each frame: eased in and out of the seen runs
        trust = smooth(ok.astype(float), np.ones(len(ok), bool), 2.0)[:T2] * ok[:T2]
        par = rig.parent[hi_]
        folds, turned = [], []
        for t in range(T2):
            if not ok[t]:
                continue
            pose = H.fingers_from(kp[t])
            for n, e in pose.items():
                j = names.index(n)
                E[t, j] = np.array(default[n]) + trust[t] * (np.array(e) - np.array(default[n]))
            folds.append(np.mean([math.degrees(pose[H.bones[(fn, k)]["name"]][0]) for fn in FINGERS if fn != "thumb" for k in (1, 2)]))
            if turn > 0:
                F = palm_frame(kp[t, 0], kp[t, 5], kp[t, 9], kp[t, 17], H.right)
                want = F @ H.rest_frame.T @ H.B                      # the hand bone's world turn that matches the measured palm
                pre = P[t, par, :3, :3] @ off[hi_][:3, :3]
                solved = pre @ euler_xyz(torch.tensor(new_euler[t, hi_])).numpy()
                R = slerp_mat(solved, want, turn * trust[t])
                e = np.clip(euler_from_matrix(pre.T @ R), lo[hi_], hi[hi_])
                turned.append(math.degrees(math.acos(max(-1, min(1, (np.trace(solved.T @ want) - 1) / 2)))))
                new_euler[t, hi_] = e
        idx = [names.index(b["name"]) for b in H.bones.values()]
        E[:, idx] = smooth(E[:, idx], np.ones(T, bool), 0.8)
        if turn > 0:
            new_euler[:, hi_] = smooth(new_euler[:, hi_], np.ones(T, bool), 1.0)
        rep.update({"mean_fold_deg": float(np.mean(folds)) if folds else None, "wrist_weight": turn,
                    "wrist_vs_solver_deg": float(np.mean(turned)) if turned else None})
        report[hand] = rep
        log(f"{hand}: tracked in {rep['seen']:.0%} of frames ({rep['filled']:.0%} with short gaps filled), front view led in "
            f"{rep['front_share']:.0%}; fingers fold {rep['mean_fold_deg'] or 0:.0f} deg on average"
            + (f"; measured palm and solved hand differ by {rep['wrist_vs_solver_deg']:.0f} deg on average" if turned else ""))
    motion["fingers"] = {"bones": names, "euler": E.round(4).tolist()}
    if not np.array_equal(new_euler, np.array(motion["euler"], float)):
        motion["euler"] = new_euler.round(5).tolist()
        J, _ = rig.forward(trans, Rr, torch.tensor(new_euler))        # the knuckles went with their wrists
        motion["joints"] = J.numpy().round(4).tolist()
    return report, first


def sheet(frames, tracked, path, n=12, zoom=4):
    """Crops of each tracked hand in both views with the keypoints the model found."""
    cols = [(255, 80, 80), (255, 200, 60), (80, 230, 80), (80, 200, 255), (220, 100, 255)]
    rows = []
    for hand, per in tracked.items():
        for view in ("front", "side"):
            v = per[view]
            T = len(v["ok"])
            cells = []
            for t in np.linspace(0, T - 1, n).round().astype(int):
                c, sz = v["c"][t], v["size"][t]
                M = np.array([[zoom, 0, -zoom * (c[0] - sz / 2)], [0, zoom, -zoom * (c[1] - sz / 2)]], np.float32)
                side = int(round(sz * zoom))
                img = cv2.warpAffine(frames[t], M, (side, side), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                if v["ok"][t]:
                    k = (v["kp2d"][t] - (c - sz / 2)) * zoom
                    for ids, col in zip(FINGERS.values(), cols):
                        pts = np.round(k[[0, *ids]]).astype(int)
                        for a, b in zip(pts[:-1], pts[1:]):
                            cv2.line(img, tuple(int(x) for x in a), tuple(int(x) for x in b), col, 2, cv2.LINE_AA)
                img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
                cv2.putText(img, f"{hand} {view} {t}" + ("" if v["ok"][t] else " hidden"), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cells.append(img)
            rows.append(np.concatenate(cells, 1))
    if rows:
        cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("det"); ap.add_argument("info"); ap.add_argument("motion")
    ap.add_argument("--method", default="wilor", choices=("wilor", "mediapipe"),
                    help="mediapipe needs no MANO, but rarely finds a hand of this size (docs/hands.md)")
    ap.add_argument("--wrist", type=float, default=0.8, help="how far to turn the solved wrist toward the measured palm (0 to 1)")
    ap.add_argument("--scale", type=float, default=2.3, help="first crop's side as a multiple of the hand's length")
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    info = json.load(open(A.info))
    log = lambda *a: print("hands:", *a, flush=True)  # noqa: E731  tools/capture.sh shows the lines that start so
    if not info.get("fingers"):
        log("this rig has no fingers (rig.json \"fingers\"): nothing to do")
        return
    cfg = pathlib.Path(os.environ.get("MOCAP_CREATURE", "")) / "creature.json"
    creature = json.load(open(cfg)) if os.environ.get("MOCAP_CREATURE") and cfg.exists() else None
    det = dict(np.load(A.det, allow_pickle=True))
    frames = read_frames(A.clip)
    motion = json.load(open(A.motion))
    tracked = track(frames, det, info, creature, A.method, A.scale, log)
    report, first = apply(motion, info, tracked, A.wrist, log)
    motion["hands"] = {"method": A.method, "wrist_weight": A.wrist, "hands": report, "solved_euler": first}
    text = json.dumps(motion)               # whole before the file is touched: a failure here leaves the solver's motion
    dst = pathlib.Path(A.out or A.motion)
    part = dst.with_name(dst.name + ".part")
    part.write_text(text)
    part.replace(dst)
    if A.sheet:
        sheet(frames, tracked, A.sheet)
    log(f"[{A.method}] written to", A.out or A.motion)


if __name__ == "__main__":
    main()
