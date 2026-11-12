"""An exam for the hands step: a made-up motion with known fingers, filmed by both cameras, read back and scored.

python tools/hands_exam.py make  rig_info.json OUT_DIR     OUT_DIR/truth.json (arms and fingers moving), body.json (same, fingers left out)
blender -b -P blender/apply_motion.py -- rigged.blend rig_info.json OUT_DIR/truth.json OUT_DIR/render --views front,side --paint --samples 16
python tools/hands_exam.py video OUT_DIR                   OUT_DIR/exam.mp4 (the two-camera clip) and det.npz (the true joints, as if read)
HANDS_PY -m mocap.hands OUT_DIR/exam.mp4 OUT_DIR/det.npz rig_info.json OUT_DIR/body.json --out OUT_DIR/hands.json --wrist 1
python tools/hands_exam.py score rig_info.json OUT_DIR     fingertip and fold error against the truth, per finger
python tools/hands_exam.py world rig_info.json OUT_DIR SOLVED.json    a whole capture of exam.mp4 (reader, solver,
                                                           hands) against the truth: hand turn, wrist, fingertips in the world.
                                                           Read exam.mp4 with `python -m mocap.detect ... --no-calib`: it
                                                           doesn't start from the reference pose, so there's nothing to align
docs/hands.md has every command in order.

The free hand opens and closes three times while its arm and wrist turn it to face every way, so it is seen from the
palm, the back and edge-on, sometimes behind the body. The joints are the true ones, so the score is the hand model's
and the retarget's alone, on a clean render at the real size of a hand in the frame: the best an AI clip can do.
"""
import json, math, pathlib, subprocess, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make(info_path, out):
    info = json.load(open(info_path))
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = [b["name"] for b in info["bones"]]
    f = info["fingers"]
    fnames = [b["name"] for b in f["bones"]]
    T, fps = 168, 24.0
    u = np.arange(T) / T
    s = lambda k, ph=0.0: np.sin(2 * math.pi * (k * u + ph))  # noqa: E731
    E = np.zeros((T, len(names), 3))
    # degrees, for a biped's bones as the knight and the woman name them; other names are skipped. X bends, Y twists,
    # Z swings sideways
    sets = {
        "upperarm_L": (-35 + 30 * s(1), 10 * s(2, 0.2), -15 + 20 * s(2, 0.4)),
        "forearm_L": (-55 + 40 * s(1.5, 0.1), 45 * s(1, 0.3), 0 * u),
        "hand_L": (30 * s(2.5), 22 * s(1.7, 0.2), 25 * s(2, 0.6)),
        "upperarm_R": (-15 + 15 * s(1, 0.5), 0 * u, 8 * s(1.5)),
        "forearm_R": (-25 + 20 * s(2, 0.2), 15 * s(1, 0.1), 0 * u),
        "spine2": (4 * s(1), 8 * s(1, 0.25), 0 * u),
    }
    for n, (x, y, z) in sets.items():
        if n in names:
            E[:, names.index(n)] = np.radians(np.stack([x, y, z], 1))
    yaw = np.radians(22 * s(1, 0.1))
    # the pelvis bone's own axes are not the world's: the body turns about world Z
    B = np.array(info["bones"][0]["matrix_local"])[:3, :3]
    axis = B.T @ np.array([0, 0, 1.0])
    root_quat = np.concatenate([np.cos(yaw / 2)[:, None], np.sin(yaw / 2)[:, None] * axis[None]], 1)
    F = np.zeros((T, len(fnames), 3))
    lag = {"index": 0.0, "middle": 0.03, "ring": 0.06, "pinky": 0.09, "thumb": 0.04}
    for i, b in enumerate(f["bones"]):
        hand = b["hand"]
        if hand in f["hold"]:
            F[:, i] = f["hold"][hand][b["name"]]
            continue
        amt = 0.5 - 0.5 * np.cos(2 * math.pi * (3 * u - lag[b["finger"]]))            # 0 open ... 1 fist
        fist = f["fist"][hand][b["name"]][0]
        F[:, i, 0] = fist * amt
        if b["k"] == 1 and b["finger"] != "thumb":
            F[:, i, 2] = math.radians(6) * s(2, 0.3) * {"index": 1, "middle": 0.3, "ring": -0.4, "pinky": -1}[b["finger"]] * (1 - amt)
    base = {"fps": fps, "bones": names, "trans": np.zeros((T, 3)).tolist(), "root_quat": root_quat.round(6).tolist(),
            "euler": E.round(5).tolist()}
    json.dump({**base, "fingers": {"bones": fnames, "euler": F.round(5).tolist()}}, open(out / "truth.json", "w"))
    json.dump(base, open(out / "body.json", "w"))
    print("wrote", out / "truth.json", "and body.json:", T, "frames")


def video(out):
    import cv2
    from mocap import layout

    out = pathlib.Path(out)
    fr, sd = sorted((out / "render/front").glob("f_*.png")), sorted((out / "render/side").glob("f_*.png"))
    (out / "frames").mkdir(exist_ok=True)
    for i, (a, b) in enumerate(zip(fr, sd)):
        f = cv2.cvtColor(cv2.imread(str(a), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        s = cv2.cvtColor(cv2.imread(str(b), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        cv2.imwrite(str(out / "frames" / f"f_{i:04d}.png"), cv2.cvtColor(layout.compose(f, s), cv2.COLOR_RGB2BGR))
    j = json.load(open(out / "render/joints.json"))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(j["fps"]), "-i", str(out / "frames/f_%04d.png"),
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(out / "exam.mp4")], check=True)
    J = np.array(j["joints"])
    fuv, suv = layout.project(J)
    np.savez(out / "det.npz", front_uv=fuv, side_uv=suv, front_conf=np.ones(fuv.shape[:2], np.float32),
             side_conf=np.ones(suv.shape[:2], np.float32), fps=float(j["fps"]), names=np.array(j["names"]),
             calib=np.array([[1, 0, 0], [0, 1, 0]], np.float32))
    print("wrote", out / "exam.mp4", "and det.npz:", len(fr), "frames")


def score(info_path, out):
    from mocap.hands import FINGERS, Hand

    info = json.load(open(info_path))
    out = pathlib.Path(out)
    truth, got = json.load(open(out / "truth.json")), json.load(open(out / "hands.json"))
    names = truth["fingers"]["bones"]
    A, B = np.array(truth["fingers"]["euler"]), np.array(got["fingers"]["euler"])
    T = min(len(A), len(B))
    res = {}
    for hand in info["fingers"]["hands"]:
        if hand in info["fingers"]["hold"]:
            continue
        H = Hand(info, hand)
        tips_t = np.array([H.tips(dict(zip(names, A[t]))) for t in range(T)])
        tips_g = np.array([H.tips(dict(zip(names, B[t]))) for t in range(T)])
        default = np.array([H.tips(info["fingers"]["relaxed"][hand])] * T)
        err = np.linalg.norm(tips_g - tips_t, axis=-1) * 100
        still = np.linalg.norm(default - tips_t, axis=-1) * 100
        fold_t = np.array([[A[t][names.index(H.bones[(f, k)]["name"])][0] for f in FINGERS for k in (1, 2, 3)] for t in range(T)])
        fold_g = np.array([[B[t][names.index(H.bones[(f, k)]["name"])][0] for f in FINGERS for k in (1, 2, 3)] for t in range(T)])
        # how open or closed the hand is, as one number: mean fold of the four fingers' first two joints
        idx = [i for i, (f, k) in enumerate((f, k) for f in FINGERS for k in (1, 2, 3)) if f != "thumb" and k < 3]
        res[hand] = {"fingertip_cm_mean": float(err.mean()), "fingertip_cm_median": float(np.median(err)),
                     "fingertip_cm_p95": float(np.percentile(err, 95)),
                     "per_finger_cm": {f: float(err[:, i].mean()) for i, f in enumerate(FINGERS)},
                     "untracked_hand_cm": float(still.mean()),
                     "fold_error_deg_mean": float(np.degrees(np.abs(fold_g - fold_t)).mean()),
                     "open_close_corr": float(np.corrcoef(fold_t[:, idx].mean(1), fold_g[:, idx].mean(1))[0, 1]),
                     "wrist_error_deg": (got.get("hands", {}).get("hands", {}).get(hand, {}) or {}).get("wrist_vs_solver_deg"),
                     "seen": (got.get("hands", {}).get("hands", {}).get(hand, {}) or {}).get("seen")}
        r = res[hand]
        print(f"{hand}: fingertips {r['fingertip_cm_mean']:.2f} cm from the truth on average (median {r['fingertip_cm_median']:.2f}, "
              f"95th percentile {r['fingertip_cm_p95']:.2f}); a hand left in its relaxed pose would be {r['untracked_hand_cm']:.2f} cm off. "
              f"Per finger: " + ", ".join(f"{f} {v:.2f}" for f, v in r["per_finger_cm"].items())
              + f". Folds off by {r['fold_error_deg_mean']:.1f} deg per joint; open/close follows the truth with correlation {r['open_close_corr']:.2f}"
              + (f"; the measured palm is {r['wrist_error_deg']:.1f} deg from the true hand" if r["wrist_error_deg"] is not None else ""))
    res["method"] = got.get("hands", {}).get("method")
    json.dump(res, open(out / "score.json", "w"), indent=1)


def world(info, motion):
    """Per hand: the hand bone's world turn (T, 3, 3), wrist (T, 3) and fingertips (T, 5, 3) of a motion."""
    import torch
    from mocap.hands import Hand
    from mocap.solve import Rig

    rig = Rig(info)
    q = np.array(motion["root_quat"])
    w, x, y, z = q.T
    Rr = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                   2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
    _, P = rig.forward(torch.tensor(motion["trans"], dtype=torch.float64), torch.tensor(Rr), torch.tensor(motion["euler"], dtype=torch.float64))
    P = P.numpy()
    f = info["fingers"]
    names = (motion.get("fingers") or {}).get("bones", [])
    default = {}
    for hand in f["hands"]:
        default.update(f["hold"].get(hand) or f["relaxed"][hand])
    out = {}
    for hand in f["hands"]:
        H = Hand(info, hand)
        i = rig.names.index(hand)
        rest = np.array(info["bones"][i]["matrix_local"])
        tips = []
        for t in range(len(P)):
            pose = dict(zip(names, motion["fingers"]["euler"][t])) if names else default
            tr = H.tips(pose)
            M = P[t, i] @ np.linalg.inv(rest)
            tips.append(tr @ M[:3, :3].T + M[:3, 3])
        out[hand] = {"R": P[:, i, :3, :3], "wrist": P[:, i, :3, 3], "tips": np.array(tips)}
    return out


def score_world(info_path, out, solved):
    """A whole capture (reader, solver, hands) against the truth, in the world: where the hands and fingertips ended up."""
    info = json.load(open(info_path))
    out = pathlib.Path(out)
    A, B = world(info, json.load(open(out / "truth.json"))), world(info, json.load(open(solved)))
    res = {}
    for hand in A:
        T = min(len(A[hand]["R"]), len(B[hand]["R"]))
        Rt, Rg = A[hand]["R"][:T], B[hand]["R"][:T]
        ang = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", Rt, Rg) - 1) / 2, -1, 1)))
        res[hand] = {"hand_turn_deg_mean": float(ang.mean()), "hand_turn_deg_median": float(np.median(ang)),
                     "wrist_cm": float(np.linalg.norm(A[hand]["wrist"][:T] - B[hand]["wrist"][:T], axis=-1).mean() * 100),
                     "fingertip_cm": float(np.linalg.norm(A[hand]["tips"][:T] - B[hand]["tips"][:T], axis=-1).mean() * 100)}
        r = res[hand]
        print(f"{hand}: hand turned {r['hand_turn_deg_mean']:.1f} deg from the truth on average (median {r['hand_turn_deg_median']:.1f}); "
              f"wrist {r['wrist_cm']:.2f} cm off, fingertips {r['fingertip_cm']:.2f} cm off in the world")
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"make": lambda: make(*sys.argv[2:4]), "video": lambda: video(sys.argv[2]), "score": lambda: score(*sys.argv[2:4]),
     "world": lambda: score_world(*sys.argv[2:5])}[cmd]()
