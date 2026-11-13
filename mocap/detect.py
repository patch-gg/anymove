"""Read every joint in both halves of a two-camera video with a trained joint reader.

python -m mocap.detect VIDEO.mp4 MODEL.pt REF.png OUT.npz [overlay.mp4]
The clip is first aligned to the reference frame (the video model may resize or crop the canvas), then each frame is
split into its front and side views. Writes pixel positions in layout coordinates and peak confidences per view.
"""
import sys

import cv2
import numpy as np
import torch

from mocap import layout
from mocap.model import HM, SIZE, JointReader, decode, square


def read_frames(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[..., ::-1])
    return np.array(frames), fps


def calibrate(first, ref, target_w=None, target_h=None):
    """Affine map from video pixels to layout pixels, found by aligning the first frame to the reference image.

    Returns (A, converged). `converged` is False when ECC could not lock onto the
    reference and the plain stretch was used instead. That is not a cosmetic
    detail: the stretch puts the subject at the wrong scale in the layout, so the
    rig fit comes back with a much larger residual. The caller must carry the
    flag through to the result so an indicative fit is never mistaken for a
    contract fit.
    """
    h, w = first.shape[:2]
    tw = target_w or layout.W
    th = target_h or layout.H
    S = np.array([[tw / w, 0, 0], [0, th / h, 0]], np.float32)  # plain stretch to start
    a = cv2.cvtColor(cv2.resize(ref, (640, 360)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(cv2.warpAffine(first, S, (tw, th)), (640, 360)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    converged = True
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error:
        print("ECC did not converge; using a plain stretch")
        converged = False
    # warp maps reference (half-size) coords to frame coords; we want frame -> layout
    W2 = np.vstack([warp, [0, 0, 1]])
    W2[:2, 2] *= 2  # half-size -> full-size translation
    A = np.linalg.inv(W2) @ np.vstack([S, [0, 0, 1]])
    return A[:2].astype(np.float32), converged


def prep(view):
    _, ox, oy, side = square(view, view[2, 2])
    k = SIZE / side
    A = np.array([[k, 0, ox * k], [0, k, oy * k]], np.float32)
    fill = tuple(float(x) for x in view[2, 2])
    out = cv2.warpAffine(view, A, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    return out, (ox, oy, side)


@torch.no_grad()
def run(model, views, bs=32, device="cuda"):
    uv_all, conf_all = [], []
    for i in range(0, len(views), bs):
        batch = [prep(v) for v in views[i:i + bs]]
        x = torch.from_numpy(np.stack([b[0] for b in batch])).permute(0, 3, 1, 2).float().to(device) / 255
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hm = model(x).float()
        else:
            hm = model(x).float()
        uv, conf = decode(hm)
        uv = uv.cpu().numpy() * SIZE / HM
        for (_, (ox, oy, side)), u in zip(batch, uv):
            uv_all.append(u * side / SIZE - np.array([ox, oy]))
        conf_all.append(conf.cpu().numpy())
    return np.array(uv_all), np.concatenate(conf_all)


def load(model_path, device="cuda"):
    ck = torch.load(model_path, map_location="cpu", weights_only=True)  # tensors, names and numbers only: safe for downloaded readers
    model = JointReader(len(ck["names"])).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    return model, ck["names"]


def overlay(frames, fuv, suv, fconf, names, path, fps, single=False):
    h, w = (layout.H, layout.W_SIDE) if single else (layout.H, layout.W)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    col = lambda n: (209, 60, 245) if n.endswith("_L") else (255, 107, 47) if n.endswith("_R") else (255, 255, 255)  # BGR
    for t, f in enumerate(frames):
        img = np.ascontiguousarray(f[..., ::-1])
        for j, n in enumerate(names):
            if single:
                for (u, v), x0 in ((suv[t, j], 0),):
                    cv2.circle(img, (int(u + x0), int(v)), 4, col(n), -1, cv2.LINE_AA)
            else:
                for (u, v), x0 in ((fuv[t, j], 0), (suv[t, j], layout.W_FRONT)):
                    cv2.circle(img, (int(u + x0), int(v)), 4, col(n), -1, cv2.LINE_AA)
        vw.write(img)
    vw.release()


def main():
    # A flag's VALUE must not be mistaken for a positional argument. `--device
    # cpu` used to leave "cpu" in the positional list, which silently pushed the
    # overlay argument out of position (and wrote nothing, because a path with
    # no extension is not a container cv2 can open).
    argv = list(sys.argv[1:])
    device = "cuda"
    if "--device" in argv:
        i = argv.index("--device")
        if i + 1 < len(argv):
            device = argv[i + 1]
            argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv if not a.startswith("--")]
    video, model_path, ref_path, out = args[:4]
    single = "--single-view" in argv
    # The reader needs ~2.4 GB of VRAM. On a box where another ComfyUI lane
    # holds the card, that is the difference between a read and a wedge at the
    # device->host copy, so the reader can be asked to stay off the GPU.
    if device not in ("cuda", "cpu"):
        raise SystemExit(f"--device must be cuda or cpu, not {device!r}")
    print(f"device: {device}")
    frames, fps = read_frames(video)
    ref = cv2.imread(ref_path)[..., ::-1]
    converged = True
    if single:
        # ONE camera, not two. The reference is a single view (the side panel of
        # the layout, 860x720), so the whole frame IS that view and the
        # calibration maps video pixels straight into that view's coordinates.
        tw, th = layout.W_SIDE, layout.H
        if "--no-calib" in argv:
            A = np.array([[tw / frames.shape[2], 0, 0], [0, th / frames.shape[1], 0]], np.float32)
        else:
            A, converged = calibrate(frames[0], ref, tw, th)
    elif "--no-calib" in sys.argv:  # already a layout frame (the exam)
        A = np.array([[layout.W / frames.shape[2], 0, 0], [0, layout.H / frames.shape[1], 0]], np.float32)
    else:
        A, converged = calibrate(frames[0], ref)
    print("calibration (video px -> layout px):", A.round(4).tolist())
    if not converged:
        print("WARNING: the calibration fell back to a plain stretch - the clip is "
              "not framed like the reference, so the rig fit will carry a larger "
              "residual than a two-camera read. The result is indicative, not "
              "contract-grade.")
    frames = np.array([cv2.warpAffine(f, A, (tw if single else layout.W, th if single else layout.H),
                                       flags=cv2.INTER_AREA) for f in frames])
    model, names = load(model_path, device)
    if single:
        suv, sconf = run(model, [f for f in frames], device=device)
        # No front view exists: the front observation is EMPTY, not zero-and-
        # confident. mocap.solve --single-view reads the flag and fits the rig
        # in 2.5D (Y and Z from the side view, X held at the rest pose).
        fuv = np.zeros_like(suv)
        fconf = np.zeros_like(sconf)
        np.savez(out, front_uv=fuv, side_uv=suv, front_conf=fconf, side_conf=sconf,
                 fps=fps, names=np.array(names), calib=A, single_view=True,
                 calib_converged=converged)
        print(f"{len(frames)} frames at {fps:.2f} fps; SINGLE VIEW; "
              f"mean confidence side {sconf.mean():.2f}")
        if len(args) > 4:
            overlay(frames, fuv, suv, fconf, names, args[4], fps, single=True)
        return
    fuv, fconf = run(model, [f[:, :layout.W_FRONT] for f in frames], device=device)
    suv, sconf = run(model, [f[:, layout.W_FRONT:] for f in frames], device=device)
    np.savez(out, front_uv=fuv, side_uv=suv, front_conf=fconf, side_conf=sconf, fps=fps, names=np.array(names), calib=A, calib_converged=converged)
    print(f"{len(frames)} frames at {fps:.2f} fps; mean confidence front {fconf.mean():.2f} side {sconf.mean():.2f}")
    if len(args) > 4:
        overlay(frames, fuv, suv, fconf, names, args[4], fps)


if __name__ == "__main__":
    main()
