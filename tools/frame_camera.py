"""A 3/4 camera that frames a whole take, or several, for blender/apply_motion.py: front-left, a little above.

python tools/frame_camera.py [--creature creature.json] MOTION.json [MOTION.json ...]   prints "cam target lens"
python tools/frame_camera.py --write creature.json MOTION.json [MOTION.json ...]        saves it as creature.json "camera"
With --creature, the creature's own "camera" wins when it has one. Otherwise every joint of every frame, padded for
the mesh, is projected through the camera, and the distance is set so they fill about 80% of a 16:9 frame.
"""
import json, re, sys

import numpy as np


def frame(paths):
    J = np.concatenate([np.array(json.load(open(p))["joints"]).reshape(-1, 3) for p in paths])
    height = max(J[:, 2].max(), 0.3)
    pad = 0.1 * height
    lo, hi = J.min(0) - pad, J.max(0) + pad
    lo[2] = 0
    # corners of the padded box, plus the joints themselves
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pts = np.concatenate([corners, J])
    tgt = (lo + hi) / 2
    d = np.array([-0.78, -0.58, 0.24])  # front-left, slightly above
    d /= np.linalg.norm(d)
    lens, sw, sh = 40.0, 36.0, 36.0 * 9 / 16
    right = np.cross(-d, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, -d)

    def fits(dist):
        rel = pts - (tgt + d * dist)
        z = rel @ -d
        if (z <= 0.1).any():
            return False
        u, v = rel @ right / z * lens, rel @ up / z * lens
        return np.abs(u).max() < 0.4 * sw and np.abs(v).max() < 0.4 * sh

    dist = np.linalg.norm(hi - lo)
    while not fits(dist):
        dist *= 1.08
    for _ in range(30):  # come back in as far as it still fits
        if fits(dist / 1.03):
            dist /= 1.03
        else:
            break
    fmt = lambda v: ",".join(f"{x:.3f}" for x in v)
    return {"cam": fmt(tgt + d * dist), "target": fmt(tgt), "lens": int(lens)}


def main():
    a = sys.argv[1:]
    creature = write = None
    if "--creature" in a:
        i = a.index("--creature")
        creature = a[i + 1]
        del a[i:i + 2]
    if "--write" in a:
        i = a.index("--write")
        write = a[i + 1]
        del a[i:i + 2]
    if creature:
        c = json.load(open(creature)).get("camera")
        if c:
            print(c["cam"], c["target"], c.get("lens", 50))
            return
    cam = frame(a)
    if write:
        text = open(write, newline="").read()
        line = json.dumps(cam)
        if '"camera"' in text:  # keep the file's own layout: only the camera line changes
            text = re.sub(r'"camera":\s*\{[^}]*\}', '"camera": ' + line, text)
        else:
            text = re.sub(r"\s*\}\s*$", ',\n  "camera": ' + line + "\n}\n", text)
        json.loads(text)
        open(write, "w", newline="\n").write(text)
    print(cam["cam"], cam["target"], cam["lens"])


if __name__ == "__main__":
    main()
