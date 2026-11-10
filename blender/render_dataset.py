"""Render the creature in random poses from both layout cameras, with the true pixel position of every joint.

blender -b -P blender/render_dataset.py -- rigged.blend rig_info.json out_dir start count seed [scale]
Each sample i writes out_dir/{i:05d}_front.png, _side.png (RGBA) and _meta.json (joints in metres and in
full-size layout pixels for both views, plus the pose). About half the samples carry motion blur from a random
velocity, the way a video model smears fast limbs. Lights, turn, position and airtime are random too.
A rig with fingers gets a random hand in every sample, from open to a fist, so the reader learns that a knuckle
stays where it is whatever the fingers do; a hand that holds something keeps its grip.
"""
import json, math, pathlib, sys

import bpy
import numpy as np
from mathutils import Euler, Vector

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
blend, info_path, out = args[:3]
start, count, seed = int(args[3]), int(args[4]), int(args[5])
scale = float(args[6]) if len(args) > 6 else 0.6
out = pathlib.Path(out)
out.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=blend)
scene = bpy.context.scene
info = json.load(open(info_path))
rig, body = bpy.data.objects["rig"], bpy.data.objects["body"]
bones = [b["name"] for b in info["bones"]]
lim = {b["name"]: np.radians(np.array(b["limits_deg"], float)) for b in info["bones"]}
for pb in rig.pose.bones:
    pb.rotation_mode = "XYZ"
common.studio(scene, samples=16)
scene.cycles.use_adaptive_sampling = True
scene.cycles.denoising_use_gpu = True  # OIDN on the GPU; on the CPU it was the bottleneck
scene.render.threads_mode = "FIXED"
scene.render.threads = 4
cams = common.cameras(scene)
key, fill, rim = (bpy.data.objects[n] for n in ("key", "fill", "rim"))
bg = scene.world.node_tree.nodes["Background"]
M, H, ZC = common.M_PER_PX, common.H, common.Z_CENTER


def sample_pose(rng):
    """Euler angles per bone: half uniform inside the limits, half a narrower spread around the rest pose."""
    pose = {}
    wide = rng.random() < 0.5
    for name in bones:
        lo, hi = lim[name][:, 0], lim[name][:, 1]
        if name == "pelvis" and rng.random() < 0.15:  # sometimes rear up or dive
            lo = lo.copy(); lo[0] = math.radians(-60)
        if wide:
            a = rng.uniform(lo, hi)
        else:
            a = np.clip(rng.normal(0, 0.3 * (hi - lo) / 2), lo, hi)
        pose[name] = a
    return pose


FING = info.get("fingers")
FINGER_OF = {b["name"]: (b["finger"], b["k"]) for b in FING["bones"]} if FING else {}


def sample_fingers(rng):
    """{finger bone: Euler}: a held grip (a little loose or tight), or anything from an open hand to a fist."""
    pose = {}
    if not FING:
        return pose
    for hand in FING["hands"]:
        hold, relaxed, fist = FING["hold"].get(hand), FING["relaxed"][hand], FING["fist"][hand]
        if hold:
            k = rng.uniform(0.9, 1.06)
            pose.update({b: (e[0] * k, 0.0, 0.0) for b, e in hold.items()})
            continue
        r = rng.random()
        u = 0.0 if r < 0.2 else 1.0 if r < 0.4 else rng.uniform(-0.25, 1.1)    # relaxed, fist, or anywhere between
        spread = rng.normal(0, math.radians(5))
        per = {}
        for b in relaxed:
            f, k = FINGER_OF[b]
            per.setdefault(f, float(np.clip(u + rng.normal(0, 0.12), -0.3, 1.1)))
            x = relaxed[b][0] + (fist[b][0] - relaxed[b][0]) * per[f]
            z = spread * {"index": 1.0, "middle": 0.3, "ring": -0.4, "pinky": -1.0, "thumb": 0.0}[f] if k == 1 else 0.0
            pose[b] = (max(x, math.radians(-8)), 0.0, z)
    return pose


def apply(pose, loc, yaw, frame):
    for name, a in pose.items():
        pb = rig.pose.bones[name]
        pb.rotation_euler = Euler(a.tolist(), "XYZ")
        if frame is not None:
            pb.keyframe_insert("rotation_euler", frame=frame)
    rig.location = loc
    rig.rotation_euler = (0, 0, yaw)
    if frame is not None:
        rig.keyframe_insert("location", frame=frame)
        rig.keyframe_insert("rotation_euler", frame=frame)


def lowest_z():
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    co = co.reshape(-1, 3) @ np.array(body.matrix_world)[:3, :3].T + np.array(body.matrix_world)[:3, 3]
    return co[:, 2].min()


def joints_world():
    return np.array([list(common.keypoint_world(rig, info, kp)) for kp in info["keypoints"]])


rng = np.random.default_rng(seed + start)
for i in range(start, start + count):
    stem = out / f"{i:05d}"
    if (out / f"{i:05d}_meta.json").exists():
        continue
    rig.animation_data_clear()
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)
    for pb in rig.pose.bones:
        pb.location = (0, 0, 0)
    pose = sample_pose(rng)
    fingers = sample_fingers(rng)
    common.set_fingers(rig, fingers)
    r = rng.random()
    yaw = math.radians(rng.uniform(-30, 30) if r < 0.75 else rng.uniform(-70, 70) if r < 0.85 else 0)
    loc = Vector((rng.uniform(-0.08, 0.08), rng.uniform(-0.25, 0.25), 0))
    apply(pose, loc, yaw, None)
    bpy.context.view_layer.update()
    loc.z = -lowest_z() + (rng.uniform(0.05, 0.45) if rng.random() < 0.2 else 0)
    blur = rng.random() < 0.5
    if blur:  # key the pose at frames 0, 1, 2 along a random velocity; render frame 1 with motion blur
        vel = {n: rng.normal(0, math.radians(6 if any(k in n for k in ("arm", "meta", "shin", "thigh", "paw", "tail")) else 2.5), 3) for n in bones}
        vloc = Vector(rng.normal(0, 0.015, 3).tolist())
        vyaw = rng.normal(0, math.radians(3))
        for f, s in ((0, -1), (1, 0), (2, 1)):
            apply({n: pose[n] + s * vel[n] for n in bones}, loc + s * vloc, yaw + s * vyaw, f)
        scene.frame_set(1)
        scene.render.use_motion_blur = True
        scene.render.motion_blur_shutter = rng.uniform(0.5, 1.0)
    else:
        apply(pose, loc, yaw, None)
        scene.render.use_motion_blur = False
        bpy.context.view_layer.update()
    # lights and world
    key.rotation_euler = (math.radians(rng.uniform(20, 70)), 0, rng.uniform(-math.pi, math.pi))
    key.data.energy = rng.uniform(1.5, 4.5)
    fill.rotation_euler = (math.radians(rng.uniform(30, 80)), 0, rng.uniform(-math.pi, math.pi))
    fill.data.energy = rng.uniform(0.2, 1.5)
    rim.data.energy = rng.uniform(0.0, 2.5)
    bg.inputs["Strength"].default_value = rng.uniform(0.3, 0.9)
    g = rng.uniform(0.4, 0.6)
    bg.inputs["Color"].default_value = (g * rng.uniform(0.9, 1.1), g, g * rng.uniform(0.9, 1.1), 1)

    jw = joints_world()
    front = np.stack([common.W_FRONT / 2 + jw[:, 0] / M, H / 2 - (jw[:, 2] - ZC) / M], 1)
    side = np.stack([common.W_SIDE / 2 - jw[:, 1] / M, H / 2 - (jw[:, 2] - ZC) / M], 1)
    for name, (cam, w) in cams.items():
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = round(w * scale), round(H * scale)
        scene.render.filepath = f"{stem}_{name.split('_')[1]}.png"
        bpy.ops.render.render(write_still=True)
    meta = {"joints_world": jw.round(5).tolist(), "front_uv": front.round(3).tolist(), "side_uv": side.round(3).tolist(),
            "scale": scale, "blur": blur, "yaw": yaw, "root": list(loc),
            "pose": {n: np.round(a, 5).tolist() for n, a in pose.items()},
            "fingers": {n: [round(float(x), 4) for x in e] for n, e in fingers.items()}}
    (out / f"{i:05d}_meta.json").write_text(json.dumps(meta))
    print("sample", i, flush=True)
