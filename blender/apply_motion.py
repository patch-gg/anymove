"""Put a solved (or made-up) motion on the rig: keyframes, true joint positions, renders and a GLB.

blender -b -P blender/apply_motion.py -- rigged.blend rig_info.json motion.json out_dir [options]
  --views front,side,persp   which cameras to render (default none)
  --engine cycles|workbench  (default cycles)
  --paint                    keep the leg paint (the exam needs it; the showcase does not)
  --blur                     motion blur from the keyframes, like a video model's fast limbs
  --glb                      export out_dir/animation.glb
  --frames 10,40,80          render only these frames (stills), for quick comparisons
  --springs fur.json         simulate the fur bones (blender/add_fur_bones.py) as damped springs
  --scale 0.75               render at this fraction of the layout size
  --ghost                    render the mesh as an x-ray ghost (workbench), for drawing a skeleton over
  --follow knuckle_L         with the view "follow": a close camera that travels with this joint (a hand, a paw),
                             --follow-size metres across (0.45), --follow-dir x,y,z from the joint to the camera,
                             --follow-res pixels (540). Several joints: --follow a,b renders follow_a/ and follow_b/
Always writes out_dir/joints.json, the world position of every joint per frame from Blender itself, and for every
rendered view out_dir/cam_<view>.json, its pixel projection (common.camera_json).
Fingers: motion.json "fingers" {"bones": [names], "euler": [T][bones][x, y, z]} (mocap/hands.py) is keyed as it is;
finger bones it does not name hold their default pose (closed around what the hand holds, loosely curled otherwise).
"""
import json, math, pathlib, sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
blend, info_path, motion_path, out = args[:4]
opts = args[4:]
opt = lambda k, d=None: opts[opts.index(k) + 1] if k in opts else d
out = pathlib.Path(out)
out.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=blend)
scene = bpy.context.scene
info = json.load(open(info_path))
motion = json.load(open(motion_path))
rig, body = bpy.data.objects["rig"], bpy.data.objects["body"]
names = motion["bones"]
T = len(motion["euler"])
fps = motion.get("fps", 24)
scene.render.fps = round(fps)
scene.render.fps_base = round(fps) / fps
scene.frame_start, scene.frame_end = 0, T - 1

if "--paint" not in opts:
    attr = body.data.color_attributes["legpaint"]
    rgba = np.zeros(len(attr.data) * 4)
    attr.data.foreach_set("color", rgba)

root = names[0]
B = rig.data.bones[root].matrix_local.to_3x3()
for pb in rig.pose.bones:
    pb.rotation_mode = "XYZ"
rig.pose.bones[root].rotation_mode = "QUATERNION"
for t in range(T):
    pr = rig.pose.bones[root]
    pr.rotation_quaternion = Quaternion(motion["root_quat"][t])
    pr.location = B.transposed() @ Vector(motion["trans"][t])
    pr.keyframe_insert("rotation_quaternion", frame=t)
    pr.keyframe_insert("location", frame=t)
    for i, n in enumerate(names[1:], 1):
        pb = rig.pose.bones[n]
        pb.rotation_euler = motion["euler"][t][i]
        pb.keyframe_insert("rotation_euler", frame=t)
if info.get("fingers"):
    common.set_fingers(rig, common.default_fingers(info))
    fm = motion.get("fingers")
    if fm:
        for t in range(T):
            for i, n in enumerate(fm["bones"]):
                pb = rig.pose.bones[n]
                pb.rotation_euler = fm["euler"][t][i]
                pb.keyframe_insert("rotation_euler", frame=t)
    for b in info["fingers"]["bones"]:          # untracked fingers: one key, so the pose is in the exported clip
        if not fm or b["name"] not in fm["bones"]:
            rig.pose.bones[b["name"]].keyframe_insert("rotation_euler", frame=0)
for fc in rig.animation_data.action.fcurves if hasattr(rig.animation_data.action, "fcurves") else []:
    for k in fc.keyframe_points:
        k.interpolation = "LINEAR"

if opt("--springs"):
    # Secondary motion for the fur bones: each tip is a damped spring chasing a target, so fur lags behind the
    # body, overshoots and settles. The target is the rigid (sculpted) direction pulled part of the way back toward
    # the bone's rest direction in the world ("gravity": fur keeps hanging down when the body rears or sits).
    # Sub-stepped, the tip kept at bone length and within max_deg of the target.
    springs = json.load(open(opt("--springs")))
    sub = 6
    dt = 1.0 / (fps * sub)
    state, swing = {}, {n: [] for n in springs}
    rest_world = {}
    for name in springs:
        b = rig.data.bones[name]
        rest_world[name] = (rig.matrix_world.to_3x3() @ (b.tail_local - b.head_local)).normalized()
    for t in range(T):
        scene.frame_set(t)
        for name, s in springs.items():
            pb = rig.pose.bones[name]
            par = pb.parent
            rigid = rig.matrix_world @ par.matrix @ (par.bone.matrix_local.inverted() @ pb.bone.matrix_local)
            L = pb.bone.length
            head = rigid.translation.copy()
            rdir = (rigid @ Vector((0, L, 0)) - head).normalized()
            gdir = rdir.slerp(rest_world[name], s.get("gravity", 0.0))
            target = head + gdir * L
            if name not in state:
                state[name] = {"x": target.copy(), "v": Vector((0, 0, 0)), "h": head, "g": target}
            st = state[name]
            w = 2 * math.pi * s["freq"]
            k, c = w * w, 2 * s["damping"] * w
            cos_max = math.cos(math.radians(s["max_deg"]))
            for i in range(1, sub + 1):
                a = i / sub
                h, g = st["h"].lerp(head, a), st["g"].lerp(target, a)
                st["v"] += (k * (g - st["x"]) - c * st["v"]) * dt
                st["x"] += st["v"] * dt
                d = (st["x"] - h).normalized()
                gd = (g - h).normalized()
                if d.dot(gd) < cos_max:  # past the swing limit: put it on the cone's edge
                    axis = gd.cross(d).normalized()
                    d = gd.copy()
                    d.rotate(Quaternion(axis, math.radians(s["max_deg"])))
                st["x"] = h + d * L
            st["h"], st["g"] = head, target
            final = (st["x"] - head).normalized()
            swing[name].append(math.degrees(math.acos(max(-1.0, min(1.0, final.dot(rdir))))))
            local = rigid.to_3x3().inverted() @ final
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Vector((0, 1, 0)).rotation_difference(local)
            pb.keyframe_insert("rotation_quaternion", frame=t)
    print("springs simulated (swing from the sculpted direction, mean / max deg):",
          {n: (round(sum(v) / len(v), 1), round(max(v), 1)) for n, v in swing.items()})

joints = []
for t in range(T):
    scene.frame_set(t)
    mw = rig.matrix_world
    joints.append([list(common.keypoint_world(rig, info, kp)) for kp in info["keypoints"]])
json.dump({"names": [k["name"] for k in info["keypoints"]], "joints": joints, "fps": fps}, open(out / "joints.json", "w"))
print("joints written", T, "frames")

views = [v for v in opt("--views", "").split(",") if v]
if views:
    engine = "ghost" if "--ghost" in opts else opt("--engine", "cycles")
    if engine == "ghost":
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "SINGLE"
        scene.display.shading.single_color = (0.55, 0.56, 0.6)
        scene.display.shading.show_xray = True
        scene.display.shading.xray_alpha = 0.18
        scene.render.film_transparent = True
    elif engine == "cycles":
        common.studio(scene, samples=int(opt("--samples", 32)))
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.denoising_use_gpu = True
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 4
    else:
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "TEXTURE"
        scene.render.film_transparent = True
    scene.render.use_motion_blur = "--blur" in opts
    scene.render.motion_blur_shutter = 0.5
    cams = common.cameras(scene)
    if "persp" in views:  # a three-quarter showcase camera (--cam x,y,z --target x,y,z --lens mm --res WxH)
        cam = bpy.data.objects.new("cam_persp", bpy.data.cameras.new("cam_persp"))
        scene.collection.objects.link(cam)
        cam.data.lens = float(opt("--lens", 50))
        vec = lambda k, d: Vector([float(x) for x in opt(k, d).split(",")])
        target = vec("--target", "0,0,0.5")
        cam.location = vec("--cam", "-3.2,-3.0,1.6")
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        cam.data.clip_end = 200
        pw, ph = (int(x) for x in opt("--res", "1280x720").split("x"))
        cams["cam_persp"] = (cam, (pw, ph))
    if "follow" in views:   # a close camera per followed joint, riding its (smoothed) path at a fixed angle
        views.remove("follow")
        J = np.array(joints)
        kn = [k["name"] for k in info["keypoints"]]
        fdir = Vector([float(x) for x in opt("--follow-dir", "-0.45,-0.85,0.3").split(",")]).normalized()
        fres = int(opt("--follow-res", 540))
        ker = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)
        ker /= ker.sum()
        for jn in opt("--follow", "").split(","):
            path = J[:, kn.index(jn)]
            path = np.stack([np.convolve(np.pad(path[:, i], 9, mode="edge"), ker, mode="valid") for i in range(3)], 1)
            cam = bpy.data.objects.new(f"cam_follow_{jn}", bpy.data.cameras.new(f"cam_follow_{jn}"))
            scene.collection.objects.link(cam)
            cam.data.type = "ORTHO"
            cam.data.ortho_scale = float(opt("--follow-size", 0.45))
            cam.data.clip_end = 100
            cam.rotation_euler = (-fdir).to_track_quat("-Z", "Y").to_euler()
            for t in range(T):
                cam.location = Vector(path[t].tolist()) + fdir * 4.0
                cam.keyframe_insert("location", frame=t)
            cams[f"cam_follow_{jn}"] = (cam, (fres, fres))
            views.append(f"follow_{jn}")
    floor = None
    if "persp" in views and engine == "cycles":   # a shadow catcher, so the creature stands on the ground in 3/4 shots
        bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, 0))
        floor = bpy.context.object
        floor.is_shadow_catcher = True
    for v in views:
        cam, w = cams[f"cam_{v}"]
        scene.camera = cam
        if floor is not None:
            floor.hide_render = v != "persp"
        k = float(opt("--scale", 1.0))
        w, hgt = w if isinstance(w, tuple) else (w, common.H)
        scene.render.resolution_x = round(w * k)
        scene.render.resolution_y = round(hgt * k)
        json.dump(common.camera_json(cam, scene.render.resolution_x, scene.render.resolution_y), open(out / f"cam_{v}.json", "w"))
        d = out / v
        d.mkdir(exist_ok=True)
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        if opt("--frames"):
            for f in [int(x) for x in opt("--frames").split(",")]:
                scene.frame_set(f)
                scene.render.filepath = str(d / f"f_{f:04d}.png")
                bpy.ops.render.render(write_still=True)
        else:
            scene.render.filepath = str(d / "f_")
            bpy.ops.render.render(animation=True)
        print("rendered", v, flush=True)

if "--glb" in opts:
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    body.select_set(True)
    for o in list(scene.objects):
        if o.type in ("CAMERA", "LIGHT"):
            o.select_set(False)
    bpy.ops.export_scene.gltf(filepath=str(out / "animation.glb"), use_selection=True, export_animations=True,
                              export_force_sampling=True, export_frame_range=True)
    print("glb written")
