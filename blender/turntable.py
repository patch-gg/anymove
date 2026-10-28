"""A slow orbit around the creature at rest, for the explainer: textured, painted, or x-ray with its bones.

blender -b -P blender/turntable.py -- rigged.blend rig.json out_dir mode [frames] [WxH] [deg_from,deg_to]
mode: plain (texture, no paint), paint (limb colours), bones (a ghost of the mesh, then the skeleton as a second
pass in out_dir/sticks; tools/stack_passes.py lays the solid bones over the ghost)
The camera circles the creature's bounding box at a three-quarter height, framing it whole in every frame.
"""
import json, math, pathlib, sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
blend, rig_json, out, mode = args[:4]
frames = int(args[4]) if len(args) > 4 else 96
w, h = (int(x) for x in (args[5] if len(args) > 5 else "1408x792").split("x"))
a0, a1 = (float(x) for x in (args[6] if len(args) > 6 else "-60,-20").split(","))
out = pathlib.Path(out)
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=blend)
scene = bpy.context.scene
rig, body = bpy.data.objects["rig"], bpy.data.objects["body"]
for pb in rig.pose.bones:
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (0, 0, 0)
if mode != "paint":
    attr = body.data.color_attributes["legpaint"]
    attr.data.foreach_set("color", np.zeros(len(attr.data) * 4))
co = np.array([body.matrix_world @ v.co for v in body.data.vertices])
lo, hi = co.min(0), co.max(0)
centre = Vector(((lo + hi) / 2).tolist())
radius = float(np.linalg.norm(hi - lo)) / 2

if mode == "bones":
    joints, bones, _ = common.load_rig(rig_json)
    before = set(bpy.data.objects)
    common.stick_figure(joints, bones, radius=0.022 * radius / 1.4)
    sticks = list(set(bpy.data.objects) - before)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_xray = True
    scene.display.shading.xray_alpha = 0.18
    scene.render.film_transparent = True
    for m in bpy.data.materials:
        if m.name.startswith("stick_"):
            m.diffuse_color = (1, 1, 1, 1)   # one ink for the bones: the patch is the only pink in a view
    body.active_material.diffuse_color = (0.55, 0.56, 0.6, 1)
else:
    common.studio(scene, samples=24)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.denoising_use_gpu = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4

cam = bpy.data.objects.new("orbit", bpy.data.cameras.new("orbit"))
scene.collection.objects.link(cam)
cam.data.lens = 50
scene.camera = cam
scene.render.resolution_x, scene.render.resolution_y = w, h
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
fov = 2 * math.atan(36 / 2 / cam.data.lens)          # horizontal, 36 mm sensor
dist = radius / math.tan(min(fov, fov * h / w) / 2) * float(__import__("os").environ.get("TURNTABLE_FRAME", 0.64))  # 0.64: a close-up
passes = [(out, None)]
if mode == "bones":   # the ghost (mesh only, x-ray), then the bones alone and solid
    passes = [(out, False), (out / "sticks", True)]
for d, show_sticks in passes:
    d.mkdir(parents=True, exist_ok=True)
    if show_sticks is not None:
        for o in sticks:
            o.hide_render = not show_sticks
        body.hide_render = show_sticks
        scene.display.shading.show_xray = not show_sticks
    for f in range(frames):
        a = math.radians(a0 + (a1 - a0) * f / max(1, frames - 1))
        cam.location = centre + Vector((math.sin(a) * math.cos(0.32), -math.cos(a) * math.cos(0.32), math.sin(0.32))) * dist
        cam.rotation_euler = (centre - cam.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(d / f"f_{f:04d}.png")
        bpy.ops.render.render(write_still=True)
print("turntable", mode, frames, "frames")
