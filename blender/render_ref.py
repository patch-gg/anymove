"""Render the rest pose from both layout cameras (RGBA), for the video model's first frame.

blender -b -P blender/render_ref.py -- rigged.blend out_dir [samples]
"""
import pathlib, sys

import bpy

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
out = pathlib.Path(args[1])
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=args[0])
for pb in bpy.data.objects["rig"].pose.bones:
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (0, 0, 0)
    pb.location = (0, 0, 0)
# creature.json "start_pose" (degrees per bone): the clip's first frame, e.g. a biped's arms lowered from its A-pose
start = common._c.get("start_pose", {}) if hasattr(common, "_c") else {}
if start and "rest" not in args[2:]:
    import math
    from mathutils import Euler
    for name, deg in start.items():
        pb = bpy.data.objects["rig"].pose.bones[name]
        pb.rotation_euler = Euler([math.radians(v) for v in deg], "XYZ")
common.studio(bpy.context.scene, samples=int(args[2]) if len(args) > 2 and args[2].isdigit() else 128)
common.render_views(out, "ref")
