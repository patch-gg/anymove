"""One GLB per creature: the textured, rigged mesh and every clip as its own named animation.

blender -b -P blender/pack_glb.py -- rigged.blend rig_info.json out.glb Name=motion.json [Name=motion.json ...]
The limb paint (a colour attribute mixed over the texture for the video model) is taken out first, so the GLB
carries the creature's own texture. A rig with fingers gets them in every clip: as mocap/hands.py tracked them
(motion.json "fingers"), or in their default pose (closed around what the hand holds, loosely curled otherwise).
"""
import json, pathlib, sys

import bpy
from mathutils import Quaternion, Vector

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
blend, info_path, out = args[:3]
clips = [a.split("=", 1) for a in args[3:]]
bpy.ops.wm.open_mainfile(filepath=blend)
scene = bpy.context.scene
info = json.load(open(info_path))
rig, body = bpy.data.objects["rig"], bpy.data.objects["body"]

# undo common.paint_legs: feed the texture straight into the shader again and drop the paint attribute
for slot in body.material_slots:
    nt = slot.material.node_tree
    for mix in [n for n in nt.nodes if n.type == "MIX" and any(l.from_node.type == "VERTEX_COLOR" for l in n.inputs["Factor"].links)]:
        bsdf = next(nd for nd in nt.nodes if nd.type == "BSDF_PRINCIPLED")
        src = mix.inputs["A"].links[0].from_socket if mix.inputs["A"].links else None
        for n in [l.from_node for l in mix.inputs["Factor"].links]:
            nt.nodes.remove(n)
        nt.nodes.remove(mix)
        if src is not None:
            nt.links.new(src, bsdf.inputs["Base Color"])
if "legpaint" in body.data.color_attributes:
    body.data.color_attributes.remove(body.data.color_attributes["legpaint"])

for pb in rig.pose.bones:
    pb.rotation_mode = "XYZ"
root = info["bones"][0]["name"]
rig.pose.bones[root].rotation_mode = "QUATERNION"
B = rig.data.bones[root].matrix_local.to_3x3()
finger_default = common.default_fingers(info)
rig.animation_data_create()
fps = None
for name, path in clips:
    m = json.load(open(path))
    fps = fps or m.get("fps", 24)
    names = m["bones"]
    act = bpy.data.actions.new(name)
    rig.animation_data.action = act
    T = len(m["euler"])
    fm = m.get("fingers") or {"bones": [], "euler": []}
    for n, e in finger_default.items():  # fingers nothing tracked: one key, so the clip holds their pose
        if n not in fm["bones"]:
            rig.pose.bones[n].rotation_euler = e
            rig.pose.bones[n].keyframe_insert("rotation_euler", frame=0)
    for t in range(T):
        for i, n in enumerate(fm["bones"]):
            pb = rig.pose.bones[n]
            pb.rotation_euler = fm["euler"][t][i]
            pb.keyframe_insert("rotation_euler", frame=t)
        pr = rig.pose.bones[root]
        pr.rotation_quaternion = Quaternion(m["root_quat"][t])
        pr.location = B.transposed() @ Vector(m["trans"][t])
        pr.keyframe_insert("rotation_quaternion", frame=t)
        pr.keyframe_insert("location", frame=t)
        for i, n in enumerate(names[1:], 1):
            pb = rig.pose.bones[n]
            pb.rotation_euler = m["euler"][t][i]
            pb.keyframe_insert("rotation_euler", frame=t)
    act.use_frame_range = True
    act.frame_start, act.frame_end = 0, T - 1
    act.use_fake_user = True
    track = rig.animation_data.nla_tracks.new()
    track.name = name
    strip = track.strips.new(name, 0, act)
    track.mute = True
    rig.animation_data.action = None
    print("clip", name, T, "frames", flush=True)
for pb in rig.pose.bones:  # rest pose for the bind
    pb.rotation_quaternion = (1, 0, 0, 0)
    pb.rotation_euler = (0, 0, 0)
    pb.location = (0, 0, 0)
scene.render.fps = round(fps or 24)
bpy.ops.object.select_all(action="DESELECT")
rig.select_set(True)
body.select_set(True)
bpy.ops.export_scene.gltf(filepath=str(pathlib.Path(out).resolve()), use_selection=True, export_animations=True,
                          export_animation_mode="ACTIONS", export_force_sampling=True, export_frame_range=False,
                          export_nla_strips=True, export_anim_single_armature=True, export_reset_pose_bones=True,
                          export_image_format="JPEG", export_jpeg_quality=90)
print("glb written", out)
