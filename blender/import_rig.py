"""Bring a Rodin-rigged GLB into anymove: scale/facing, limb paint, rig_info.json and rigged.blend.

blender -b -P blender/import_rig.py -- rigged.glb rig.json creature.json out_dir
"""
import json, math, pathlib, sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common

src, rig_file, creature_file, out = sys.argv[sys.argv.index("--") + 1:]
out = pathlib.Path(out)
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src, bone_heuristic="BLENDER", guess_original_bind_pose=False)
arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
if len(arms) != 1:
    raise RuntimeError(f"expected one armature, got {len(arms)}")
arm = arms[0]
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for obj in meshes:
    obj.select_set(True)
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
body = bpy.context.view_layer.objects.active
body.name, arm.name = "body", "rig"

# Apply the same world-space transform to skin and skeleton, as prep_mesh.py does for a static mesh.
prep = json.loads(pathlib.Path(creature_file).read_text()).get("prep", {})
turn = Matrix.Rotation(math.radians(prep.get("rotz_deg", 0)), 4, "Z")
vertices = np.array([(turn @ body.matrix_world @ v.co)[:] for v in body.data.vertices])
lo, hi = vertices.min(0), vertices.max(0)
axis = "xyz".index(prep.get("axis", "y"))
scale = prep.get("length_m", 1.0) / (hi[axis] - lo[axis])
move = Matrix.Translation(Vector((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2])))
transform = Matrix.Scale(scale, 4) @ move @ turn
arm_world, body_world = transform @ arm.matrix_world, transform @ body.matrix_world
arm.matrix_world, body.matrix_world = arm_world, body_world
bpy.ops.object.select_all(action="DESELECT")
arm.select_set(True)
body.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

joints, bones, limits = common.load_rig(rig_file)
missing = [name for name, *_ in bones if name not in arm.data.bones]
if missing:
    raise RuntimeError(f"Rodin rig is missing bones: {missing}")
for joint in joints:
    source = next(((name, "head_local") for name, head, tail, parent in bones if head == joint), None)
    source = source or next((name, "tail_local") for name, head, tail, parent in bones if tail == joint)
    joints[joint] = list(getattr(arm.data.bones[source[0]], source[1]))
for name, head, tail, parent in bones:
    actual = arm.data.bones[name].parent
    if (actual.name if actual else None) != parent:
        raise RuntimeError(f"Rodin changed the parent of {name}")
# Untextured uploads still need a shader for the limb colours in the reference frame.
if not body.data.materials or any(slot.material is None for slot in body.material_slots):
    material = bpy.data.materials.new("body")
    material.use_nodes = True
    if not body.data.materials:
        body.data.materials.append(material)
    for slot in body.material_slots:
        if slot.material is None:
            slot.material = material
common.paint_legs(body)
common.write_rig_info(arm, joints, bones, limits, out / "rig_info.json")
bpy.ops.wm.save_as_mainfile(filepath=str(out / "rigged.blend"))
print("saved", out / "rigged.blend", "bones", len(bones))
