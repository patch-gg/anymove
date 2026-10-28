"""Build the creature's armature from rig.json, skin it with Blender's automatic weights, paint the legs and save.

blender -b -P blender/build_rig.py -- static.blend rig.json out_dir [preview]
"preview" only renders the joints over an x-ray of the mesh (side, front, top) so they can be checked.
Otherwise writes rigged.blend, rig_info.json (rest matrices, parents, keypoints for the solver) and test renders.
"""
import json, math, pathlib, sys

import bpy
import numpy as np
from mathutils import Euler, Vector

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import common  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
static_blend, rig_json, out = args[:3]
preview = len(args) > 3 and args[3] == "preview"
out = pathlib.Path(out)
out.mkdir(parents=True, exist_ok=True)

bpy.ops.wm.open_mainfile(filepath=static_blend)
body = bpy.data.objects["body"]
joints, bones, limits = common.load_rig(rig_json)

if preview:
    common.stick_figure(joints, bones)
    common.xray_views(out, "joints")
    raise SystemExit

# Tripo's GLB splits vertices along every texture seam (185 pieces). Merge them back into one surface first, or the
# automatic weights stop at each seam and neighbouring patches pull apart when a joint bends. UVs live on face
# corners, so the texture is unaffected.
bpy.context.view_layer.objects.active = body
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
n0 = len(body.data.vertices)
bpy.ops.mesh.remove_doubles(threshold=1e-5)
bpy.ops.object.mode_set(mode="OBJECT")
print("merged seam vertices:", n0, "->", len(body.data.vertices))

arm_data = bpy.data.armatures.new("rig")
arm = bpy.data.objects.new("rig", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
eb = {}
for name, h, t, parent in bones:
    b = arm_data.edit_bones.new(name)
    b.head, b.tail = Vector(joints[h]), Vector(joints[t])
    d = (b.tail - b.head).normalized()
    z = Vector((1, 0, 0)).cross(d)
    if z.length < 0.5:  # runs mostly along X: local X horizontal and across the bone
        xl = Vector((0, 0, 1)).cross(d)
        z = xl.normalized().cross(d) if xl.length > 1e-4 else Vector((0, 0, 1))
    b.align_roll(z.normalized())  # local X = world X for bones in the body's plane: flexion is about X
    eb[name] = b
for name, h, t, parent in bones:
    if parent:
        eb[name].parent = eb[parent]
        eb[name].use_connect = False
bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_AUTO")

# vertices the heat solve missed go fully to the nearest bone segment
names = [b[0] for b in bones]
segs = np.array([[joints[h], joints[t]] for _, h, t, _ in bones])
groups = {g.index: g.name for g in body.vertex_groups}
missing = 0
for v in body.data.vertices:
    if sum(g.weight for g in v.groups) < 1e-3:
        p = np.array(v.co[:])
        a, b = segs[:, 0], segs[:, 1]
        t = np.clip(((p - a) * (b - a)).sum(1) / ((b - a) ** 2).sum(1), 0, 1)
        d = np.linalg.norm(a + (b - a) * t[:, None] - p, axis=1)
        body.vertex_groups[names[int(d.argmin())]].add([v.index], 1.0, "REPLACE")
        missing += 1
print("vertices without weights fixed:", missing)

# soften the weight borders (a few Laplacian passes over mesh edges), then keep at most 4 bones per vertex (what
# game engines skin with) and normalise
nv, ng = len(body.data.vertices), len(body.vertex_groups)
W = np.zeros((nv, ng))
for v in body.data.vertices:
    for g in v.groups:
        W[v.index, g.group] = g.weight
E = np.array([e.vertices[:] for e in body.data.edges])
deg = np.bincount(E.ravel(), minlength=nv)[:, None].clip(1)
for _ in range(3):
    acc = np.zeros_like(W)
    np.add.at(acc, E[:, 0], W[E[:, 1]])
    np.add.at(acc, E[:, 1], W[E[:, 0]])
    W = 0.5 * W + 0.5 * acc / deg
if hasattr(common, "_c") and common._c.get("rigid"):
    # rigid parts (creature.json "rigid": true, e.g. a mech): every separate piece of the mesh follows the one bone
    # that holds most of its weight, and the rest of the surface follows its strongest bone, with no blending
    lab = np.arange(nv)
    def root(a):
        while lab[a] != a:
            lab[a] = lab[lab[a]]
            a = lab[a]
        return a
    for a, b in E:
        ra, rb = root(a), root(b)
        if ra != rb:
            lab[ra] = rb
    comp = np.array([root(i) for i in range(nv)])
    hard = W.argmax(1)
    for c in np.unique(comp):
        idx = np.nonzero(comp == c)[0]
        if len(idx) < 0.15 * nv:
            hard[idx] = W[idx].sum(0).argmax()
    W = np.zeros_like(W)
    W[np.arange(nv), hard] = 1.0
    print("rigid skin:", len(np.unique(comp)), "pieces")
keep = np.argsort(-W, axis=1)[:, :4]
mask = np.zeros_like(W, bool)
np.put_along_axis(mask, keep, True, axis=1)
W = np.where(mask, W, 0)
W /= W.sum(1, keepdims=True).clip(1e-9)
for gi, vg in enumerate(body.vertex_groups):
    vg.remove(list(range(nv)))
    idx = np.nonzero(W[:, gi] > 1e-4)[0]
    for i in idx:
        vg.add([int(i)], float(W[i, gi]), "REPLACE")
print("weights smoothed; max bones per vertex:", int((W > 1e-4).sum(1).max()))

common.paint_legs(body)
common.write_rig_info(arm, joints, bones, limits, out / "rig_info.json")
bpy.ops.wm.save_as_mainfile(filepath=str(out / "rigged.blend"))
print("saved", out / "rigged.blend")

# a test pose: front-left leg folded, hind-right stepping, head down, tail up
test = {"upperarm_L": (-30, 0, 0), "forearm_L": (-70, 0, 0), "metacarpus_L": (90, 0, 0),
        "thigh_R": (-35, 0, 0), "shin_R": (60, 0, 0), "metatarsus_R": (-60, 0, 0),
        "neck1": (25, 0, 0), "head": (20, 0, 15), "tail1": (-30, 0, 0), "tail2": (-20, 0, 10)}
if CREATURE_TESTPOSE := (common._c.get("testpose") if hasattr(common, "_c") else None):  # creature.json "testpose"
    test = CREATURE_TESTPOSE
for name, deg in [(n, d) for n, d in test.items() if n in arm.pose.bones]:
    pb = arm.pose.bones[name]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = Euler([math.radians(a) for a in deg], "XYZ")
common.studio(bpy.context.scene)
common.render_views(out, "testpose", res=(640, 480))
