"""Build the creature's armature from rig.json, skin it with Blender's automatic weights, paint the legs and save.

rig.json may also carry hands (see common.load_rig_extras and docs/hands.md): three bones per finger, folding about
their X axis; a prop bone that alone moves the prop's vertices; and a handle a hand holds, around which its fingers
are closed here.

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
extras = common.load_rig_extras(rig_json)
unit = lambda v: np.asarray(v, float) / (np.linalg.norm(v) + 1e-12)  # noqa: E731

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
# fingers: base to tip under the hand. Each bone's Z points the way the finger folds, so +X folds it toward the palm.
finger_bones = []      # (name, hand, finger, k)
for hand, spec in extras["fingers"].items():
    n = unit(spec["palm_out"])
    names = common.finger_bone_names(hand)
    pinky_base = np.array(spec["joints"]["pinky"][0])
    for f in common.FINGERS:
        pts = [np.array(q, float) for q in spec["joints"][f]]
        for k in range(3):
            b = arm_data.edit_bones.new(names[f][k])
            b.head, b.tail = Vector(pts[k]), Vector(pts[k + 1])
            d = unit(pts[k + 1] - pts[k])
            fold = n if f != "thumb" else unit(0.55 * unit(pinky_base - pts[k]) + 0.45 * n)   # a thumb folds across the palm
            b.align_roll(Vector(unit(fold - d * float(fold @ d))))
            b.parent = eb[hand] if k == 0 else eb[names[f][k - 1]]
            b.use_connect = False
            eb[names[f][k]] = b
            finger_bones.append((names[f][k], hand, f, k + 1))
bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_AUTO")

# vertices the heat solve missed go fully to the nearest bone segment
names = [b[0] for b in bones] + [fb[0] for fb in finger_bones]
segs = np.array([[joints[h], joints[t]] for _, h, t, _ in bones]
                + [[list(arm.data.bones[fb[0]].head_local), list(arm.data.bones[fb[0]].tail_local)] for fb in finger_bones])
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
gname = [vg.name for vg in body.vertex_groups]
co = np.array([v.co[:] for v in body.data.vertices])


def seg_dist(p, a, b):
    t = np.clip(((p - a) * (b - a)).sum(1) / ((b - a) ** 2).sum().clip(1e-12), 0, 1)
    return np.linalg.norm(a + (b - a) * t[:, None] - p, axis=1)


# A finger's bones move only the skin nearest that finger. Heat also warms the fingers next door, which then drag
# along when one finger folds; here each vertex keeps finger weights only for the finger it is closest to.
for hand in extras["fingers"]:
    fn = common.finger_bone_names(hand)
    D = np.stack([np.min([seg_dist(co, np.array(arm.data.bones[b].head_local), np.array(arm.data.bones[b].tail_local))
                          for b in fn[f]], 0) for f in common.FINGERS], 1)
    own = D.argmin(1)
    for fi, f in enumerate(common.FINGERS):
        for b in fn[f]:
            gi = gname.index(b)
            stray = (own != fi) & (W[:, gi] > 0)
            W[stray, gname.index(hand)] += W[stray, gi]      # what a neighbour finger held goes to the palm
            W[stray, gi] = 0
    print(f"{hand}: finger weights kept to their own finger")

# A prop is rigid: every vertex of it follows the prop's bone alone, and nothing else follows that bone. Which
# vertices are the prop comes with the rig (rig.json "props": {bone: points file}), since neither heat nor the
# mesh's pieces (creature.json "rigid" above) can tell a held handle from the fingers around it.
for bone, src in extras["props"].items():
    from mathutils.kdtree import KDTree
    pts = np.load(pathlib.Path(rig_json).parent / src)
    tree = KDTree(len(pts))
    for i, q in enumerate(pts):
        tree.insert(Vector(q.tolist()), i)
    tree.balance()
    owned = np.array([tree.find(Vector(c.tolist()))[2] < 1e-4 for c in co])
    gi = gname.index(bone)
    W[owned] = 0
    W[owned, gi] = 1
    lost = (~owned) & (W[:, gi] > 0)
    W[lost, gi] = 0
    empty = lost & (W.sum(1) < 1e-6)
    W[empty, gname.index(arm.data.bones[bone].parent.name)] = 1
    W[lost] /= W[lost].sum(1, keepdims=True).clip(1e-9)
    print(f"prop {bone}: {int(owned.sum())} vertices follow it alone; {int(lost.sum())} others let go of it")
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

fingers_info = None
if finger_bones:
    def Rx(a):
        c, s_ = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0, 0], [0, c, -s_, 0], [0, s_, c, 0], [0, 0, 0, 1.0]])

    def seg_seg(a0, a1, b0, b1, n=24):
        """Shortest distance between two segments (sampled along the first, exact to the second)."""
        ts = np.linspace(0, 1, n)[:, None]
        pts = a0 + (a1 - a0) * ts
        return float(seg_dist(pts, b0, b1).min())

    rest = {fb[0]: np.array(arm.data.bones[fb[0]].matrix_local) for fb in finger_bones}
    length = {fb[0]: arm.data.bones[fb[0]].length for fb in finger_bones}

    def chain(hand, f, angles):
        """Head and tail of each of a finger's three bones with its joints folded by `angles` (hand at rest)."""
        fn = common.finger_bone_names(hand)[f]
        M, segs_, prev = None, [], None
        for name, a in zip(fn, angles):
            M = rest[name] @ Rx(a) if prev is None else M @ np.linalg.inv(rest[prev]) @ rest[name] @ Rx(a)
            segs_.append((M[:3, 3].copy(), M[:3, 3] + M[:3, 1] * length[name]))
            prev = name
        return segs_

    hold = {}
    for hand, h in extras["hold"].items():
        spec = extras["fingers"][hand]
        h0, h1, hr = np.array(h["from"], float), np.array(h["to"], float), float(h["radius"])
        fn = common.finger_bone_names(hand)
        pose, placed, report = {}, [], {}
        for f in ("index", "middle", "ring", "pinky", "thumb"):     # the thumb last: it closes over the fingers
            lim = common.FOLD["max"]["thumb" if f == "thumb" else "other"]
            ang = [0.0, 0.0, 0.0]
            for k in range(3):
                r = float(spec["radii"].get(f"{f}{k + 1}", 0.008))
                touched = False
                for deg in range(0, lim[k] + 1):
                    ang[k] = math.radians(deg)
                    a0, a1 = chain(hand, f, ang)[k]
                    gap = seg_seg(a0, a1, h0, h1) - (r + hr)
                    if f == "thumb":
                        gap = min([gap] + [seg_seg(a0, a1, b0, b1) - (r + br) for b0, b1, br in placed])
                    if gap <= -0.001:           # a millimetre into the handle: skin gives
                        touched = True
                        break
                if not touched and k == 0 and f != "thumb":
                    ang[k] = math.radians(lim[k] * 0.85)
            for name, a in zip(fn[f], ang):
                pose[name] = (a, 0.0, 0.0)
            placed += [(a0, a1, float(spec["radii"].get(f"{f}{k + 1}", 0.008))) for k, (a0, a1) in enumerate(chain(hand, f, ang))]
            report[f] = [round(math.degrees(a)) for a in ang]
        hold[hand] = pose
        print(f"{hand} closed around its handle (degrees, base to tip):", report)
    fingers_info = {
        "hands": list(extras["fingers"]),
        "bones": [{"name": n_, "hand": hand, "finger": f, "k": k, "parent": arm.data.bones[n_].parent.name,
                   "matrix_local": [list(r) for r in arm.data.bones[n_].matrix_local], "length": arm.data.bones[n_].length,
                   "radius": float(extras["fingers"][hand]["radii"].get(f"{f}{k}", 0.008))}
                  for n_, hand, f, k in finger_bones],
        "palm_out": {hand: list(unit(spec["palm_out"])) for hand, spec in extras["fingers"].items()},
        "hold": hold,
        "relaxed": {hand: common.finger_pose(hand, "relaxed") for hand in extras["fingers"]},
        "fist": {hand: common.finger_pose(hand, "fist") for hand in extras["fingers"]},
    }

common.paint_legs(body)
common.write_rig_info(arm, joints, bones, limits, out / "rig_info.json", extras, fingers_info)
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
if fingers_info:
    common.set_fingers(arm, common.default_fingers(fingers_info))
common.studio(bpy.context.scene)
common.render_views(out, "testpose", res=(640, 480))
