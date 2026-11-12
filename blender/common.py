"""Shared Blender helpers: the rig file, the two-camera layout, the studio, limb paint and debug renders.

The layout is the contract between every stage. One 1280 x 720 frame holds two orthographic views at the same scale:
the front camera (left) looks along +Y at the creature's face, the side camera (right) looks along +X at its right
flank, head to the right. Both share the vertical axis, so a joint's height is the same row in both halves; that is
what makes two views enough for 3D and what the sync check measures.

The creature is chosen with MOCAP_CREATURE=<assets dir>, whose creature.json can set its own framing (a dragon's
wingspan needs a wider front view) and limb paint groups. Without it, the wolf's settings below apply.
"""
import json, math, os, pathlib

import bpy
import numpy as np
from mathutils import Vector

H = 720
W_FRONT, W_SIDE = 420, 860
M_PER_PX = 1.9 / H        # 1.9 m of height in frame
GROUND_V = 630            # pixel row of the ground (z = 0); the 90 rows below it are the floor band
LEG_COLORS = {"FL": "#F53CD1", "FR": "#2F6BFF", "HL": "#FFD21F", "HR": "#19D3E0"}
LEG_BONES = {"FL": ["upperarm_L", "forearm_L", "metacarpus_L", "fpaw_L"],
             "FR": ["upperarm_R", "forearm_R", "metacarpus_R", "fpaw_R"],
             "HL": ["thigh_L", "shin_L", "metatarsus_L", "hpaw_L"],
             "HR": ["thigh_R", "shin_R", "metatarsus_R", "hpaw_R"]}
CREATURE = os.environ.get("MOCAP_CREATURE")
if CREATURE and (pathlib.Path(CREATURE) / "creature.json").exists():
    _c = json.load(open(pathlib.Path(CREATURE) / "creature.json"))
    _l = _c.get("layout", {})
    W_FRONT, W_SIDE = _l.get("W_FRONT", W_FRONT), _l.get("W_SIDE", W_SIDE)
    M_PER_PX = _l.get("M_PER_PX", M_PER_PX)
    GROUND_V = _l.get("GROUND_V", GROUND_V)
    if "paint" in _c:
        LEG_COLORS = {k: v["color"] for k, v in _c["paint"].items()}
        LEG_BONES = {k: v["bones"] for k, v in _c["paint"].items()}
Z_CENTER = (GROUND_V - H / 2) * M_PER_PX
LAYOUT = {"H": H, "W_FRONT": W_FRONT, "W_SIDE": W_SIDE, "M_PER_PX": M_PER_PX, "GROUND_V": GROUND_V,
          "Z_CENTER": Z_CENTER, "Y_CENTER": 0.0, "X_CENTER": 0.0}


def mirror(name):
    return name[:-2] + "_R" if name and name.endswith("_L") else name


def load_rig(path):
    """Joints and bones from rig.json with every _L mirrored to _R, and per-bone Euler limits in degrees."""
    rig = json.load(open(path))
    joints = dict(rig["joints"])
    for k, v in rig["joints"].items():
        if k.endswith("_L") and mirror(k) not in rig["joints"]:
            joints[mirror(k)] = [-v[0], v[1], v[2]]
    bones = []
    for b in rig["bones"]:
        bones.append(b)
        if b[0].endswith("_L"):
            bones.append([mirror(x) for x in b])
    limits = {}
    for name, *_ in bones:
        key = name if name in rig["limits_deg"] else name[:-2]
        limits[name] = rig["limits_deg"][key]
    return joints, bones, limits


def load_rig_extras(path):
    """The optional parts of rig.json, for creatures with hands and props (docs/hands.md):
    "points"  {joint: bone}: a joint the reader reads that rides on a bone without being one of its ends (a knuckle)
    "fingers" {hand bone: {"palm_out": [x, y, z], "joints": {finger: four points, base to tip}, "radii": {bone: m}}}:
              three bones per finger under the hand. The reader never reads them; a hand tracker poses them.
              "radii" is optional (8 mm without it).
    "props"   {bone: "points.npy"}: a prop's bone and the vertices that are the prop (positions, beside rig.json).
              They follow that bone alone and nothing else does, so a sword never bends with the hand around it.
    "hold"    {hand bone: {"from": p, "to": p, "radius": m}}: the handle that hand is closed around."""
    rig = json.load(open(path))
    ex = {k: dict(rig.get(k, {})) for k in ("points", "fingers", "props", "hold")}
    # like joints and bones, a left point or hand is mirrored to the right unless the right one is given
    for j, bone in list(ex["points"].items()):
        if j.endswith("_L") and mirror(j) not in ex["points"]:
            ex["points"][mirror(j)] = mirror(bone)
    for hand, spec in list(ex["fingers"].items()):
        if hand.endswith("_L") and mirror(hand) not in ex["fingers"]:
            ex["fingers"][mirror(hand)] = {"palm_out": [-spec["palm_out"][0], *spec["palm_out"][1:]], "radii": spec.get("radii", {}),
                                           "joints": {f: [[-q[0], q[1], q[2]] for q in pts] for f, pts in spec["joints"].items()}}
    for spec in ex["fingers"].values():
        spec.setdefault("radii", {})
    return ex


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# how far each finger bone folds, base to tip, in degrees about its own X axis (positive folds toward the palm)
FOLD = {"relaxed": {"thumb": (6, 10, 12), "other": (16, 24, 16)},
        "fist": {"thumb": (22, 34, 48), "other": (82, 96, 62)},
        "max": {"thumb": (45, 60, 80), "other": (95, 110, 80)}}


def finger_bone_names(hand):
    """thumb1_L ... pinky3_L for hand_L, base to tip per finger."""
    side = hand[hand.rindex("_"):] if "_" in hand else ""
    return {f: [f"{f}{k}{side}" for k in (1, 2, 3)] for f in FINGERS}


def finger_pose(hand, kind):
    """{bone: (x, y, z) Euler radians} for one hand: "relaxed" or "fist"."""
    out = {}
    for f, names in finger_bone_names(hand).items():
        for name, deg in zip(names, FOLD[kind]["thumb" if f == "thumb" else "other"]):
            out[name] = (math.radians(deg), 0.0, 0.0)
    return out


def default_fingers(info):
    """The pose fingers take when nothing tracks them: closed around what the hand holds, loosely curled otherwise."""
    f = info.get("fingers")
    if not f:
        return {}
    pose = {}
    for hand in f["hands"]:
        pose.update(f["hold"].get(hand) or f["relaxed"][hand])
    return pose


def set_fingers(rig, pose):
    for name, e in pose.items():
        pb = rig.pose.bones[name]
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = e


def keypoint_world(rig, info, kp):
    """A keypoint's world position from the posed rig: a bone's head, its tail, or a fixed spot that rides on it."""
    pb = rig.pose.bones[info["bones"][kp["bone"]]["name"]]
    if kp["end"] == "offset":
        return rig.matrix_world @ (pb.matrix @ Vector(kp["local"]))
    return rig.matrix_world @ (pb.head if kp["end"] == "head" else pb.tail)


def srgb_to_linear(hexcol):
    c = np.array([int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def paint_legs(body, threshold=0.55):
    """Colour each leg from the shoulder or hip down to the paw, as a color attribute mixed over the texture."""
    idx = {g.name: g.index for g in body.vertex_groups}
    n = len(body.data.vertices)
    w = {leg: np.zeros(n) for leg in LEG_BONES}
    for v in body.data.vertices:
        for g in v.groups:
            for leg, names in LEG_BONES.items():
                if body.vertex_groups[g.group].name in names:
                    w[leg][v.index] += g.weight
    rgba = np.zeros((n, 4))
    for leg, col in LEG_COLORS.items():
        m = w[leg] > threshold
        rgba[m, :3] = srgb_to_linear(col)
        rgba[m, 3] = 1.0
    attr = body.data.color_attributes.get("legpaint") or body.data.color_attributes.new("legpaint", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", rgba.ravel())
    for slot in body.material_slots:
        nt = slot.material.node_tree
        bsdf = next(nd for nd in nt.nodes if nd.type == "BSDF_PRINCIPLED")
        src = bsdf.inputs["Base Color"].links[0].from_socket if bsdf.inputs["Base Color"].links else None
        ca = nt.nodes.new("ShaderNodeVertexColor")
        ca.layer_name = "legpaint"
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        nt.links.new(ca.outputs["Alpha"], mix.inputs["Factor"])
        if src is not None:
            nt.links.new(src, mix.inputs["A"])
        else:
            mix.inputs["A"].default_value = bsdf.inputs["Base Color"].default_value
        nt.links.new(ca.outputs["Color"], mix.inputs["B"])
        nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    print("painted vertices", {leg: int((w[leg] > threshold).sum()) for leg in w})


def write_rig_info(arm, joints, bones, limits, path, extras=None, fingers=None):
    """Everything the solver needs to redo Blender's forward kinematics, plus where each joint sits: a bone's head or
    tail, or (rig.json "points") a fixed spot on a bone. `fingers` (build_rig) is written as is: the finger bones and
    their poses, which the solver leaves alone."""
    order = [b[0] for b in bones]
    points = (extras or {}).get("points", {})
    info = {"bones": [], "keypoints": [], "layout": LAYOUT}
    for name, h, t, parent in bones:
        bb = arm.data.bones[name]
        info["bones"].append({"name": name, "parent": order.index(parent) if parent else -1,
                              "matrix_local": [list(r) for r in bb.matrix_local], "length": bb.length,
                              "head_joint": h, "tail_joint": t, "limits_deg": limits[name]})
    for j in joints:
        src = next(((i, "head") for i, b in enumerate(bones) if b[1] == j), None) or \
              next(((i, "tail") for i, b in enumerate(bones) if b[2] == j), None)
        if src is None:
            if points.get(j) not in order:
                raise SystemExit(f'rig.json: joint "{j}" is not an end of any bone; name the bone it rides on in "points"')
            i = order.index(points[j])
            local = arm.data.bones[points[j]].matrix_local.inverted() @ Vector(joints[j])
            info["keypoints"].append({"name": j, "bone": i, "end": "offset", "local": list(local), "rest": joints[j]})
            continue
        info["keypoints"].append({"name": j, "bone": src[0], "end": src[1], "rest": joints[j]})
    if fingers:
        info["fingers"] = fingers
    json.dump(info, open(path, "w"), indent=1)


def cameras(scene):
    """The front and side orthographic cameras of the layout (created once, reused)."""
    cams = {}
    for name, loc, rot, w in (("cam_front", (0, -20, Z_CENTER), (math.pi / 2, 0, 0), W_FRONT),
                              ("cam_side", (-20, 0, Z_CENTER), (math.pi / 2, 0, -math.pi / 2), W_SIDE)):
        cam = bpy.data.objects.get(name)
        if cam is None:
            cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
            scene.collection.objects.link(cam)
        cam.location, cam.rotation_euler = loc, rot
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = M_PER_PX * max(w, H)
        cam.data.clip_end = 100
        cams[name] = (cam, w)
    return cams


def camera_json(cam, w, h):
    """The camera's pixel projection, so 2D drawings can be laid exactly over its renders:
    pc = R @ p + t; perspective u = cx + f * pc.x / -pc.z, v = cy - f * pc.y / -pc.z; ortho u = cx + f * pc.x."""
    bpy.context.view_layer.update()   # a camera placed from Python has a stale matrix_world until the next update
    mw = cam.matrix_world.normalized()
    R = mw.to_3x3().transposed()
    t = -(R @ mw.translation)
    if cam.data.type == "ORTHO":
        f = max(w, h) / cam.data.ortho_scale
    else:
        f = cam.data.lens / cam.data.sensor_width * max(w, h)
    return {"type": cam.data.type, "f": f, "cx": w / 2, "cy": h / 2, "res": [w, h],
            "R": [list(r) for r in R], "t": list(t), "location": list(cam.location)}


def studio(scene, samples=32, seed=None):
    """Cycles on CUDA, transparent film, Standard view transform (keeps the leg paint saturated), three suns."""
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "CUDA"
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    world = scene.world or bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1)
    bg.inputs["Strength"].default_value = 0.55
    for name, rot, energy in (("key", (math.radians(50), 0, math.radians(-35)), 3.2),
                              ("fill", (math.radians(60), 0, math.radians(120)), 1.0),
                              ("rim", (math.radians(35), 0, math.radians(200)), 1.6)):
        ob = bpy.data.objects.get(name)
        if ob is None:
            ob = bpy.data.objects.new(name, bpy.data.lights.new(name, "SUN"))
            scene.collection.objects.link(ob)
        ob.rotation_euler = rot
        ob.data.energy = energy
        ob.data.angle = math.radians(8)


def render_views(out, prefix, scale=1.0, res=None):
    """Render the front and side views as RGBA PNGs at the layout's pixel scale times `scale`."""
    scene = bpy.context.scene
    paths = {}
    for name, (cam, w) in cameras(scene).items():
        scene.camera = cam
        scene.render.resolution_x = round(w * scale)
        scene.render.resolution_y = round(H * scale)
        scene.render.resolution_percentage = 100
        view = name.split("_")[1]
        scene.render.filepath = str(out / f"{prefix}_{view}.png")
        bpy.ops.render.render(write_still=True)
        paths[view] = scene.render.filepath
    return paths


def stick_figure(joints, bones, radius=0.012):
    """Spheres at joints and rods along bones (left pink, right blue, middle orange), drawn over the mesh."""
    mats = {}
    for side, col in (("L", (1, 0.1, 0.8, 1)), ("R", (0.1, 0.3, 1, 1)), ("M", (1, 0.5, 0, 1))):
        m = bpy.data.materials.new(f"stick_{side}")
        m.diffuse_color = col
        mats[side] = m
    side = lambda n: "L" if n.endswith("_L") else "R" if n.endswith("_R") else "M"
    for name, p in joints.items():
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius * 1.6, location=p, segments=12, ring_count=8)
        bpy.context.object.data.materials.append(mats[side(name)])
    for name, h, t, _ in bones:
        a, b = Vector(joints[h]), Vector(joints[t])
        bpy.ops.mesh.primitive_cylinder_add(radius=radius * 0.6, depth=(b - a).length, location=(a + b) / 2, vertices=8)
        rod = bpy.context.object
        rod.rotation_euler = (b - a).to_track_quat("Z", "Y").to_euler()
        rod.data.materials.append(mats[side(name)])


def xray_views(out, prefix):
    """Workbench x-ray renders (side, front, top) so joints inside the mesh are visible."""
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_xray = True
    scene.display.shading.xray_alpha = 0.35
    scene.render.film_transparent = False
    cams = cameras(scene)
    top = bpy.data.objects.new("cam_top", bpy.data.cameras.new("cam_top"))
    scene.collection.objects.link(top)
    top.location, top.rotation_euler = (0, 0, 20), (0, 0, 0)
    top.data.type, top.data.ortho_scale = "ORTHO", M_PER_PX * W_SIDE
    for name, cam, w, h in (("front", cams["cam_front"][0], W_FRONT, H), ("side", cams["cam_side"][0], W_SIDE, H),
                            ("top", top, 600, W_SIDE)):
        scene.camera = cam
        scene.render.resolution_x, scene.render.resolution_y = w, h
        scene.render.filepath = str(out / f"{prefix}_{name}.png")
        bpy.ops.render.render(write_still=True)
