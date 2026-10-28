---
name: anymove-rig
description: Set up a new creature for anymove - get a textured mesh, scale and face it, place its skeleton in rig.json with grid views, build and skin the rig in Blender, paint each limb its own colour, choose the two-camera framing, and render the reference frame the video model starts from. Use when adding a creature to anymove, when a rig tears or deforms badly, when the creature is cut off or tiny in the two-camera frame, or when someone asks how to rig a non-humanoid for anymove.
license: Apache-2.0
compatibility: Blender 5.x on PATH (or BLENDER=...), Python 3.11 venv from requirements.txt. Rodin uses the public vibecoding evaluation key or RODIN_API_KEY; Tripo fallback and video use FAL_KEY.
---

# anymove-rig: one creature, ready to film

Everything for creature `<c>` lives in `assets/<c>/`: `model.glb` (the mesh), `creature.json` (framing, paint, words
for the prompt), `rig.json` (the skeleton). Copy a close example to start: `assets/wolf` (quadruped), `assets/dragon`
(quadruped with wings), `assets/spider` (eight legs), `assets/demon` (biped with wings and tail).

## 1. The mesh

- Any textured GLB works. To make one, copy a close example config, then run `tools/mesh.sh <c> concept.png`.
  Rodin Gen-2.5 runs first with the public `vibecoding` evaluation key (usage limits); `RODIN_API_KEY` in `.env`
  or the environment overrides it. Rejected/failed generations fall back to Tripo H3.1 on fal (`FAL_KEY`, about
  $0.40 with detailed textures). The mesh lands in `assets/<c>/model.glb`.
- Use a plain 3/4 concept picture on a neutral background: whole body in frame, limbs apart and clearly separate.
  The bundled meshes are Tripo H3.1; a new mesh needs new joint placement.
- After an interruption, repeat with `--resume`. Tasks are saved in `runs/<c>/rodin.json` or `rodin.tripo.json`;
  keep them private. A pending task or download error is resumed, without submitting another generation.
- Prefer a design where every limb reads separately. Fire, smoke, capes, weapons and long hair hide joints from the
  reader later; leave them to game VFX.

## 2. Face it, scale it, stand it on the ground

Fill in `creature.json` → `"prep": {"length_m": ..., "rotz_deg": ..., "axis": "y"}`: the real size along `axis` (y is
nose to tail; x for creatures wider than long, like a spider's leg span; z for tall bipeds), and the turn about Z that
makes it face -Y. Then:

```bash
tools/prep.sh <c>        # assets/<c>/prep/static.blend, verts.npy, and views.png (side, front, top, 10 cm grid)
```

Check `assets/<c>/prep/views.png`: in the side view the head is on the right, the feet sit on z = 0, and both feet of a
pair are level in the front view. Generated meshes can arrive at a different orientation; fix
`rotz_deg` until the feet are level (the demon needed -73, not -90). If unsure which way it faces, run
`blender -b -P blender/inspect_mesh.py -- assets/<c>/model.glb runs/<c>/inspect` and look at the four views.

## 3. The skeleton: rig.json

For automatic placement and skinning, keep the closest example's bone names, hierarchy, limits and paint groups,
then run `tools/rig.sh <c> rodin`. Rodin uses the named hierarchy from `rig.json` (including mirrored `_R` bones),
places it on the mesh, and writes the usual `rigged.blend`, `rig_info.json` and reference frame. The joint positions
in `rig.json` are used by the Blender route; the Rodin route records its predicted positions in `rig_info.json`.
It reuses a saved Rodin generation and its textures. Other meshes are uploaded without textures, or pass
`--image concept.png` to generate PBR textures. Use `--resume` after an interruption; the task is saved privately
in `runs/<c>/rigging.json`. Check facing, paint and joint limits before training a new reader. Own-key rigging costs
0.5 credit; the public evaluation key has usage limits.

For hand placement with Blender:

Metres in the prepped mesh's frame: the creature faces -Y, its left is +X, Z is up, the ground is z = 0.

```json
{
  "joints": {"hips": [0, 0.32, 0.86], "spine1": [0, 0.05, 0.95], "shoulder_L": [0.21, -0.55, 1.02]},
  "bones": [["pelvis", "hips", "spine1", null], ["upperarm_L", "shoulder_L", "elbow_L", "chest"]],
  "limits_deg": {"pelvis": [[-30, 30], [-20, 20], [-30, 30]], "upperarm": [[-70, 50], [-10, 10], [-25, 25]]}
}
```

- `joints`: name to [x, y, z]. Every `_L` joint is mirrored to `_R` (x negated) unless you give the `_R` joint
  yourself; give both for asymmetric meshes (the dragon's wings).
- `bones`: [name, head joint, tail joint, parent]. The first bone is the root (free to turn and move). `_L` bones are
  mirrored too. Joints at bone ends are what the reader learns to find, so put one at every place the video should
  control: paw tips, wing tips, tail segments (6 for a long tail), jaw if the mouth opens.
- `limits_deg`: per bone name, or per stem without `_L`/`_R`: [[x lo, x hi], [y lo, y hi], [z lo, z hi]] in degrees.
  Local X is the bending axis (world X for bones in the body's plane), Y runs along the bone (twist: small for spine
  and legs, ±8-20; arms can need ±40), Z swings sideways. Limits shape the training poses as well as the fit: too wide
  and the reader learns impossible poses, too narrow and it can't read the move.

Place joints from the grid views. After each edit:

```bash
tools/prep.sh <c>               # redraws views.png with the joints and bones on the grid
tools/rig.sh <c> preview        # joints over an x-ray of the mesh: assets/<c>/prep/preview/joints_{side,front,top}.png
```

Joints belong inside the limb, at the pivot: the elbow at the bend, not on the skin. For legs, take the centre line
from the side and front views; for wings, put the wrist and finger tips on the membrane's edge.

## 4. Build, skin, paint, render the reference

Fill in `creature.json` → `"paint"`: one group per limb with a colour and the bones it covers, left and right in
different colours, all colours far apart (pink #F53CD1, blue #2F6BFF, yellow #FFD21F, cyan #19D3E0, orange #FF7A1A,
green #2BD94A, purple #9B4DFF, red #FF2B2B). Set `"feet"` to the paw or foot joints (the sync score uses them).

```bash
tools/rig.sh <c>     # assets/<c>/rigged/{rigged.blend, rig_info.json, testpose_*.png}, runs/<c>/ref/ref.png
```

`build_rig.py` merges the seam vertices first (Tripo splits the mesh at every texture seam; skipping this tears the
thighs open when a joint bends), skins with Blender's automatic weights, smooths them and keeps 4 bones per vertex.
Check `testpose_front.png` and `testpose_side.png`: the bent limbs stay whole and only the painted part moves.

## 5. The two-camera framing: creature.json "layout"

`{"W_FRONT": 640, "W_SIDE": 640, "M_PER_PX": 0.006875, "GROUND_V": 630}`: widths of the two views in pixels (they sum
to 1280; 420 + 860 suits a long, narrow quadruped, 640 + 640 a wide one), metres per pixel, and the ground's pixel
row in the 720-row frame. The creature in its biggest pose must fit both views with margin: height under about
`GROUND_V × M_PER_PX × 0.85`, width under `W_FRONT × M_PER_PX`, length under `W_SIDE × M_PER_PX`. A dragon rearing
with wings up needs more headroom than one standing. Re-run `tools/rig.sh <c>` after changing it and look at
`runs/<c>/ref/ref.png`: the whole creature, both views, a dark divider between them, nothing touching an edge.

## 6. Words for the prompt

Also in `creature.json`: `name` ("spider"), `noun` ("giant spider"), `describe` ("a black armoured giant spider with
bone-coloured markings"), `colour_code` ("first left leg pink, first right leg blue; ..."), `count` ("Exactly eight
legs, two fangs, one body, one round abdomen."), `flank` ("flank" or "side") and `limbs` ("legs" or "limbs").
`tools/make_prompt.py` builds every prompt from these (anymove-film).

Last, `"camera": {"cam": "-5.4,-6.0,1.4", "target": "0,-0.3,1.5", "lens": 32}` is the 3/4 view that
`tools/capture.sh` renders each move from (anymove-capture says how to frame it). Without it, the wolf's applies.

## Failure modes seen

- Torn or gaping thighs when a leg bends: seam vertices weren't merged (a mesh from a tool other than build_rig.py),
  or a joint sits outside the limb.
- A limb half painted: the paint group misses a bone, or the weights spill onto the body; move the joint, rebuild.
- The rig bends the wrong way: the bone's local X isn't the bending axis. Keep bones of a limb in one plane.
- Rodin rigging keeps bone names but predicts positions and local axes. Check limb paint and adjust local joint
  limits for the new rig; train a new reader instead of reusing one trained on another mesh or skeleton.
