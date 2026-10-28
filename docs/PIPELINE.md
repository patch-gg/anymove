# pipeline.

How the pieces fit, file by file, and what each stage reads and writes. The skills in `skills/` say how to run it;
this says what's inside.

## The contract: one frame, two cameras

Every stage agrees on one 1280 x 720 layout (`mocap/layout.py`, mirrored in `blender/common.py`):

- **Front view**, left, `W_FRONT` px wide: an orthographic camera at -Y looking along +Y at the creature's face. Image
  u grows with world X.
- **Side view**, right, `W_SIDE` px wide: an orthographic camera at -X looking along +X at the creature's right flank,
  head to the right. Image u grows with world -Y.
- Both share the vertical axis: v grows downward with -Z, the ground (z = 0) is row `GROUND_V`, and one pixel is
  `M_PER_PX` metres in both views. A joint at height z sits on the same row in both halves. That is what makes two
  views enough for 3D (x from the front, y from the side, z from both) and what the sync check measures.
- The studio: grey wall (128, 130, 134), a lighter floor band below the ground row (152, 154, 158), a dark divider.

`MOCAP_CREATURE=assets/<c>` selects a creature; its `creature.json` sets the layout, the paint and the feet. Without
it, every module falls back to the wolf's layout (420 + 860 px, 1.9 m of height in frame).

## Files a creature brings

| file | written by | holds |
|---|---|---|
| `assets/<c>/model.glb` | `tools/mesh.sh` (Rodin first, fal Tripo fallback), or your own generator | the textured mesh |
| `assets/<c>/creature.json` | you | `prep` (length_m, rotz_deg, axis), `layout`, `paint` (limb groups: colour + bones), `feet`, prompt words (`name`, `noun`, `describe`, `colour_code`, `count`, `flank`, `limbs`), the 3/4 showcase `camera` |
| `assets/<c>/rig.json` | you | `joints` (metres, -Y forward, +X left), `bones` [name, head, tail, parent], `limits_deg` per bone or stem |
| `assets/<c>/prep/static.blend` | `blender/prep_mesh.py` | the mesh at real size, facing -Y, on the ground |
| `assets/<c>/rigged/rigged.blend` | `blender/build_rig.py` | armature, skinned mesh, limb paint as a colour attribute (`legpaint`) |
| `assets/<c>/rigged/rig_info.json` | `blender/build_rig.py` | rest matrices, parents, lengths, limits, and the keypoints (which bone end each joint is) for the solver |

## Stages

| stage | code | reads | writes |
|---|---|---|---|
| mesh | `tools/mesh.sh`, `mocap/mesh.py`, `mocap/rodin.py` | concept image | `assets/<c>/model.glb`, private Rodin task in `runs/<c>/rodin.json`, fallback task in `runs/<c>/rodin.tripo.json` |
| rig (optional API) | `mocap/rig.py`, `blender/import_rig.py` | model or saved Rodin asset + rig.json hierarchy | rigged.blend, rig_info.json; private task in `runs/<c>/rigging.json` |
| reference frame | `blender/render_ref.py`, `tools/compose_ref.py` | rigged.blend | `runs/<c>/ref/ref.png` |
| prompt | `tools/make_prompt.py` | creature.json + one action sentence | `prompts/<c>/<move>.txt` |
| video | `mocap/generate.py` (fal queue client in `mocap/fal.py`) | ref.png + prompt | `runs/<c>/videos/<move>/s<seed>/<model>.mp4` |
| dataset | `blender/render_dataset.py` | rigged.blend + rig_info.json | `runs/<c>/dataset/<i>_{front,side}.png`, `<i>_meta.json` (true joints in metres and layout px) |
| reader | `mocap/model.py`, `mocap/train.py` | dataset | `runs/<c>/reader.pt` (state dict, joint names, held-out error) |
| read | `mocap/detect.py` | clip + reader + ref.png | `det.npz`: `front_uv`, `side_uv` (T x J x 2, layout px), `front_conf`, `side_conf`, `fps`, `names`, `calib` |
| fit | `mocap/solve.py` | det.npz + rig_info.json | `motion.json`: `bones`, `trans`, `root_quat`, `euler` (T x B x 3), `joints` (T x J x 3 m), `reproj_px`, `sync` (incl. `paw_agreement`) |
| apply | `blender/apply_motion.py` | rigged.blend + motion.json | keyframed rig, `joints.json` (true joint positions from Blender), renders, `animation.glb` |
| compare | `mocap/showcase.py` | clip + det.npz + renders | `showcase.mp4` |
| exam | `mocap/exam.py` | rig_info.json | a random motion, its two-camera video, and a per-joint error score in cm |
| LoRA | `tools/build_lora_set.py`, `mocap/mirror.py`, `mocap/lora.py` | captured motions + prompts | training clips, the fal run, `lora.json` |

## Inside the reader

DINOv2 ViT-B/14 with registers (`facebook/dinov2-with-registers-base`, Apache-2.0), fine-tuned end to end (backbone at
2e-5, head at 1e-3, AdamW, cosine schedule with warm-up). Each view is padded to a square with the backdrop grey and
resized to 448 x 448 (32 x 32 patches); the patch tokens go through a small conv head that upsamples to one 128 x 128
heatmap per joint, trained against Gaussian targets (sigma 2). Peaks are refined to sub-pixel from their neighbours.
Training images are laid on random backdrops and degraded like video; 4% of samples are held out.

## Inside the fit

`mocap/solve.py` redoes Blender's forward kinematics in torch (pose = parent pose x rest offset x Euler XYZ), with the
root free and every other bone's Euler angles squashed into its limits by a sigmoid. Before fitting it scores the
clip (paw agreement, height gap and best lag between views) and time-warps the front view onto the side view when
that cuts the height gap by 15% or more. Joints are triangulated (x from the front, y from the side, height weighted
0.25 front / 1 side by confidence), the root rotation comes from a weighted Kabsch fit of the points riding on the
root bone, and an analytic IK pose aimed at the triangulated joints is the starting point. The loss is a robust
(Cauchy) reprojection error in both views, its scale annealed from 40 px to 6 px over the first 60% of 1,500 Adam
steps, plus joint acceleration, angular velocity, a small pull to rest, and a ground term for paws.

## Known gaps in the code

- The ground term and `fix_swaps` know the wolf's and dragon's limb names (`fpaw`, `ftoe`, `hpaw`, `htoe`; `LEGS` in
  `solve.py`). Other creatures fit without a ground penalty, and `--fix-swaps` does nothing for them.
- `mocap/exam.py`'s per-group scores use the wolf's joint names; the mean, median and p95 work for any creature.
- `tools/build_lora_set.py` mirrors motions assuming a symmetric rig.
