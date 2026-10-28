<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/readme/header-evening.png">
  <img alt="patch. anymove. motion capture for any creature, from one AI video." src="docs/readme/header-day.png" width="800">
</picture>

<br><br>

<img alt="Six panels: a dragon, a giant spider, a fire demon and a wolf, all moving; then an AI video shows the dragon from the front and the side at once; every joint is found in both views; the two views stand up as walls and pin each joint in 3D; the rig lands on the points; a fire demon, a spider and a wolf move the same way." src="docs/anymove.gif" width="800">

**@patch_gg** · dragon, giant spider, fire demon, wolf · one AI clip per move, $0.26 each, read and fitted in under
a minute.

anymove turns one AI video into an animation for any rigged creature, wings, tails and eight legs included. A video
model films the creature from two cameras in one frame, front on the left and side on the right. A joint reader
trained on that one creature finds every joint in both views. A joint sits at the same height in both, so the pair
pins it in 3D, and the rig is fitted to those joints frame by frame and exported as a GLB.

The method is [@CanItCode](https://www.youtube.com/watch?v=AS92HYsEBqo)'s; the MiniMax H3 LoRA step is
[@Graalitoo](https://x.com/Graalitoo/status/2112548557576380659)'s idea. This repo is our working version of both, with
the agent skills to run it.

## how it works.

| | |
|---|---|
| <img src="docs/steps/step-1.png" width="320" alt="A concept picture of a dragon, an arrow, and the textured 3D model it became."> | **1 · a 3d model.**<br>One picture becomes a textured mesh at real size.<br><sub><picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/bfl-white.png"><img src="docs/readme/marks/bfl-black.png" height="14" alt=""></picture> flux 3 picture, <picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/tripo-white.png"><img src="docs/readme/marks/tripo-black.png" height="14" alt=""></picture> tripo h3.1 mesh · $0.40</sub> |
| <img src="docs/steps/step-2.png" width="320" alt="The dragon as a faint x-ray with its white skeleton inside."> | **2 · a skeleton.**<br>Bones placed on grid views, skinned in <picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/blender-white.png"><img src="docs/readme/marks/blender-black.png" height="14" alt=""></picture> Blender.<br><sub>42 bones: legs, wings, tail</sub> |
| <img src="docs/steps/step-3.png" width="320" alt="The dragon with each leg and wing painted its own colour, the right wing labelled."> | **3 · limb colours.**<br>Every limb its own colour, so left and right never swap.<br><sub>6 limbs, 6 colours</sub> |
| <img src="docs/steps/step-4.png" width="320" alt="The dragon from above, one camera at its face and one at its right flank."> | **4 · two cameras.**<br>One camera at the front, one at the side, both views in one frame; a video model animates both at once from a prompt.<br><sub><picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/minimax-white.png"><img src="docs/readme/marks/minimax-black.png" height="14" alt=""></picture> minimax h3 max on <picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/fal-white.png"><img src="docs/readme/marks/fal-black.png" height="14" alt=""></picture> fal · 5 s, $0.26</sub> |
| <img src="docs/steps/step-5.png" width="320" alt="A random training pose of the dragon in both views with its true joints drawn on."> | **5 · teach a reader.**<br>A vision model learns this one dragon from random poses whose joints are known; its average miss falls from 40 px to 6 px.<br><sub><picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/meta-white.png"><img src="docs/readme/marks/meta-black.png" height="14" alt=""></picture> dinov2 · 2,200 poses · 20 min on one gpu</sub> |
| <img src="docs/steps/step-6.png" width="320" alt="The AI clip with a white skeleton drawn over the dragon in both views."> | **6 · read every joint.**<br>The reader finds every joint in both views, in every frame.<br><sub>50 joints · 10 s a clip</sub> |
| <img src="docs/steps/step-7.png" width="320" alt="The two views stood up as walls at a right angle, the nose ringed on both, and the two lines meeting at the nose in 3D."> | **7 · two views, 3d.**<br>Stand the two views up as walls at a right angle: each joint is where its two lines meet. The front gives side to side, the side gives nose to tail.<br><sub>x from the front, y from the side, z from both</sub> |
| <img src="docs/steps/step-8.png" width="320" alt="The rig as an x-ray, its bones landed on the 3D points, stray points in grey."> | **8 · the rig follows.**<br>The rig bends its bones onto the points, ignores the strays, and exports for any engine.<br><sub>glb, 24 fps, no hand edits · under a minute a clip</sub> |

[docs/PIPELINE.md](docs/PIPELINE.md) has the file-by-file version.

## quick start.

You need an NVIDIA GPU with CUDA (24 GB to train a reader; reading and fitting need much less), Python 3.11,
[Blender](https://www.blender.org) 5.x on your `PATH` (or `BLENDER=/path/to/blender`), and a [fal](https://fal.ai) key
to make new clips.

```bash
git clone https://github.com/Patch-gg/anymove && cd anymove
python3.11 -m venv .venv
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
cp .env.example .env                                  # then add your FAL_KEY; git ignores .env
```

Fit the dragon's roar on the CPU, no Blender or downloads needed. The repo carries the joints its reader found in the
clip; this fits the rig to them (about 20 seconds) and prints how well the two views agree:

```bash
mkdir -p runs/dragon/clips/roar
MOCAP_CREATURE=assets/dragon .venv/bin/python -m mocap.solve examples/dragon_roar/det.npz \
  assets/dragon/rigged/rig_info.json runs/dragon/clips/roar/motion.json
```

The whole capture, from the AI clip to a GLB, with an example creature's mesh, trained reader and clip from the
[release](https://github.com/Patch-gg/anymove/releases/tag/examples-v1) (13 MB mesh, 355 MB reader, 4 MB clip):

```bash
tools/fetch_examples.sh dragon --readers --clips
tools/prep.sh dragon && tools/rig.sh dragon          # real size, skeleton, skin, limb paint, reference frame
tools/capture.sh dragon runs/dragon/videos/roar/s1/h3max.mp4 roar
# runs/dragon/clips/roar/render/animation.glb, and showcase.mp4 to compare it with the clip
```

A new move for the same dragon (two clips, about $0.52 on fal):

```bash
.venv/bin/python tools/make_prompt.py assets/dragon "the dragon whips its tail from side to side, then stands still." prompts/dragon/tailwhip.txt
tools/film.sh dragon tailwhip h3max 1,2
tools/capture.sh dragon runs/dragon/videos/tailwhip/s1/h3max.mp4 tailwhip
```

## your own creature.

Copy the closest example in `assets/` (wolf: four legs; dragon: four legs and wings; spider: eight legs; demon: two
legs, two arms, wings and a tail). Bring your own `assets/<name>/model.glb`, or make one from a concept picture:
Rodin Gen-2.5 runs first with the public `vibecoding` evaluation key; rejected or failed generations fall back to
Tripo H3.1 on fal (`FAL_KEY`, about $0.40 with detailed textures). Use a plain 3/4 view, whole body in frame, limbs
apart on a neutral background.

```bash
tools/mesh.sh <name> concept.png  # optional: generate model.glb if you do not have one
tools/prep.sh <name>              # set "prep" in creature.json first: real size, facing -Y
# place joints in assets/<name>/rig.json on the grid views; check with: tools/rig.sh <name> preview
tools/rig.sh <name>               # build, skin, paint, reference frame
# or: tools/rig.sh <name> rodin    # Rodin rigging, same rigged.blend/rig_info.json and reference frame
tools/dataset.sh <name>           # 2,200 random poses, about 25 min
tools/train.sh <name>             # the joint reader, about 20 min
tools/film.sh <name> <move>       # after tools/make_prompt.py writes prompts/<name>/<move>.txt
tools/capture.sh <name> runs/<name>/videos/<move>/s1/h3max.mp4 <move>
```

The skills below walk an agent (or you) through each step, with the checks that tell a good capture from a bad one.

Set `RODIN_API_KEY` in `.env` or the environment to use your own key; otherwise the public evaluation key has
usage limits. If a run is interrupted, repeat `tools/mesh.sh` with `--resume`: it reuses the saved Rodin or Tripo
task in `runs/<name>/rodin*.json`. Keep those task files private. New meshes need new joint placement
with the Blender route; the Rodin route places the bones from the named hierarchy in `rig.json`.

Rodin rigging reuses a saved Rodin mesh task and its textures when available. For a mesh from another generator,
it uploads `model.glb` without textures; pass `--image concept.png` to make new PBR textures. Repeat
`tools/rig.sh <name> rodin --resume` after an interruption. Check facing, limb paint and joint limits before
training a new reader; an old reader was trained on the old skeleton. Rigging costs 0.5 credit with your own key.

## agent skills.

anymove is a [Claude Code](https://code.claude.com) plugin and its own marketplace:

```
/plugin marketplace add Patch-gg/anymove
/plugin install anymove@anymove
```

| skill | for |
|---|---|
| `anymove` | the method, setup, which step comes next, costs |
| `anymove-rig` | mesh, facing and scale, placing joints in rig.json, building and painting the rig, framing |
| `anymove-film` | writing the prompt, generating clips on fal, checking them before capture |
| `anymove-reader` | rendering the dataset, training the reader, the exam in centimetres |
| `anymove-capture` | reading, fitting, the quality gates, fixing bad fits, showcase renders |
| `anymove-lora` | training and testing the MiniMax H3 LoRA on fal |

They're plain [Agent Skills](https://agentskills.io) (`skills/<name>/SKILL.md`), so other agents can load them from that
folder too.

## results.

| creature | skeleton | reader error, front / side (px) | how well the views agree (1 = same footfalls) |
|---|---|---|---|
| wolf, 1.6 m | 30 bones, 36 joints | 6.9 / 3.8 | trot 0.94 · exam: 0.94 cm mean 3D joint error |
| dragon, 3 m | 42 bones, 50 joints, wings | 6.2 / 5.5 | take off 1.00, roar 0.93, swipe 0.85, walk 0.81 |
| giant spider, 2.4 m span | 28 bones, 40 joints, 8 legs | 2.9 / 3.6 | rear up 0.99, bite 0.96, pounce 0.89 |
| fire demon, 3.6 m | 38 bones, 46 joints, wings, tail | 8.4 / 9.4 | wing beat 0.99, slam 0.84, roar 0.73 (feet planted) |

Reader error is the held-out mean on the 720-row frame; the demon stands 412 px tall, so its 8.4 px is about 2% of its
height. Every clip in the table is MiniMax H3 Max. The exam renders a made-up motion with known joints in the same
studio and reads it back.

## what it costs.

| | |
|---|---|
| a mesh, Rodin Gen-2.5 / Tripo H3.1 with HD texture | public evaluation key / $0.40 on fal |
| a 5 s two-camera clip, MiniMax H3 Max | about $0.26 (H3: $0.30) |
| training renders, 2,200 poses x 2 views | about 25 min on one RTX 4090 |
| the joint reader, 20 epochs | 17-20 min, 12 GB |
| reading a clip / fitting it | 10 s / under a minute (20 s for the dragon's roar on a CPU) |
| the H3 LoRA, 1,200 steps on fal | about $12, about 2 h |

Prices on fal as of October 2026.

## limits.

- **Eight-legged walking doesn't fit.** The two views disagree about which legs are up (0.36-0.49). Big single
  gestures work best: rearing, roaring, slamming, taking off, pouncing.
- **Lying down doesn't fit.** The reader learns from random poses inside the joint limits, which don't include lying on
  a side; the wolf's death and lie-down clips failed.
- **The Blender route places skeletons by hand** on grid views (`tools/prep.sh`, `tools/rig.sh preview`).
  `tools/rig.sh <name> rodin` can place and skin the named skeleton instead; check its result before training.
- **Video models drift.** Plain MiniMax H3 swapped the limb colours on 2 of 5 wolf clips; with the LoRA, 0 of 5 (one
  seed per move: a first read, not a benchmark). Seedance 2.5 played the views out of step on a trot.
- **The LoRA's data is AI-derived.** Its training clips re-render motions solved from AI clips, so the halves are in
  sync by construction but the motion itself is AI-made. Real dog mocap exists (AI4Animation) under a non-commercial
  licence.
- **Ground contact is wolf- and dragon-shaped.** The solver's ground term knows paw names `fpaw`, `ftoe`, `hpaw`,
  `htoe`; other creatures fit without it.
- **Fur and cloth** need extra bones: springs (`apply_motion.py --springs`) or tracked tips (the wolf's tracked fur read
  7-28° swings from the video).

## licence.

Apache-2.0, see [LICENSE](LICENSE). The Patch name, mark, wordmark and Patch Pixel belong to Patch and aren't covered
by it. Other companies' marks in this README belong to their owners and are shown only to credit their tools (icons
from [lobe-icons](https://github.com/lobehub/lobe-icons), MIT, and [Simple Icons](https://simpleicons.org), CC0).
MiniMax H3's open weights carry a community licence and hosted use on fal goes through fal's terms: check both before
shipping what you make with them. The wolf LoRA in the release (`lora-wolf-twocam-h3.safetensors`) is a LoRA on
MiniMax H3, so MiniMax H3's community licence applies to it, not Apache-2.0.

## credits.

- The two-camera method is [@CanItCode](https://www.youtube.com/watch?v=AS92HYsEBqo)'s, and the H3 LoRA step is
  [@Graalitoo](https://x.com/Graalitoo/status/2112548557576380659)'s.
- <picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/meta-white.png"><img src="docs/readme/marks/meta-black.png" height="16" alt=""></picture> DINOv2 by Meta (with registers, Apache-2.0) is the reader's backbone.
- <picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/tripo-white.png"><img src="docs/readme/marks/tripo-black.png" height="16" alt=""></picture> Tripo H3.1 made the meshes, from concept pictures by
  <picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/bfl-white.png"><img src="docs/readme/marks/bfl-black.png" height="16" alt=""></picture> Black Forest Labs' FLUX 3.
- <picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/minimax-white.png"><img src="docs/readme/marks/minimax-black.png" height="16" alt=""></picture> MiniMax H3 and H3 Max made the clips, all run on
  <picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/fal-white.png"><img src="docs/readme/marks/fal-black.png" height="16" alt=""></picture> fal.
- <picture><source media="(prefers-color-scheme: dark)" srcset="docs/readme/marks/blender-white.png"><img src="docs/readme/marks/blender-black.png" height="16" alt=""></picture> Blender rigs, renders and exports everything.
- Made by [Patch](https://patch.gg) ([@patch_gg](https://x.com/patch_gg)).

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/readme/footer-evening.png">
  <img alt="patch.gg. Mashups, mods and browser games." src="docs/readme/footer-day.png" width="500">
</picture>
