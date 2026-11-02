<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/readme/header-evening.png">
  <img alt="patch. anymove. motion capture for any creature, from one AI video." src="docs/readme/header-day.png" width="800">
</picture>

<br><br>

<img alt="Six panels: a dragon, a giant spider, a fire demon and a wolf, all moving; then an AI video shows the dragon from the front and the side at once; every joint is found in both views; the two views stand up as walls and pin each joint in 3D; the rig lands on the points; a fire demon, a spider and a wolf move the same way." src="docs/anymove.gif" width="800">

**@patch_gg** · dragon, giant spider, fire demon, wolf · about $3 for a usable move, all in.

anymove turns one AI video into an animation for any rigged creature, wings, tails and eight legs included. A video
model films the creature from two cameras in one frame, front on the left and side on the right. A joint reader
trained on that one creature finds every joint in both views. A joint sits at the same height in both, so the pair
pins it in 3D, and the rig is fitted to those joints frame by frame and exported as a GLB.

The method is [@CanItCode](https://www.youtube.com/watch?v=AS92HYsEBqo)'s; the MiniMax H3 LoRA step is
[@Graalitoo](https://x.com/Graalitoo/status/2112548557576380659)'s idea. This repo is our working version of both, with
the agent skills to run it.

**Just want the creatures?** [Nine are ready in Patch's Assets](https://staging.patch.gg/create/toolbox/assets?source=anymove),
with 4 to 7 moves each: T-rex, knight, horse, centaur, woman, ogre, ape, mech and scorpion.

## before you start.

- **You need:** an NVIDIA card (12 GB to train a reader, 24 GB is safer), Blender 5.x, Python 3.11 and a
  [fal](https://fal.ai) key. Without the card it doesn't run, and there's no cloud version.
- **It costs about $3 for a usable move, all in,** and about $17 and a couple of hours for a new creature with about 5
  moves. [The numbers are below](#what-it-costs).
- **Most clips don't make it.** A move took about 7 clips on average, and 58% of captured clips passed the checks.

## what it's good at, and what it isn't.

**Good at:** big single gestures on any body plan: roars, slams, rearing up, chest beats, stomps, taking off. Walks and
runs work on two and four legs. You get a real animation on a real skeleton, in a file any engine plays.

**Not good at:**

- **Eight-legged walking.** The two views disagree about which legs are up (0.36-0.49). Scorpions and spiders can rear,
  bite and strike, but they don't walk.
- **Lying down.** The reader learns from random poses inside the joint limits, which don't include lying on a side.
- **Bird wings.** The eagle's wings don't fit (15-30 px off), and it was left out of the nine. Bat-like wings do fit, on
  the dragon and the demon.
- **Tail swipes.** The two views disagree about where the tail tip is. The T-rex's was dropped after six tries.
- **Fine foot contact.** Feet slide 1 to 2 cm a step and nothing pins them yet. That's fine for most uses, and visible up
  close.
- **Fast runs.** Runs look like fast walks in places: the video model doesn't lower the body on a run.
- **Moving an animation to another creature.** An animation only fits the creature it was made for, and a new model
  needs a new reader.
- **Light files.** Generated creatures are 55,000 to 60,000 triangles with 4K textures, which is heavy for a browser
  game, and there are no lighter versions yet.

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

Then the best takes become one file, with seamless loops and root motion ([docs/PACKS.md](docs/PACKS.md)).
[docs/PIPELINE.md](docs/PIPELINE.md) has the file-by-file version.

## quick start.

You need an NVIDIA GPU with CUDA (12 GB to train a reader, 24 GB is safer; reading and fitting need much less), Python
3.11, [Blender](https://www.blender.org) 5.x on your `PATH` (or `BLENDER=/path/to/blender`), and a
[fal](https://fal.ai) key to make new clips.

```bash
git clone https://github.com/Patch-gg/anymove && cd anymove
python3.11 -m venv .venv
.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
cp .env.example .env                                  # then add your FAL_KEY; git ignores .env
```

**On Windows,** use [Git Bash](https://git-scm.com/downloads/win), and the same commands with the venv's other folder.
Every `tools/*.sh` finds the venv and handles Blender's paths and Python's line endings. Blender usually isn't on your
`PATH`, so point at it:

```bash
git clone https://github.com/Patch-gg/anymove && cd anymove
py -3.11 -m venv .venv
.venv/Scripts/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env
export BLENDER="/c/Program Files/Blender Foundation/Blender 5.2/blender.exe"
```

Where a command below says `.venv/bin/python`, use `.venv/Scripts/python.exe`.

Fit the dragon's roar on the CPU, no Blender or downloads needed. The repo carries the joints its reader found in the
clip; this fits the rig to them (about a minute and a half) and prints how well the two views agree:

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

A new move for the same dragon (two clips, about $0.52 on fal; expect to need more, since a usable move took about 7
clips on average):

```bash
.venv/bin/python tools/make_prompt.py assets/dragon "the dragon whips its tail from side to side, then stands still." prompts/dragon/tailwhip.txt
tools/film.sh dragon tailwhip h3max 1,2
tools/capture.sh dragon runs/dragon/videos/tailwhip/s1/h3max.mp4 tailwhip
```

## your own creature.

Copy the closest example in `assets/` (wolf: four legs; dragon: four legs and wings; spider: eight legs; demon: two
legs, two arms, wings and a tail), put your mesh at `assets/<name>/model.glb`, and work through:

```bash
tools/prep.sh <name>              # set "prep" in creature.json first: real size, facing -Y
# place joints in assets/<name>/rig.json on the grid views; check with: tools/rig.sh <name> preview
tools/rig.sh <name>               # build, skin, paint, reference frame
tools/dataset.sh <name>           # 2,200 random poses, about 25 min
tools/train.sh <name>             # the joint reader, about 20 min
tools/film.sh <name> <move>       # after tools/make_prompt.py writes prompts/<name>/<move>.txt
tools/capture.sh <name> runs/<name>/videos/<move>/s1/h3max.mp4 <move>_s1
.venv/bin/python tools/score_takes.py <name>   # which takes pass the checks
tools/pack.sh <name>                           # after listing the keepers in picks/<name>.json: one GLB with every move
```

`tools/make_mesh.py` makes the picture and the mesh on fal, if you don't have a model. The skills below walk an agent (or
you) through each step, with the checks that tell a good capture from a bad one.

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
| `anymove-pack` | choosing takes, seamless loops, root motion, foot slide, the final GLB |
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

Then one run made ten more creatures. What's in each pack, with the worst foot slide across its walk and run:

| creature | moves | worst slide | | creature | moves | worst slide |
|---|---|---|---|---|---|---|
| T-rex | 7 | 2.26 cm | | ogre | 6 | 1.15 cm |
| knight | 7 | 1.88 cm | | ape | 5 | 1.40 cm |
| centaur | 7 | 1.62 cm | | horse | 5 | 1.64 cm |
| mech | 7 | 2.12 cm | | woman | 4 | 1.36 cm |
| scorpion | 4 | no walk | | eagle | 1 (left out) | |

Their choices, and each clip's seed and prompt, are in [`picks/`](picks).

## what it costs.

A clip's price isn't a move's price, because most moves take several tries. These are the real numbers from the run
that made ten creatures, on fal and rented GPUs:

| | |
|---|---|
| **a usable move, all in** | **about $3** |
| its clips alone | about $1.80, about 7 clips |
| **a new creature with about 5 moves, all in** | **about $17** ($110 of AI services and about $55-65 of cloud GPU, across 10 creatures) |
| the same on your own card | about $11 of AI services |
| captured clips that pass the checks | 58% (193 of 335) |

What goes into that:

| | |
|---|---|
| a concept picture and a mesh, Tripo H3.1 with HD texture | about $0.50, more if the first mesh comes back wrong |
| a 5 s two-camera clip, MiniMax H3 Max | about $0.26 (H3: $0.30) |
| placing the skeleton | your or your agent's time; nobody has timed it |
| training renders and the joint reader | about 75 min on one RTX 4090, 10 min of training on a cloud H100 |
| reading a clip / fitting it | about 12 s / about 45 s (about 90 s for the dragon's roar on a CPU) |
| exporting the GLB and renders | about 3 min a clip |
| the optional H3 LoRA, 1,200 steps on fal | about $12, about 2 h |

Prices on fal as of October 2026. After a creature's first move, each new one only needs clips and capture.

## limits.

[What it isn't good at](#what-its-good-at-and-what-it-isnt) comes first. And:

- **Skeletons are placed by hand** on grid views (`tools/prep.sh`, `tools/rig.sh preview`). UniRig could do it; its
  install pulls prebuilt flash-attn wheels.
- **Video models drift.** Plain MiniMax H3 swapped the limb colours on 2 of 5 wolf clips; with the LoRA, 0 of 5 (one
  seed per move: a first read, not a benchmark). Seedance 2.5 played the views out of step on a trot.
- **The LoRA's data is AI-derived.** Its training clips re-render motions solved from AI clips, so the halves are in
  sync by construction but the motion itself is AI-made. Real dog mocap exists (AI4Animation) under a non-commercial
  licence.
- **Mirrored arms need their own limits.** `_L` and `_R` bones share a local Z axis, so one sideways limit is wrong for
  one side. Give `upperarm_L` and `upperarm_R` mirrored Z ranges in `rig.json`
  ([anymove-rig](skills/anymove-rig/SKILL.md)).
- **Overhead moves can leave the frame.** The layout's height needs room for the biggest move, or the prompt has to keep
  limbs low.
- **Fur and cloth** need extra bones: springs (`apply_motion.py --springs`) or tracked tips (the wolf's tracked fur read
  7-28° swings from the video).

## licence.

Apache-2.0, see [LICENSE](LICENSE). The Patch name, mark, wordmark and Patch Pixel belong to Patch and aren't covered
by it. Other companies' marks in this README belong to their owners and are shown only to credit their tools (icons
from [lobe-icons](https://github.com/lobehub/lobe-icons), MIT, and [Simple Icons](https://simpleicons.org), CC0).
MiniMax H3's open weights carry a community licence and hosted use on fal goes through fal's terms: check both before
shipping what you make with them. What you make doesn't automatically get the licence of anything here, including the
CC0 on Patch's nine creatures: its terms depend on the models you use and the keys you use them with. The wolf LoRA in the release (`lora-wolf-twocam-h3.safetensors`) is a LoRA on
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
