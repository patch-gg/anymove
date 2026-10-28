---
name: anymove
description: Motion capture for any creature from one AI video, with the anymove repo. A video model films a rigged 3D creature from two cameras in one frame, a DINOv2 joint reader trained on that creature finds every joint in both views, and the rig is fitted to them and exported as a GLB. Use when someone wants to animate a non-human creature (dragon, spider, wolf, monster) from a text prompt or an AI video, asks how anymove works, or needs to know which step comes next; it routes to anymove-rig, anymove-film, anymove-reader, anymove-capture and anymove-lora.
license: Apache-2.0
compatibility: Needs an NVIDIA GPU with CUDA (24 GB to train a reader at batch 20), Blender 5.x, Python 3.11, and a fal.ai key for video generation.
---

# anymove

Animate any rigged creature from one AI video. The trick is the frame: one image holds two orthographic views of the
creature at the same scale, front on the left and side on the right, sharing the vertical axis. A joint sits on the
same pixel row in both halves, so the front view gives left-right and height, the side view gives front-back and
height, and together they pin the joint in 3D. The method is @CanItCode's; the LoRA step is @Graalitoo's idea.

## The pipeline

| step | once per | what happens | skill |
|---|---|---|---|
| 1. mesh | creature | a textured 3D model (Rodin Gen-2.5 from one image, or your own GLB), facing -Y, real size | anymove-rig |
| 2. skeleton | creature | joints placed by hand or Rodin, rig skinned and imported into Blender | anymove-rig |
| 3. limb colours | creature | each limb painted its own colour so left and right never swap | anymove-rig |
| 4. two cameras | move | the painted reference frame plus a strict prompt go to MiniMax H3 Max on fal | anymove-film |
| 5. teach a reader | creature | 2,200 random poses rendered with known joints; DINOv2 learns this creature | anymove-reader |
| 6. read every joint | clip | the reader finds every joint in both views, every frame | anymove-capture |
| 7. two views, 3D | clip | shared heights tie the views; the pair gives each joint in 3D | anymove-capture |
| 8. the rig follows | clip | forward kinematics in torch fitted to both views; GLB out | anymove-capture |
| optional | creature | a MiniMax H3 LoRA that keeps the two-camera layout and colours more reliably | anymove-lora |

## Before anything

1. Work in an anymove checkout. If the working directory has no `tools/capture.sh`, clone it:
   `git clone https://github.com/Patch-gg/anymove && cd anymove`.
2. Set up once: `python3.11 -m venv .venv`, then
   `.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128` and
   `.venv/bin/pip install -r requirements.txt`. Blender 5.x must be on PATH, or set `BLENDER=/path/to/blender`.
3. For video generation and LoRA training, put `FAL_KEY=...` in `.env` (git ignores it) or export it. Never print it.
   New meshes use Rodin with the public `vibecoding` evaluation key, or your `RODIN_API_KEY` from `.env`/the
   environment; failed generations fall back to Tripo on fal with `FAL_KEY` (anymove-rig).
4. Every command takes a creature name; its config is `assets/<c>/creature.json` and every output lands in
   `runs/<c>/` (reference frame, dataset, `reader.pt`, videos, clips). The scripts set `MOCAP_CREATURE` for you; when
   you run a `python -m mocap...` module by hand, set `MOCAP_CREATURE=assets/<c>` first, or the wolf's layout applies.

## Which step next

- No `assets/<c>/model.glb`: make or fetch the mesh. Examples: `tools/fetch_examples.sh <c>` (wolf, dragon, spider,
  demon). New creature: anymove-rig.
- No `assets/<c>/rigged/rigged.blend` or `runs/<c>/ref/ref.png`: `tools/prep.sh <c>` then `tools/rig.sh <c>`
  (anymove-rig).
- No `runs/<c>/reader.pt`: anymove-reader (about 45 min on one RTX 4090), or `tools/fetch_examples.sh <c> --readers`.
- No clip for the move: anymove-film (about $0.26 a 5 s clip).
- A clip: anymove-capture, `tools/capture.sh <c> <video.mp4> <name>`.
- Clips keep swapping limb colours or the halves drift apart: anymove-lora.

## Try it without a GPU

The repo carries the dragon's roar as read by its reader. Fit it on the CPU in about 20 seconds:

```bash
mkdir -p runs/dragon/clips/roar
MOCAP_CREATURE=assets/dragon .venv/bin/python -m mocap.solve examples/dragon_roar/det.npz \
  assets/dragon/rigged/rig_info.json runs/dragon/clips/roar/motion.json
```

It prints the paw agreement (about 0.93) and the reprojection error per view; `motion.json` holds the keyframes.

## Costs and time (October 2026, one RTX 4090)

- Rodin mesh: public evaluation key (usage limits); own-key costs at https://hyper3d.ai/features/api. Tripo fallback:
  $0.40 with HD texture. MiniMax H3 Max clip: about $0.26 for 5 s (H3: $0.30).
- Dataset of 2,200 poses: about 25 min. Reader training: 17-20 min. Reading a clip: 10 s. Fitting: under a minute.
- A LoRA: $0.01 a training step, so 1,200 steps cost about $12 and take about 2 h on fal.

Say what a step will cost before calling fal, and confirm with the person before spending on a batch.
