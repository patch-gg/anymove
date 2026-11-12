# hands.

A hand is about forty pixels of the 720-pixel frame. The joint reader can't see fingers at that size, so a creature with
hands comes out with stiff, open ones, and anything it holds has to be pushed through the palm. This adds fingers: bones
for them in the rig, a grip on what a hand holds, and a hand model that reads them from the clip.

The hand model step is the finger tracking of [prompt2rig](https://github.com/blendi-remade/prompt2rig) (`hq/hands.py`:
WiLoR, with MediaPipe Hands as a fallback), fitted to two cameras. prompt2rig finds hands with a detector and tells them
apart by the body's wrists. Here the reader has already found each wrist and knuckle in both views and the rig knows
which hand is which, so each hand is cut out where it is and only the pose model runs.

All of it is optional. A `rig.json` without the new keys builds, trains and captures exactly as before.

## What changes

| step | without fingers | with them |
|---|---|---|
| the rig | a hand is one bone, wrist to fingertips | wrist to middle knuckle, plus three bones per finger |
| what the reader reads | wrist and fingertips | wrist and three knuckles, which stay put whatever the fingers do |
| a held prop | an open hand can't hold it, so it passes through the palm | sits in front of the palm, the fingers closed around it |
| the prop's skin | automatic weights, shared with the hand | its own vertices follow its bone alone |
| training poses | hands as modelled | a random hand in every pose, open to fist; a holding hand keeps its grip |
| after the fit | nothing | `mocap/hands.py`: fingers and wrist from the hand model, in both views |
| the GLB and the pack | body bones | the fingers too: tracked, or in their default pose |

## rig.json

```json
"bones":   [["hand_L", "wrist_L", "knuckle_L", "forearm_L"]],
"points":  {"index_L": "hand_L", "pinky_L": "hand_L"},
"fingers": {"hand_L": {"palm_out": [x, y, z],
                       "joints": {"thumb": [p1, p2, p3, p4], "index": [], "middle": [], "ring": [], "pinky": []},
                       "radii": {"index1": 0.008}}},
"props":   {"sword": "sword_points.npy"},
"hold":    {"hand_R": {"from": [x, y, z], "to": [x, y, z], "radius": 0.018}}
```

- **The hand bone** ends at the middle finger's knuckle, not at the fingertips. Its name ends in `_L` or `_R`.
- `points`: joints the reader reads that ride on a bone without being one of its ends. They're listed in `joints` like
  any other. The wrist, the middle knuckle and these two (the index and little-finger knuckles) give the hand's roll.
- `fingers`: per hand bone, four points per finger from base to tip, in the prepped mesh's frame. `build_rig.py` makes
  three bones of them (`thumb1_L` to `pinky3_L`) whose X axis folds toward `palm_out`, the way the palm faces, and
  keeps each finger's weights on its own finger. The reader never reads them. `radii` (metres, per bone, optional) is
  how thick each finger is where it closes on a handle.
- `props`: a prop's bone and a `.npy` beside `rig.json` with the prop's vertex positions (N x 3, in the prepped mesh's
  frame, matched to 0.1 mm). Those vertices follow that bone alone, and nothing else follows it. creature.json's
  `"rigid": true` can't do this for a held prop: the fingers around a handle can be the same piece of mesh.
- `hold`: the handle a hand holds, as a capsule. The builder folds each finger from base to tip until it meets the
  handle, the thumb last, and stores that grip in `rig_info.json`. The reference frame, the training poses and every
  export use it.

A left hand's `points` and `fingers` are mirrored to the right like joints and bones, unless you give the right ones.
Add the finger bones to the hand's `paint` group in `creature.json`, so the whole hand has the limb's colour.

For the first creature nobody placed the finger points by hand. Each hand was filmed from eight cameras in a ring, WiLoR
marked 21 keypoints in every picture, the points the views agreed on were kept (3 mm apart on average), and rays to the
skin centred each one in its finger. That tool reads another project's rig and isn't in this repo yet.

## Capture

```bash
export PROMPT2RIG_HQ=/path/to/prompt2rig/hq
tools/capture.sh <c> runs/<c>/videos/clench/s2/h3max.mp4 clench_s2
```

`PROMPT2RIG_HQ` is the `hq` folder of a prompt2rig checkout: its `.venv` has the hand model, and its
`setup_hands.py` fetches the weights into `weights/`. The step runs under that Python, not anymove's; `HANDS_PY` names
another one, and `HANDS_ARGS` passes options (`HANDS_ARGS="--wrist 0.5"`). Without `PROMPT2RIG_HQ` the capture still
finishes, with the fingers in their default pose: closed around what the hand holds, loosely curled otherwise.

`capture.sh` runs `mocap.hands` after the fit when the rig has fingers. For each hand and view it:

1. cuts the hand out where the reader found its wrist and knuckle, and mirrors a left hand;
2. checks the crop shows the hand's paint colour, so a hand hidden behind the body isn't read off what covers it;
3. runs the model twice, the second time on a crop centred and sized on the hand the first pass found;
4. turns the 21 keypoints into world axes (the cameras are fixed) and blends the two views by how squarely each sees
   the palm, dropping the worse one when they disagree about which way the palm faces;
5. fills gaps up to 0.4 s and smooths.

Each finger bone is then aimed along its measured segment in the palm's own frame, and kept inside what a finger can
do. The wrist is turned 80% of the way to the measured palm (`--wrist`). A hand that holds something keeps its grip and
its solved wrist. A hand whose knuckles are ground contacts keeps its solved wrist too, so what plants it stays planted.

It writes `fingers` (`bones`, `euler`: T x bones x 3) and `hands` (how often each hand was seen, and by which view) into
`motion.json`, updates the hand's Euler angles and the joints that move with it, and saves `hands.jpg`: what the model
saw. `apply_motion.py`, `tools/shape_clip.py`, `blender/pack_glb.py` and `mocap/mirror.py` carry the fingers through.
The step can be run again on the same file with other settings: it starts from the solver's wrists each time.

## The exam

`tools/hands_exam.py` makes a motion whose free hand opens and closes three times while the arm turns it every way,
films it with both cameras, and scores what comes back against the truth (168 frames, a biped holding a sword).

| the free hand | fingertips off | hand's turn off |
|---|---|---|
| reader and solver only | 7.4 cm | 23° |
| with the hands step, WiLoR | 2.4 cm | 9° |

That's the whole capture, in the world. Given the true joints instead of the reader's, WiLoR puts fingertips 1.9 cm
from the truth (a hand left in its default pose: 6.3 cm) and follows opening and closing with a correlation of 0.98.
MediaPipe Hands, given the same crops, found the hand in 35% of the front view's frames and 26% of the side's, and was
5.2 cm off. At this size it isn't usable.

This is a clean render, so it's the best an AI clip can do. On AI clips there's no truth to score: look at `hands.jpg`
and at the showcase.

```bash
c=<c>; r=assets/$c/rigged; e=runs/$c/hands_exam; export MOCAP_CREATURE=assets/$c
.venv/bin/python tools/hands_exam.py make $r/rig_info.json $e
blender -b -P blender/apply_motion.py -- $r/rigged.blend $r/rig_info.json $e/truth.json $e/render \
  --views front,side --paint --samples 16
.venv/bin/python tools/hands_exam.py video $e
# the hand model alone, on the true joints
$PROMPT2RIG_HQ/.venv/bin/python -m mocap.hands $e/exam.mp4 $e/det.npz $r/rig_info.json $e/body.json \
  --out $e/hands.json --wrist 1
.venv/bin/python tools/hands_exam.py score $r/rig_info.json $e
# the whole capture: --no-calib, since the clip doesn't start from the reference pose
.venv/bin/python -m mocap.detect $e/exam.mp4 runs/$c/reader.pt runs/$c/ref/ref.png $e/read.npz --no-calib
.venv/bin/python -m mocap.solve $e/read.npz $r/rig_info.json $e/solved.json
.venv/bin/python tools/hands_exam.py world $r/rig_info.json $e $e/solved.json      # reader and solver only
$PROMPT2RIG_HQ/.venv/bin/python -m mocap.hands $e/exam.mp4 $e/read.npz $r/rig_info.json $e/solved.json
.venv/bin/python tools/hands_exam.py world $r/rig_info.json $e $e/solved.json      # with the hands step
```

## Licence

anymove's code stays Apache-2.0, and no model or weights are bundled. `mocap/hands.py` follows prompt2rig's
`hq/hands.py` (Apache-2.0, Copyright 2026 blendi-remade) in how the models are set up and in its gap and smoothing
settings, and only calls the models through their own packages.

**The model that makes this work is not free for commercial use.** WiLoR is CC-BY-NC-ND, and it needs MANO, which is
non-commercial and needs a registration. WiLoR-mini, the package it's called through, has no licence file of its own.
Its pipeline also loads a hand detector built on Ultralytics YOLO (AGPL-3.0), which this step doesn't call. MediaPipe
Hands is Apache-2.0 and doesn't work at this size. A hand reader trained the way the joint reader is, on the creature's
own renders, would keep the whole chain clean. It isn't built.

## Not done

- **A hand the side camera can't see has no depth.** The fit places it from the front view alone, and on one kneel
  the left hand ended inside the thigh. Its fingers are then read from the front view only; a hand neither view shows
  eases back to its default pose after 0.4 s.
- **One grip per hand.** A hand can't let go of what it holds, change its grip, or hold with both hands.
- **The thumb lies along the handle** rather than wrapping it.
- **Human hands only:** five fingers, three bones each. Claws, paws and fused fingers aren't handled.
- **One pair of hands per creature.** Finger bones are named per side.
- **The tool that finds a mesh's finger points** (above).
- **A mirrored motion** (`mocap/mirror.py`, for the LoRA set) swaps the fingers with the hands, but a held prop stays
  in the hand it was built in.
