---
name: anymove-pack
description: Turn an anymove creature's captured takes into one game-ready GLB - score the takes against the gates, pick the keepers, cut seamless loops, make root-motion versions of walks and runs, trim one-shots, take the limb paint out, and measure foot slide. Use when a creature has several captured takes and someone wants the final animation file, asks which take to keep, wants loops or root motion, or wants to reproduce one of the creatures in Patch's Assets.
license: Apache-2.0
compatibility: Blender 5.x, Python 3.11 venv from requirements.txt. No GPU and no fal key. Needs assets/<c>/rigged/rigged.blend (anymove-rig) and captured takes in runs/<c>/clips/ (anymove-capture).
---

# anymove-pack: takes in, one GLB out

[docs/PACKS.md](../../docs/PACKS.md) has every field and number. The short version:

## 1. Capture every take into its own folder

Name each folder `<move>_s<seed>`, so you can tell which seed made it:

```bash
for s in 1 2 3; do tools/capture.sh <c> runs/<c>/videos/roar/s$s/h3max.mp4 roar_s$s; done
```

## 2. Score them

```bash
.venv/bin/python tools/score_takes.py <c>      # a table, and runs/<c>/takes.csv
```

A take passes with `paw` 0.8 or more (`None` is fine: planted feet), `front_x` 3 px or less and `side` 8.5 px or less,
on confident readings. Expect about 58% to pass.

A pass is a shortlist. **Look at each passing take's `showcase.mp4` before you pick it**: a run can pass and still look
like a fast walk. If a move has no good take, it usually takes two or three more seeds. If six seeds fail, the move may
be one anymove can't do: tell the person, and drop it or flag it. Don't keep spending.

For walks and runs, compare seeds by their feet:

```bash
.venv/bin/python tools/locomotion.py --candidates <c> 24,96 walk_s1 walk_s2 walk_s3
```

About 2 cm of worst slide or less is as good as it gets today. A run should be clearly faster than the walk, with a
longer stride and some time in the air.

## 3. Write picks/\<c\>.json

```json
{"Idle":   {"take": "idle_s2", "loop": true, "skip": 24, "cycle": [48, 96]},
 "Walk":   {"take": "walk_s14", "loop": true, "skip": 12, "cycle": [24, 96], "locomotion": true},
 "Run":    {"take": "run_s3", "loop": true, "skip": 12, "cycle": [12, 72], "locomotion": true},
 "Attack": {"take": "attack_s3"},
 "Hit":    {"take": "hit_s2", "flagged": true}}
```

The key is the animation's name in the GLB. Idles, walks and runs loop; attacks and hits are one-shots, trimmed
automatically, or by `start` and `end`. `flagged` puts a near miss in a separate file. `picks/` has ten real examples.

## 4. Build it

```bash
tools/pack.sh <c>       # runs/<c>/pack/<c>.glb, under a minute
```

Check three things:

- **The seams.** In `shape_report.json`, `seam_jump_vs_median_step` compares the step over the wrap with a typical step,
  so about 1 means no visible join (the 46 loops in Patch's packs are 0.54 to 1.12). Well above that, try a wider
  `cycle` or a larger `skip`.
- **The feet.** It prints each walk's and run's speed, stride, cadence and slide, and writes `locomotion.json`.
- **The file.** Load the GLB in an engine or viewer, play every clip, and confirm the limb colours are gone.

## Telling people what they're getting

Say these plainly. An animation only fits the creature it was made for. Feet slide 1 to 2 cm a step, and nothing pins
them. Files are heavy for a browser game (55,000 to 60,000 triangles and 4K textures), with no lighter versions.
