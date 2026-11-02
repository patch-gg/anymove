# packs.

A capture gives one raw clip: 5 s, starting from the reference pose. A game wants one file with the creature and its
moves, with walks that loop and travel. This is how the last step works, and how to get what's in Patch's Assets.

## The steps

```bash
tools/capture.sh <c> runs/<c>/videos/roar/s3/h3max.mp4 roar_s3   # one folder per take: <move>_s<seed>
.venv/bin/python tools/score_takes.py <c>                        # which takes pass: runs/<c>/takes.csv
# look at each passing take's showcase.mp4, then list the keepers in picks/<c>.json
tools/pack.sh <c>                                                # runs/<c>/pack/<c>.glb
```

On Windows, in Git Bash, the Python is `.venv/Scripts/python.exe`.

## Choosing takes

`tools/score_takes.py` writes a row per take. A take passes when:

| number | gate | what it means |
|---|---|---|
| `paw` | 0.8 or more | the feet's heights agree between the two views, so they tell one story |
| `front_x` | 3 px or less | the rig lands on the readings, left to right |
| `side` | 8.5 px or less | the rig lands on the readings in the side view |

All three use confident readings only (above 0.3). A limb hidden behind the body reads badly, and the fit is right to
ignore it, so it doesn't count. A foot that moves under 8 px in both views is planted, and its correlation is noise,
so it scores `None`. Also look at `pop` (one frame far from the others) and `swap_suspect`.

**A pass is a shortlist, not a yes.** In the run that made the nine creatures, 58% of captured clips passed (193 of
335), and some that passed still looked wrong. Watch `showcase.mp4`: the numbers don't see a run that looks like a walk.
For walks and runs, compare seeds by their feet before picking:

```bash
.venv/bin/python tools/locomotion.py --candidates trex 24,96 walk_s3 walk_s14
```

## picks/\<c\>.json

```json
{"Idle":   {"take": "idle_s2", "loop": true, "skip": 24, "cycle": [48, 96]},
 "Walk":   {"take": "walk_s14", "loop": true, "skip": 12, "cycle": [24, 96], "locomotion": true},
 "Attack": {"take": "attack_s3"},
 "Kick":   {"take": "kick_s1", "start": 4, "end": 118},
 "Hit":    {"take": "hit_s2", "flagged": true}}
```

| field | does |
|---|---|
| `take` | the folder in `runs/<c>/clips/`. The key is the animation's name in the GLB |
| `loop` | finds the start and length whose ends match best, then spreads what's left across the loop so the wrap has no pop |
| `skip` | frames to ignore at the start, while the move settles out of the reference pose (default 12) |
| `cycle` | the shortest and longest loop to try, in frames (default 20 to 96) |
| `locomotion` | also writes `<Name>_Root`, with root motion. The plain version stays on the spot |
| `start`, `end` | trims a one-shot by hand. Without them, it's trimmed to the action automatically |
| `flagged` | a weak clip or near miss. It goes in `<c>_flagged.glb`, not the main pack |

**Root motion.** Clips are filmed on the spot, so planted feet ride backwards together, like a treadmill. For each frame
the shaper takes the stillest planted foot, treats its speed as the ground's, and moves the root by that instead.

## What comes out

`runs/<c>/pack/`:

- `<c>.glb`: the textured, rigged creature and every clip as a named animation, with the limb paint taken out.
- `<c>_flagged.glb`: the picks marked `flagged`, if there are any.
- `clips/<Name>.json`: each shaped clip, in `motion.json`'s format, so `blender/apply_motion.py` can render it.
- `shape_report.json`: where each loop starts, how long it is, how big its seam is, and its root speed.
- `locomotion.json`: speed, stride, cadence, time in the air and foot slide, for each walk and run.

**Foot slide** is measured on the root-motion loop: a contact point is planted while it's within 2 cm (or 1% of the
creature's height) of its rest height and moving under 1 cm a frame (0.5% of height). Slide is how far the stillest
planted point travels over one footfall. It comes from moments with two feet down, where the video's feet disagree
about the ground's speed. Expect 1 to 2 cm a step. Nothing pins feet yet; a short IK pass would.

## Reproducing Patch's nine

`picks/` holds the choices behind the creatures in Assets, and `picks/provenance.json` says what made each one: the
concept prompt and model, the mesh model, and each clip's video model, seed and exact prompt. `assets/<c>/` holds their
configs and skeletons, and `prompts/<c>/` their move prompts.

The repo doesn't hold their meshes, readers or clips, so there are two ways to use this:

- **The same method on your creature.** Follow the steps above with your own takes. This is what the tools are for.
- **Those exact files.** Given the same mesh and `motion.json` files, `tools/pack.sh` reproduces a pack exactly. We rebuilt
  the T-rex and its GLB matched byte for byte. From scratch, you won't get the same creature: seeds don't promise the
  same result across model versions, so expect to reroll and pick again.

Two things happen after the pack when a creature goes into Patch's Assets: the horse's `Trot` is renamed `Run` (its run
is a trot), and materials and textures are renamed after the creature. The eagle isn't listed, since only one of its
clips was good.
