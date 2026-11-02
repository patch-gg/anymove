---
name: anymove-capture
description: Turn a two-camera AI clip into a game-ready animation with anymove - read every joint with the creature's trained reader, check that the two views agree, fit the rig in both views at once, and export a GLB with renders and a side-by-side comparison. Covers the quality gates (paw agreement, reprojection error), what to do when a fit fails, and three-quarter showcase renders. Use when someone has an anymove clip to capture, asks whether a capture is good, wants the GLB or renders of a move, or a fit looks wrong.
license: Apache-2.0
compatibility: NVIDIA GPU for reading (a few GB), CPU or GPU for the fit, Blender 5.x for the GLB and renders, Python 3.11 venv from requirements.txt. Needs runs/<c>/reader.pt (anymove-reader) and runs/<c>/ref/ref.png (anymove-rig).
---

# anymove-capture: clip in, animation out

```bash
tools/capture.sh <c> runs/<c>/videos/roar/s1/h3max.mp4 roar
```

Writes `runs/<c>/clips/roar/`: `det.npz` (every joint in both views, every frame), `overlay.mp4` (the joints drawn on
the clip), `motion.json` (the fit: root path and rotation, Euler angles per bone, joints in metres),
`render/animation.glb` (the rig with the animation, for any engine), `render/persp` and `render/side` frames, and
`showcase.mp4` (the AI clip with joints, the rig in 3/4, and clip vs rig through the same side camera).

## What happens

1. **Read.** The clip is aligned to the reference frame (video models resize or crop the canvas: an affine fit by
   ECC; "ECC did not converge" means a plain stretch was used, so check the overlay), split into its two views, and
   the reader finds every joint in each (about 10 s for 5 s of video).
2. **Check the views agree.** A joint has the same height in both halves. Per foot (creature.json `"feet"`), the
   correlation of its height between the views over the clip is the paw agreement. If the halves run out of step in
   time, the front view is time-warped onto the side view (only when that cuts the height gap by 15% or more).
3. **Fit.** Blender's forward kinematics in torch, Euler angles bounded by the joint limits, fitted to both views at
   once. Heights come mostly from the side view (the front view foreshortens leg lifts), left-right from the front
   view. It starts from an analytic IK pose aimed at the triangulated joints and anneals a robust loss from 40 px to
   6 px, so big poses (sit, rear, slam) are found. Under a minute for 5 s (about 90 s for the dragon's roar on a CPU).
4. **Export.** Blender keys the rig, writes the GLB and renders the views.

## Is it good? Read the solve output

The dragon's roar, for example:

```
paw height agreement between views: {'fpaw_L': 1.0, 'fpaw_R': 1.0, 'hpaw_L': 0.78, 'hpaw_R': None, 'score': 0.93}
sync as generated: mean height gap 4.4 px, worst 8.6 px, best constant lag 0 frames
reprojection error px: {'front': 5.92, 'front_x_only': 2.55, 'side': 7.99}
```

- **Paw agreement score:** 0.8 or more, the views tell one story: keep it. 0.5-0.8: look at `showcase.mp4`, often one
  limb is off. Under 0.5: the video model animated the views differently; reroll the clip (anymove-film). `None` for
  a foot means it barely moved (planted feet in a roar); judge that clip by the reprojection error instead.
- **Reprojection error** (layout pixels, how far the fitted rig's joints land from the readings): front_x_only of 1-3
  px and side of 2-8 px were typical on good clips. Scale by the creature's size in the frame.
- Results so far: dragon take-off 1.00, roar 0.93, swipe 0.85, walk 0.81; spider rear-up 0.99, bite 0.96, pounce
  0.89, walk 0.36-0.49 (fails); demon wing beat 0.99, slam 0.84, roar 0.73 with feet planted and 3 px reprojection;
  wolf trot 0.94 with H3 Max.

## When a fit is wrong

- Legs crossed or a limb flipped side to side in the side view: the painted colours make labels reliable, so swap
  fixing is off. `python -m mocap.solve ... --fix-swaps` swaps near/far legs where heights say the reader mixed them
  up; it "fixed" correct labels on one clip, so only use it after looking. Its pairs come from creature.json's `paint`
  groups. If the video dropped the paint and the front view swaps feet, set `"fix_front_swaps": true` in creature.json.
- The fit stays standing while the clip sits or rears: check `overlay.mp4`; if the readings are right, the limits in
  rig.json are too tight for the pose.
- Feet slide or sink: contact joints come from creature.json `contacts`, or default to every painted-limb joint near
  the ground at rest. Check the solve output's `planted contacts` line. 1 to 2 cm a step is as good as it gets for now.
- A low-confidence wrong reading the fit rightly ignored still counts in `reproj_px`. `reproj_confident_px` leaves
  those out, so judge by it. Limbs behind the body read badly from the side, so `tools/score_takes.py` also reports
  `side_near`, the near side alone.
- One joint chain won't fit, such as a tail: `--joint-weight` and `--joint-sigma` weight it by name prefix, though that
  didn't rescue the T-rex's tail swipe.
- Run `--no-align` to skip the time warp when the views were already in step and it hurt.

Debug views: `python tools/fit_check.py VIDEO runs/<c>/clips/<name> out.jpg 10 40 80` (readings as rings, fitted
joints as dots, per frame) and `python tools/plot_heights.py runs/<c>/clips/<name>/det.npz out.png` (joint heights
per view over time: lags and swaps show up as curves that don't overlap). Both need `MOCAP_CREATURE` set.

## Showcase renders (3/4 view on a shadow-catcher floor)

```bash
MOCAP_CREATURE=assets/<c> blender -b -P blender/apply_motion.py -- assets/<c>/rigged/rigged.blend \
  assets/<c>/rigged/rig_info.json runs/<c>/clips/roar/motion.json runs/<c>/clips/roar/tile \
  --views persp --samples 16 --res 1152x648 --cam -5.4,-6.0,1.4 --target 0,-0.3,1.5 --lens 32
```

`tools/capture.sh` uses creature.json `"camera"` for its 3/4 renders, and without one frames the take itself
(`tools/frame_camera.py`). So the takes share one camera, save one that frames them all:
`python tools/frame_camera.py --write assets/<c>/creature.json runs/<c>/clips/*/motion.json`. Options take a space, never
`=`. Frame the creature to its whole clip, not its rest pose: read the joint heights from the render's `joints.json` (the
demon flew to 5.3 m), then check every frame's alpha bounding box. Cameras that
worked: dragon `--cam -5.4,-6.0,1.4 --target 0,-0.3,1.5`, spider `-3.3,-3.65,1.3 / 0,-0.1,0.78`, demon
`-6.4,-7.1,1.8 / 0,-0.1,2.1` (flying: `-7.4,-8.3,2.1 / 0,-0.1,2.6`). The shadow catcher leaves a faint denoiser haze
in the alpha; when compositing, drop alpha under 10/255. `--springs assets/wolf/fur.json` adds damped spring fur on
the wolf (bones from `blender/add_fur_bones.py`).
