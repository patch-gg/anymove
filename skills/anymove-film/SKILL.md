---
name: anymove-film
description: Make the two-camera AI clip for an anymove creature - write the move's prompt from creature.json, send the reference frame to MiniMax H3 Max (or H3, Seedance, FLUX 3 Max) on fal, and check the clip kept the split screen, the limb colours and both views in step before capture. Use when someone wants a new move for a rigged anymove creature, when clips come back with swapped colours, a moving camera or a broken split screen, or when choosing a video model for two-camera clips.
license: Apache-2.0
compatibility: Needs a fal.ai key (FAL_KEY in .env or the environment) and a creature already set up with anymove-rig (runs/<c>/ref/ref.png).
---

# anymove-film: one move, two cameras, one clip

The video model gets one image, `runs/<c>/ref/ref.png` (front view left, side view right, limbs painted), and a prompt
that pins everything except the action. Both views come from one generation, which is what keeps them in step.

## 1. Write the prompt

```bash
.venv/bin/python tools/make_prompt.py assets/<c> "the dragon rears up on its hind legs, spreads its wings wide and roars, then drops back onto all four legs." prompts/<c>/roar.txt
```

The template says: animate the exact model shown twice, front camera left and side camera right, locked off, same
moment frame for frame; keep the model as it is (`describe`); limbs colour-coded and never swapping (`colour_code`);
the body count (`count`); then the action; it stays on the same spot, centred in both views; one continuous 5 s shot
at natural speed; the cameras never move.

Write the action as one sentence that starts and ends in the rest pose ("..., then drops back onto all four legs").
That keeps the creature in frame and makes clips loop. Name the limbs the way `colour_code` does. Keep it on the spot:
walks and runs work as "on the spot like a video-game running loop". Add bans to `count` when the model drifts
("No flames, no smoke, no weapon." for the fire demon). `python tools/make_prompt.py --check` confirms every prompt
under `prompts/` still matches its creature.json.

## 2. Generate

Say the cost first: about $0.26 per 5 s clip with H3 Max (October 2026), one clip per model and seed, and about 7 clips
($1.80) for a usable move on average. Two seeds is a start, not a budget: 58% of captured clips passed the checks.

```bash
tools/film.sh <c> roar h3max 1,2      # runs/<c>/videos/roar/s1/h3max.mp4 and s2/; about 1-2 min each
```

Models (`mocap/generate.py`): `h3max` (MiniMax H3 Max, best sync of the views so far), `h3` (MiniMax H3, $0.30),
`h3maxturbo`, `seedance` (ByteDance Seedance 2.5, expensive: three 720p clips cost $7.10), `flux3max` (FLUX 3 Max). Two
seeds per move is a good default; capture both and keep the one whose views agree better.

## 3. Check the clip before capture

Look at a few frames (or the whole clip) and reject it if:

- the split screen breaks: a view zooms, pans or cuts, the divider moves, the creature walks out of a view;
- a limb's colour changes or swaps sides (plain H3 swapped them on 2 of 5 wolf clips; the reader then labels the
  wrong legs); the LoRA (anymove-lora) kept them on 5 of 5;
- the body count changes (an extra leg, a lost wing), or the creature turns to face another way;
- fire, smoke, dust or a weapon covers joints.

The real gate is numeric and comes in capture: the paw agreement between the views (anymove-capture). What the
clips taught so far:

- Views disagree in the legs, not in time. Seedance's trot played the front legs in anti-phase between the views
  (paw agreement -0.04) while H3 Max scored 0.94 on the same prompt.
- Big single gestures (rear up, roar, slam, take off, pounce) read best: dragon take-off 1.00, spider rear-up 0.99,
  demon wing beat 0.99. Eight-legged walking fails (0.36-0.49): the views disagree on which legs are up.
- Lying down and dying poses failed on the wolf: the reader's random training poses don't cover them.
