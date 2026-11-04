#!/usr/bin/env python
"""
film_h3.py — the anymove film stage, running MiniMax-H3 locally on ComfyUI.

Same job as `film_matrix.py` (upstream `mocap/generate.py` sends the two-camera
reference frame to MiniMax-H3-Max), but with no paid API and no fal key: the
open-weights MiniMax-H3 denoiser is sampled by a local ComfyUI 0.43.0 instance.

The denoiser is loaded with ComfyUI's own `UNETLoader` from the int8_convrot
safetensors checkpoint by default. The Q2_K GGUF is installed and supported
(`--loader gguf`) but is ~58x slower per step in ComfyUI, because
ComfyUI-GGUF dequantizes Q2_K inside every forward pass instead of using
comfy_kitchen's fused int8_convrot kernel. Measured 2026-10-29 on this
RTX 3060 Ti: 3.74-3.85 s/step (native) vs 212-235 s/step (gguf) on the
identical graph.

Usage
-----
  python film_h3.py REF.png PROMPT.txt OUT_DIR
  python film_h3.py --creature wolf --move rear
  python film_h3.py --creature wolf --move rear --script prompts/wolf/dual/rear.txt
  python film_h3.py --creature wolf --move shake --loader gguf   # the slow route
  python film_h3.py --creature wolf --move shake --engine pika-flex
  python film_h3.py --creature wolf --move shake --engine pika-flex-r2v

Engines
-------
`--engine` selects a COMPLETE set of parts - base checkpoint, adapter, video
VAE, sampler, schedule and prompt shape - because half of one engine plus half
of another loads successfully and renders nonsense.

  base            the stock MiniMax-H3 route we have been filming: turbo LoRA,
                  res_multistep, a step count, plain prose prompts.
  pika-flex       Pika's H3 Flex adapters on the fl2va base: a FIXED knot
                  schedule (not a step count), the lcm sampler, the int8_convrot
                  video VAE, and Pika's structured prompt.
  pika-flex-r2v   the reference-to-video route: a different base checkpoint
                  (ref2va), a different adapter, and a reference image that
                  conditions the model instead of becoming the first frame.

The clip lands at OUT_DIR/<out-name>.mp4 with its request and response in
OUT_DIR/<out-name>.json, matching the upstream layout so `mocap.detect` and
`mocap.solve` read it unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_SERVER = "http://127.0.0.1:8189"

# The denoiser checkpoint depends on the loader, and so does the speed.
#
#   native -> ComfyUI's own UNETLoader with the int8_convrot safetensors.
#             comfy_kitchen's fused int8_convrot_linear makes this the FAST
#             route: measured 3.74-3.85 s/step on this card vs 212-235 s/step
#             for the GGUF on the identical graph (2026-10-29).
#   gguf   -> ComfyUI-GGUF UnetLoaderGGUF with the Q2_K GGUF. Same picture
#             quality, but ComfyUI-GGUF dequantizes Q2_K inside every forward
#             pass (eager PyTorch, no fused kernel - py-spy showed
#             dequantize_blocks_Q2_K in the hot stack), so the GGUF saves
#             download size, not compute. ~58x slower per step.
#
# The GGUF is kept installed and verified because it is the open-weights
# artefact that was asked for; it is simply not the route you film on.
UNETS = {
    "native": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "gguf": "minimax_h3_fl2va_pruned-Q2_K.gguf",
}

DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

# ---------------------------------------------------------------------------
# The engines
# ---------------------------------------------------------------------------
# An engine is a COMPLETE, self-consistent set of: the base checkpoint, the
# adapter, the video VAE, the sampler, the schedule and the prompt shape. They
# travel together, because taking half of one and half of another produces a
# graph that loads and then renders nonsense.
#
#   base          the stock MiniMax-H3 route: turbo LoRA, res_multistep, a
#                 step count, plain prose prompts. What we have been filming.
#   pika-flex     Pika's H3 Flex adapters on the fl2va base: SIX STEPS on a
#                 FIXED knot schedule (not a step count), the lcm sampler,
#                 the int8_convrot video VAE (2.62 GiB instead of 4.85), and
#                 H3 Flex's structured prompt.
#   pika-flex-r2v the reference-to-video route: a different BASE checkpoint
#                 (ref2va), a different adapter, and a reference image that
#                 conditions the model instead of becoming the first frame.
#
# The Pika adapters are LoRA - they belong in models/loras/ and load through
# LoraLoaderModelOnly. They are native-only: ComfyUI-GGUF's loader does not
# take them, and the GGUF route is the base engine's business.
PIKA_SIGMAS = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"

ENGINES = {
    "base": {
        "node": "MiniMaxH3ImageToVideo",
        "unet": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "video_vae": "minimax_h3_video_vae_fp16.safetensors",
        "sampler": "res_multistep",
        "scheduler": "simple",
        "steps": 8,
        "sigmas": None,
        "prompt_style": "plain",
        "route": "first_frame",
    },
    "pika-flex": {
        "node": "MiniMaxH3ImageToVideo",
        "unet": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "lora": "pika_h3_flex_fl2va_v1_bf16.safetensors",
        "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
        "sampler": "lcm",
        "scheduler": None,
        "steps": 6,
        "sigmas": PIKA_SIGMAS,
        "prompt_style": "h3flex",
        "route": "first_frame",
    },
    "pika-flex-r2v": {
        "node": "MiniMaxH3ReferenceToVideo",
        "unet": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "lora": "pika_h3_flex_ref2va_v1_bf16.safetensors",
        "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
        "sampler": "lcm",
        "scheduler": None,
        "steps": 6,
        "sigmas": PIKA_SIGMAS,
        "prompt_style": "subject_definitions",
        "route": "reference",
    },
}

# The anymove composite canvas is 1280x720. H3 wants /32 dimensions, so the
# upstream-sized request would be 1280x736 (a 2.2% stretch, applied identically
# to both panels). We film at 864x480 instead - the official 0.4 MP setting -
# because the cost scales with the latent token count, and `mocap.detect`
# calibrates any resolution back to the 1280x720 layout before reading.
CANVAS_W = 1280
CANVAS_H = 736
FPS = 24.0


def frames_for(seconds: float) -> int:
    """H3's length grid is 17k+5 frames at 24 fps (5, 22, 39, ... 124, ...)."""
    n = max(5, round(seconds * FPS))
    n += (5 - n % 17) % 17
    return n


def post_json(url: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_json(url: str, timeout: int = 120) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def upload_image(server: str, path: Path) -> str:
    boundary = uuid.uuid4().hex
    blob = path.read_bytes()
    body = b""
    for k, v in (("overwrite", "true"), ("type", "input")):
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
             f"filename=\"{path.name}\"\r\nContent-Type: image/png\r\n\r\n").encode()
    body += blob + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{server}/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["name"]


# ---------------------------------------------------------------------------
# The H3 Flex prompt shapes
# ---------------------------------------------------------------------------
# Pika's H3 Flex models read a STRUCTURED prompt: the picture, the soundscape
# and the score are separate labelled blocks. The shapes below are taken from
# Pika's own published ComfyUI templates, not invented here - the labelled
# blocks are the instruction, and a bare paragraph makes the model guess which
# half of the text is the picture and which half is the sound.


def h3flex_prompt(description: str, soundscape: str | None = None,
                  music: str | None = None, alignment: str | None = None) -> str:
    """The H3 Flex prompt for text-to-video and first-frame-to-video."""
    blocks = []
    if alignment:
        blocks.append(alignment.strip())
    blocks.append(f"integrated_multimodal_description: [Shot 1] {description.strip()}")
    if soundscape:
        blocks.append(f"overall_soundscape: {soundscape.strip()}")
    if music:
        blocks.append(f"non_diegetic_music: {music.strip()}")
    return "\n\n".join(blocks)


def first_frame_alignment() -> str:
    """How the reference picture maps onto the target video's timeline.

    The alignment line is Pika's own: it tells the model that the picture it
    was given is the 0.00-second mark of the shot it is about to generate.
    """
    return ("How the reference pictures align with the target video - "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the "
            "target video.")


def _first_sentence(text: str, limit: int = 240) -> str:
    """The opening sentence of a description, for the summary block."""
    flat = " ".join(text.split())
    for stop in (". ", "! ", "? "):
        idx = flat.find(stop)
        if 0 < idx <= limit:
            return flat[:idx + 1]
    return flat[:limit].rstrip(",;: ") + "."


def subject_definitions_prompt(subject: str, role: str, features: str,
                                description: str, soundscape: str | None = None,
                                music: str | None = None) -> str:
    """The H3 Flex reference-to-video prompt.

    The reference image does NOT become frame 1: it CONDITIONS the model, and
    this prompt says what the model should take from it. The blocks are the
    ones Pika's own reference template uses.
    """
    blocks = [
        "subject_definitions:\n"
        f"<Subject 1> is {subject} in <Picture 1>. Reference role: {role}. "
        f"Features to follow exactly: {features}.",
        "summary:\n"
        f"[reference generation] {_first_sentence(description)}",
        "retention_analysis:\n"
        f"<Subject 1> (appears in [Shot 1]): fully_preserved - {features}.",
        "detailed_description:\n"
        f"{description.strip()}",
    ]
    if soundscape:
        blocks.append(f"overall_soundscape: {soundscape.strip()}")
    if music:
        blocks.append(f"non_diegetic_music: {music.strip()}")
    return "\n\n".join(blocks)


def build_graph(args, image_name: str) -> dict:
    g: dict = {}
    if args.loader == "native":
        g["10"] = {"class_type": "UNETLoader",
                   "inputs": {"unet_name": args.unet, "weight_dtype": args.weight_dtype}}
    else:
        g["10"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": args.unet}}

    if args.lora and not args.no_turbo:
        g["11"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["10", 0], "lora_name": args.lora,
                              "strength_model": args.lora_strength}}
        model_ref = ["11", 0]
    else:
        model_ref = ["10", 0]

    # A SECOND adapter, chained after the first. The Licon VBVR LoRA is a
    # QUALITY adapter, not a speed adapter, so it stacks ON TOP of the engine's
    # own adapter rather than replacing it: the turbo LoRA keeps the 8-step
    # schedule and VBVR's weights are applied on top of the same base. Two
    # LoraLoaderModelOnly nodes in series is ComfyUI's own stacking form.
    if getattr(args, "extra_lora", None):
        g["11b"] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": model_ref,
                               "lora_name": args.extra_lora,
                               "strength_model": args.extra_lora_strength}}
        model_ref = ["11b", 0]

    g["12"] = {"class_type": "CLIPLoader",
               "inputs": {"clip_name": args.clip, "type": "minimax", "device": args.clip_device}}
    g["13"] = {"class_type": "VAELoader", "inputs": {"vae_name": args.video_vae}}
    g["14"] = {"class_type": "VAELoader", "inputs": {"vae_name": args.audio_vae}}
    g["15"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}

    if args.route == "reference":
        # REFERENCE TO VIDEO. The reference image does NOT become the first
        # frame: it conditions the model through ref_images, and the prompt
        # says what the reference contributes. This is the only route the base
        # MiniMax-H3 tree cannot do.
        #
        # The key is FLAT: `ref_images.ref_image_0`, not a nested dict. The
        # group is COMFY_AUTOGROW_V3 with prefix `ref_image_`, and ComfyUI's
        # API format addresses an autogrow member as `<group>.<member>` -
        # Pika's own reference template uses exactly this spelling.
        g["16"] = {"class_type": args.node,
                   "inputs": {"clip": ["12", 0], "vae": ["13", 0], "audio_vae": ["14", 0],
                              "ref_images.ref_image_0": ["15", 0],
                              "ref_image_size": args.ref_image_size,
                              "prompt": args.prompt,
                              "width": args.width, "height": args.height,
                              "length": args.length}}
    else:
        g["16"] = {"class_type": args.node,
                   "inputs": {"clip": ["12", 0], "vae": ["13", 0], "prompt": args.prompt,
                              "width": args.width, "height": args.height, "length": args.length,
                              "first_frame": ["15", 0]}}
    g["17"] = {"class_type": "BasicGuider",
               "inputs": {"model": model_ref, "conditioning": ["16", 0]}}
    g["18"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}}
    if args.sigmas:
        # Pika H3 Flex: a FIXED six-step knot schedule. A step count would be
        # meaningless here - the knots are the schedule.
        g["19"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": args.sigmas}}
    else:
        g["19"] = {"class_type": "BasicScheduler",
                   "inputs": {"model": model_ref, "scheduler": args.scheduler,
                              "steps": args.steps, "denoise": 1.0}}
    g["20"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": args.sampler}}
    g["21"] = {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["18", 0], "guider": ["17", 0], "sampler": ["20", 0],
                          "sigmas": ["19", 0], "latent_image": ["16", 1]}}
    g["22"] = {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["13", 0]}}
    g["23"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["21", 0], "vae": ["14", 0]}}
    g["24"] = {"class_type": "CreateVideo",
               "inputs": {"images": ["22", 0], "fps": FPS, "audio": ["23", 0]}}
    g["25"] = {"class_type": "SaveVideo",
               "inputs": {"video": ["24", 0], "filename_prefix": args.prefix,
                          "format": args.video_format}}
    return g


def find_prompt(creature: str, move: str) -> Path | None:
    for sub in ("", "heldout", "train"):
        p = REPO / "prompts" / creature / sub / f"{move}.txt"
        if p.is_file():
            return p
    return None


def apply_script(prompt: str, script: str, seconds: float | None = None) -> str:
    """Replace the loose 'Action: ...' clause with a timed beat script.

    Same rationale as film_matrix.apply_script: naming the times removes the
    room for the two panels to animate different actions.

    `seconds` also makes the shot length honest. The upstream prompt says "One
    continuous 5-second shot", which is wrong for a short local clip and would
    invite the model to compress a 5 s beat script into 1.6 s.
    """
    import re
    if seconds is not None:
        prompt = re.sub(r"One continuous \d+(?:\.\d+)?-second shot",
                        f"One continuous {seconds:g}-second shot", prompt)
    m = re.search(r"Action:\s*(.*?)\s*It stays on the same spot", prompt, re.S)
    if not m:
        return prompt
    return prompt[:m.start()] + "Action: " + script.strip() + "\n" + prompt[m.end():]


def main() -> int:
    ap = argparse.ArgumentParser(description="film an anymove creature with local MiniMax-H3 on ComfyUI")
    ap.add_argument("ref", nargs="?", type=Path, help="reference frame PNG")
    ap.add_argument("prompt_file", nargs="?", type=Path, help="prompt .txt")
    ap.add_argument("out_dir", nargs="?", type=Path, help="output directory for the clip")
    ap.add_argument("--creature", help="use the upstream layout: runs/<c>/ref/ref.png + prompts/<c>/<move>.txt")
    ap.add_argument("--move", help="move name (prompt under prompts/<creature>/<move>.txt)")
    ap.add_argument("--model", default="MiniMax-H3-local",
                    help="label recorded in the receipt")
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--script", type=Path, default=None,
                    help="timed beat script to use instead of the prompt's Action clause")
    ap.add_argument("--out-name", default=None, help="file name for the clip (default: the model name)")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--engine", default="base", choices=sorted(ENGINES),
                    help="the complete, indivisible set: base checkpoint + adapter + "
                         "video VAE + sampler + schedule + prompt shape. "
                         "base = stock MiniMax-H3 (what we have been filming). "
                         "pika-flex = Pika H3 Flex on the fl2va base. "
                         "pika-flex-r2v = Pika H3 Flex reference-to-video on ref2va.")
    ap.add_argument("--loader", default="native", choices=["gguf", "native"],
                    help="native = ComfyUI UNETLoader with the int8_convrot checkpoint "
                         "(fast: ~3.8 s/step measured); "
                         "gguf = ComfyUI-GGUF UnetLoaderGGUF with the Q2_K GGUF "
                         "(same quality, ~58x slower per step: dequant in the forward pass). "
                         "The Pika engines are native-only and force this.")
    ap.add_argument("--weight-dtype", default="default",
                    help="UNETLoader weight_dtype (native loader only)")
    ap.add_argument("--unet", default=None,
                    help="denoiser checkpoint; default follows the engine "
                         "(the base engine follows --loader)")
    ap.add_argument("--clip", default=DEFAULT_CLIP)
    ap.add_argument("--video-vae", default=None,
                    help="video VAE; default follows the engine (the Pika engines use "
                         "the int8_convrot VAE: 2.62 GiB instead of 4.85)")
    ap.add_argument("--audio-vae", default=DEFAULT_AUDIO_VAE)
    ap.add_argument("--lora", default=None,
                    help="adapter; default follows the engine")
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--no-turbo", action="store_true",
                    help="skip the adapter. REFUSED for the Pika engines, whose "
                         "adapter IS the engine - the request is refused, not ignored")
    ap.add_argument("--extra-lora", default=None,
                    help="a SECOND adapter, chained after the engine's own. The Licon "
                         "VBVR LoRA is a quality adapter (object consistency, no "
                         "duplication, no interpenetration), not a speed adapter, so "
                         "it stacks on top of the turbo LoRA instead of replacing it")
    ap.add_argument("--extra-lora-strength", type=float, default=1.0)
    ap.add_argument("--clip-device", default="cpu")
    ap.add_argument("--width", type=int, default=864,
                    help="H3 width; 864x480 is the official 0.4 MP setting and the "
                         "one this card can film at a sane speed")
    ap.add_argument("--height", type=int, default=480,
                    help="H3 height; the upstream fal clip is 1344x768, but the reader "
                         "calibrates any resolution back to the 1280x720 layout")
    ap.add_argument("--steps", type=int, default=None,
                    help="step count; default follows the engine. IGNORED by the Pika "
                         "engines - their schedule is a fixed knot list, not a count")
    ap.add_argument("--sampler", default=None, help="default follows the engine")
    ap.add_argument("--scheduler", default=None, help="default follows the engine")
    ap.add_argument("--sigmas", default=None,
                    help="fixed knot schedule; default follows the engine")
    ap.add_argument("--ref-image-size", default="match", choices=["match", "max"],
                    help="reference-to-video only. match = scale the reference down to "
                         "the generation's pixel area; max = 2048 px short edge, best "
                         "identity fidelity but 'several times slower'")
    ap.add_argument("--soundscape", default=None,
                    help="overall_soundscape: line for the H3 Flex prompt")
    ap.add_argument("--soundscape-file", type=Path, default=None,
                    help="read the overall_soundscape line from a .txt instead")
    ap.add_argument("--music", default=None,
                    help="non_diegetic_music: line for the H3 Flex prompt")
    ap.add_argument("--music-file", type=Path, default=None,
                    help="read the non_diegetic_music line from a .txt instead")
    ap.add_argument("--subject", default=None,
                    help="reference-to-video: what the subject IS, e.g. 'a grey wolf'")
    ap.add_argument("--subject-role", default=None,
                    help="reference-to-video: what the reference contributes, e.g. "
                         "'identity and form anchor'")
    ap.add_argument("--subject-features", default=None,
                    help="reference-to-video: the features to follow exactly")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default="anymove")
    ap.add_argument("--video-format", default="auto",
                    help="SaveVideo container: auto | mp4 | mkv | webm")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--dry-run", action="store_true",
                    help="build the graph, print it, and exit without uploading or "
                         "submitting anything")
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Resolve the engine: one indivisible set of parts.
    # -----------------------------------------------------------------------
    eng = ENGINES[args.engine]
    args.node = eng["node"]
    args.route = eng["route"]
    args.prompt_style = eng["prompt_style"]
    args.is_pika = args.engine.startswith("pika")

    if args.is_pika:
        # The Pika adapters are safetensors LoRAs. ComfyUI-GGUF's loader does not
        # take them, so the GGUF route is not merely slower here - it cannot
        # load the engine at all.
        if args.loader == "gguf":
            print("WARNING: --engine " + args.engine + " is native-only "
                  "(the Pika adapters are safetensors LoRAs, which the GGUF loader "
                  "cannot take); switching to the native loader.", file=sys.stderr)
            args.loader = "native"
        if args.no_turbo:
            print("ERROR: --no-turbo is refused for --engine " + args.engine +
                  ". The Pika adapter IS the engine - without it the base model "
                  "runs with the base schedule and the base prompt shape, which "
                  "is a different engine, not a cheaper one.",
                  file=sys.stderr)
            return 1

    # Every part defaults to the engine's own value. An explicit flag wins.
    if args.unet is None:
        args.unet = UNETS[args.loader] if args.engine == "base" else eng["unet"]
    if args.video_vae is None:
        args.video_vae = eng["video_vae"]
    if args.lora is None:
        args.lora = eng["lora"]
    if args.sampler is None:
        args.sampler = eng["sampler"]
    if args.scheduler is None:
        args.scheduler = eng["scheduler"]
    if args.steps is None:
        args.steps = eng["steps"]
    if args.sigmas is None:
        args.sigmas = eng["sigmas"]

    # The denoiser follows the loader unless one was named explicitly.
    if args.loader == "gguf" and not args.unet.endswith(".gguf"):
        print(f"WARNING: --loader gguf with a non-.gguf checkpoint ({args.unet}); "
              f"UnetLoaderGGUF will not find it. Did you mean --loader native?",
              file=sys.stderr)
    elif args.loader == "native" and args.unet.endswith(".gguf"):
        print(f"WARNING: --loader native with a .gguf checkpoint ({args.unet}); "
              f"UNETLoader ignores .gguf files. Did you mean --loader gguf?",
              file=sys.stderr)

    if args.sigmas:
        print(f"[film_h3] schedule: FIXED knots ({args.sigmas}) "
              f"- --steps is ignored on this engine")
    for attr, flag in (("soundscape", "soundscape_file"), ("music", "music_file")):
        src = getattr(args, flag)
        if src:
            p = Path(src)
            if not p.is_file():
                print(f"ERROR: --{flag.replace('_', '-')} points at a missing file: {p}",
                      file=sys.stderr)
                return 1
            setattr(args, attr, p.read_text(encoding="utf-8").strip())
    if args.subject is None and args.creature:
        args.subject = f"the {args.creature}"

    if args.creature and args.move:
        ref = REPO / "runs" / args.creature / "ref" / "ref.png"
        if not ref.is_file():
            ref = REPO / "assets" / args.creature / "ref" / "ref.png"
        prompt_file = find_prompt(args.creature, args.move)
        out_dir = REPO / "runs" / args.creature / "videos" / args.move / "s1"
    else:
        ref, prompt_file, out_dir = args.ref, args.prompt_file, args.out_dir
        if not (ref and prompt_file and out_dir):
            ap.error("give REF.png PROMPT.txt OUT_DIR, or --creature and --move")

    ref, prompt_file, out_dir = Path(ref), Path(prompt_file), Path(out_dir)
    if not ref.is_file():
        print(f"ERROR: no reference frame at {ref}", file=sys.stderr)
        return 1
    if not prompt_file.is_file():
        print(f"ERROR: no prompt at {prompt_file}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if args.script:
        script = Path(args.script).read_text(encoding="utf-8").strip()
        prompt = apply_script(prompt, script, seconds=args.seconds)
        print(f"[film_h3] beat script: {args.script}")

    args.length = frames_for(args.seconds)
    args.prefix = f"{args.prefix}/{args.out_name or args.model}"

    # The prompt the node actually receives, in the engine's own shape.
    if args.prompt_style == "h3flex":
        args.prompt = h3flex_prompt(
            description=prompt,
            soundscape=args.soundscape,
            music=args.music,
            alignment=first_frame_alignment(),
        )
    elif args.prompt_style == "subject_definitions":
        missing = [n for n in ("subject", "subject_role", "subject_features")
                   if not getattr(args, n)]
        if missing:
            print(f"ERROR: --engine {args.engine} needs "
                  + ", ".join("--" + m.replace("_", "-") for m in missing),
                  file=sys.stderr)
            return 1
        args.prompt = subject_definitions_prompt(
            subject=args.subject,
            role=args.subject_role,
            features=args.subject_features,
            description=prompt,
            soundscape=args.soundscape,
            music=args.music,
        )
    else:
        args.prompt = prompt

    print(f"[film_h3] engine   = {args.engine}  node={args.node}  route={args.route}")
    print(f"[film_h3] server   = {args.server}")
    print(f"[film_h3] ref      = {ref}")
    print(f"[film_h3] prompt   = {prompt_file}  [shape: {args.prompt_style}]")
    print(f"[film_h3] {args.width}x{args.height} frames={args.length} seed={args.seed}")
    print(f"[film_h3] schedule = {args.sigmas if args.sigmas else f'{args.sampler}/{args.scheduler}/{args.steps} steps'}")
    print(f"[film_h3] adapter  = {args.lora}")
    if getattr(args, "extra_lora", None):
        print(f"[film_h3] adapter2 = {args.extra_lora}  "
              f"(strength {args.extra_lora_strength}, chained AFTER the engine adapter)")
    print(f"[film_h3] video VAE= {args.video_vae}")

    if args.dry_run:
        graph = build_graph(args, "<DRY-RUN>")
        print("[film_h3] dry run: the graph below was NOT uploaded or submitted")
        print(json.dumps(graph, indent=1))
        return 0

    t0 = time.time()
    try:
        get_json(f"{args.server}/system_stats", timeout=30)
    except Exception as e:
        print(f"ERROR: ComfyUI not reachable at {args.server} ({e}).\n"
              f"       Start it with D:\\workspace\\ComfyUI_h3_portable\\run_h3.bat", file=sys.stderr)
        return 2

    image_name = upload_image(args.server, ref)
    graph = build_graph(args, image_name)
    resp = post_json(f"{args.server}/prompt", {"prompt": graph})
    if "error" in resp:
        print("[film_h3] PROMPT ERROR:", file=sys.stderr)
        print(json.dumps(resp.get("error"), indent=1)[:3000], file=sys.stderr)
        for node, err in (resp.get("node_errors") or {}).items():
            print(f"  node {node}: {json.dumps(err)[:800]}", file=sys.stderr)
        return 1

    pid = resp["prompt_id"]
    print(f"[film_h3] prompt_id={pid}")

    deadline = time.time() + args.timeout
    entry = None
    while time.time() < deadline:
        hist = get_json(f"{args.server}/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("completed") or entry.get("outputs"):
                break
            if status.get("status_str") == "error":
                print("[film_h3] execution error:", file=sys.stderr)
                print(json.dumps(status)[:2000], file=sys.stderr)
                return 1
        print(f"[film_h3] {time.time() - t0:7.1f}s", flush=True)
        time.sleep(10)

    if entry is None:
        print(f"[film_h3] TIMEOUT after {args.timeout}s", file=sys.stderr)
        return 3

    wall = round(time.time() - t0, 1)
    print(f"[film_h3] finished in {wall}s")

    out_name = args.out_name or args.model
    dest = out_dir / f"{out_name}.mp4"

    # Copy out of the ComfyUI output tree.
    comfy_out = Path(os.environ.get("H3_COMFY_OUTPUT", r"D:\workspace\h3-runs"))
    candidates = []
    for node_out in (entry.get("outputs") or {}).values():
        for key in ("videos", "gifs", "images"):
            for item in node_out.get(key, []) or []:
                fn = item.get("filename")
                sub = item.get("subfolder", "")
                if fn:
                    candidates.append(comfy_out / sub / fn)
    if not candidates:
        print("[film_h3] no output file in the result", file=sys.stderr)
        return 1
    src = candidates[0]
    if not src.is_file():
        print(f"[film_h3] output not found on disk: {src}", file=sys.stderr)
        return 1
    shutil.copy2(src, dest)

    receipt = out_dir / f"{out_name}.json"
    receipt.write_text(json.dumps({
        "model": args.model,
        "engine": args.engine,
        "node": args.node,
        "route": args.route,
        "prompt_style": args.prompt_style,
        "source": f"local ComfyUI 0.43.0 ({args.loader} loader, no paid API)",
        "loader": args.loader,
        "denoiser": args.unet, "text_encoder": args.clip,
        "video_vae": args.video_vae, "audio_vae": args.audio_vae,
        "lora": None if args.no_turbo else args.lora,
        "lora_strength": None if args.no_turbo else args.lora_strength,
        "extra_lora": getattr(args, "extra_lora", None),
        "extra_lora_strength": getattr(args, "extra_lora_strength", None),
        "width": args.width, "height": args.height, "frames": args.length, "fps": FPS,
        "steps": args.steps, "sampler": args.sampler, "scheduler": args.scheduler,
        "sigmas": args.sigmas,
        "ref_image_size": args.ref_image_size if args.route == "reference" else None,
        "seed": args.seed, "seconds": args.seconds,
        "reference_type": "reference_image" if args.route == "reference" else "first_frame",
        "ref": str(ref), "prompt_file": str(prompt_file),
        "script": None if not args.script else str(args.script),
        "prompt": prompt,
        "node_prompt": args.prompt,
        "soundscape": args.soundscape, "music": args.music,
        "subject": args.subject, "subject_role": args.subject_role,
        "subject_features": args.subject_features,
        "prompt_id": pid, "wall_s": wall,
        "comfy_output": str(src),
    }, indent=1), encoding="utf-8")

    print(f"[film_h3] wrote {dest} ({dest.stat().st_size / 2**20:.1f} MB)")
    print(f"[film_h3] wrote {receipt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
