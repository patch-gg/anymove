"""Make a textured creature mesh: Hyper3D first, fal Tripo after a confirmed failure.

python -m mocap.mesh IMAGE OUTPUT.glb TASK.json [--resume]
TASK.json saves the Rodin task; TASK.tripo.json saves the fal fallback, if used.
"""
import json, pathlib, sys

import requests

from mocap import fal, rodin

TRIPO = "tripo3d/h3.1/image-to-3d"


def tripo(image, output, task_file, resume=False, poll=5, timeout=1800):
    # Reuse the repository's FAL_KEY environment/.env handling and queue client.
    fal.key()
    if resume:
        task = json.loads(task_file.read_text())
    else:
        url = fal.upload(image)
        task = fal.submit(TRIPO, {"image_url": url, "texture": True, "pbr": True,
                                 "texture_quality": "detailed", "orientation": "align_image",
                                 "quad": False})
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(json.dumps(task))
    print(f"fal Tripo task {task['request_id']}; saved in {task_file}", flush=True)
    result, _ = fal.wait(task, poll=poll, timeout=timeout)
    # Prefer the explicit GLB variant. quad=False requests triangle/GLB output.
    model = (result.get("model_urls") or {}).get("glb") or result.get("model_mesh")
    if not model or not model.get("url"):
        raise RuntimeError("fal Tripo returned no model file")
    partial = output.with_suffix(output.suffix + ".part")
    try:
        fal.download(model["url"], partial)
        with partial.open("rb") as downloaded:
            if downloaded.read(4) != b"glTF":
                raise RuntimeError("fal Tripo returned a file that is not a GLB")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Saved {output} (fal Tripo)")
    return output


def generate(image, output, task_file, resume=False, poll=5, timeout=1800):
    image, output, task_file = pathlib.Path(image), pathlib.Path(output), pathlib.Path(task_file)
    if output.exists():
        raise FileExistsError(f"{output} already exists; move it before generating")
    if poll <= 0 or timeout <= 0:
        raise ValueError("poll and timeout must be positive")
    tripo_task = task_file.with_suffix(".tripo.json")
    if tripo_task.exists():
        if not resume:
            raise FileExistsError(f"{tripo_task} already exists; use --resume")
        return tripo(image, output, tripo_task, resume=True, poll=poll, timeout=timeout)
    # Check input before either provider, except when resuming an existing task.
    if not resume:
        with image.open("rb"):
            pass
    try:
        return rodin.generate(image, output, task_file, resume=resume, poll=poll, timeout=timeout)
    except rodin.GenerationFailed as error:
        print(f"{error}; falling back to fal Tripo (requires FAL_KEY; billed by fal)", flush=True)
    return tripo(image, output, tripo_task, poll=poll, timeout=timeout)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) not in (3, 4) or (len(args) == 4 and args[3] != "--resume"):
        raise SystemExit(__doc__)
    try:
        generate(*args[:3], resume="--resume" in args)
    except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException) as error:
        # Request exceptions may contain signed URLs; keep them out of the console.
        raise SystemExit("network request failed; use --resume if a task was saved"
                         if isinstance(error, requests.RequestException) else str(error)) from None
