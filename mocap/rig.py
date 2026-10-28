"""Rig a creature with Rodin, keeping rig.json's bone names for anymove's paint and solver.

python -m mocap.rig MODEL.glb RIG.json OUTPUT.glb TASK.json [--asset UUID] [--image IMAGE] [--resume]
An existing Rodin asset keeps its textures. Uploaded models need --image for new PBR textures; otherwise untextured.
"""
import json, pathlib, sys
from contextlib import ExitStack

import requests

from mocap import rodin


def hierarchy(rig_file):
    rig = json.loads(pathlib.Path(rig_file).read_text())
    bones = []
    for bone in rig["bones"]:
        bones.append(bone)
        if bone[0].endswith("_L"):
            bones.append([value[:-2] + "_R" if value and value.endswith("_L") else value for value in bone])
    names = [bone[0] for bone in bones]
    return [{"name": name, "parent": names.index(parent) if parent else -1} for name, _, _, parent in bones]


def generate(model, rig_file, output, task_file, asset=None, image=None, resume=False, poll=5, timeout=1800):
    def submit(session):
        data = {"bone_template": "custom", "hierarchy": json.dumps(hierarchy(rig_file)),
                "skinning_method": "shell", "geometry_file_format": "glb"}
        with ExitStack() as stack:
            files = {}
            if asset:
                data["asset_id"] = asset
            else:
                model_path = pathlib.Path(model)
                files["model"] = (model_path.name, stack.enter_context(model_path.open("rb")), "model/gltf-binary")
                data["material"] = "PBR" if image else "None"
                if image:
                    image_path = pathlib.Path(image)
                    files["image"] = (image_path.name, stack.enter_context(image_path.open("rb")))
            # asset_id mode still uses multipart/form-data, even without file uploads.
            files.update({name: (None, value) for name, value in data.items()})
            return rodin.post(session, "rigging", files=files)
    return rodin.run(output, task_file, submit, resume=resume, poll=poll, timeout=timeout)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 4:
        raise SystemExit(__doc__)
    options = {}
    for flag in ("--asset", "--image"):
        if flag in args:
            i = args.index(flag)
            if i + 1 >= len(args):
                raise SystemExit(__doc__)
            options[flag[2:]] = args[i + 1]
            del args[i:i + 2]
    if len(args) not in (4, 5) or (len(args) == 5 and args[4] != "--resume"):
        raise SystemExit(__doc__)
    try:
        generate(*args[:4], resume="--resume" in args[4:], **options)
    except (OSError, ValueError, KeyError, RuntimeError, requests.RequestException) as error:
        raise SystemExit("network request failed; use --resume if a task was saved"
                         if isinstance(error, requests.RequestException) else str(error)) from None
