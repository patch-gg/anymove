"""A concept picture and a textured 3D model for a new creature, on fal. Costs money: say so first.

python tools/make_mesh.py image NAME TAG "prompt" [aspect=4:3]   -> runs/NAME/concepts/TAG.png, about $0.11 (FLUX 3)
python tools/make_mesh.py mesh NAME IMAGE.png TAG                -> runs/NAME/meshes/TAG.glb, about $0.40 (Tripo H3.1)
Count the legs on the mesh, not the picture, then copy the one you keep to assets/NAME/model.glb. For many-legged
creatures, a steep top-down 3/4 concept that says how many legs are on each side comes back with the right count.
"""
import json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mocap import fal  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGE = "blackforestlabs/flux-3/text-to-image"
MESH = "tripo3d/h3.1/image-to-3d"


def image(name, tag, prompt, aspect="4:3"):
    out = ROOT / "runs" / name / "concepts" / f"{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = fal.submit(IMAGE, {"prompt": prompt, "resolution": "1.5k", "aspect_ratio": aspect, "output_format": "png"})
    res, info = fal.wait(sub, timeout=900)
    fal.download(res["images"][0]["url"], out)
    out.with_suffix(".json").write_text(json.dumps({"endpoint": IMAGE, "prompt": prompt, "response": res}, indent=1))
    print("image", out, round(info["wall_s"]), "s")


def mesh(name, img, tag):
    out = ROOT / "runs" / name / "meshes" / f"{tag}.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    sub = fal.submit(MESH, {"image_url": fal.upload(img), "geometry_quality": "detailed", "texture_quality": "detailed",
                            "pbr": True, "texture_alignment": "original_image", "face_limit": 60000})
    res, info = fal.wait(sub, timeout=1800)
    out.with_suffix(".json").write_text(json.dumps({"endpoint": MESH, "image": str(img), "response": res}, indent=1))
    glb = res.get("pbr_model") or res.get("model_mesh") or res.get("base_model")
    fal.download(glb["url"] if isinstance(glb, dict) else glb, out)
    print("mesh", out, round(info["wall_s"]), "s")


if __name__ == "__main__":
    {"image": image, "mesh": mesh}[sys.argv[1]](*sys.argv[2:])
