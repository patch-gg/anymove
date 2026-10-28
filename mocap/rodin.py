"""Hyper3D Rodin image-to-GLB client, used by mocap.mesh.

RODIN_API_KEY comes from the environment or .env; otherwise use the public vibecoding evaluation key.
Save the task before polling so an interrupted run can resume without generating again.
"""
import json, mimetypes, os, pathlib, time

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = "https://api.hyper3d.com/api/v2"


class GenerationFailed(RuntimeError):
    """A rejected request or terminal task failure permits another provider."""


def key():
    """Use a private key if configured, otherwise the public evaluation key."""
    value = os.environ.get("RODIN_API_KEY")
    env = ROOT / ".env"
    if not value and env.exists():
        value = next((line.split("=", 1)[1].strip() for line in env.read_text().splitlines()
                      if line.startswith("RODIN_API_KEY=")), None)
    return value or "vibecoding"


def post(session, endpoint, **kwargs):
    response = session.post(f"{BASE}/{endpoint}", timeout=120, **kwargs)
    failure = GenerationFailed if endpoint == "rodin" else RuntimeError
    if not response.ok:
        raise failure(f"Rodin {endpoint}: HTTP {response.status_code}; check API access and quota")
    data = response.json()
    if data.get("error"):
        raise failure(f"Rodin {endpoint}: {data['error']}")
    return data


def run(output, task_file, submit, resume=False, poll=5, timeout=1800):
    """Submit once, save the task, wait for packaging and download its GLB."""
    output, task_file = pathlib.Path(output), pathlib.Path(task_file)
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose another output or move it first")
    if poll <= 0 or timeout <= 0:
        raise ValueError("poll and timeout must be positive")
    if not resume and task_file.exists():
        raise FileExistsError(f"{task_file} already exists; use --resume or choose another task file")

    with requests.Session() as session:
        session.headers["Authorization"] = f"Bearer {key()}"
        if resume:
            task = json.loads(task_file.read_text())
        else:
            # Prepare the destination before submitting; do not retry a generation POST.
            task_file.parent.mkdir(parents=True, exist_ok=True)
            task = submit(session)
            # The subscription key is a credential; keep it out of console output.
            with open(task_file, "w", opener=lambda path, flags: os.open(path, flags, 0o600)) as saved:
                json.dump(task, saved)
        task_id = task["uuid"]
        subscription_key = task["jobs"]["subscription_key"]
        print(f"Rodin task {task_id}; saved in {task_file}", flush=True)
        deadline = time.monotonic() + timeout
        while True:
            status = post(session, "status", json={"subscription_key": subscription_key})
            states = [job["status"] for job in status["jobs"]]
            if any(state in {"Failed", "Canceled", "Cancelled"} for state in states):
                raise GenerationFailed(f"Rodin task {task_id} failed or was canceled")
            if states and all(state == "Done" for state in states):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Rodin task {task_id} is still pending; rerun with --resume")
            time.sleep(min(poll, remaining))
        files = post(session, "download", json={"task_uuid": task_id})["list"]

    model = next((item for item in files if item["name"].lower().endswith(".glb")), None)
    if model is None:
        raise RuntimeError(f"Rodin task {task_id} returned no GLB")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    try:
        # Signed storage downloads must not receive the API Authorization header.
        with requests.get(model["url"], stream=True, timeout=600) as response:
            if not response.ok:
                raise RuntimeError(f"GLB download: HTTP {response.status_code}; rerun with --resume")
            with partial.open("wb") as dest:
                for chunk in response.iter_content(1 << 20):
                    dest.write(chunk)
        with partial.open("rb") as downloaded:
            if downloaded.read(4) != b"glTF":
                raise RuntimeError("Downloaded file is not a GLB; rerun with --resume")
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Saved {output}; check its facing and scale with tools/prep.sh before placing joints")
    return output


def generate(image, output, task_file, resume=False, poll=5, timeout=1800):
    image = pathlib.Path(image)
    def submit(session):
        data = {"tier": "Gen-2.5-Medium", "mesh_mode": "Raw", "quality": "low",
                "geometry_file_format": "glb", "material": "PBR"}
        with image.open("rb") as source:
            files = {"images": (image.name, source, mimetypes.guess_type(image.name)[0] or "application/octet-stream")}
            return post(session, "rodin", files=files, data=data)
    return run(output, task_file, submit, resume=resume, poll=poll, timeout=timeout)
