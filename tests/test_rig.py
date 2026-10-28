"""Rodin custom skeleton and multipart request checks, without network calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from mocap import rig


class RigTests(unittest.TestCase):
    def test_mirrored_hierarchy_keeps_names_and_parent_indices(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "rig.json"
            path.write_text(json.dumps({"bones": [["body", "hips", "chest", None],
                                                  ["arm_L", "shoulder_L", "hand_L", "body"]]}))
            self.assertEqual(rig.hierarchy(path), [{"name": "body", "parent": -1},
                                                   {"name": "arm_L", "parent": 0},
                                                   {"name": "arm_R", "parent": 0}])

    def test_all_bundled_skeletons_have_unique_bones_and_one_root(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / "assets").glob("*/rig.json"):
            with self.subTest(creature=path.parent.name):
                nodes = rig.hierarchy(path)
                self.assertEqual(len({n["name"] for n in nodes}), len(nodes))
                self.assertEqual(sum(n["parent"] == -1 for n in nodes), 1)
                self.assertTrue(all(-1 <= n["parent"] < i for i, n in enumerate(nodes)))

    def request(self, **kwargs):
        with patch.object(rig.rodin, "run") as run, patch.object(rig, "hierarchy", return_value=[{"name": "root", "parent": -1}]), \
             patch.object(rig.rodin, "post", return_value={"uuid": "rig-task"}) as post:
            rig.generate("model.glb", "rig.json", "output.glb", "task.json", **kwargs)
            submit = run.call_args.args[2]
            submit(MagicMock())
            return post.call_args

    def test_asset_input_preserves_textures_without_upload(self):
        call = self.request(asset="source-task")
        self.assertEqual(call.args[1], "rigging")
        files = call.kwargs["files"]
        self.assertEqual(files["asset_id"], (None, "source-task"))
        self.assertEqual(files["bone_template"], (None, "custom"))
        self.assertNotIn("model", files)
        self.assertNotIn("image", files)
        self.assertNotIn("material", files)

    def test_uploaded_model_uses_none_or_pbr_with_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            model, image = Path(temp) / "model.glb", Path(temp) / "image.png"
            model.write_bytes(b"glTFmodel")
            image.write_bytes(b"image")
            for reference in (None, str(image)):
                with self.subTest(reference=reference), patch.object(rig.rodin, "run") as run, \
                     patch.object(rig, "hierarchy", return_value=[{"name": "root", "parent": -1}]), \
                     patch.object(rig.rodin, "post") as post:
                    rig.generate(model, "rig.json", "output.glb", "task.json", image=reference)
                    run.call_args.args[2](MagicMock())
                    files = post.call_args.kwargs["files"]
                    self.assertEqual(files["material"], (None, "PBR" if reference else "None"))
                    self.assertNotIn("asset_id", files)
                    self.assertEqual("image" in files, reference is not None)
                    self.assertTrue(files["model"][1].closed)
