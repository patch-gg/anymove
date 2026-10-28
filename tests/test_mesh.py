"""Provider order, fallback and task recovery without calling either service."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import requests

from mocap import mesh, rodin


class MeshTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = self.root / "concept.png"
        self.image.write_bytes(b"image")
        self.output = self.root / "model.glb"
        self.task_file = self.root / "rodin.json"
        self.primary = self.enterContext(patch.object(mesh.rodin, "generate"))
        self.secondary = self.enterContext(patch.object(mesh, "tripo"))

    def generate(self, **kwargs):
        return mesh.generate(self.image, self.output, self.task_file, **kwargs)

    def test_primary_success_never_calls_fal(self):
        self.primary.return_value = self.output
        self.assertEqual(self.generate(), self.output)
        self.secondary.assert_not_called()

    def test_confirmed_failure_uses_tripo_after_primary(self):
        calls = []
        def failed(*args, **kwargs):
            calls.append("rodin")
            raise rodin.GenerationFailed("task failed")
        self.primary.side_effect = failed
        self.secondary.side_effect = lambda *a, **kw: calls.append("tripo")
        self.generate()
        self.assertEqual(calls, ["rodin", "tripo"])
        self.assertEqual(self.secondary.call_args.args[2], self.root / "rodin.tripo.json")

    def test_uncertain_task_or_download_error_does_not_submit_fallback(self):
        for error in (TimeoutError("pending"), requests.Timeout("unknown submission outcome"),
                      RuntimeError("status query or download unavailable")):
            with self.subTest(error=error):
                self.primary.side_effect = error
                with self.assertRaises(type(error)):
                    self.generate()
        self.secondary.assert_not_called()

    def test_resume_tripo_does_not_reenter_primary(self):
        (self.root / "rodin.tripo.json").write_text("{}")
        self.generate(resume=True)
        self.primary.assert_not_called()
        self.assertTrue(self.secondary.call_args.kwargs["resume"])

    def test_existing_mesh_or_task_blocks_new_generation(self):
        self.output.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            self.generate()
        self.output.unlink()
        (self.root / "rodin.tripo.json").write_text("{}")
        with self.assertRaises(FileExistsError):
            self.generate()
        self.primary.assert_not_called()
        self.secondary.assert_not_called()


class TripoTests(unittest.TestCase):
    def test_queue_contract_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output, state = root / "model.glb", root / "tripo.json"
            task = {"request_id": "fal-1", "status_url": "https://queue/status",
                    "response_url": "https://queue/result"}
            with patch.object(mesh.fal, "key"), patch.object(mesh.fal, "upload", return_value="https://input") as upload, \
                 patch.object(mesh.fal, "submit", return_value=task) as submit, \
                 patch.object(mesh.fal, "wait", return_value=({"model_urls": {"glb": {"url": "https://glb"}},
                                                              "model_mesh": {"url": "https://other"}}, {})), \
                 patch.object(mesh.fal, "download", side_effect=lambda url, path: path.write_bytes(b"glTFmodel")) as download:
                mesh.tripo(root / "image.png", output, state)
                self.assertEqual(submit.call_args.args[0], "tripo3d/h3.1/image-to-3d")
                self.assertEqual(submit.call_args.args[1]["image_url"], "https://input")
                self.assertFalse(submit.call_args.args[1]["quad"])
                self.assertEqual(download.call_args.args[0], "https://glb")
                self.assertEqual(json.loads(state.read_text()), task)
                output.unlink()
                mesh.tripo(root / "image.png", output, state, resume=True)
                self.assertEqual(submit.call_count, 1)
                self.assertEqual(upload.call_count, 1)

    def test_missing_fal_key_never_uploads_or_submits(self):
        with patch.object(mesh.fal, "key", side_effect=SystemExit("set FAL_KEY")), \
             patch.object(mesh.fal, "upload") as upload, patch.object(mesh.fal, "submit") as submit:
            with self.assertRaises(SystemExit):
                mesh.tripo(Path("image.png"), Path("out.glb"), Path("task.json"))
            upload.assert_not_called()
            submit.assert_not_called()
