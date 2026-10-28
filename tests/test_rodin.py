"""Offline API-flow checks: python -m unittest discover -s tests."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from mocap import rodin


class RodinTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = self.root / "concept.png"
        self.image.write_bytes(b"image")
        self.output = self.root / "assets/model.glb"
        self.task_file = self.root / "runs/rodin.json"
        self.task = {"uuid": "task-1", "jobs": {"subscription_key": "private-subscription"}}
        self.download = {"list": [{"name": "preview.webp", "url": "https://storage/preview"},
                                  {"name": "model.glb", "url": "https://storage/model?signed=true"}]}
        self.session = self.enterContext(patch.object(rodin.requests, "Session")).return_value.__enter__.return_value
        self.session.headers = {}
        self.get = self.enterContext(patch.object(rodin.requests, "get"))
        self.get.return_value.__enter__.return_value.iter_content.return_value = [b"glTF", b"mock model"]
        self.enterContext(patch.object(rodin, "key", return_value="vibecoding"))
        self.enterContext(patch.object(rodin.time, "sleep"))

    def responses(self, *data):
        self.session.post.side_effect = [MagicMock(ok=True, json=MagicMock(return_value=d)) for d in data]

    def generate(self, **kwargs):
        return rodin.generate(self.image, self.output, self.task_file, **kwargs)

    def save_task(self):
        self.task_file.parent.mkdir(parents=True)
        self.task_file.write_text(json.dumps(self.task))

    def test_submit_poll_download(self):
        self.responses(self.task, {"jobs": [{"status": "Running"}]},
                       {"jobs": [{"status": "Done"}]}, self.download)
        self.generate()
        submit, pending, done, download = self.session.post.call_args_list
        self.assertEqual(submit.args, (rodin.BASE + "/rodin",))
        self.assertEqual(submit.kwargs["data"]["geometry_file_format"], "glb")
        self.assertEqual(pending.kwargs["json"], {"subscription_key": "private-subscription"})
        self.assertEqual(download.kwargs["json"], {"task_uuid": "task-1"})
        self.assertEqual(self.output.read_bytes(), b"glTFmock model")
        self.assertEqual(self.task_file.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.session.headers["Authorization"], "Bearer vibecoding")
        self.assertNotIn("headers", self.get.call_args.kwargs)
        self.assertEqual(self.get.call_args.args[0], "https://storage/model?signed=true")

    def test_resume_never_submits(self):
        self.save_task()
        self.image.unlink()
        self.responses({"jobs": [{"status": "Done"}]}, self.download)
        self.generate(resume=True)
        self.assertEqual([call.args[0] for call in self.session.post.call_args_list],
                         [rodin.BASE + "/status", rodin.BASE + "/download"])

    def test_failed_or_canceled_jobs_stop(self):
        for state in ("Failed", "Canceled", "Cancelled"):
            with self.subTest(state=state):
                self.responses(self.task, {"jobs": [{"status": state}]})
                self.task_file.unlink(missing_ok=True)
                with self.assertRaises(RuntimeError):
                    self.generate()
                self.assertFalse(self.output.exists())
        self.get.assert_not_called()

    def test_empty_jobs_do_not_count_as_done(self):
        self.responses(self.task, {"jobs": []})
        with patch.object(rodin.time, "monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(TimeoutError, "--resume"):
                self.generate(timeout=1)
        self.assertTrue(self.task_file.exists())
        self.get.assert_not_called()

    def test_saved_task_or_existing_mesh_prevents_submission(self):
        self.save_task()
        with self.assertRaises(FileExistsError):
            self.generate()
        self.task_file.unlink()
        self.output.parent.mkdir(parents=True)
        self.output.write_bytes(b"existing model")
        with self.assertRaises(FileExistsError):
            self.generate()
        self.assertEqual(self.output.read_bytes(), b"existing model")
        self.session.post.assert_not_called()

    def test_invalid_download_does_not_leave_a_mesh(self):
        self.responses(self.task, {"jobs": [{"status": "Done"}]}, self.download)
        self.get.return_value.__enter__.return_value.iter_content.return_value = [b"<html>error</html>"]
        with self.assertRaisesRegex(RuntimeError, "not a GLB"):
            self.generate()
        self.assertFalse(self.output.exists())
        self.assertFalse(self.output.with_suffix(".glb.part").exists())

    def test_submission_error_is_not_retried(self):
        self.session.post.return_value = MagicMock(ok=False, status_code=429)
        with self.assertRaisesRegex(RuntimeError, "HTTP 429"):
            self.generate()
        self.assertEqual(self.session.post.call_count, 1)
        self.assertFalse(self.task_file.exists())

    def test_json_error_with_success_http_status_is_a_rejection(self):
        self.responses({"error": "API_PARALLELISM_LIMIT_REACHED"})
        with self.assertRaises(rodin.GenerationFailed):
            self.generate()
        self.assertFalse(self.task_file.exists())


class KeyTests(unittest.TestCase):
    def test_public_default_and_private_overrides(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(rodin, "ROOT", Path(temp)):
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(rodin.key(), "vibecoding")
                (Path(temp) / ".env").write_text("RODIN_API_KEY=file-key\n")
                self.assertEqual(rodin.key(), "file-key")
                with patch.dict(os.environ, {"RODIN_API_KEY": "env-key"}):
                    self.assertEqual(rodin.key(), "env-key")


if __name__ == "__main__":
    unittest.main()
