from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

from filelock import FileLock

from tts_audiobook_tool import web_model_setup as setup
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class TestModelProfiles(unittest.TestCase):
    def test_catalog_covers_local_models_and_uses_upstream_requirements(self):
        profiles = {profile.catalog_id: profile for profile in setup.model_profiles()}
        local_models = {model.id for model in TtsModelType.all() if model.value.local_module_test}
        self.assertEqual(set(profiles), local_models)
        for profile in profiles.values():
            requirements = setup.ROOT / profile.requirements
            self.assertTrue(requirements.is_file())
            contents = requirements.read_text(encoding="utf-8").replace("Python v", "Python ")
            self.assertIn(f"Python {profile.python_version}", contents)
        self.assertEqual(profiles["breeze_tts_2_local"].batch_size, 10)
        self.assertEqual(profiles["qwen3tts_local"].batch_size, 5)
        self.assertEqual(profiles["fish_s2_local"].batch_size, 1)
        self.assertEqual(profiles["mira_local"].batch_size, 10)
        self.assertEqual(profiles["vibevoice_local"].batch_size, 10)

    def test_batch_targets_respect_free_vram_validation_and_variant(self):
        gib = 1024**3
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with patch.object(setup, "get_nv_vram", return_value=(2 * gib, 24 * gib)):
                self.assertEqual(setup.recommended_batch_size("venv-breeze"), 54)
                self.assertEqual(setup.recommended_batch_size("venv-qwen3tts"), 5)
                self.assertEqual(setup.recommended_batch_size("venv-moss"), 1)
                target = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
                self.assertEqual(setup.recommended_batch_size("venv-moss", model_target=target), 1)
                self.assertEqual(setup.recommended_batch_size("venv-moss", model_target=target, validate_audio=False), 20)
                self.assertEqual(setup.recommended_batch_size("venv-fish-s2"), 1)
            # Driver reporting may put a 24 GiB card slightly below 24 GiB total.
            with patch.object(setup, "get_nv_vram", return_value=(3500 * 1024**2, 24564 * 1024**2)):
                self.assertEqual(setup.recommended_batch_size("venv-breeze"), 54)
                self.assertEqual(setup.recommended_batch_size("venv-breeze", validate_audio=False), 54)
            with patch.object(setup, "get_nv_vram", return_value=(7 * gib, 24 * gib)):
                self.assertEqual(setup.recommended_batch_size("venv-breeze"), 10)
            with patch.object(setup, "get_nv_vram", return_value=(11 * gib, 24 * gib)):
                self.assertEqual(setup.recommended_batch_size("venv-breeze"), 1)
                self.assertEqual(setup.recommended_batch_size("venv-breeze", validate_audio=False), 10)
            with patch.object(setup, "get_nv_vram", return_value=None):
                self.assertEqual(setup.recommended_batch_size("venv-qwen3tts"), 1)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            with patch.object(setup, "get_nv_vram", return_value=(0, 24 * gib)):
                self.assertEqual(setup.recommended_batch_size("venv-breeze"), 1)


@unittest.skipUnless(setup.model_profiles()[0].installable, "Browser setup runs on x86-64 Linux/WSL.")
class TestModelInstallation(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name)
        self.repo = workspace / "checkout"
        self.repo.mkdir()
        self.private = workspace / "private-models"
        profile = next(profile for profile in setup.model_profiles() if profile.model_id == "venv-breeze")
        self.link = self.repo / profile.model_id
        self.requirements = self.repo / profile.requirements
        self.requirements.write_text(
            "# Original requirements\n--only-binary=av,soundfile\nav==18.1.0\n", encoding="utf-8",
        )
        self.steps = []
        self.temporary_requirements = []
        self.stage = None
        for target, value in (("ROOT", self.repo), ("INSTALL_ROOT", self.private)):
            patcher = patch.object(setup, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(setup, "_find_uv", return_value="/local/bin/uv")
        self.find_uv = patcher.start()
        self.addCleanup(patcher.stop)

    def _record_step(self, argv, description, on_progress, cancel_event, **kwargs):
        self.assertFalse(self.link.is_symlink())
        self.steps.append(argv)
        if argv[1] == "venv":
            self.stage = Path(argv[-1])
            (self.stage / "pyvenv.cfg").write_text("home = /fake", encoding="utf-8")
        if "-r" in argv:
            self.temporary_requirements.append(Path(argv[-1]).read_text(encoding="utf-8"))
        if on_progress is not None:
            on_progress(description)

    def test_publishes_only_after_verification_and_checkpoint(self):
        progress = []

        def record_progress(update):
            self.assertEqual(self.link.is_symlink(), update.completed == update.total)
            progress.append(update)

        with patch.object(setup, "_run_step", side_effect=self._record_step):
            with patch.object(setup, "_qualified", return_value=True) as qualified:
                result = setup.install_model("venv-breeze", on_progress=record_progress)
        self.assertEqual(result, self.link)
        self.assertEqual(self.link.resolve(), self.stage)
        self.assertEqual(self.stage.parent, self.private)
        self.assertEqual(self.stage.stat().st_mode & 0o777, 0o700)
        self.assertFalse((self.stage / ".upstream-requirements.txt").exists())
        self.assertEqual(self.requirements.read_text(encoding="utf-8").count("--only-binary="), 1)
        self.assertEqual(self.temporary_requirements, ["# Original requirements\nav==18.1.0\n"])
        self.assertEqual(self.steps[0][-3:], ["--python", "3.12", str(self.stage)])
        self.assertIn("https://download.pytorch.org/whl/cu126", self.steps[1])
        self.assertIn("torch==2.8.0+cu126", self.steps[1])
        self.assertEqual(self.steps[2][-6:-2], ["--only-binary", "av", "--only-binary", "soundfile"])
        self.assertEqual(self.steps[4][-2:], ["3.12", "2.8.0+cu126"])
        self.assertEqual(self.steps[-1][-2], "BreezeBlue/Breeze-TTS-2")
        total = len(self.steps) + 1
        self.assertEqual([(item.completed, item.total) for item in progress], [(i, total) for i in range(total + 1)])
        self.assertIn("ETA measuring", progress[0].description)
        self.assertRegex(progress[1].description, r"ETA \d")
        self.assertNotIn("ETA", progress[-1].description)
        self.assertIn("Download", progress[-3].stage)
        self.assertIn("Publish", progress[-2].stage)
        self.assertEqual(qualified.call_args.args[0], self.stage)

    def test_recipe_counts_access_check_and_each_checkpoint(self):
        profile = next(item for item in setup.model_profiles() if item.model_id == "venv-breeze")
        checkpoint = replace(profile.checkpoints[0], repo_id="example/second-checkpoint")
        profile = replace(profile, gated_repo="example/gated", checkpoints=(*profile.checkpoints, checkpoint))
        progress = []
        with (
            patch.dict(setup._BY_MODEL_ID, {profile.model_id: profile}),
            patch.object(setup, "_run_step", side_effect=self._record_step),
            patch.object(setup, "_qualified", return_value=True),
        ):
            setup.install_model(profile.model_id, on_progress=progress.append)

        total = len(self.steps) + 1
        self.assertEqual([(item.completed, item.total) for item in progress], [(i, total) for i in range(total + 1)])
        self.assertIn("example/gated", progress[0].stage)
        self.assertIn(checkpoint.repo_id, progress[-3].stage)

    def test_cancellation_before_publication_does_not_report_completion(self):
        cancelled = Event()
        progress = []

        def record_progress(update):
            progress.append(update)
            if update.stage.startswith("Publish"):
                cancelled.set()

        with (
            patch.object(setup, "_run_step", side_effect=self._record_step),
            patch.object(setup, "_qualified", return_value=True),
        ):
            with self.assertRaisesRegex(setup.WebModelSetupError, "stopped"):
                setup.install_model("venv-breeze", on_progress=record_progress, cancel_event=cancelled)

        self.assertEqual(progress[-1].completed, progress[-1].total - 1)
        self.assertFalse(self.link.exists())
        self.assertFalse(self.stage.exists())

    def test_failed_qualification_cleans_stage_and_allows_retry(self):
        with patch.object(setup, "_run_step", side_effect=self._record_step):
            with patch.object(setup, "_qualified", side_effect=[False, True]):
                with self.assertRaisesRegex(setup.WebModelSetupError, "did not qualify"):
                    setup.install_model("venv-breeze")
                first_stage = self.stage
                self.assertFalse(first_stage.exists())
                self.assertFalse(self.link.exists())
                setup.install_model("venv-breeze")
        self.assertNotEqual(first_stage, self.link.resolve())

    def test_remote_install_stages_native_marker_and_requires_launcher_verification(self):
        self.link = self.repo / "venv-remote-tts"
        source = Path(setup.__file__).resolve().parents[1]
        shutil.copyfile(source / "requirements-remote.txt", self.repo / "requirements-remote.txt")
        marker = self.repo / "launcher_markers" / "remote_client"
        shutil.copytree(source / "launcher_markers" / "remote_client", marker)
        progress = []
        staged_manifests = []

        def record(argv, description, on_progress, cancel_event, **kwargs):
            self._record_step(argv, description, on_progress, cancel_event, **kwargs)
            if "-r" in argv:
                staged_marker = Path(argv[-1]).parent / ".remote-client-marker"
                staged_manifests.append((staged_marker / "pyproject.toml").read_bytes())
                normalized = self.temporary_requirements[-1]
                self.assertIn(f"tts-audiobook-tool-remote-client-marker @ {staged_marker.as_uri()}", normalized)
                self.assertIn("torch==2.8.0", normalized)
                self.assertNotIn("./launcher_markers/remote_client", normalized)

        with (
            patch.object(setup, "_run_step", side_effect=record),
            patch("launch.has_remote_client_marker", side_effect=[False, True]) as native_probe,
        ):
            with self.assertRaisesRegex(setup.WebModelSetupError, "did not qualify"):
                setup.install_remote_client(on_progress=progress.append)
            failed_stage = self.stage
            self.assertFalse(failed_stage.exists())
            self.assertFalse(self.link.exists())
            result = setup.install_remote_client(on_progress=progress.append)

        self.assertEqual(result, self.link)
        self.assertEqual(result.resolve(), self.stage)
        self.assertEqual([args.args[0] for args in native_probe.call_args_list], [str(failed_stage), str(self.stage)])
        self.assertEqual(staged_manifests, [(marker / "pyproject.toml").read_bytes()] * 2)
        self.assertFalse((self.stage / ".remote-client-marker").exists())
        self.assertFalse((self.stage / ".upstream-requirements.txt").exists())
        self.assertEqual([item.completed for item in progress], [*range(4), *range(6)])
        self.assertTrue(all(item.total == 5 for item in progress))
        self.assertTrue(all("snapshot_download" not in " ".join(step) for step in self.steps))
        self.assertEqual(self.steps[-1][-1], "3.12")

    def test_remote_install_reuses_an_existing_marked_environment(self):
        existing = self.repo / "venv-client" / "bin" / "python"
        progress = []
        with patch.object(setup, "remote_client_python", return_value=existing):
            self.assertEqual(setup.install_remote_client(on_progress=progress.append), existing.parent.parent)
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0].completed, progress[0].total)
        self.find_uv.assert_not_called()
        self.assertFalse(self.private.exists())

    def test_existing_paths_and_duplicate_jobs_are_preserved(self):
        partial = self.repo / "unfinished"
        partial.mkdir()
        self.link.symlink_to(partial, target_is_directory=True)
        with self.assertRaisesRegex(setup.WebModelSetupError, "existing path occupies"):
            setup.install_model("venv-breeze")
        self.assertEqual(self.link.resolve(), partial)
        self.link.unlink()
        with FileLock(str(self.private / "venv-breeze.lock"), timeout=0):
            with self.assertRaisesRegex(setup.WebModelSetupError, "already running"):
                setup.install_model("venv-breeze")
        self.find_uv.assert_not_called()

    def test_rejects_forged_model_ids_and_symlinked_requirements(self):
        with self.assertRaisesRegex(setup.WebModelSetupError, "supported local speech model"):
            setup.install_model("../other")
        self.assertFalse(self.private.exists())
        redirected = self.repo.parent / "elsewhere.txt"
        self.requirements.rename(redirected)
        self.requirements.symlink_to(redirected)
        with self.assertRaisesRegex(setup.WebModelSetupError, "upstream requirements file"):
            setup.install_model("venv-breeze")
        self.assertFalse(self.link.exists())

    def test_rejects_setup_root_inside_checkout(self):
        nested = self.repo / "staged"
        with patch.object(setup, "INSTALL_ROOT", nested):
            with self.assertRaisesRegex(setup.WebModelSetupError, "private Linux directory"):
                setup.install_model("venv-breeze")
        self.assertFalse(nested.exists())

    def test_cancelled_child_is_stopped(self):
        cancelled = Event()

        def timeout(*args, **kwargs):
            cancelled.set()
            raise subprocess.TimeoutExpired("uv", 0.25)

        with patch.object(setup.subprocess, "Popen") as popen:
            child = popen.return_value.__enter__.return_value
            child.wait.side_effect = timeout
            with patch.object(setup, "_stop_process_group") as stop:
                with self.assertRaisesRegex(setup.WebModelSetupError, "stopped"):
                    setup._run_step(["uv", "version"], "Verify CLI", None, cancelled)
        stop.assert_called_once_with(child)
