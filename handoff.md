# Audiobook generator handoff

Continue on the `gradio-interface` branch of `anethema/tts-audiobook-tool`. The current work adds automatic recovery for local audiobook renders after a batch timeout, an unhealthy model, or an unexpected worker exit. The 180-second deadline that stopped the long Breeze TTS 2 render covers the whole synthesis and validation batch.

## Set up the next machine

Use Linux or WSL with NVIDIA CUDA available (`nvidia-smi`), Python 3.12, `uv`, and FFmpeg. From a shell on that machine:

```bash
git clone --branch gradio-interface https://github.com/anethema/tts-audiobook-tool.git
cd tts-audiobook-tool
uv venv --python 3.12 venv-breeze
uv pip install --python venv-breeze/bin/python -r requirements-breeze.txt -r requirements-gradio.txt
venv-breeze/bin/python -m tts_audiobook_tool.gradio_app --port 7862 --terminal-log
```

Open `http://127.0.0.1:7862` locally. Allow Breeze and Whisper to download their model files as needed. Configure the dialogue analysis endpoint and API key locally if you need to run analysis again. Saved project snapshots omit the API key.

## Move the current book privately

The original run stopped at 2,849 out of 50,604 segments, then resumed in its temporary Gradio workspace. The private transfer archive captures 3,972 completed lines. GitHub carries the app and this handoff. Transfer the project, dialogue analysis, voice references, and audio checkpoints through private storage.

1. When generation is stopped and the original book is still open, use **Saved projects → Save new project**. Use **Open current project folder** to locate the snapshot. Saving a new project from the original tab copies matching render directories into `render/renders/` alongside the snapshot's text and voice assets. Verify that the saved project includes the latest run before closing the original app.
2. Copy the entire ID-named saved project folder to `~/.local/share/tts-audiobook-tool/web-projects/` on the destination, keeping its original directory name. On Linux or WSL, create the root with `mkdir -p -m 700 ~/.local/share/tts-audiobook-tool/web-projects` and ensure it is private (`chmod 700`).
3. Open the transferred project via **Saved projects → Open project**. Use the same Breeze TTS 2 model, text and segmentation, cast and reference clips, delivery settings, and validation settings as the stopped run. Start with batch size **54**, keep **Generate new takes for this range** off, and select **Generate or resume audiobook**. Check the console's `Selected segments include … reused` line before committing to the long run.

If the temporary project can no longer be saved from the original tab, use the private `audiobook-checkpoint-transfer.tar` backup from the source Desktop. It contains the original render directory, FLAC files, matching timing sidecars, and dialogue files. Extract to a private scratch directory. Recreate or restore the matching browser project, then copy the entire hash-named directory under the archive's `renders/` into that project's `render/renders/`. Retain the original hash directory name. It is derived from the book and render settings. The archive is a checkpoint backup and does not contain the browser's full saved-project snapshot.

## Recovery behavior and verification

- On a recoverable failure, the browser render restarts the model worker and submits only unfinished lines. It halves the batch size each time, down to one, and keeps the original render directory. A retry budget of two resets each time more audio is saved. If recovery stalls, the log shows the last attempted and outstanding lines.
- An isolated CPU test stalled two real worker processes with synthetic audio, including the first batch of a replacement worker. It kept four saved takes, stepped batch size from four to two to one, and exported a valid M4B with all ten lines. A real CUDA/Whisper retry on the full book remains to be verified.
- For targeted regression checks, run `pytest tests/ui/test_web_render_recovery.py tests/ui/test_web_render.py tests/ui/test_web_generation_telemetry.py tests/workers/test_gen_timeout.py` using the environment described in `docs-dev/test-routing.md`. The broader model-worker suite has five existing failures in an exact field-list assertion that omits the existing `aac_bitrate` field.
