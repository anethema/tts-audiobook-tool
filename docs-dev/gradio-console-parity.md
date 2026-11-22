# Gradio console parity

The browser interface should cover each console workflow with the same project, audio, and model semantics. The console menus and their supporting modules are the reference for behavior. Use a disposable project and synthetic audio for browser checks. Reserve provider calls, model downloads, and expensive renders for cases that require them.

## Project and text

| Console function | Browser coverage |
| --- | --- |
| Open and create projects | Named browser projects and a copy action for current or recent console projects. |
| Create from current project settings | **Save new project** copies the current text, voice, analysis, model and generation settings, and available browser takes. |
| Create from ABR audio settings | Reads a native snapshot and creates a new project with its supported settings. Import source text and supply reference audio afterward. |
| Set or clear project language | Supported independently of the applied segmentation language. |
| Open project directory | Supported for saved projects, including Windows Explorer from WSL. |
| Import EPUB | Supported, including chapter structure. |
| Import text file and enter text manually | UTF-8 upload and pasted text use the native segmenter. |
| Maximum words, segmentation strategy, dialog segmentation | Available on import and resegmentation. |
| View and edit text lines | Section and line selection, within-section search, editing, split, eligible join, deletion and resegmentation. Bulk editing, whole-book search, staged undo and reordering remain pending. |
| Book title and EPUB formatting repair | Book and section titles are editable. Reapply formatting from the source EPUB and re-export matching takes to update the reader. |
| Word substitutions | Ordered substitutions persist with the project and apply in the native synthesis worker. |

## Models and voices

| Console function | Browser coverage |
| --- | --- |
| Select and install a local speech model | Discovers existing launch environments and offers setup recipes for the local catalog on x86-64 Linux and WSL. Setup and publication checks use mocked installs. |
| Select remote endpoint, refresh remote models, unload remote models | SGL-Omni and audio.cpp discovery and exact model selection. Supports audio.cpp unload. The installable remote client handles sample, audiobook and playback jobs. Verification used mocked servers and setup. |
| Model parameters and model-specific source or checkpoint selection | Catalog parameters and selected models' stock checkpoints. Additional custom sources and model-specific controls remain pending. |
| About model, Force CPU, and unload models | Catalog details and Force CPU are available. Local browser synthesis unloads its worker after each job. |
| Text sample and compare model outputs | Supported, including playback of individual results. |
| List, add, name, modify, reorder, and remove voices | Named library supports all actions and importing named console voices. |
| Set or clear sample and edit the transcript | Direct upload, microphone, automatic cached Whisper transcription, manual edit and retry. |
| Trim and play sample | Range selection, playback and edited transcript persist with the voice. |
| Delivery instructions by model | Stored on named voices and supplied to the active native model. |
| Stereo position, volume, and room reverb | Stored on named voices and applied by native generation and export. |
| LoRA | Adapter options in Speech Models for Breeze, OmniVoice and VibeVoice. |
| Assign voice selections to arbitrary lines | Inclusive line and range assignment from the named library, including clear and review. Console manual assignments copy where references are available. |
| Build character roster and attribute dialogue | Supported with OpenAI. Oversized roster requests are tracked as a separate follow-up. |
| Review attribution and map named characters to voices | Per-section correction, direct character references, and named library selection. |

## Generate and export

| Console function | Browser coverage |
| --- | --- |
| Generate audiobook, stop, and resume missing audio | Supported. |
| Select lines, review takes, preview, and regenerate a range | Inclusive range generation and regeneration with native take ranking and playback. |
| Batch size, validation tolerance, retries, and auto-concatenate | Available, including benchmark-based batch recommendations using current free VRAM. |
| Choose export line range | Supported for existing complete or partial renders. |
| Mark sections and bookmarks | Chapter split, join and title editing, plus export bookmarks. Multi-file exports can split at explicit lines or existing section boundaries. |
| Export AAC/M4B or FLAC | Supported with native ABR metadata. |
| Pause modulation, segment and section pauses, and silence limits | Native pause presets, minimum sentence pause and silence limits are selectable. |
| Section sound effect, phrase subdivisions, normalization and loudness | Available in Export options. |
| Upsampling, stereo reverb, room tone, and treble EQ | Available in Export options. Live LavaSR is unverified. |
| Open generated book in synchronized player | Served locally from the bundled reader and launched from the export history. |

## Playback, chat, preferences, and tools

| Console function | Browser coverage |
| --- | --- |
| Realtime playback, line range, and save output | Browser streaming for a selected range with cached-take reuse, temporary audio generation, and optional FLAC download. |
| Voice chat, microphone and text inputs, echo, prompts, streaming, and saved input/output | Session chat with editable microphone transcripts, typed messages, echo, custom prompts, streamed text and downloadable reply audio. |
| Whisper model and device | Runtime selectors for generation validation and CPU reference transcription. |
| LLM endpoint, model, key, and request settings | OpenAI-compatible endpoint, pasted key, model, reasoning effort and separate extra parameters for casting and chat. |
| AAC bitrate and imported voice silence limit | AAC bitrate is available for range export and as a saved automatic export preference. Named voices can cap internal silence in reference audio. |
| Save generation logs and contextual hints | Project generation logs can be captured and downloaded. Console hint preferences remain pending. |
| Enhance an existing audiobook with synchronized text | Available under Tools with a cached CPU Whisper model. Alignment tested with mocked transcription. |
| Convert EPUB to text | Available under Tools. |
| Concatenate audio files | Available under Tools. |
| Transcode app FLAC to M4B and preserve metadata | Available under Tools. |
| Speed up a voice sample | Available under Tools. |

## Verification

Record the console entry point, browser entry point, and observed outcome when a feature is ported. For persistence, create and reopen a project and verify the exported artifact or generated audio when the feature changes rendering. Test cancellation and failures where the console offers recovery.
