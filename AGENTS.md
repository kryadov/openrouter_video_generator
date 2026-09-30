# AGENTS.md

Guidance for coding agents working on **clipgen**. README.md is the user documentation. The maintainer works in Russian, so reply in Russian unless they switch languages.

## What the project does

It makes short-form vertical music videos (20–40 s) from a track in `input/<Track>/`, in two parts:

- **Part 1, `compare/`:** automatic excerpt, a default shot list, the same brief rendered by N OpenRouter video models (`COMPARE_MODELS`), and a labelled grid.
- **Part 2, `refine/`:** versioned shot lists on one model (`REFINE_MODEL`). Only changed shots are paid for; `--assemble-only` re-edits for free.

`legacy/` is the earlier local-GPU shader/SDXL visualizer. The maintainer rejected its output, so keep it working but don't extend it unless asked.

## Hard constraints

- **Cloud generation goes through OpenRouter only.** No fal.ai, Replicate or other providers, even when they offer something OpenRouter lacks. You can mention such a capability, but don't build for it.
- **Money.** Every paid call goes through `core/videogen.py` (ledger `output/_ledger.json`, cap `VIDEO_BUDGET_USD`). Always `--dry-run` first and state the cost before spending. Budgets come in small increments from the maintainer, so spend carefully and prefer fixes that need no generation (reorder, `window`, `stutter`, `pick` another seed, `--assemble-only`).
- **Never delete or overwrite paid shots** in `output/**/shots/` unless explicitly asked.
- **Minimal tokens, no LLM in the pipeline.** Shot lists are deterministic (`compare/autoshot.py`) or hand-edited JSON.
- **Secrets.** `.env` is gitignored, and `.env.example` documents every key. Never print the API key.
- **Slow network.** Don't run speed tests or add large downloads without saying so.

## Layout

```
core/config.py     .env + env-var settings (key, budget, model lists, resolution, encoder, cast)
core/analyze.py    audio -> 60 fps features, tempo (fine-grid refined), beat/bar grid, sections (cached npz + json)
core/excerpt.py    short-form excerpt scoring -> excerpt.json (+ m4a previews)
core/videogen.py   OpenRouter /api/v1/videos client: estimate, submit (budget check), poll, download, ledger
core/shotlist.py   shot-list model: load (Ctx), {placeholder} prompts, bar timing, cache paths, generate
core/edit.py       beat-fitted trims, stutter cuts, pre-roll/tail, final assembly, comparison grid
core/encode.py     NVENC-if-working else libx264
compare/           part 1 CLI (__main__) + default shot list builder (autoshot)
refine/            part 2 CLI (__main__) + styles/ (approved looks as reusable JSON)
legacy/            old local GPU pipeline (clipgen, render, keyframes, planner, style, presets, shaders)
input/, output/    user data (gitignored)
```

## Invariants

- **Time base:** 60 fps analysis. Bars are `beats[::4]`. A shot's `bars` index the bar grid from the excerpt's first bar; `tail_bars` extend past the excerpt with fades.
- **Shot cache key** = md5(model, fully substituted prompt [+ first-frame size], gen seconds, aspect, resolution, seed), and the path is `<shotlist dir>/shots/<model>/<id>_s<seed>_<key>.mp4`. Changing a shared placeholder re-keys every shot that uses it. When building a new version that reuses shots, freeze their prompts as literals (`prompt_of(old_sl, shot)`) and keep their `id`.
- **Generation seconds** = the smallest supported duration ≥ shot length + 0.35 s (`supported_durations` from the models endpoint).
- **Determinism:** hash with md5 or crc32, never Python `hash()`.
- **Output layout:** part 1 writes to `output/<Track>/compare/`, part 2 to `output/<Track>/short/`. The finished file is `<model-slug><take>.mp4`.

## Creative rules (maintainer feedback, enforce by default)

- Casting follows `VIDEO_CAST` from the local `.env` (the maintainer's rule lives there); always state casting explicitly in prompts.
- No visible singing mouths (text-to-video can't lip-sync) and no burned-in lyric captions.
- The opening frame must be the most striking image. No back-view openers.
- Match the energy: peak choruses get rhythmic material, never romance or calm holds. Calm songs get long, slow shots.
- Vary locations and framing between neighbouring shots, and keep one visual climax.
- No abrupt endings (tail bar + fades).
- Describe arrangements instead of counts. Check hands on instruments or buttons, and odd artifacts (lights on clothes, back-and-forth camera swings).
- An approved look must be preserved. Save it under `refine/styles/`.

## Adding things

- **New model:** nothing to code as long as OpenRouter lists it. Pricing is read from `pricing_skus` (`estimate()` handles per-second, per-token (Seedance) and cents SKUs). Add a new SKU shape there if needed.
- **New edit feature:** add a per-shot field and handle it in `core/edit.assemble`, then document it in README's shot-list reference.
- **New default shot vocabulary:** edit `compare/autoshot.py` (`ENERGETIC` / `CALM`).

## Verifying changes (free)

- `python -m compare "input/<Track>" --dry-run` runs analysis, excerpt, shot list and pricing without spending.
- `python -m refine "<Track>" <shotlist> --dry-run` must report `$0.00` when nothing changed.
- `python -m refine "<Track>" <shotlist> --assemble-only --take _check` re-edits from the cache. Check the result with `ffprobe` (1080×1920, h264 + aac, expected duration) and grab frames with `ffmpeg -ss T -frames:v 1`.
- Test the CPU path with `VIDEO_ENCODER=x264`.

## Environment notes

- Windows: `.venv\Scripts\python`, and the console is cp1252, so CLIs reconfigure stdout to UTF-8. Track names can contain spaces and Cyrillic, so quote paths.
- `truststore` is injected in `core/videogen.py` (TLS-inspecting proxies and antivirus).
- NVENC may be listed by ffmpeg without a usable GPU. `core/encode.py` test-encodes before using it.
- `legacy/` needs `GpuPreference=2` for python.exe on Optimus laptops, or moderngl picks the iGPU.
