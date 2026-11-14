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
- **`VIDEO_BUDGET_USD` is a lifetime cap on the ledger total across all tracks and keys**, not a per-key balance. Report it as "spent $X of $Y, $Z left, this run $N". If the maintainer switches to a new OpenRouter key, set the cap to ledger total + the new key's balance. Don't edit or reset the ledger.
- **Paid runs need an explicit go.** Before any run that costs money, show the maintainer the audio analysis, the shot list and the `--dry-run` cost, then wait for their "go". A budget answer is not a go. Don't put a paid run in the background while questions are still open.
- **An interrupted run is already paid.** Jobs marked `"status": "submitted"` in the ledger are billed. Never rerun them, which pays twice. Recover them for free by polling their `job_id`: build `{"id", "model", "out": Path, "polling_url": f"{videogen.API}/videos/{id}"}` from each submitted ledger entry and pass the list to `videogen.wait_all`.
- **Never delete or overwrite paid shots** in `output/**/shots/` unless explicitly asked.
- **Minimal tokens, no LLM in the pipeline.** Shot lists are deterministic (`compare/autoshot.py`) or hand-edited JSON.
- **Secrets.** `.env` is gitignored, and `.env.example` documents every key. Never print the API key.
- **Commit and push as you go.** After every change to code, README.md or AGENTS.md, commit it and push it to `origin main` without being asked (`git pull --ff-only` first). Never commit `.env`, `input/` or `output/`.
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

## Starting a new track (the maintainer's usual workflow; order matters)

1. **Budget status.** State the ledger total, the cap and what's left before proposing anything. Ask about details (location, hero, look) with concrete options.
2. **Analyse the wav and show the result.** Use the section list from `analysis/features_*.json` plus a per-bar table (rms, kick, bass, mid, high, air, kick count) built from the `.npz` over the excerpt and its neighbours. Name the drop hit, the build-ups, where new layers enter, the loudest bars and the natural ending.
3. **Send the excerpt audio for approval.** The maintainer approves by ear, not from a table. Cut the proposed excerpt with ffmpeg into `output/<Track>/excerpts/*.m4a`, on the bar grid with the tail fade, and list planned cut times relative to the clip start. If the maintainer's timecode is ambiguous (track time or clip time?), cut both variants and let them choose. The automatic excerpt must not end before the loudest bars.
4. **Write the shot list from the brief.** Read `input/<Track>/notes.txt` and `style.txt`. The `autoshot` default (stage, hands on instrument) ignores the brief. Put shot cuts on structural changes and the visual climax on the loudest bars of the chosen excerpt.
5. **Print every substituted prompt** with `prompt_of` and read it. Placeholders such as `{hero}` must be noun phrases that still read grammatically wherever they land. Put constraints like "bare hands, no weapon" in `rules`, not inside the noun phrase.
6. **Run the model probe first, never the full compare first.** `python -m compare "input/<Track>" --probe <shot id>` renders one shot of about 5 s on every `COMPARE_MODELS` model and builds `compare/probe.mp4`, a labelled 2×2 grid in one frame. It costs about $1.7 for 4 models instead of about $8. The maintainer picks the model from it. The full shot list then runs on that one model (part 2, `refine/`), not on all four.
   The maintainer has picked `bytedance/seedance-2.0-fast` (`REFINE_MODEL`) on every track so far. Recommend it up front, and treat the probe as a confirmation, not an open question.
7. **Reuse the probe shot in part 2.** Copy (don't move) the chosen model's probe clip from `compare/shots/<model>/` to `short/shots/<model>/`. Give that shot in `short/shotlist_v1.json` the same prompt and the same bar count, so that its cache key matches and it costs $0. Set `"model"` in the shot list.
8. Show the dry-run cost, then wait for the go.

## Pitfalls already hit

- **Visible cuts and samey extras (maintainer rejected v1 and v2 of Last of Trance).** Separate text-to-video shots jump in light, place and costume, and a shared `{enemies}` placeholder gives every shot the same cloned faces. Default for story clips: build the shot list as one continuous movement with `"chain": true` (see README). Describe different extras in each shot (age, build, hair, clothes) and add "no two people alike" to `rules`. Seedance still clones a described extra (twice "a dark-haired woman in a biker jacket"). Pin each extra to a place ("by the left pillar", "by the right wall"), say "only these attackers are inside, and they look nothing alike", and give every extra at least two features nobody else has (hair and clothing colour). Check the frames for twins before accepting a shot. Plan the hero's route as one natural path and state it in `rules`. Turns are fine (toward the camera, then a side tracking shot) when the space motivates them: a doorway, a bend in the path. The hero enters the frame naturally (he runs in at a human pace from the very first second; no empty intro) and leaves it naturally (he runs past the camera and out of frame at the very end of the shot, with the fade following at once; no empty hold). No pauses or idle holds anywhere: there is action on screen every second. The maintainer rejects heroes who stand waiting, drop from the sky, fly like rockets or jump into the lens ("neuro-slop"): make climaxes real stunts, not shockwaves. Never turn back the way he came, and keep the screen direction consistent (side views always left to right). v3 failed on this: the hero ran down the stairs, then the model put the next door at the top, so he ran back up. Keep the hero in motion in every shot, with no standing or walking holds except the final exit. A chain re-keys everything after a changed shot, so check the first shots before paying for the rest.

- **Content filters.** Veo filters scenes like "enemies lying defeated" and returns no output (cost $0, but the shot is lost). Show defeat indirectly: empty space, fog, the hero walking away.
- **Rounding of generation seconds.** A shot of 3.65 s becomes 3.65 + 0.35 = 4.0000x and rounds up to the next duration, which re-keys an already paid 4 s shot. When re-timing a shot list, run `plan_cost` per model, and pin `gen_seconds` on shots whose paid clip still covers the new length.
- **The `[auto excerpt]` line** in compare output is the scorer's candidate. The shot list's own `excerpt` (printed on the `[shotlist]` line) is what gets rendered.

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
