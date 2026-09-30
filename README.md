# clipgen: short-form music videos from a track

Give it a song and it makes a 20–40 s vertical video for Shorts, Reels or TikTok. Cuts land on the beat, the excerpt is picked for the hook, and the footage comes from cloud video models on [OpenRouter](https://openrouter.ai).

Work happens in two parts:

1. **`compare`**: renders the same brief with 4, 6 or 8 video models and puts the results side by side, so you can pick a model.
2. **`refine`**: iterates on the chosen model through versioned shot lists. Only the shots you change are paid for again. Re-editing (reordering, trimming, changing timing) costs nothing.

Audio analysis, excerpt selection and editing all run locally. The only paid step is video generation, and a hard budget cap stops it before it can overspend.

```
input/<Track>/*.wav ─► analysis (tempo, beat/bar grid, sections)
                    ─► best short-form excerpt (hook first, loopable, phrase-aligned)
                    ─► shot list (JSON, editable)
                    ─► OpenRouter video models (cached per shot, budget-capped)
                    ─► bar-exact edit + beat-synced trims + excerpt audio with fade-out
                    ─► output/<Track>/{compare|short}/*.mp4
```

## System requirements

| | compare / refine (parts 1–2) | legacy/ (optional, local GPU) |
|---|---|---|
| OS | Linux, macOS, Windows | Windows or Linux |
| CPU / RAM | 2 vCPU, 4 GB RAM (a 4-minute track uses ~1–2 GB during analysis) | 8+ GB RAM |
| GPU | **not required**. An NVIDIA GPU, if present, is used for NVENC encoding | NVIDIA with ≥ 8 GB VRAM, CUDA 12.x, OpenGL 3.3 |
| Python | 3.10+ (developed on 3.14) | same |
| ffmpeg | 6+ with `libx264`; `libfreetype` for labels in the comparison grid | + `h264_nvenc` |
| Disk | ~200 MB per track (each 720p shot is ~2–5 MB) | + ~10 GB of models (PyTorch CUDA, SDXL) |
| Network | outbound HTTPS to `openrouter.ai`, ~3–5 MB download per shot | one-time model downloads |

**CPU-only Linux VM.** Fully supported for both parts. With `VIDEO_ENCODER=auto` the tool checks whether NVENC actually works by doing a test encode, and falls back to `libx264` when it doesn't. Rough timings on 2 vCPU:

- analysis: 10–30 s per track
- motion analysis: ~1 s per shot
- final 30 s encode: 20–60 s

Generation happens on OpenRouter's side and usually takes 1–3 minutes per batch of shots, whatever hardware you run on.

### Setup (Ubuntu/Debian VM)

```bash
sudo apt update && sudo apt install -y python3-venv ffmpeg fonts-dejavu-core git
git clone https://github.com/kryadov/openrouter_video_generator.git && cd openrouter_video_generator
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add OPENROUTER_API_KEY, set VIDEO_BUDGET_USD
```

On Windows, use `.venv\Scripts\python` and put ffmpeg on `PATH` (`winget install ffmpeg`).

## Configuration: `.env`

| Key | Meaning |
|---|---|
| `OPENROUTER_API_KEY` | the only secret |
| `VIDEO_BUDGET_USD` | hard cap on the **total** spend recorded in `output/_ledger.json`; any job that would exceed it is refused before it is submitted |
| `COMPARE_MODELS` | comma-separated model ids for part 1 (4, 6 or 8 fill the grid best) |
| `REFINE_MODEL` | the model chosen for part 2 |
| `VIDEO_RESOLUTION` | `480p` / `720p` for new shot lists |
| `VIDEO_ENCODER` | `auto` / `nvenc` / `x264` |
| `VIDEO_CAST` | optional casting rule added to every generated prompt |

Environment variables override `.env`. The model catalogue with prices is at `GET https://openrouter.ai/api/v1/videos/models`; it is cached in `output/_video_models.json`.

## Input folder

```
input/<Track name>/
  <anything>.wav   required
  lyrics.txt       optional; section tags like [Verse] / [Chorus]. Empty or missing means instrumental
  style.txt        optional; genre and sound description in any language
  notes.txt        optional; free-form notes for the human or agent editing shot lists
  *.jpg / *.png    optional; e.g. cover art to use as an image-to-video first frame
```

## Part 1: compare models

```bash
python -m compare "input/My Track" --dry-run    # analysis + excerpt + default shot list + price per model
python -m compare "input/My Track"              # renders every model, builds the grid
python -m compare "input/My Track" --models google/veo-3.1-lite bytedance/seedance-2.0-fast
```

Outputs in `output/<Track>/`:

- `excerpt.json` with the three best excerpt candidates (`excerpts/*.m4a` to listen to them)
- `compare/shotlist.json`, the default brief: two-bar shots with shot types chosen for the track's energy, the style text and first chorus as context, plus casting and "no singing faces" rules
- `compare/<model>.mp4`, one finished short per model
- `compare/compare.mp4`, all models in a labelled grid

Pick another excerpt with `--candidate 2`. Edit `compare/shotlist.json` and rerun: shots that already exist are reused.

## Part 2: refine on the chosen model

```bash
mkdir -p "output/My Track/short"
cp "output/My Track/compare/shotlist.json" "output/My Track/short/shotlist_v1.json"
# edit prompts / timing, then:
python -m refine "My Track" shotlist_v1.json --dry-run     # what would be generated, and the cost
python -m refine "My Track" shotlist_v1.json               # generate the new or changed shots, then edit
python -m refine "My Track" shotlist_v2.json --assemble-only   # re-edit only (reorder, trim): free
```

Keep one file per version (`shotlist_v1.json`, `_v2`, …). Each version renders to `short/<model><take>.mp4`.

### Shot list reference

```jsonc
{
  "excerpt": {"start": 117.28, "bars": 16, "tail_bars": 1,   // bar grid; tail bar = fade-out ending
              "audio_start": 53.55},                          // optional: start audio in the gap before a vocal pickup
  "model": "bytedance/seedance-2.0-fast", "aspect_ratio": "9:16", "resolution": "720p",
  "music": "...", "look": "...", "rules": "...",             // any string field becomes a {placeholder}
  "grade": "eq=contrast=1.05:saturation=1.08,noise=alls=5:allf=t",
  "shots": [
    {"id": "s1", "bars": [0, 2], "prompt": "{music}. Vertical 9:16 frame. ... {look}. {rules}",
     "seeds": [7, 21], "pick": 21,          // several takes, choose one
     "first_frame": "cover_9x16.jpg",       // image-to-video from a file next to the shot list
     "window": [0.2, 1.0],                  // manual source [offset, speed] instead of beat-fitting
     "stutter": true}                       // cut one slice per beat on motion peaks (drums etc.)
  ]
}
```

**Cache rule.** A shot is identified by model, fully substituted prompt, duration, aspect ratio, resolution, seed and first-frame image. To reuse a paid shot in a new version, keep its `id` and its exact final prompt text. Changing a shared `{placeholder}` changes every shot that uses it, so freeze reused prompts as literal strings.

## Budget and ledger

Every submitted job is written to `output/_ledger.json` with its estimate, then updated with the actual cost reported by OpenRouter. `--dry-run` prices everything up front, and a run that would exceed `VIDEO_BUDGET_USD` stops before it starts. Typical cost of a 30 s 720p short with 8 shots on Seedance 2.0 Fast: **~$3–4**. A full 4-model comparison costs **~$10**.

## Creative guidelines (learned the hard way)

- **The first 1–2 s decide retention.** Open on the most striking image of the video, never on a back view or a slow build.
- **Text-to-video can't lip-sync to your audio.** Don't show singing faces. Use backs, silhouettes, hands, wide shots and people who aren't singing.
- **No burned-in lyric captions** unless they are timed to the real vocal.
- **Match the energy.** Peak choruses need rhythmic material (crowds, drums, lights), not romance or calm pauses. Calm songs need long, slow shots.
- **Keep it varied.** No two neighbouring shots should share a location or framing. Use one visual climax, not three.
- **Don't end abruptly.** End on a tail bar with an audio and video fade. A last shot that rhymes with the first helps the loop.
- **Continuity** is easiest with one location and a consistent description of the people. An image-to-video first frame locks the look.
- **Models are bad at counting and fine motor actions.** Describe arrangements ("two in front, two in the back") rather than numbers, and check hands on instruments or buttons.

## Troubleshooting

- **`SSL: CERTIFICATE_VERIFY_FAILED`** behind a corporate proxy or antivirus: `truststore` (installed) makes Python use the OS certificate store.
- **NVENC errors:** set `VIDEO_ENCODER=x264`.
- **No labels in the comparison grid:** ffmpeg was built without `libfreetype`, or no fonts are installed (`fonts-dejavu-core`).

## legacy/: local GPU visualizer (experimental)

This is the first iteration: an audio-reactive GLSL renderer (moderngl) with SDXL keyframes, genre presets and fully local rendering. It is kept for reference and needs an NVIDIA GPU (`pip install -r legacy/requirements.txt` plus CUDA PyTorch). Run it with `python -m legacy.clipgen "input/My Track"`.

## License

[MIT](LICENSE) © 2026 Konstantin Ryadov
