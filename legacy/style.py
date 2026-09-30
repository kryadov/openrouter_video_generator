"""Style detection -> style.json.

Priority: style.txt text  >  CLAP zero-shot audio tags (local GPU)  >  tempo/energy heuristic.
The chosen preset is the one whose keywords best match the style text/tags.
"""
import json
from pathlib import Path

import numpy as np

PRESETS_DIR = Path(__file__).resolve().parent / "presets"

CLAP_MODEL = "laion/larger_clap_music"
GENRE_TAGS = ["psytrance", "techno", "house", "trance", "drum and bass", "dubstep", "synthwave", "funk", "disco",
              "ambient", "drone", "lo-fi hip hop", "hip hop", "rock", "metal", "pop", "jazz", "classical orchestral",
              "cinematic soundtrack", "folk acoustic", "r&b soul", "electronic dance music", "downtempo chill"]
MOOD_TAGS = ["energetic", "dark", "happy", "melancholic", "dreamy", "aggressive", "calm", "epic", "playful", "romantic"]


def presets():
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(PRESETS_DIR.glob("*.json"))}


def clap_tags(audio_path, top=6):
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClapModel.from_pretrained(CLAP_MODEL).to(dev).eval()
    proc = ClapProcessor.from_pretrained(CLAP_MODEL)
    y, sr = librosa.load(audio_path, sr=48000, mono=True)
    L = 10 * sr
    starts = np.linspace(0, max(0, len(y) - L), 5).astype(int)[1:-1] if len(y) > 3 * L else [0]
    clips = [y[s:s + L] for s in starts]
    tags = GENRE_TAGS + MOOD_TAGS
    texts = [f"{t} music" for t in tags]
    with torch.no_grad():
        a = proc(audios=clips, sampling_rate=48000, return_tensors="pt").to(dev)
        ae = model.get_audio_features(**a)
        t = proc(text=texts, return_tensors="pt", padding=True).to(dev)
        te = model.get_text_features(**t)
        ae = ae / ae.norm(dim=-1, keepdim=True)
        te = te / te.norm(dim=-1, keepdim=True)
        sim = (ae @ te.T).mean(0).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    g = sorted(zip(GENRE_TAGS, sim[:len(GENRE_TAGS)]), key=lambda x: -x[1])[:top]
    m = sorted(zip(MOOD_TAGS, sim[len(GENRE_TAGS):]), key=lambda x: -x[1])[:3]
    return [(k, float(v)) for k, v in g + m]


def heuristic_text(meta):
    segs = meta["segments"]
    kick_share = sum((s["end"] - s["start"]) for s in segs if s["has_kick"]) / meta["duration"]
    energy = float(np.mean([s["energy"] for s in segs]))
    tempo = meta["tempo"]
    if kick_share < 0.35:
        return "ambient atmospheric calm"
    if 136 <= tempo <= 150:
        return "psychedelic trance hypnotic"
    return f"electronic dance {'energetic' if energy > 0.5 else 'downtempo'}"


def score(text, preset):
    """Keyword hits; genres named earlier in the description weigh more."""
    t = text.lower()
    total = 0.0
    for k, w in [(k, 1.0) for k in preset["keywords"]] + [(k, 0.25) for k in preset.get("weak_keywords", [])]:
        pos = t.find(k)
        if pos >= 0:
            total += w * (1.0 + 0.5 * (len(k) > 6)) * (1.0 + 1.0 * (1 - pos / max(len(t), 1)))
    return round(total, 3)


def detect(track_dir, workdir, meta, audio_path, force=False):
    out = Path(workdir) / "style.json"
    if out.exists() and not force:
        st = json.loads(out.read_text(encoding="utf-8"))
        notes_p = Path(track_dir) / "notes.txt"
        st["notes"] = notes_p.read_text(encoding="utf-8").strip() if notes_p.exists() else ""
        return st
    txt_p = Path(track_dir) / "style.txt"
    text = txt_p.read_text(encoding="utf-8").strip() if txt_p.exists() else ""
    source, tags = "style.txt", []
    if not text:
        try:
            tags = clap_tags(audio_path)
            text = ", ".join(t for t, _ in tags)
            source = "clap"
        except Exception as e:  # no torch/transformers yet -> heuristic
            print("  CLAP unavailable:", e)
            text, source = heuristic_text(meta), "heuristic"
    P = presets()
    scores = {k: score(text, p) for k, p in P.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        h = heuristic_text(meta)
        scores = {k: score(h, p) for k, p in P.items()}
        best = max(scores, key=scores.get) if max(scores.values()) > 0 else "electronic"
    lyr_p = Path(track_dir) / "lyrics.txt"
    lyrics = lyr_p.read_text(encoding="utf-8").strip() if lyr_p.exists() else ""
    notes_p = Path(track_dir) / "notes.txt"
    notes = notes_p.read_text(encoding="utf-8").strip() if notes_p.exists() else ""
    st = {"source": source, "text": text, "tags": tags, "preset": best, "scores": scores,
          "has_lyrics": bool(lyrics), "notes": notes, "tempo": meta["tempo"]}
    out.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    return st
