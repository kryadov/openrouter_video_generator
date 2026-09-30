"""Pick the best short-form excerpt (Shorts / Reels / TikTok) -> excerpt.json.

Windows sit on the bar grid, last a whole number of phrases and 20-40 s (song-dependent).
Scoring favours: energy in the first 2 s (no swipe-away), starting on a drop/chorus,
chorus coverage, loopability (last bar resembles the first), no silence, ~30 s length.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

from core import analyze

MIN_S, MAX_S, TARGET_S = 20.0, 40.0, 30.0
W = {"hook": 3.0, "drop_start": 2.0, "coverage": 1.5, "loop": 1.5, "silence": 2.0, "length": 1.0}


def _bar_frames(feats, n):
    bars = list(feats["beats"][::4])
    return bars + [n]


def candidates(feats, meta, fps=60, top=3):
    n = len(feats["rms"])
    bars = _bar_frames(feats, n)
    bar_s = float(np.median(np.diff(bars[:-1]))) / fps
    segs = meta["segments"]
    drop_bars = [s["bar0"] for s in segs if s["kind"] == "drop"]
    in_drop = np.zeros(len(bars) - 1, bool)
    for s in segs:
        if s["kind"] == "drop":
            in_drop[s["bar0"]:s["bar0"] + s["bars"]] = True
    keys = ["kick", "bass", "mid", "high", "rms"]
    V = np.array([[feats[k][a:b].mean() for k in keys] for a, b in zip(bars[:-1], bars[1:])])
    inten = 0.6 * feats["rms"] + 0.4 * feats["high"]
    p90 = np.percentile(inten, 90) + 1e-6
    phrase = 4
    lengths = [L for L in range(phrase, len(bars), phrase) if MIN_S <= L * bar_s <= MAX_S]
    if not lengths:  # very slow/fast tempo: fall back to the closest phrase multiple
        lengths = [max(phrase, int(round(TARGET_S / bar_s / phrase)) * phrase)]
    out = []
    for L in lengths:
        for b0 in range(0, len(bars) - 1 - L + 1, 2):
            b1 = b0 + L
            f0, f1 = bars[b0], bars[b1]
            hook = float(np.clip(inten[f0:f0 + 2 * fps].mean() / p90, 0, 1))
            d = min([abs(b0 - db) for db in drop_bars] or [99])
            drop_start = 1.0 if d == 0 else 0.7 if d <= 1 else 0.3 if any(0 < b0 - db <= 4 for db in drop_bars) else 0.0
            coverage = float(in_drop[b0:b1].mean())
            a, z = V[b0], V[b1 - 1]
            loop = float(a @ z / (np.linalg.norm(a) * np.linalg.norm(z) + 1e-9))
            loop = float(np.clip((loop - 0.9) / 0.1, 0, 1))
            sec_rms = [feats["rms"][i:i + fps].mean() for i in range(f0, f1 - fps + 1, fps)]
            silence = 1.0 if min(sec_rms) < 0.12 else 0.0
            dur = (f1 - f0) / fps
            length = float(max(0.0, 1 - abs(dur - TARGET_S) / 12))
            parts = {"hook": hook, "drop_start": drop_start, "coverage": coverage, "loop": loop,
                     "silence": -silence, "length": length}
            score = sum(W[k] * v for k, v in parts.items())
            out.append({"bar0": b0, "bars": L, "start": round(f0 / fps, 3), "end": round(f1 / fps, 3),
                        "duration": round(dur, 2), "score": round(score, 3),
                        "parts": {k: round(v, 2) for k, v in parts.items()}})
    out.sort(key=lambda c: -c["score"])
    picked = []
    for c in out:  # non-overlapping (<50%) alternates
        if all(min(c["end"], p["end"]) - max(c["start"], p["start"]) < 0.5 * c["duration"] for p in picked):
            picked.append(c)
        if len(picked) == top:
            break
    for c in picked:
        kinds = [s["kind"] for s in segs if s["end"] > c["start"] and s["start"] < c["end"]]
        c["sections"] = kinds
    return picked, bar_s


def excerpt_audio(audio, start, end, out):
    """Bar-exact cut with 20 ms edge fades (clean loop point)."""
    dur = end - start
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(audio),
                    "-af", f"afade=t=in:d=0.02,afade=t=out:st={dur - 0.02:.3f}:d=0.02",
                    "-c:a", "aac", "-b:a", "256k", str(out)], check=True)


def select(audio, workdir, fps=60, force=False):
    workdir = Path(workdir)
    path = workdir / "excerpt.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    feats = analyze.load(audio, fps, workdir / "analysis")
    meta = analyze.meta(audio, fps, workdir / "analysis")
    cands, bar_s = candidates(feats, meta, fps)
    d = workdir / "excerpts"
    d.mkdir(exist_ok=True)
    for i, c in enumerate(cands):
        c["audio"] = str(d / f"cand{i + 1}_{c['start']:.1f}-{c['end']:.1f}.m4a")
        excerpt_audio(audio, c["start"], c["end"], c["audio"])
    res = {"chosen": 0, "bar_seconds": round(bar_s, 4), "tempo": meta["tempo"], "candidates": cands}
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    return res
