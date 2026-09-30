"""Audio analysis -> per-frame feature curves, cached in cache/analysis/.

Usage: python -m core.analyze <audio.wav> [--fps 60]
"""
import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "analysis"


def _norm(x, pct=98):
    hi = np.percentile(x, pct)
    return np.clip(x / (hi + 1e-9), 0.0, 1.0)


def _envelope(x, attack, release):
    """One-pole attack/release follower (per-frame units)."""
    out = np.zeros_like(x)
    v = 0.0
    for i, s in enumerate(x):
        k = attack if s > v else release
        v += (s - v) * k
        out[i] = v
    return out


def analyze(audio, fps=60, cache_dir=None):
    audio = Path(audio)
    cdir = Path(cache_dir) if cache_dir else CACHE / audio.stem.replace(' ', '_')
    cdir.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(f"{audio.stat().st_size}-{audio.stat().st_mtime}-{fps}-v6".encode()).hexdigest()[:10]
    out_path = cdir / f"features_{fps}fps_{key}.npz"
    if out_path.exists():
        return out_path

    sr = 48000
    y, _ = librosa.load(audio, sr=sr, mono=True)
    duration = len(y) / sr
    n_frames = int(np.ceil(duration * fps))
    hop = sr // fps  # 800 samples @60fps

    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

    def band(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return np.sqrt(S[m].sum(axis=0))

    bands = {
        "kick": band(35, 110),
        "bass": band(110, 300),
        "mid": band(300, 2500),     # guitar / vocal body
        "high": band(2500, 9000),   # arps, acid, hats
        "air": band(9000, 16000),
    }
    rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=hop)[0]

    def fit(a):
        a = np.asarray(a, dtype=np.float32)
        if len(a) >= n_frames:
            return a[:n_frames]
        return np.pad(a, (0, n_frames - len(a)), mode="edge")

    feats = {}
    for k, v in bands.items():
        feats[k] = fit(_norm(v))
    feats["rms"] = fit(_norm(rms))

    # Kick onsets: positive flux in kick band, peak picked
    kflux = np.maximum(0, np.diff(bands["kick"], prepend=bands["kick"][0]))
    kflux = _norm(kflux)
    peaks = librosa.util.peak_pick(kflux, pre_max=3, post_max=3, pre_avg=6, post_avg=6, delta=0.12, wait=int(fps * 0.2))
    kick_trig = np.zeros(n_frames, np.float32)
    peaks = peaks[peaks < n_frames]
    kick_trig[peaks] = 1.0
    feats["kick_trig"] = kick_trig
    feats["kick_env"] = fit(_envelope(kick_trig, 1.0, 0.12))

    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    feats["onset"] = fit(_norm(onset))

    # Tempo with a dance-music prior; fold into 95..190 BPM range
    tempo = float(librosa.feature.tempo(onset_envelope=onset, sr=sr, hop_length=hop, start_bpm=128)[0])
    while tempo < 95:
        tempo *= 2
    while tempo > 190:
        tempo /= 2
    # Refine tempo + phase on a fine-resolution flux (~2.7 ms) so the grid doesn't drift
    fhop = 128
    Sf = np.abs(librosa.stft(y, n_fft=2048, hop_length=fhop))
    ff = librosa.fft_frequencies(sr=sr, n_fft=2048)
    has_kick = feats["kick"].mean() > 0.08
    lowf = Sf[(ff >= 35) & (ff < 110)].sum(0) if has_kick else Sf.sum(0)
    fine = np.maximum(0, np.diff(lowf, prepend=lowf[0]))
    fine = uniform_filter1d(fine / (np.percentile(fine, 99) + 1e-9), 5)
    frate = sr / fhop
    best = (-1, tempo, 0.0)
    for bpm in np.arange(tempo * 0.96, tempo * 1.04, 0.05):
        per = 60.0 / bpm * frate
        for ph in np.linspace(0, per, 40, endpoint=False):
            idx = np.round(np.arange(ph, len(fine), per)).astype(int)
            s = fine[idx[idx < len(fine)]].mean()
            if s > best[0]:
                best = (s, bpm, ph)
    _, tempo, ph = best
    tempo = float(tempo)
    beat_times = np.arange(ph / frate, duration, 60.0 / tempo)
    beats = np.round(beat_times * fps).astype(int)
    beats = beats[beats < n_frames]
    ref = fit(kflux if has_kick else _norm(onset))
    beat_trig = np.zeros(n_frames, np.float32)
    beat_trig[beats] = 1
    # beat phase 0..1 between beats
    phase = np.zeros(n_frames, np.float32)
    for a, b in zip(beats[:-1], beats[1:]):
        phase[a:b] = np.linspace(0, 1, b - a, endpoint=False)
    feats["beat_trig"] = beat_trig
    feats["beat_phase"] = phase

    # Smoothed intensity for section-level dynamics
    feats["energy"] = fit(_norm(uniform_filter1d(rms, size=fps * 2)))
    for k in ("kick", "bass", "mid", "high", "air"):
        feats[k + "_s"] = fit(_envelope(feats[k], 0.5, 0.08))

    feats["cumkick"] = np.cumsum(feats["kick_env"]).astype(np.float32) / fps

    # Kick triggers snapped to the beat grid (cleaner than raw peak picking)
    grid_kick = np.zeros(n_frames, np.float32)
    kick_on = fit(bands["kick"] / (np.percentile(bands["kick"], 98) + 1e-9))
    for b in beats:
        w = ref[max(0, b - 3): b + 4]
        if kick_on[b] > 0.25 and w.max() > 0.15:
            grid_kick[b] = min(1.0, w.max() * 1.5)
    feats["gkick_trig"] = grid_kick
    feats["gkick_env"] = fit(_envelope(grid_kick, 1.0, 0.14))

    segments = segment(feats, beats, fps, n_frames)

    np.savez_compressed(out_path, fps=fps, duration=duration, tempo=tempo,
                        beats=beats.astype(np.int32), kick_peaks=peaks.astype(np.int32), **feats)
    meta = {"fps": fps, "duration": duration, "tempo": tempo, "n_frames": n_frames,
            "n_beats": int(len(beats)), "n_kicks": int(len(peaks)), "segments": segments}
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out_path


def segment(feats, beats, fps, n_frames, bar_beats=4, min_bars=4):
    """Structural segmentation on the bar grid. Returns list of dicts with
    start/end (seconds), bars, energy, has_kick, kind."""
    bars = beats[::bar_beats]
    if len(bars) < 3:
        return [{"start": 0.0, "end": n_frames / fps, "kind": "main", "energy": 0.5, "has_kick": False}]
    edges = list(bars) + [n_frames]
    keys = ["kick", "bass", "mid", "high", "rms"]
    V = np.array([[feats[k][a:b].mean() for k in keys] for a, b in zip(edges[:-1], edges[1:])])
    raw = V.copy()
    V = (V - V.mean(0)) / (V.std(0) + 1e-6) * 0.25  # z-score: flat-energy mixes still segment
    nb = len(V)
    w = 4
    nov = np.zeros(nb)
    for i in range(1, nb):
        a = V[max(0, i - w):i].mean(0)
        b = V[i:min(nb, i + w)].mean(0)
        nov[i] = np.linalg.norm(a - b) * (1.0 + 2.0 * abs(a[0] - b[0]))  # kick in/out weighs more
    # prefer 4-bar phrase boundaries
    cand = sorted(range(1, nb), key=lambda i: -nov[i] * (1.3 if i % 4 == 0 else 1.0))
    thr = np.percentile(nov[1:], 60)
    chosen = []
    for i in cand:
        if nov[i] < thr:
            break
        if all(abs(i - c) >= min_bars for c in chosen) and i >= min_bars and nb - i >= 2:
            chosen.append(i)
    chosen = sorted(chosen)
    bounds = [0] + chosen + [nb]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        s, e = edges[a], edges[b]
        energy = float(feats["rms"][s:e].mean())
        has_kick = bool(feats["kick"][s:e].mean() > 0.15)
        segs.append({"start": round(s / fps, 3), "end": round(e / fps, 3), "bar0": int(a), "bars": int(b - a),
                     "energy": round(energy, 3), "has_kick": has_kick})
    # "intensity" = loudness + brightness + body, ranked relative to this track
    for x in segs:
        a, b = x["bar0"], x["bar0"] + x["bars"]
        r = raw[a:b].mean(0)  # kick, bass, mid, high, rms
        x["intensity"] = round(float(0.55 * r[4] + 0.3 * r[3] + 0.15 * r[2]), 3)
    # scale from inner sections only: a quiet intro/outro must not squash the range
    inner = segs[1:-1] if len(segs) > 3 else segs
    kick_int = [x["intensity"] for x in inner if x["has_kick"]] or [x["intensity"] for x in segs] or [0, 1]
    lo, hi = min(kick_int), max(kick_int)
    first_main = next((i for i, x in enumerate(segs) if x["has_kick"]), None)
    for i, x in enumerate(segs):
        rel = (x["intensity"] - lo) / (hi - lo + 1e-6)
        x["rel"] = round(rel, 3)
        if i == 0 and not x["has_kick"] and x["intensity"] < hi * 0.8:
            x["kind"] = "intro"
        elif i == len(segs) - 1 and x["intensity"] < hi * 0.8:
            x["kind"] = "outro"
        elif not x["has_kick"]:
            x["kind"] = "breakdown"
        elif i == first_main and x["bars"] >= 8 and rel < 0.85:
            x["kind"] = "groove"
        elif rel >= 0.6 or (i > 0 and rel - segs[i - 1].get("rel", rel) >= 0.25 and x["bars"] >= 6):
            x["kind"] = "drop"  # loud overall, or a clear lift over the previous section
        else:
            x["kind"] = "groove"
    return segs


def load(audio, fps=60, cache_dir=None):
    return dict(np.load(analyze(audio, fps, cache_dir)))


def meta(audio, fps=60, cache_dir=None):
    p = analyze(audio, fps, cache_dir)
    return json.loads(p.with_suffix(".json").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--fps", type=int, default=60)
    a = ap.parse_args()
    p = analyze(a.audio, a.fps)
    d = load(a.audio, a.fps)
    for sg in meta(a.audio, a.fps)["segments"]:
        print(sg)
    raise SystemExit
    print(p, "tempo", d["tempo"], "dur", d["duration"])
    # coarse energy map, 2s bins
    e = d["rms"]; k = d["kick"]; h = d["high"]; m = d["mid"]
    step = a.fps * 2
    for i in range(0, len(e), step):
        t = i / a.fps
        print(f"{int(t//60)}:{t%60:04.1f}  rms {e[i:i+step].mean():.2f} kick {k[i:i+step].mean():.2f} "
              f"mid {m[i:i+step].mean():.2f} high {h[i:i+step].mean():.2f}  " + "#" * int(e[i:i+step].mean() * 40))
