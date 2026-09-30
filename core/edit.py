"""Editing: bar-exact assembly of generated shots with the excerpt audio, plus comparison grids.

Beat-sync: each shot's source window (offset + speed 0.92-1.08) is picked so motion peaks land on
beats; shot["stutter"] cuts one slice per beat starting on motion peaks (e.g. drum hits);
shot["window"] = [offset, speed] overrides. The excerpt can start early (audio_start /
pre_roll_beats) for a vocal pickup, and tail_bars fade out, so nothing ends abruptly.
"""
import subprocess
from pathlib import Path

import numpy as np

from core import encode
from core.shotlist import first_frame, gen_seconds, seeds_of, shot_path, shot_span, slug

OUT_W, OUT_H, OUT_FPS = 1080, 1920, 30
FONTS = ["C:/Windows/Fonts/arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
         "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]


def motion_energy(path, fps=OUT_FPS):
    w, h = 144, 256
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf",
                          f"fps={fps},scale={w}:{h},format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
    m = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    m = np.concatenate([[m[0]], m])
    return (m - m.mean()) / (m.std() + 1e-6)


def fit_window(path, dur, beat_offsets, beat_len, fps=OUT_FPS):
    """Source offset + speed so motion peaks fall on beats. Returns (offset, speed, score)."""
    m = motion_energy(path, fps)
    L = len(m) / fps
    best = (0.0, 1.0, -9.0)
    bo = np.asarray(beat_offsets)
    for sp in (0.92, 0.96, 1.0, 1.04, 1.08):
        need = dur * sp
        if need > L - 1 / fps:
            continue
        for off in np.arange(0, L - need, 1 / fps):
            src = off + bo * sp
            on = np.clip(np.round(src * fps).astype(int), 0, len(m) - 1)
            offb = np.clip(np.round((src + 0.5 * beat_len * sp) * fps).astype(int), 0, len(m) - 1)
            score = m[on].mean() - m[offb].mean() - 1.5 * abs(sp - 1)
            if score > best[2]:
                best = (float(off), sp, float(score))
    return best


def stutter_slices(path, dur, beat_offsets, lead=2 / OUT_FPS, fps=OUT_FPS):
    """One slice per beat, each starting just before a strong motion peak. [(src_start, length)]."""
    m = motion_energy(path, fps)
    L = len(m) / fps
    peaks = [i for i in range(1, len(m) - 1) if m[i] >= m[i - 1] and m[i] > m[i + 1] and m[i] > 0.3]
    peaks.sort(key=lambda i: -m[i])
    marks = sorted(set(round(x, 4) for x in [0.0, dur] + [b for b in beat_offsets if 0.05 < b < dur - 0.05]))
    out, used = [], []
    for a, z in zip(marks[:-1], marks[1:]):
        seg = z - a
        cands = [p / fps - lead for p in peaks if p / fps - lead >= 0 and p / fps - lead + seg <= L - 1 / fps]
        cands = [x for x in cands if all(abs(x - u) > 0.25 for u in used[-2:])] or cands or [0.0]
        used.append(cands[0])
        out.append((cands[0], seg))
    return out


def assemble(c, mid, take=None):
    sl, ex = c.sl, c.sl["excerpt"]
    tail = ex.get("tail_bars", 0)
    t_start = c.bars[c.b0] / 60
    total = c.bars[c.b0 + ex["bars"] + tail] / 60 - t_start
    tail_s = total - (c.bars[c.b0 + ex["bars"]] / 60 - t_start)
    beat_len = float(np.median(np.diff(c.beats)))
    pre = ex.get("pre_roll_beats", 0) * beat_len
    if "audio_start" in ex:
        pre = max(0.0, t_start - ex["audio_start"])
    take = take if take is not None else sl.get("take", "")
    inputs, filt, labels, plan = [], [], [], []
    for i, s in enumerate(sl["shots"]):
        a, dur = shot_span(c, s)
        sec = gen_seconds(sl, mid, s, dur)
        seeds = [s["pick"]] if s.get("pick") is not None else seeds_of(s)
        f = next((p for p in (shot_path(c, mid, s, sd, sec) for sd in seeds) if p.exists()), None)
        if not f:
            raise SystemExit(f"missing shot {s['id']} for {mid}")
        bo = [b - a for b in c.beats if a - 1e-3 <= b < a + dur]
        inputs += ["-i", str(f)]
        post = (f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},"
                f"fps={OUT_FPS},setsar=1,{sl.get('grade', 'null')}")
        if s.get("stutter"):
            sls = stutter_slices(f, dur, bo)
            plan.append((s["id"], "stutter", len(sls)))
            k = len(sls)
            parts = [f"[{i}:v]split={k}" + "".join(f"[s{i}_{j}]" for j in range(k))]
            parts += [f"[s{i}_{j}]trim=start={st:.4f}:duration={ln:.4f},setpts=PTS-STARTPTS[t{i}_{j}]"
                      for j, (st, ln) in enumerate(sls)]
            parts.append("".join(f"[t{i}_{j}]" for j in range(k)) + f"concat=n={k}:v=1:a=0,{post}[v{i}]")
            filt.append(";".join(parts))
        else:
            dur_out = dur
            if i == 0 and pre > 0:  # cover the pickup with moving footage, slowed only if the source is short
                src_len = len(motion_energy(f)) / OUT_FPS
                dur_out = dur + pre
                off, sp = 0.0, round(min(1.0, (src_len - 1 / OUT_FPS) / dur_out), 4)
            elif s.get("window"):
                off, sp = s["window"]
            elif first_frame(c, s):
                off, sp = 0.0, 1.0
            else:
                off, sp, _ = fit_window(f, dur, bo, beat_len)
            plan.append((s["id"], round(off, 2), sp))
            filt.append(f"[{i}:v]trim=start={off:.4f}:duration={dur_out * sp:.4f},"
                        f"setpts=(PTS-STARTPTS)/{sp},{post}[v{i}]")
        labels.append(f"[v{i}]")
    n = len(sl["shots"])
    filt.append("".join(labels) + f"concat=n={n}:v=1:a=0[vc]")
    total += pre
    filt.append(f"[vc]fade=t=out:st={total - 0.8:.3f}:d=0.8[v]" if tail else "[vc]null[v]")
    inputs += ["-ss", f"{t_start - pre:.4f}", "-t", f"{total:.4f}", "-i", str(c.wav)]
    fo = max(tail_s, 0.03)
    out = c.base / f"{slug(mid)}{take}.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filt),
           "-map", "[v]", "-map", f"{n}:a", "-af", f"afade=t=in:d=0.02,afade=t=out:st={total - fo:.3f}:d={fo:.3f}",
           *encode.h264_args(), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
           "-movflags", "+faststart", "-t", f"{total:.4f}", str(out)]
    subprocess.run(cmd, check=True)
    print(f"  -> {out} ({total:.2f}s)")
    return out


def compare_grid(items, out):
    """items: [(label, video_path)] of equal-length vertical clips -> one labelled grid (2x2 / 3x2 / 4x2)."""
    k = len(items)
    cols = 2 if k <= 4 else 3 if k <= 6 else 4
    rows = (k + cols - 1) // cols
    w, h = (OUT_W // 2, OUT_H // 2) if cols == 2 else (360, 640)
    font = next((f for f in FONTS if Path(f).exists()), None)
    ff = f"fontfile='{font.replace(':', chr(92) + ':')}':" if font else ""
    inputs, filt = [], []
    for i, (label, path) in enumerate(items):
        inputs += ["-i", str(path)]
        filt.append(f"[{i}:v]scale={w}:{h},drawtext={ff}text='{label}':x=12:y=12:fontsize={max(18, w // 16)}:"
                    f"fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6[g{i}]")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(items[0][1])], capture_output=True, text=True).stdout or 30)
    for i in range(k, rows * cols):
        filt.append(f"color=black:s={w}x{h}:d={dur:.3f}:r={OUT_FPS}[g{i}]")
    layout = "|".join(f"{(i % cols) * w}_{(i // cols) * h}" for i in range(rows * cols))
    filt.append("".join(f"[g{i}]" for i in range(rows * cols)) + f"xstack=inputs={rows * cols}:layout={layout}[v]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filt),
                    "-map", "[v]", "-map", "0:a", *encode.h264_args(21), "-pix_fmt", "yuv420p", "-c:a", "copy",
                    "-t", f"{dur:.3f}", str(out)], check=True)
    print(f"  -> {out}")
    return out
