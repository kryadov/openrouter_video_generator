"""Shot lists: load, prompt substitution, bar timing, cache paths, generation via OpenRouter.

A shot list is a JSON file inside output/<track>/<stage>/ (stage = compare | short).
Generated clips go to <that dir>/shots/<model>/, finished videos to <that dir>/<model><take>.mp4.

Fields:
  excerpt {start, bars, tail_bars, audio_start?, pre_roll_beats?}
  model, aspect_ratio, resolution
  any string field (music, look, rules, cast, setting, pair, ...) -> {placeholder} in shot prompts
  grade                         ffmpeg filter applied to every shot
  shots[] {id, bars, prompt, lyric?, seeds?, pick?, gen_seconds?, first_frame?, chain?, window?, stutter?}

chain: true starts the shot (image-to-video) on the last frame the edit uses from the previous shot,
so the cut is seamless. Chained shots are generated one after another, and both sides of a chained cut
are edited with window [0, 1] (no beat-fitting) unless a window is given.
"""
import base64
import subprocess
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from core import analyze, videogen
from core.config import ROOT


@dataclass
class Ctx:
    track: str
    work: Path      # output/<track>
    base: Path      # directory of the shot list
    sl: dict
    wav: Path
    bars: list      # bar start frames (60 fps grid)
    b0: int         # index of the excerpt's first bar
    beats: object   # beat times in seconds (numpy array)


def track_wav(track):
    wavs = sorted((ROOT / "input" / track).glob("*.wav"))
    if not wavs:
        raise SystemExit(f"no .wav in input/{track}")
    return wavs[0]


def load(track, shotlist_rel):
    work = ROOT / "output" / track
    path = work / shotlist_rel
    sl = json.loads(path.read_text(encoding="utf-8"))
    wav = track_wav(track)
    feats = analyze.load(wav, 60, work / "analysis")
    bars = list(feats["beats"][::4])
    b0 = min(range(len(bars)), key=lambda i: abs(bars[i] / 60 - sl["excerpt"]["start"]))
    return Ctx(track, work, path.parent, sl, wav, bars, b0, feats["beats"] / 60.0)


def slug(model_id):
    return model_id.replace("/", "__")


def prompt_of(sl, shot):
    ctx = defaultdict(str, {k: v for k, v in sl.items() if isinstance(v, str)})
    ctx["lyric"] = shot.get("lyric", "")
    return shot["prompt"].format_map(ctx)


def shot_span(c, shot):
    a, z = c.bars[c.b0 + shot["bars"][0]] / 60, c.bars[c.b0 + shot["bars"][1]] / 60
    return a, z - a


def gen_seconds(sl, mid, shot, dur):
    want = shot.get("gen_seconds") or sl.get("gen_seconds") or math.ceil(dur + 0.35)
    durs = videogen.models()[mid]["supported_durations"] or [want]
    return min([d for d in durs if d >= want] or [max(durs)])


def seeds_of(shot):
    return shot.get("seeds") or [7]


def first_frame(c, shot):
    ff = shot.get("first_frame")
    return (c.base / ff) if ff else None


def chosen_seed(shot):
    return shot["pick"] if shot.get("pick") is not None else seeds_of(shot)[0]


def clip_path(c, mid, shot, seed=None):
    """Cache path of a shot's clip for its chosen (or the given) seed."""
    sec = gen_seconds(c.sl, mid, shot, shot_span(c, shot)[1])
    return shot_path(c, mid, shot, chosen_seed(shot) if seed is None else seed, sec)


def chain_source(c, mid, shot):
    """(previous shot's clip, source time of its last used frame) for a chained shot, else None."""
    if not shot.get("chain"):
        return None
    i = [s["id"] for s in c.sl["shots"]].index(shot["id"])
    if i == 0:
        raise SystemExit(f"shot {shot['id']}: chain needs a previous shot")
    prev = c.sl["shots"][i - 1]
    off, sp = prev.get("window") or [0.0, 1.0]
    return clip_path(c, mid, prev), off + shot_span(c, prev)[1] * sp


def chain_frame(c, mid, shot):
    """Extract (once) the previous clip's last used frame as a jpg next to it."""
    src, t = chain_source(c, mid, shot)
    jpg = src.with_name(f"{src.stem}_at{t:.3f}.jpg")
    if not jpg.exists():
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(0.0, t - 1 / 30):.4f}", "-i", str(src),
                        "-frames:v", "1", "-q:v", "2", str(jpg)], check=True)
    return jpg


def shot_path(c, mid, shot, seed, sec):
    ff = first_frame(c, shot)
    tag = f"|ff:{ff.stat().st_size}" if ff else ""
    ch = chain_source(c, mid, shot)
    if ch:
        tag = f"|chain:{ch[0].name}@{ch[1]:.3f}"
    key = videogen.job_key(mid, prompt_of(c.sl, shot) + tag, sec, c.sl["aspect_ratio"], c.sl["resolution"], seed)
    return c.base / "shots" / slug(mid) / f"{shot['id']}_s{seed}_{key}.mp4"


def plan_cost(c, mid):
    """[(shot, seed, seconds, est, out_path)] for shots not yet generated."""
    todo = []
    for s in c.sl["shots"]:
        _, dur = shot_span(c, s)
        sec = gen_seconds(c.sl, mid, s, dur)
        for seed in seeds_of(s):
            out = shot_path(c, mid, s, seed, sec)
            if not out.exists():
                todo.append((s, seed, sec, videogen.estimate(mid, sec, c.sl["resolution"]), out))
    return todo


def generate(c, mid, budget=None, dry=False, upto=None):
    todo = plan_cost(c, mid)
    if upto:  # stage a chain: pay only up to this shot, review it, then continue
        ids = [s["id"] for s in c.sl["shots"]]
        if upto not in ids:
            raise SystemExit(f"no shot {upto!r} in the shot list")
        todo = [t for t in todo if ids.index(t[0]["id"]) <= ids.index(upto)]
    total = sum(t[3] for t in todo)
    jobs = []
    for s, seed, sec, est, out in todo:
        ch = chain_source(c, mid, s)
        if dry:
            after = f" (chained after {ch[0].name})" if ch else ""
            print(f"  would generate {mid} {s['id']} seed {seed}: {sec}s, est ${est:.2f}{after}")
            continue
        if ch and not ch[0].exists() and jobs:
            videogen.wait_all(jobs)  # the previous shot must exist before its last frame can seed this one
            jobs = []
        if ch and not ch[0].exists():
            print(f"  skip {s['id']}: previous clip {ch[0].name} is missing")
            continue
        extra = None
        ff = chain_frame(c, mid, s) if ch else first_frame(c, s)
        if ff:
            uri = "data:image/jpeg;base64," + base64.b64encode(ff.read_bytes()).decode()
            extra = {"frame_images": [{"type": "image_url", "image_url": {"url": uri}, "frame_type": "first_frame"}]}
        jobs.append(videogen.submit(mid, prompt_of(c.sl, s), out, sec, c.sl["aspect_ratio"], c.sl["resolution"],
                                    seed, budget, extra))
    print(f"{mid}: new shots est ${total:.2f}; ledger ${videogen.spent():.2f}")
    if not dry and jobs:
        videogen.wait_all(jobs)
    return total
