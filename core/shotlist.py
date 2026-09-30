"""Shot lists: load, prompt substitution, bar timing, cache paths, generation via OpenRouter.

A shot list is a JSON file inside output/<track>/<stage>/ (stage = compare | short).
Generated clips go to <that dir>/shots/<model>/, finished videos to <that dir>/<model><take>.mp4.

Fields:
  excerpt {start, bars, tail_bars, audio_start?, pre_roll_beats?}
  model, aspect_ratio, resolution
  any string field (music, look, rules, cast, setting, pair, ...) -> {placeholder} in shot prompts
  grade                         ffmpeg filter applied to every shot
  shots[] {id, bars, prompt, lyric?, seeds?, pick?, gen_seconds?, first_frame?, window?, stutter?}
"""
import base64
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


def shot_path(c, mid, shot, seed, sec):
    ff = first_frame(c, shot)
    tag = f"|ff:{ff.stat().st_size}" if ff else ""
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


def generate(c, mid, budget=None, dry=False):
    todo = plan_cost(c, mid)
    total = sum(t[3] for t in todo)
    jobs = []
    for s, seed, sec, est, out in todo:
        if dry:
            print(f"  would generate {mid} {s['id']} seed {seed}: {sec}s, est ${est:.2f}")
            continue
        extra = None
        ff = first_frame(c, s)
        if ff:
            uri = "data:image/jpeg;base64," + base64.b64encode(ff.read_bytes()).decode()
            extra = {"frame_images": [{"type": "image_url", "image_url": {"url": uri}, "frame_type": "first_frame"}]}
        jobs.append(videogen.submit(mid, prompt_of(c.sl, s), out, sec, c.sl["aspect_ratio"], c.sl["resolution"],
                                    seed, budget, extra))
    print(f"{mid}: new shots est ${total:.2f}; ledger ${videogen.spent():.2f}")
    if not dry and jobs:
        videogen.wait_all(jobs)
    return total
