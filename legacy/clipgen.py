"""clipgen — music video generator (local GPU).

  python -m legacy.clipgen "input/My Track"                    # full pipeline, both formats
  python -m legacy.clipgen "input/My Track" --until plan       # stop after writing plans (edit them, then rerun)
  python -m legacy.clipgen "input/My Track" --variants 1       # just the best-matching look (default 3)
  python -m legacy.clipgen "input/My Track" --plan paper_world --formats 16x9

Input folder: <track>/ with one *.wav (required), lyrics.txt (empty = instrumental / no lyrics), style.txt (empty = detect),
notes.txt (optional free-form notes; carried into style.json and plan meta, not interpreted by the pipeline).
Output: output/<track>/ — analysis/, analysis.json, style.json, plans/*.json, keyframes/, chunks/, *.mp4
Every stage is cached; edit any JSON artifact and rerun — only affected work is redone.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = ["analyze", "style", "plan", "keyframes", "render"]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track_dir")
    ap.add_argument("--until", choices=STAGES, default="render")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--plan", nargs="*", help="only these plan names (default: all plans in output/<track>/plans)")
    ap.add_argument("--formats", nargs="+", default=["16x9", "9x16"])
    ap.add_argument("--steps", type=int, default=30, help="diffusion steps")
    ap.add_argument("--restyle", action="store_true", help="redo style detection")
    ap.add_argument("--replan", action="store_true", help="overwrite generated plans")
    ap.add_argument("--preview", nargs=2, type=float, metavar=("START", "LEN"))
    ap.add_argument("--draft", action="store_true", help="skip keyframe generation; render with placeholders")
    a = ap.parse_args()

    track = Path(a.track_dir).resolve()
    wavs = sorted(track.glob("*.wav"))
    if not wavs:
        sys.exit(f"no .wav in {track}")
    audio = wavs[0]
    work = ROOT / "output" / track.name
    work.mkdir(parents=True, exist_ok=True)
    stop = STAGES.index(a.until)

    from core import analyze
    print(f"[analyze] {audio.name}")
    meta = analyze.meta(audio, 60, work / "analysis")
    (work / "analysis.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  tempo {meta['tempo']:.1f} BPM, {meta['duration']:.1f}s, {len(meta['segments'])} sections: "
          + " ".join(s["kind"][:5] for s in meta["segments"]))
    if stop < 1:
        return

    from legacy import style as style_mod
    st = style_mod.detect(track, work, meta, audio, force=a.restyle)
    print(f"[style] {st['source']}: preset={st['preset']}  ({st['text'][:80]})")
    if st.get("notes"):
        print(f"  notes.txt: {st['notes'][:200]}")
    if stop < 2:
        return

    from legacy import planner
    made = planner.build(track, work, meta, st, audio, a.variants, a.replan)
    plans = sorted((work / "plans").glob("*.json"))
    if a.plan:
        plans = [p for p in plans if p.stem in a.plan]
    print(f"[plan] {', '.join(p.stem for p in plans)}")
    if stop < 3:
        return

    from legacy.render import load_clip, render
    for p in plans:
        clip = load_clip(p)
        if not a.draft:
            print(f"[keyframes] {p.stem}")
            from legacy import keyframes
            keyframes.generate(clip, a.formats, steps=a.steps)
        if stop < 4:
            continue
        for fmt in a.formats:
            print(f"[render] {p.stem} {fmt}")
            render(clip, fmt, a.preview, draft=a.draft)


if __name__ == "__main__":
    main()
