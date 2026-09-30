"""Deterministic planner: analysis + style + lyrics -> plans/<preset>_<world>.json.

No LLM. Sections come from the audio segmentation, effects from the genre preset,
image prompts from lyric lines (if any) blended with the preset world's motifs.
A plan is an editable artifact: existing plans are never overwritten unless replan=True.
"""
import json
import math
import re
from pathlib import Path

from legacy.style import presets

LYRIC_KIND = {"verse": "verse", "pre-chorus": "pre", "prechorus": "pre", "pre chorus": "pre", "build": "pre",
              "chorus": "chorus", "hook": "chorus", "final chorus": "chorus", "refrain": "chorus", "drop": "chorus",
              "bridge": "breakdown", "breakdown": "breakdown", "interlude": "breakdown", "solo": "solo",
              "intro": "intro", "outro": "outro"}
MOTIF_FALLBACK = {"pre": "verse", "breakdown": "verse", "solo": "chorus", "intro": "verse", "outro": "breakdown"}
MAX_UNIQUE_PER_SECTION = 6


def parse_lyrics(text):
    """-> list of (kind, [lines]) in order."""
    blocks, kind, lines = [], None, []
    for raw in text.splitlines():
        s = raw.strip()
        m = re.match(r"^\[(.+?)\]$", s)
        if m:
            if kind and lines:
                blocks.append((kind, lines))
            tag = re.sub(r"\s*\d+$", "", m.group(1).strip().lower())
            kind = next((v for k, v in LYRIC_KIND.items() if tag.startswith(k)), "verse")
            lines = []
        elif s:
            lines.append(s)
    if kind and lines:
        blocks.append((kind, lines))
    if not blocks and text.strip():
        blocks = [("verse", [l.strip() for l in text.splitlines() if l.strip()])]
    return blocks


def section_kinds(segs):
    kinds = []
    for i, s in enumerate(segs):
        k = {"intro": "intro", "outro": "outro", "drop": "chorus", "breakdown": "breakdown"}.get(s["kind"], "verse")
        kinds.append(k)
    for i in range(len(segs) - 1):
        if kinds[i] in ("verse", "breakdown") and kinds[i + 1] == "chorus" and segs[i]["bars"] <= 8:
            kinds[i] = "pre"
    return kinds


def split_long(segs, max_bars=16):
    """Split very long segments into phrase-sized chunks so visuals evolve."""
    out = []
    for s in segs:
        n = max(1, round(s["bars"] / 16)) if s["bars"] > max_bars else 1
        step = s["bars"] / n
        for j in range(n):
            b0 = s["bar0"] + round(j * step)
            b1 = s["bar0"] + round((j + 1) * step)
            out.append(dict(s, bar0=b0, bars=b1 - b0))
    return out


def choose_variants(style, n=3):
    """(preset, world) pairs: 2 worlds of the best-matching genre + 1 from the runner-up genre,
    then fill from the remaining worlds. Each variant keeps its own preset's editing style."""
    allp = presets()
    ctx = (style.get("text", "") + " " + style.get("title", "")).lower()
    for p in allp.values():  # within a genre, prefer worlds whose keywords match the style text / title
        p["worlds"] = sorted(p["worlds"], key=lambda w: -sum(k in ctx for k in w.get("keywords", [])))
    ranked = sorted(allp, key=lambda k: -style["scores"].get(k, 0))
    best = style["preset"]
    ranked.remove(best)
    ranked.insert(0, best)
    order = [(best, w) for w in allp[best]["worlds"][:2]]
    runner = next((k for k in ranked[1:] if style["scores"].get(k, 0) > 0), None)
    if runner:
        order.append((runner, allp[runner]["worlds"][0]))
    for k in ranked:
        for w in allp[k]["worlds"]:
            if (k, w) not in order:
                order.append((k, w))
    return [(allp[k], w) for k, w in order[:max(1, n)]]


def build(track_dir, workdir, meta, style, audio_path, variants=3, replan=False):
    title = Path(track_dir).name
    lyr_p = Path(track_dir) / "lyrics.txt"
    blocks = parse_lyrics(lyr_p.read_text(encoding="utf-8")) if lyr_p.exists() else []
    segs = split_long(meta["segments"])
    kinds = section_kinds(segs)
    plans_dir = Path(workdir) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(plans_dir.glob("*.json"))
    if len(existing) >= variants and not replan:
        return existing  # plans are user-editable artifacts; don't pile up extras
    made = []
    for P, world in choose_variants(dict(style, title=title), variants):
        name = f"{P['name']}_{world['name']}"
        path = plans_dir / f"{name}.json"
        made.append(path)
        if path.exists() and not replan:
            continue
        queues = {}
        for k, lines in blocks:
            queues.setdefault(k, []).append(lines)
        used_block = {}
        motif_i = {}
        prompts, sections = {}, []
        n_chorus = kinds.count("chorus")
        chorus_seen = 0
        for i, (s, kind) in enumerate(zip(segs, kinds)):
            spec = P["kinds"].get(kind) or P["kinds"]["verse"]
            every = spec.get("every", 2)
            n_img = min(MAX_UNIQUE_PER_SECTION, max(1, math.ceil(s["bars"] / every)))
            # lyric block for this section (repeat choruses reuse the same block -> same images)
            q = queues.get(kind, [])
            if kind == "chorus" and q:
                block = q[0]
            elif q:
                idx = used_block.get(kind, 0)
                block = q[min(idx, len(q) - 1)]
                used_block[kind] = idx + 1
            else:
                block = []
            mk = kind if kind in world["motifs"] else MOTIF_FALLBACK.get(kind, "verse")
            motifs = world["motifs"][mk]
            keys = []
            for j in range(n_img):
                if kind == "chorus":
                    m = motifs[j % len(motifs)]
                else:
                    c = motif_i.get(mk, 0)
                    m = motifs[c % len(motifs)]
                    motif_i[mk] = c + 1
                m = m.replace("{title}", f"visual metaphor of '{title}'")
                line = block[j % len(block)] if block else ""
                prompt = f"{line} — {m}" if line else m
                key = next((k for k, v in prompts.items() if v == prompt), None)
                if key is None:
                    key = f"{kind}{len([k for k in prompts if k.startswith(kind)]) + 1:02d}"
                    prompts[key] = prompt
                keys.append(key)
            fx = dict(spec.get("fx", {}))
            if kind == "chorus":
                chorus_seen += 1
                if chorus_seen == n_chorus and n_chorus > 1:  # last chorus hits hardest
                    for k, mul in (("uKaleido", 1.25), ("spin_speed", 1.4), ("uSparks", 1.15), ("uNeon", 1.8)):
                        if k in fx:
                            fx[k] = round(min(1.0 if k != "spin_speed" else 3.0, fx[k] * mul), 3)
                    fx["kick_flash"] = fx.get("kick_flash", 0.1) + 0.06
            sec = {"name": f"{i:02d}_{kind}", "bars": [s["bar0"], s["bar0"] + s["bars"]], "images": keys,
                   "every": every, "transition": spec.get("transition", "dissolve"),
                   "burn_beats": spec.get("burn_beats", 2), "fx": fx}
            if "blend_beats" in spec:
                sec["blend_beats"] = spec["blend_beats"]
            sections.append(sec)
        sections[-1]["bars"][1] = None
        plan = {
            "name": name,
            "audio": str(Path(audio_path).resolve()),
            "fps": 60,
            "meta": {"track": title, "preset": P["name"], "world": world["name"], "generator": "planner v1",
                     "style_source": style["source"], "tempo": meta["tempo"],
                     "notes": style.get("notes", "")},
            "style": world["style"],
            "negative": world["negative"],
            "prompts": prompts,
            "seeds": {},
            "defaults": dict(P.get("defaults", {}), **world.get("fx", {})),
            "sections": sections,
        }
        path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return made
