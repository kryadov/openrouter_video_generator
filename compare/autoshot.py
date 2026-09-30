"""Default shot list for the model comparison — deterministic, no LLM.

Two-bar shots over the chosen excerpt; the shot vocabulary depends on the track's energy.
The style description (style.txt) and the first chorus (lyrics.txt) go into every prompt as
context. The result is an editable JSON artifact; carry it over to refine/ once a model is chosen.
"""
import re

from core import config

ENERGETIC = [
    "On the very first frame, a striking wide establishing shot that captures the mood of the track, with a sudden burst of light and motion",
    "a performer seen from behind on a stage in dramatic light, playing with full energy",
    "a crowd moving and jumping in rhythm, wide shot, beams of light sweeping over them",
    "close-up of hands on an instrument, light flashing on the beat",
    "a fast dynamic camera move through the location of the video, light streaks flying past",
    "an aerial top-down shot over the scene, patterns of light and moving people",
    "the visual climax of the video: the single biggest moment of light and energy",
    "a calmer closing wide shot as the scene settles, lights glowing, the camera slowly rising",
]
CALM = [
    "On the very first frame, a beautiful atmospheric wide shot that captures the mood of the track",
    "a person seen from behind in a quiet evocative place that matches the mood of the song",
    "a meaningful close-up of a small object in soft light, moving gently",
    "a slow camera move through an empty evocative location at the right time of day",
    "two silhouettes walking away from the camera in soft light",
    "a wide view over a city or landscape with gentle motion in the light",
    "an intimate detail: light, steam, reflections or curtains moving slowly",
    "a calm closing shot, the camera slowly pulling back as the light fades",
]
CALM_WORDS = ("chill", "ambient", "laid-back", "lo-fi", "lofi", "calm", "ballad", "downtempo", "эмбиент", "спокойн")


def first_chorus(lyrics):
    blocks = re.split(r"^\[(.+?)\]\s*$", lyrics, flags=re.M)
    for i in range(1, len(blocks) - 1, 2):
        if "chorus" in blocks[i].lower() or "припев" in blocks[i].lower():
            return " / ".join(l.strip() for l in blocks[i + 1].splitlines() if l.strip())
    return ""


def build(title, excerpt, tempo, style_text, lyrics, model, bars=None):
    bars = bars or excerpt["bars"]
    calm = tempo < 110 or any(w in style_text.lower() for w in CALM_WORDS)
    vocab = CALM if calm else ENERGETIC
    pace = ("calm and dreamy, slow smooth camera movement, gentle unhurried motion" if calm
            else "high energy, fast-paced dynamic motion, movement on the beat")
    chorus = first_chorus(lyrics) if lyrics else ""
    music = f"Shot for the music video of '{title}', a {tempo:.0f} BPM track described as: {style_text.strip()[:400]}; {pace}"
    if chorus:
        music += f". The song's chorus: \"{chorus[:300]}\""
    n = max(1, bars // 2)
    shots = []
    for j in range(n):
        b1 = (j + 1) * 2 if j < n - 1 else bars + excerpt.get("tail_bars", 0)
        idea = vocab[round(j * (len(vocab) - 1) / max(1, n - 1))]
        shots.append({"id": f"c{j + 1}", "bars": [j * 2, b1],
                      "prompt": "{music}. Vertical 9:16 frame. " + idea + ". {look}. {rules}"})
    cast = config.get("VIDEO_CAST")
    rules = ("nobody sings or speaks on camera; no close-ups of faces; keep the main action in the centre of the frame; "
             "no text, no letters, no subtitles, no watermark")
    if cast:
        rules = f"{cast}; {rules}"
    return {
        "excerpt": excerpt,
        "aspect_ratio": "9:16",
        "resolution": config.get("VIDEO_RESOLUTION"),
        "model": model,
        "music": music,
        "look": "photorealistic live-action music video, shot on 35mm film, cinematic lighting, rich contrast",
        "rules": rules,
        "grade": "eq=contrast=1.05:saturation=1.08,noise=alls=5:allf=t",
        "shots": shots,
    }
