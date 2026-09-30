"""GPU renderer: clip config + audio features -> per-section cached chunks -> final mp4.

Usage: python -m legacy.render output/<track>/plans/<plan>.json [--format 16x9|9x16] [--preview SECONDS_START SECONDS_LEN]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

from core import analyze

ROOT = Path(__file__).resolve().parent.parent
SHADERS = Path(__file__).resolve().parent / "shaders"

FORMATS = {"16x9": (1920, 1080), "9x16": (1080, 1920)}

# Every scene uniform and its default. Clip sections override any of these.
DEFAULTS = {
    "uZoom": 1.0, "uRot": 0.0, "uKickZoom": 0.03,
    "uKaleido": 0.0, "uKSeg": 6.0, "uSpin": 0.0, "uWarp": 0.0, "uChroma": 0.3,
    "uSparks": 0.0, "uFlames": 0.0, "uPaper": 0.5, "uFold": 0.0, "uNeon": 0.0,
    "uSat": 1.0, "uContrast": 1.05, "uBright": 1.0, "uVignette": 0.6, "uTunnel": 0.0,
    "uGlitch": 0.0, "uLanterns": 0.0,
    "tintR": 1.0, "tintG": 1.0, "tintB": 1.0,
    "sparkR": 1.6, "sparkG": 0.7, "sparkB": 0.25,
    "bloom": 0.6, "grain": 0.035,
    # time-varying helpers (not direct uniforms)
    "spin_speed": 0.0,    # uSpin += spin_speed * cumulative kick seconds
    "rot_speed": 0.0,     # radians / s
    "drift": 0.06,        # ken-burns zoom growth over an image's lifetime
    "pan": 0.04,          # ken-burns pan amount
    "kick_flash": 0.0,    # flash on strong kicks
    "kaleido_pulse": 0.0, # kaleido amount added by kick envelope
}
TRANSITIONS = {"burn": 0.0, "dissolve": 1.0, "cut": 2.0}


def load_clip(path):
    """Plan JSON (output/<track>/plans/<name>.json). workdir defaults to the plan's grandparent."""
    path = Path(path).resolve()
    clip = json.loads(path.read_text(encoding="utf-8"))
    clip.setdefault("workdir", str(path.parent.parent))
    return clip


def workdir(clip):
    return Path(clip["workdir"])


def audio_path(clip):
    a = Path(clip["audio"])
    return a if a.is_absolute() else (workdir(clip) / a).resolve()


def features(clip):
    fps = clip.get("fps", 60)
    return analyze.load(audio_path(clip), fps, workdir(clip) / "analysis")


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


class Timeline:
    """Turns sections into per-frame arrays for every uniform + image events."""

    def __init__(self, clip, feats, n_frames, fps):
        self.clip, self.f, self.n, self.fps = clip, feats, n_frames, fps
        beats = feats["beats"]
        self.beats = beats
        bb = clip.get("bar_beats", 4)
        self.beat_len = float(np.median(np.diff(beats)))

        def bar_frame(b):
            i = int(b * bb)
            if i < len(beats):
                return int(beats[i])
            return int(beats[-1] + (i - len(beats) + 1) * self.beat_len)

        self.bar_frame = bar_frame
        secs = []
        for s in clip["sections"]:
            b0, b1 = s["bars"]
            f0 = 0 if b0 == 0 else bar_frame(b0)
            f1 = n_frames if b1 is None else min(n_frames, bar_frame(b1))
            secs.append(dict(s, f0=f0, f1=f1))
        secs[-1]["f1"] = n_frames
        self.sections = secs
        self._build_params()
        self._build_images()

    def _build_params(self):
        n = self.n
        keys = list(DEFAULTS)
        P = {k: np.full(n, DEFAULTS[k], np.float32) for k in keys}
        base = dict(DEFAULTS, **self.clip.get("defaults", {}))
        prev = base
        for s in self.sections:
            cur = dict(base, **s.get("fx", {}))
            f0, f1 = s["f0"], s["f1"]
            blend = int(s.get("blend_beats", 2) * self.beat_len)
            idx = np.arange(f0, f1)
            w = smooth((idx - f0) / max(blend, 1)) if blend > 0 else np.ones(len(idx))
            for k in keys:
                P[k][f0:f1] = prev[k] + (cur[k] - prev[k]) * w
            prev = cur
        t = np.arange(n) / self.fps
        f = self.f
        # global rotation accumulates (smooth, uniform); twist stays bounded and pulses with the kick
        P["uRot"] = P["uRot"] + np.cumsum(P["rot_speed"] + 0.35 * P["spin_speed"] * f["gkick_env"]) / self.fps
        P["uSpin"] = P["uSpin"] + P["spin_speed"] * (0.12 + 0.18 * f["gkick_env"])
        P["uKaleido"] = np.clip(P["uKaleido"] + P["kaleido_pulse"] * f["gkick_env"], 0, 1)
        P["uFlash"] = P["kick_flash"] * f["gkick_env"] ** 4
        P["t"] = t.astype(np.float32)
        self.P = P

    def _build_images(self):
        """Image events: (frame, key, transition, dur_frames)."""
        ev = []
        for s in self.sections:
            imgs = s.get("images", [])
            if not imgs:
                continue
            every = s.get("every", 4)  # bars per image
            tr = s.get("transition", "burn")
            dur = int(s.get("burn_beats", 2) * self.beat_len)
            b0, b1 = s["bars"]
            b1 = b1 if b1 is not None else b0 + 999
            b, i = b0, 0
            while b < b1:
                fr = s["f0"] if b == b0 else self.bar_frame(b)
                if fr >= s["f1"]:
                    break
                ev.append((fr, imgs[i % len(imgs)], tr, dur))
                b += every
                i += 1
        self.events = ev
        n = self.n
        A = np.zeros(n, np.int32)
        B = np.zeros(n, np.int32)
        mix = np.zeros(n, np.float32)
        tr = np.zeros(n, np.float32)
        seed = np.zeros(n, np.float32)
        local = np.zeros(n, np.float32)  # 0..1 progress within the image lifetime
        pan = np.zeros((n, 2), np.float32)
        for i, (fr, key, t, dur) in enumerate(ev):
            end = ev[i + 1][0] if i + 1 < len(ev) else n
            rng = np.random.default_rng(zlib.crc32(key.encode()) + i)
            d = rng.normal(size=2)
            d /= np.linalg.norm(d) + 1e-9
            sl = slice(fr, end)
            A[sl] = i
            B[sl] = i
            L = max(end - fr, 1)
            local[sl] = np.arange(end - fr) / L
            pan[sl] = d * (local[sl, None] - 0.5)
            if i > 0 and dur > 0:
                te = min(fr + dur, end)
                A[fr:te] = i - 1
                mix[fr:te] = np.linspace(0, 1, te - fr, endpoint=False)
                tr[fr:te] = TRANSITIONS[t]
                seed[fr:te] = (i * 0.618) % 1.0 * 10
        self.imgA, self.imgB, self.mix, self.trans, self.seed = A, B, mix, tr, seed
        self.local, self.panv = local, pan

    def image_keys(self):
        return [e[1] for e in self.events]


class Renderer:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.ctx = moderngl.create_standalone_context(require=330)
        info = self.ctx.info
        print("GL:", info["GL_RENDERER"], info["GL_VERSION"], flush=True)
        vs = """#version 330
        in vec2 in_pos; void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }"""
        self.scene = self.ctx.program(vertex_shader=vs, fragment_shader=(SHADERS / "scene.frag").read_text())
        self.post = self.ctx.program(vertex_shader=vs, fragment_shader=(SHADERS / "post.frag").read_text())
        quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        self.vao_s = self.ctx.vertex_array(self.scene, [(quad, "2f", "in_pos")])
        self.vao_p = self.ctx.vertex_array(self.post, [(quad, "2f", "in_pos")])
        self.tex_scene = self.ctx.texture((W, H), 4, dtype="f2")
        self.tex_scene.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.fbo_scene = self.ctx.framebuffer([self.tex_scene])
        self.tex_out = self.ctx.texture((W, H), 3)
        self.fbo_out = self.ctx.framebuffer([self.tex_out])
        self.textures = {}

    def texture(self, path):
        path = str(path)
        if path not in self.textures:
            im = Image.open(path).convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
            t = self.ctx.texture(im.size, 3, im.tobytes())
            t.build_mipmaps()
            t.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            t.anisotropy = 8.0
            t.repeat_x = t.repeat_y = False
            self.textures[path] = (t, im.size[0] / im.size[1])
        return self.textures[path]

    def set(self, prog, name, val):
        if name in prog:
            prog[name].value = val

    def frame(self, u, texA, texB):
        s = self.scene
        (ta, aa), (tb, ab) = texA, texB
        ta.use(0)
        tb.use(1)
        self.set(s, "uTexA", 0)
        self.set(s, "uTexB", 1)
        self.set(s, "uAspA", aa)
        self.set(s, "uAspB", ab)
        self.set(s, "uRes", (float(self.W), float(self.H)))
        for k, v in u.items():
            self.set(s, k, v)
        self.fbo_scene.use()
        self.vao_s.render(moderngl.TRIANGLE_STRIP)
        self.tex_scene.build_mipmaps()
        self.fbo_out.use()
        self.tex_scene.use(0)
        p = self.post
        self.set(p, "uScene", 0)
        self.set(p, "uRes", (float(self.W), float(self.H)))
        self.set(p, "uBloom", u["bloom"])
        self.set(p, "uGrain", u["grain"])
        self.set(p, "uTime", u["uTime"])
        self.vao_p.render(moderngl.TRIANGLE_STRIP)
        return self.fbo_out.read(components=3, alignment=1)


def placeholder(path, key, W, H, prompt="", tint=(1.0, 1.0, 1.0)):
    """Procedural stand-in when a keyframe hasn't been generated yet: layered
    colour noise + a caption naming the keyframe that will replace it."""
    from PIL import ImageDraw, ImageFilter, ImageFont
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    img = np.zeros((H, W, 3), np.float32)
    for octave, amp in ((6, 0.55), (18, 0.3), (54, 0.15)):
        g = rng.random((octave * H // W + 2, octave + 2, 3)).astype(np.float32)
        up = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC), np.float32) / 255
        img += up * amp
    base = rng.uniform(0.15, 1.0, 3) * np.array(tint)
    img = np.clip((img - 0.25) * 1.6, 0, 1) * base + 0.05
    y, x = np.mgrid[0:H, 0:W]
    glow = np.exp(-(((x - W * rng.uniform(0.3, 0.7)) / W) ** 2 + ((y - H * rng.uniform(0.3, 0.7)) / H) ** 2) * 6)
    img = np.clip(img + glow[..., None] * rng.uniform(0.3, 1.0, 3) * 0.6, 0, 1)
    im = Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 34)
        small = ImageFont.truetype("arial.ttf", 20)
    except OSError:
        font = small = ImageFont.load_default()
    d.text((30, H - 110), f"DRAFT · {key}", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((30, H - 60), prompt[:110], fill=(230, 230, 230), font=small, stroke_width=1, stroke_fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def keyframe_path(clip, key, fmt):
    d = workdir(clip) / "keyframes" / clip["name"]
    for cand in (d / f"{key}_{fmt}.png", d / f"{key}_16x9.png", d / f"{key}.png"):
        if cand.exists():
            return cand
    return None


def file_hash(p):
    st = Path(p).stat()
    return f"{p}:{st.st_size}:{st.st_mtime_ns}"


def ffmpeg_writer(path, W, H, fps):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-vf", "vflip",
           "-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "0",
           "-maxrate", "80M", "-bufsize", "160M", "-profile:v", "high", "-pix_fmt", "yuv420p",
           "-g", str(fps * 2), str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render(clip, fmt="16x9", preview=None, out=None, draft=False):
    clip = load_clip(clip) if not isinstance(clip, dict) else clip
    audio = audio_path(clip)
    fps = clip.get("fps", 60)
    feats = features(clip)
    n = int(np.ceil(float(feats["duration"]) * fps))
    tl = Timeline(clip, feats, n, fps)
    W, H = FORMATS[fmt]
    r = Renderer(W, H)

    kf = {}
    for key in set(tl.image_keys()):
        p = keyframe_path(clip, key, fmt)
        if p is None:
            p = workdir(clip) / "placeholders" / f"{key}.png"
            if not p.exists():
                d = clip.get("defaults", {})
                placeholder(p, key, 1344, 768, clip.get("prompts", {}).get(key, ""),
                            (d.get("tintR", 1.0), d.get("tintG", 1.0), d.get("tintB", 1.0)))
            print("  placeholder for", key)
        kf[key] = p

    shader_src = (SHADERS / "scene.frag").read_text() + (SHADERS / "post.frag").read_text()
    chunk_dir = workdir(clip) / "chunks" / clip["name"] / fmt
    chunk_dir.mkdir(parents=True, exist_ok=True)

    ranges = [(s["f0"], s["f1"], s["name"]) for s in tl.sections]
    if preview:
        a, l = int(preview[0] * fps), int(preview[1] * fps)
        ranges = [(max(f0, a), min(f1, a + l), nm) for f0, f1, nm in ranges if f1 > a and f0 < a + l]

    feat_keys = ["gkick_env", "bass_s", "mid_s", "high_s", "energy", "beat_phase", "cumkick"]
    uni_map = {"gkick_env": "uKick", "bass_s": "uBass", "mid_s": "uMid", "high_s": "uHigh",
               "energy": "uEnergy", "beat_phase": "uBeatPhase", "cumkick": "uCumKick"}
    chunks = []
    t_start = time.time()
    for f0, f1, name in ranges:
        sl = slice(f0, f1)
        h = hashlib.md5()
        h.update(shader_src.encode())
        h.update(f"{W}x{H}@{fps}".encode())
        for k in sorted(tl.P):
            h.update(tl.P[k][sl].tobytes())
        for arr in (tl.imgA, tl.imgB, tl.mix, tl.trans, tl.seed, tl.local, tl.panv):
            h.update(arr[sl].tobytes())
        for k in feat_keys:
            h.update(feats[k][sl].tobytes())
        used = sorted({tl.events[i][1] for i in np.unique(np.concatenate([tl.imgA[sl], tl.imgB[sl]]))})
        for k in used:
            h.update(file_hash(kf[k]).encode())
        cpath = chunk_dir / f"{f0:06d}_{f1:06d}_{name}_{h.hexdigest()[:10]}.mp4"
        chunks.append(cpath)
        if cpath.exists():
            print(f"  [cached] {name} {f0/fps:.1f}-{f1/fps:.1f}s")
            continue
        print(f"  rendering {name} {f0/fps:.1f}-{f1/fps:.1f}s ({f1-f0} frames)", flush=True)
        tmp = cpath.with_suffix(".tmp.mp4")
        w = ffmpeg_writer(tmp, W, H, fps)
        t0 = time.time()
        for fi in range(f0, f1):
            P = tl.P
            ev_a, ev_b = tl.events[tl.imgA[fi]], tl.events[tl.imgB[fi]]
            u = {k: float(P[k][fi]) for k in DEFAULTS if k.startswith("u")}
            zoom = P["uZoom"][fi] * (1.0 + P["drift"][fi] * tl.local[fi])
            u.update({
                "uTime": float(P["t"][fi]), "uZoom": float(zoom),
                "uPan": tuple(float(v) for v in tl.panv[fi] * P["pan"][fi]),
                "uMix": float(tl.mix[fi]), "uSeed": float(tl.seed[fi]), "uTransition": float(tl.trans[fi]),
                "uFlash": float(P["uFlash"][fi]),
                "uTint": (float(P["tintR"][fi]), float(P["tintG"][fi]), float(P["tintB"][fi])),
                "uSparkColor": (float(P["sparkR"][fi]), float(P["sparkG"][fi]), float(P["sparkB"][fi])),
                "bloom": float(P["bloom"][fi]), "grain": float(P["grain"][fi]),
            })
            for k in feat_keys:
                u[uni_map[k]] = float(feats[k][fi])
            buf = r.frame(u, r.texture(kf[ev_a[1]]), r.texture(kf[ev_b[1]]))
            w.stdin.write(buf)
        w.stdin.close()
        w.wait()
        tmp.replace(cpath)
        dt = time.time() - t0
        print(f"    {(f1-f0)/dt:.0f} fps", flush=True)

    out_dir = workdir(clip)
    suffix = ("_draft" if draft or any("placeholders" in str(v) for v in kf.values()) else "") +              ("_preview" if preview else "")
    out = Path(out) if out else out_dir / f"{clip['name']}_{fmt}{suffix}.mp4"
    lst = chunk_dir / "concat.txt"
    lst.write_text("".join(f"file '{c.as_posix()}'\n" for c in chunks))
    a_off = ranges[0][0] / fps
    dur = (ranges[-1][1] - ranges[0][0]) / fps
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
           "-ss", f"{a_off:.4f}", "-t", f"{dur:.4f}", "-i", str(audio),
           "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
           "-movflags", "+faststart", "-shortest", str(out)]
    subprocess.run(cmd, check=True)
    print(f"done -> {out}  ({time.time()-t_start:.0f}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--format", default="16x9", choices=list(FORMATS))
    ap.add_argument("--preview", nargs=2, type=float, metavar=("START", "LEN"))
    ap.add_argument("--out")
    a = ap.parse_args()
    render(a.clip, a.format, a.preview, a.out)
