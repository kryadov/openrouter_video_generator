"""Local keyframe generation (SDXL via diffusers), cached by prompt hash.

Usage: python -m legacy.keyframes output/<track>/plans/<plan>.json [--format 16x9 9x16] [--only key1 key2] [--steps 30]
A key is regenerated only when its prompt/style/seed/size changes (or with --force).
Per-key seeds can be pinned in the clip config: CLIP["seeds"] = {"key": 1234}.
"""
import argparse
import hashlib
import json
import time
import zlib
from pathlib import Path

import torch

from legacy.render import load_clip, workdir

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
SIZES = {"16x9": (1344, 768), "9x16": (768, 1344)}

_pipe = None


def pipe():
    global _pipe
    if _pipe is None:
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        from huggingface_hub import snapshot_download
        try:  # load from the local snapshot dir: only model_index components are read, no extra downloads
            src = snapshot_download(MODEL, local_files_only=True)
        except Exception:
            src = MODEL
        p = StableDiffusionXLPipeline.from_pretrained(src, torch_dtype=torch.float16, variant="fp16",
                                                      use_safetensors=True)
        p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config, use_karras_sigmas=True)
        p.enable_model_cpu_offload()   # fits 8 GB VRAM
        p.vae.enable_tiling()
        _pipe = p
    return _pipe


def job_hash(prompt, negative, seed, size, steps, cfg):
    return hashlib.md5(json.dumps([MODEL, prompt, negative, seed, size, steps, cfg]).encode()).hexdigest()[:12]


def generate(clip, formats=("16x9",), only=None, steps=30, cfg=6.5, force=False):
    clip = load_clip(clip) if not isinstance(clip, dict) else clip
    out = workdir(clip) / "keyframes" / clip["name"]
    out.mkdir(parents=True, exist_ok=True)
    style, neg = clip.get("style", ""), clip.get("negative", "")
    seeds = clip.get("seeds", {})
    # only keys actually used by sections, in order of appearance
    used = []
    for s in clip["sections"]:
        for k in s.get("images", []):
            if k not in used:
                used.append(k)
    todo = [k for k in used if not only or k in only]
    for fmt in formats:
        W, H = SIZES[fmt]
        for key in todo:
            prompt = f"{clip['prompts'][key]}, {style}"
            seed = seeds.get(key, zlib.crc32(key.encode()) % 100000)
            h = job_hash(prompt, neg, seed, (W, H), steps, cfg)
            img_p, meta_p = out / f"{key}_{fmt}.png", out / f"{key}_{fmt}.json"
            if not force and img_p.exists() and meta_p.exists() and json.loads(meta_p.read_text()).get("hash") == h:
                continue
            t0 = time.time()
            g = torch.Generator("cuda").manual_seed(seed)
            img = pipe()(prompt=prompt, negative_prompt=neg, width=W, height=H, num_inference_steps=steps,
                         guidance_scale=cfg, generator=g).images[0]
            img.save(img_p)
            meta_p.write_text(json.dumps({"hash": h, "prompt": prompt, "negative": neg, "seed": seed,
                                          "size": [W, H], "steps": steps, "cfg": cfg, "model": MODEL}, indent=2))
            print(f"  {key}_{fmt}  {time.time()-t0:.1f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--format", nargs="+", default=["16x9"])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    generate(a.clip, a.format, a.only, a.steps, a.cfg, a.force)
