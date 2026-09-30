"""Settings from .env (repo root) with environment-variable override. No extra dependencies."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "OPENROUTER_API_KEY": "",
    "VIDEO_BUDGET_USD": "10",
    "COMPARE_MODELS": "google/veo-3.1-lite,google/veo-3.1-fast,kwaivgi/kling-v3.0-std,bytedance/seedance-2.0-fast",
    "REFINE_MODEL": "bytedance/seedance-2.0-fast",
    "VIDEO_RESOLUTION": "720p",
    "VIDEO_ENCODER": "auto",
}


def _dotenv():
    p = ROOT / ".env"
    vals = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"\'')
    return vals


def get(key):
    return os.environ.get(key) or _dotenv().get(key) or DEFAULTS.get(key, "")


def models_list(key="COMPARE_MODELS"):
    return [m.strip() for m in get(key).split(",") if m.strip()]


def budget():
    return float(get("VIDEO_BUDGET_USD") or 10)
