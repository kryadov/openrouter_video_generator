"""OpenRouter video generation: cached, budget-capped.

Every paid call is recorded in output/_ledger.json. A job is refused if the ledger total plus
its estimated cost would exceed the budget. Results are cached by (model, prompt, params) hash,
so a shot that was paid for is never generated again.
"""
import hashlib
import json
import time
from pathlib import Path

import requests

from core import config

try:  # use the OS certificate store (corporate proxies / TLS-inspecting antivirus add roots there)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent.parent
API = "https://openrouter.ai/api/v1"
LEDGER = ROOT / "output" / "_ledger.json"
MODELS_CACHE = ROOT / "output" / "_video_models.json"


def api_key():
    k = config.get("OPENROUTER_API_KEY")
    if not k:
        raise SystemExit("OPENROUTER_API_KEY is not set (copy .env.example to .env)")
    return k


def headers():
    return {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}


def models():
    if not MODELS_CACHE.exists():
        r = requests.get(f"{API}/videos/models", timeout=60)
        r.raise_for_status()
        MODELS_CACHE.write_text(r.text, encoding="utf-8")
    d = json.loads(MODELS_CACHE.read_text(encoding="utf-8"))
    return {m["id"]: m for m in d.get("data", d)}


def ledger():
    return json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"entries": []}


def spent():
    return round(sum(e.get("cost") or e.get("estimate", 0) for e in ledger()["entries"]), 4)


def _record(entry):
    L = ledger()
    L["entries"].append(entry)
    LEDGER.write_text(json.dumps(L, indent=2, ensure_ascii=False), encoding="utf-8")


def _update(job_id, **kw):
    L = ledger()
    for e in L["entries"]:
        if e.get("job_id") == job_id:
            e.update(kw)
    LEDGER.write_text(json.dumps(L, indent=2, ensure_ascii=False), encoding="utf-8")


def estimate(model_id, seconds, resolution="720p", audio=False):
    m = models()[model_id]
    sk = {k: float(v) for k, v in (m.get("pricing_skus") or {}).items()}
    if any(k.startswith("video_tokens") for k in sk):  # Seedance: tokens ~ w*h*fps*s/1024
        w, h = {"480p": (480, 854), "720p": (720, 1280), "1080p": (1080, 1920)}.get(resolution, (720, 1280))
        rate = sk.get("video_tokens_without_audio", sk.get("video_tokens"))
        return round(w * h * 24 * seconds / 1024 * rate, 4)
    au = "with_audio" if audio else "without_audio"
    for k in (f"duration_seconds_{au}_{resolution}", f"duration_seconds_{au}", f"duration_seconds_{resolution}",
              f"text_to_video_duration_seconds_{resolution}", "duration_seconds"):
        if k in sk:
            return round(sk[k] * seconds, 4)
    for k in (f"cents_per_video_output_second_{resolution}", "cents_per_second_output"):
        if k in sk:
            return round(sk[k] / 100 * seconds, 4)
    raise ValueError(f"no price for {model_id}: {sk}")


def job_key(model_id, prompt, seconds, aspect, resolution, seed):
    return hashlib.md5(json.dumps([model_id, prompt, seconds, aspect, resolution, seed]).encode()).hexdigest()[:10]


def submit(model_id, prompt, out_path, seconds, aspect="9:16", resolution="720p", seed=7, budget=None, extra=None):
    """Returns job dict (or None if cached / refused)."""
    out_path = Path(out_path)
    if out_path.exists():
        return None
    budget = config.budget() if budget is None else budget
    est = estimate(model_id, seconds, resolution)
    if spent() + est > budget:
        print(f"  BUDGET: skip {model_id} {out_path.name} (spent ${spent():.2f} + est ${est:.2f} > ${budget})")
        return None
    body = {"model": model_id, "prompt": prompt, "duration": seconds, "aspect_ratio": aspect,
            "resolution": resolution, "generate_audio": False, "seed": seed}
    m = models()[model_id]
    if m.get("generate_audio") is False:
        body.pop("generate_audio")
    if extra:
        body.update(extra)
    r = requests.post(f"{API}/videos", headers=headers(), json=body, timeout=120)
    if r.status_code >= 400:
        print(f"  ERROR {model_id}: {r.status_code} {r.text[:300]}")
        return None
    j = r.json()
    _record({"job_id": j["id"], "model": model_id, "out": str(out_path), "seconds": seconds,
             "estimate": est, "cost": None, "status": "submitted", "t": time.strftime("%Y-%m-%d %H:%M:%S")})
    print(f"  submitted {model_id} -> {out_path.name} (est ${est:.2f}, spent ${spent():.2f})")
    return {"id": j["id"], "polling_url": j.get("polling_url", f"{API}/videos/{j['id']}"), "out": out_path,
            "model": model_id}


def wait_all(jobs, poll=15, timeout=1800):
    pending = [j for j in jobs if j]
    t0 = time.time()
    while pending and time.time() - t0 < timeout:
        time.sleep(poll)
        for j in list(pending):
            try:
                s = requests.get(j["polling_url"], headers=headers(), timeout=60).json()
            except Exception as e:
                print("  poll error", e)
                continue
            st = s.get("status")
            if st == "completed":
                url = (s.get("unsigned_urls") or [f"{API}/videos/{j['id']}/content?index=0"])[0]
                v = requests.get(url, headers=headers(), timeout=300)
                j["out"].parent.mkdir(parents=True, exist_ok=True)
                j["out"].write_bytes(v.content)
                cost = (s.get("usage") or {}).get("cost")
                _update(j["id"], status="completed", cost=cost)
                print(f"  done {j['model']} {j['out'].name} cost ${cost} ({len(v.content)//1024} KB)")
                pending.remove(j)
            elif st == "failed":
                _update(j["id"], status="failed", cost=(s.get("usage") or {}).get("cost", 0), error=str(s.get("error")))
                print(f"  FAILED {j['model']} {j['out'].name}: {s.get('error')}")
                pending.remove(j)
    for j in pending:
        print(f"  TIMEOUT {j['model']} {j['out'].name} (job {j['id']}, still billable; re-poll later)")
    return spent()
