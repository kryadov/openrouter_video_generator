"""Part 2 — refine a short on the chosen model, iterating on versioned shot lists.

  python -m refine "My Track" shotlist_v2.json --dry-run
  python -m refine "My Track" shotlist_v2.json
  python -m refine "My Track" shotlist_v2.json --assemble-only   # re-edit for free

Shot lists live in output/<track>/short/. Only shots whose prompt/seed/params changed are
generated; everything else comes from the cache, so fixes cost only the shots you touch.
Model: --model > shot list "model" > REFINE_MODEL (.env).
"""
import argparse
import sys

from core import config, videogen
from core.edit import assemble
from core.shotlist import generate, load


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track", help="track name (folder name in input/)")
    ap.add_argument("shotlist", help="shot list file in output/<track>/short/")
    ap.add_argument("--model")
    ap.add_argument("--budget", type=float, help="USD cap for the ledger total (default VIDEO_BUDGET_USD)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--take", help="output file suffix override")
    a = ap.parse_args()
    track = a.track.rstrip("/\\").split("/")[-1].split("\\")[-1]
    c = load(track, f"short/{a.shotlist}")
    mid = a.model or c.sl.get("model") or config.get("REFINE_MODEL")
    if not a.assemble_only:
        generate(c, mid, a.budget, a.dry_run)
    if not a.dry_run:
        assemble(c, mid, a.take)
        print(f"ledger total ${videogen.spent():.2f}")


if __name__ == "__main__":
    main()
