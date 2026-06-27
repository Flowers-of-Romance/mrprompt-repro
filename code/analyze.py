"""
analyze.py -- Phase B step: read scores.jsonl, report per-condition mean FA with
bootstrap CIs and the paired contrasts that answer the questions:
  replication (RQ2/RQ3): mrprompt vs card vs base
  FU:                    mrprompt vs mrprompt_noscene
  cue-key ablation:      mrprompt vs mrprompt_nokey / mrprompt_wrongkey   <-- our question
  FA swap-sensitivity:   true-facet adherence under full vs anti
"""
import os, json
import numpy as np

BASE = os.path.expanduser("~/mdrp-repro")

def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d, float)
    b = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return d.mean(), np.percentile(b, [2.5, 97.5])

def main():
    import sys
    scores_path = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/scores.jsonl"
    rows = [json.loads(l) for l in open(scores_path)]
    piv = {}
    for r in rows:
        piv.setdefault(r["id"], {})[r["condition"]] = r.get("fa_true")
    conds = ["base", "card", "mrprompt", "mrprompt_noscene",
             "mrprompt_nokey", "mrprompt_wrongkey", "mrprompt_anti"]
    print("=== mean FA (adherence to TRUE cued facet), per condition, [95% CI] ===")
    for c in conds:
        vals = [p[c] for p in piv.values() if p.get(c) is not None]
        if vals:
            m, (lo, hi) = boot(vals)
            print(f"  {c:20s} n={len(vals):3d}  {m:5.2f}  [{lo:.2f}, {hi:.2f}]")

    print("\n=== paired contrasts (bootstrap CI; * = excludes 0) ===")
    def contrast(a, b, label):
        d = [piv[i][a] - piv[i][b] for i in piv
             if piv[i].get(a) is not None and piv[i].get(b) is not None]
        if not d:
            print(f"  {label:36s} (no paired data)"); return
        m, (lo, hi) = boot(d); sig = "" if lo <= 0 <= hi else "  *"
        print(f"  {label:36s} {m:+.2f} [{lo:+.2f}, {hi:+.2f}]{sig}  (n={len(d)})")

    contrast("mrprompt", "card", "replication: mrprompt - card")
    contrast("card", "base", "replication: card - base")
    contrast("mrprompt", "base", "replication: mrprompt - base")
    contrast("mrprompt", "mrprompt_noscene", "FU: full - no_scene")
    contrast("mrprompt", "mrprompt_nokey", "cue-key: full - nokey")
    contrast("mrprompt", "mrprompt_wrongkey", "cue-key: full - wrongkey")
    contrast("mrprompt", "mrprompt_anti", "FA swap: full - anti (true-facet)")

    # swap follow-through: does anti response adhere to the anti facet?
    fa_anti = [r["fa_anti"] for r in rows if r.get("fa_anti") is not None]
    if fa_anti:
        m, (lo, hi) = boot(fa_anti)
        print(f"\n  anti response -> ANTI-facet adherence: {m:.2f} [{lo:.2f}, {hi:.2f}] "
              f"(high = model follows the swap)")

if __name__ == "__main__":
    main()
