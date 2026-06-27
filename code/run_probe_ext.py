"""
run_probe_ext.py -- LEVER A: cue-routing null + 16-22 localization at n=30,
with bootstrap CIs over items (the n=12 run had no error bars on the band).

  Tier-1 : faithfulness Delta per condition + key-routing contrasts (bootstrap CI).
  Tier-2 : per-item attention (cuekey/body) and residual norm-delta sweeps;
           band[16-22] concentration ratio reported WITH bootstrap CI over items
           (ratio CI excluding 1.0 => localization is real, not noise).
"""
import numpy as np
import cue_routing_probe as P
from probe_ext import EXT

BAND = (16, 23)

def boot_profiles(profiles, n=10000, seed=0):
    """profiles: [N, L] per-item per-layer values. Returns (mean_profile,
    band_ratio mean, band_ratio CI) where band_ratio = mean(band)/mean(all)."""
    A = np.asarray(profiles, float)
    N, L = A.shape
    bl, bh = BAND[0], min(BAND[1], L)
    rng = np.random.default_rng(seed)
    ratios = np.empty(n)
    for k in range(n):
        idx = rng.integers(0, N, N)
        m = A[idx].mean(0)
        ratios[k] = m[bl:bh].mean() / (m.mean() + 1e-12)
    m = A.mean(0)
    base_ratio = m[bl:bh].mean() / (m.mean() + 1e-12)
    return m, base_ratio, tuple(np.percentile(ratios, [2.5, 97.5]))

def show(name, profiles):
    m, r, (lo, hi) = boot_profiles(profiles)
    arg = int(np.argmax(m))
    sig = "  *band concentrated (CI excludes 1.0)*" if lo > 1.0 else ""
    print(f"  {name}: argmax=L{arg} ({m[arg]:.3f}); band[16-22]/overall ratio={r:.2f}x "
          f"[{lo:.2f}, {hi:.2f}]{sig}")
    print("    L:  " + " ".join(f"{i:>5d}" for i in range(len(m))))
    print("       " + " ".join(f"{v:5.3f}" for v in m))

def main():
    P.DATA = P.DATA + EXT
    n = len(P.DATA)
    print(f"=== LEVER A: n={n} characters (12 original + {len(EXT)} ext) ===")
    tok, model = P.load("Qwen/Qwen3-8B", want_attn=True)

    # ---- Tier-1: behavioral Delta + key-routing contrasts ----
    deltas = {c: [] for c in P.CONDITIONS}
    for it in P.DATA:
        for c in P.CONDITIONS:
            deltas[c].append(P.delta(model, tok, it, P.MODE[c]))
    print("\n=== Tier-1: faithfulness Delta (nats), mean [95% CI] ===")
    for c in P.CONDITIONS:
        m, (lo, hi) = P.bootstrap_ci(deltas[c])
        print(f"  {c:16s}  {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print("\n=== paired contrasts (key-routing hypotheses) ===")
    for a, b, label in [
        ("facet_nokey", "flat", "decomposition: nokey - flat"),
        ("facet_key", "facet_nokey", "keys at all:   key - nokey"),
        ("facet_key", "facet_wrongkey", "MATCHING cue:  key - wrongkey"),
    ]:
        d = [x - y for x, y in zip(deltas[a], deltas[b])]
        m, (lo, hi) = P.bootstrap_ci(d)
        sig = "" if lo <= 0 <= hi else "  *CI excludes 0*"
        print(f"  {label:28s}  {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")

    # ---- Tier-2a: attention localization (per item) ----
    print("\n=== Tier-2a: attention mass gen-query -> span (per-item, band CI) ===")
    acc = {"cuekey": [], "body": []}
    for it in P.DATA:
        m = P.attn_mass(model, tok, it, mode="key")
        for span in acc:
            if span in m:
                acc[span].append(m[span])
    for span in ("cuekey", "body"):
        if acc[span]:
            show(f"attn:{span}", acc[span])

    # ---- Tier-2b: residual norm-delta localization (per item) ----
    print("\n=== Tier-2b: residual ||h(orig)-h(anti)|| normalized, per-item (band CI) ===")
    import torch
    Ln = model.config.num_hidden_layers
    per_item = []
    with torch.no_grad():
        for it in P.DATA:
            po, _, _ = P.build(tok, it, "key", "orig")
            pa, _, _ = P.build(tok, it, "key", "anti")
            ho = model(po.to(model.device), output_hidden_states=True).hidden_states
            ha = model(pa.to(model.device), output_hidden_states=True).hidden_states
            prof = []
            for i in range(Ln):
                do, da = ho[i + 1][0, -1].float(), ha[i + 1][0, -1].float()
                prof.append((do - da).norm().item() / (do.norm().item() + 1e-6))
            per_item.append(prof)
    show("resid:norm", per_item)

if __name__ == "__main__":
    main()
