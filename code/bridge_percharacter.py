"""
bridge_percharacter.py -- LEVER B: the CORRECT same-axis test.

The generic AFFECT/SECRECY bridge was null because a facet swap is a multi-axis
mixture and a single generic axis only catches one strand. Here, for EACH character
we build that character's OWN steering vector by diff-in-means over PARAPHRASES of its
orig vs anti disposition (para_orig / para_anti -- deliberately NOT the body text, to
avoid circularity), then cosine it against that character's facet residual delta.

If facet language-push and disposition steering move the SAME axis, the per-character
cosine should be positive in the 16-22 band -- with no generic-axis confound.
"""
import numpy as np
import torch
import cue_routing_probe as P
import steering as S
from probe_ext import EXT

BAND = (16, 23)

def boot_band(C, n=10000, seed=0):
    """C: [N, L] per-character per-layer cosine. Bootstrap CI of band[16-22] mean."""
    A = np.asarray(C, float)
    N, L = A.shape
    bl, bh = BAND[0], min(BAND[1], L)
    rng = np.random.default_rng(seed)
    vals = np.array([A[rng.integers(0, N, N)][:, bl:bh].mean() for _ in range(n)])
    return A[:, bl:bh].mean(), tuple(np.percentile(vals, [2.5, 97.5]))

def main():
    print(f"=== LEVER B: per-character bridge, n={len(EXT)} characters ===")
    tok, model = P.load("Qwen/Qwen3-8B", want_attn=False)

    print("computing per-character facet residual deltas (mode=key, gen pos) ...")
    deltas = S.facet_deltas(model, tok, EXT)            # list of [L, Hd]

    print("building per-character diff-in-means steering from paraphrases ...")
    L = model.config.num_hidden_layers
    cos = np.zeros((len(EXT), L))
    for j, it in enumerate(EXT):
        pairs = list(zip(it["para_orig"], it["para_anti"]))   # 3 (pos, neg) pairs
        steer = S.diff_in_means(model, tok, pairs)            # [L, Hd], this char only
        for i in range(L):
            cos[j, i] = S._cos(deltas[j][i], steer[i])

    m = cos.mean(0)
    arg = int(np.argmax(m))
    band_mean, (lo, hi) = boot_band(cos)
    sig = "  *CI excludes 0 -> same axis*" if lo > 0 else ""
    print("\n=== per-character bridge: cosine(facet delta, own-disposition steering) ===")
    print(f"  peak cos=L{arg} ({m[arg]:+.3f}); band[16-22] mean={band_mean:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}")
    print("    L:  " + " ".join(f"{i:>5d}" for i in range(L)))
    print("       " + " ".join(f"{v:+5.2f}" for v in m))

    print("\n=== per-character band[16-22] cosine (which characters align) ===")
    bl, bh = BAND[0], min(BAND[1], L)
    order = np.argsort(-cos[:, bl:bh].mean(1))
    for j in order:
        label = EXT[j]["situation"][:34]
        print(f"  {cos[j, bl:bh].mean():+6.3f}  {label}")

if __name__ == "__main__":
    main()
