# -*- coding: utf-8 -*-
"""
analyze_faithful.py -- Stage E: evaluate the three contested conclusions from
scores_faithful.jsonl, under the faithful (confound-free) setup.

  (1) Structure effect : adh[mrprompt] vs adh[card] vs adh[base]
  (2) Cue-key null     : adh[mrprompt] vs adh[mrprompt_nokey] vs adh[mrprompt_wrongkey]
  (3) CoT / "card_think +0.41" : adh[card_think]-adh[card], adh[mrprompt_think]-adh[mrprompt],
      and MS-FA(ON)-MS-FA(OFF). If these ~0, the original CoT gain was a confound artifact.

Reports paired mean delta, sem, and a sign test (wins/losses) over shared instances.
No scipy dependency.
"""
import os, json, math, sys

BASE = os.path.expanduser("~/mdrp-repro")

def load(path):
    return [json.loads(l) for l in open(path)]

def col(recs, getter):
    """dict id -> value (skip None)."""
    out = {}
    for r in recs:
        v = getter(r)
        if isinstance(v, (int, float)):
            out[r["id"]] = v
    return out

def mean_sem(xs):
    n = len(xs)
    if n == 0:
        return (None, None, 0)
    m = sum(xs) / n
    if n < 2:
        return (round(m, 3), None, n)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return (round(m, 3), round(sd / math.sqrt(n), 3), n)

def paired(a, b):
    """paired deltas b-a over shared ids."""
    ids = [i for i in a if i in b]
    d = [b[i] - a[i] for i in ids]
    return d, ids

def report_contrast(label, a, b):
    d, ids = paired(a, b)
    if not d:
        print(f"  {label}: no shared instances"); return
    m, sem, n = mean_sem(d)
    wins = sum(1 for x in d if x > 0); losses = sum(1 for x in d if x < 0); ties = sum(1 for x in d if x == 0)
    ma = mean_sem([a[i] for i in ids])[0]; mb = mean_sem([b[i] for i in ids])[0]
    semstr = f"±{sem}" if sem is not None else ""
    print(f"  {label:32} Δ={m:+}{semstr}  (n={n}; {ma}→{mb}; win/tie/loss={wins}/{ties}/{losses})")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/scores_faithful.jsonl"
    recs = load(path)
    print(f"== faithful re-run analysis ({len(recs)} instances) ==\n")

    adh = {c: col(recs, lambda r, c=c: r.get("adh", {}).get(c)) for c in
           ["base", "card", "mrprompt", "mrprompt_noscene",
            "mrprompt_nokey", "mrprompt_wrongkey", "card_think", "mrprompt_think"]}
    msfa_off = col(recs, lambda r: r.get("msfa_off"))
    msfa_on = col(recs, lambda r: r.get("msfa_on"))

    print("-- condition means (single-response facet adherence, 1-10) --")
    for c, d in adh.items():
        m, sem, n = mean_sem(list(d.values()))
        print(f"  adh[{c:18}] = {m} ±{sem}  (n={n})")
    print(f"  MS-FA(OFF) = {mean_sem(list(msfa_off.values()))[0]}  (n={len(msfa_off)})")
    print(f"  MS-FA(ON)  = {mean_sem(list(msfa_on.values()))[0]}  (n={len(msfa_on)})")

    print("\n-- (1) STRUCTURE effect (paired) --")
    report_contrast("mrprompt - base", adh["base"], adh["mrprompt"])
    report_contrast("mrprompt - card", adh["card"], adh["mrprompt"])
    report_contrast("card - base", adh["base"], adh["card"])

    print("\n-- (2) CUE-KEY null (paired) --")
    report_contrast("mrprompt - nokey", adh["mrprompt_nokey"], adh["mrprompt"])
    report_contrast("mrprompt - wrongkey", adh["mrprompt_wrongkey"], adh["mrprompt"])
    report_contrast("mrprompt - noscene", adh["mrprompt_noscene"], adh["mrprompt"])

    print("\n-- (3) CoT / thinking effect (paired; confound-free) --")
    report_contrast("card_think - card", adh["card"], adh["card_think"])
    report_contrast("mrprompt_think - mrprompt", adh["mrprompt"], adh["mrprompt_think"])
    report_contrast("MS-FA(ON) - MS-FA(OFF)", msfa_off, msfa_on)

if __name__ == "__main__":
    main()
