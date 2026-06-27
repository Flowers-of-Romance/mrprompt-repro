"""
dryrun.py -- zero-download plumbing validation for cue_routing_probe.py
Injects a stub tokenizer + stub model (no HF download, no real weights) and
asserts the harness invariants: span construction, teacher-forcing index, and
Delta sign direction. Run: python dryrun.py
"""
import math
from types import SimpleNamespace
import torch
import torch.nn.functional as F

import cue_routing_probe as P

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
def check(name, cond):
    print(f"  [{PASS if cond else FAIL}] {name}")
    return bool(cond)

# ---------------------------------------------------------------- stub tokenizer
class StubTok:
    """whitespace tokenizer; deterministic word->id. add_special_tokens ignored."""
    def __init__(self): self.vocab = {}
    def _id(self, w): return self.vocab.setdefault(w, len(self.vocab) + 10)
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[self._id(w) for w in text.split()])

tok = StubTok()
item = P.DATA[0]

# ================================================================ 1. span tests (all items)
print(f"1. span construction ({len(P.DATA)} items): per-item all invariants")
ok = True
for idx, it in enumerate(P.DATA):
    ids_by, span_by = {}, {}
    item_ok = True
    for cond in P.CONDITIONS:
        ids, spans, _ = P.build(tok, it, P.MODE[cond], "orig")
        ids_by[cond], span_by[cond] = ids, spans
        T = ids.shape[1]
        segs = sorted(spans.values())
        covered = (segs[0][0] == 0 and segs[-1][1] == T and
                   all(segs[i][1] == segs[i + 1][0] for i in range(len(segs) - 1)))
        item_ok &= covered and spans["asst_head"][1] == T   # cover + query==last token
    def seg(cond, name):
        s, e = span_by[cond][name]; return ids_by[cond][0, s:e].tolist()
    # structural: cuekey present only for key/wrongkey
    item_ok &= ("cuekey" not in span_by["flat"] and "cuekey" not in span_by["facet_nokey"]
                and "cuekey" in span_by["facet_key"] and "cuekey" in span_by["facet_wrongkey"])
    # body byte-identical across key/nokey/wrongkey (only keys vary)
    item_ok &= seg("facet_key", "body") == seg("facet_nokey", "body") == seg("facet_wrongkey", "body")
    # cuekey: content differs but LENGTH matches (the new length-control invariant)
    ck, cw = seg("facet_key", "cuekey"), seg("facet_wrongkey", "cuekey")
    item_ok &= (ck != cw) and (len(ck) == len(cw))
    ok &= check(f"item {idx:2d}: cover+query, cuekey-presence, body-identical, "
                f"cuekey diff & len-matched (|cuekey|={len(ck)})", item_ok)

# ================================================================ 2. teacher-forcing index
print("\n2. answer_logprob teacher-forcing index")
V = 4096
class IndexStub:
    """rewards ONLY logits[P-1+i, answer_ids[i]] = 5.0; everything else 0."""
    device = "cpu"
    ans_len = 0
    ans_ids = []
    def __call__(self, full):
        Tt = full.shape[1]; Pn = Tt - self.ans_len
        lg = torch.zeros(1, Tt, V)
        for i, a in enumerate(self.ans_ids):
            lg[0, Pn - 1 + i, a] = 5.0
        return SimpleNamespace(logits=lg)

stub = IndexStub()
prompt_ids, _, _ = P.build(tok, item, P.MODE["facet_key"], "orig")
ans_ids = tok(item["ans_orig"]).input_ids
stub.ans_len, stub.ans_ids = len(ans_ids), ans_ids
got = P.answer_logprob(stub, prompt_ids, ans_ids)
# expected if indexing is correct: each token contributes 5 - logsumexp([5, 0*(V-1)])
per_tok = 5.0 - math.log(math.exp(5.0) + (V - 1))
expected = len(ans_ids) * per_tok
ok &= check(f"logprob matches correct-index value ({got:.3f} ~= {expected:.3f})",
            abs(got - expected) < 1e-2)
# what an off-by-one would have yielded (reads a zero-logit cell): len*(0 - logsumexp(zeros))
wrong = len(ans_ids) * (0.0 - math.log(V))
ok &= check(f"clearly separated from off-by-one value ({wrong:.3f})",
            abs(got - wrong) > 1.0)

# ================================================================ 3. Delta sign direction
print("\n3. Delta sign (faithful vs pass-through model)")
# tag bodies so a stub can detect which facet is installed (test-only copy)
tagged = dict(item)
tagged["body_orig"] = item["body_orig"] + " ORIGTAG"
tagged["body_anti"] = item["body_anti"] + " ANTITAG"
orig_tag, anti_tag = tok("ORIGTAG").input_ids[0], tok("ANTITAG").input_ids[0]
orig_ans = set(tok(item["ans_orig"]).input_ids)
anti_ans = set(tok(item["ans_anti"]).input_ids)

class FaithfulStub:
    """sees the installed facet via tag; rewards that facet's answer tokens (+8) everywhere."""
    device = "cpu"
    def __call__(self, full):
        toks = set(full[0].tolist()); Tt = full.shape[1]
        favored = orig_ans if orig_tag in toks else anti_ans if anti_tag in toks else set()
        lg = torch.zeros(1, Tt, V)
        for v in favored:
            if v < V: lg[0, :, v] = 8.0
        return SimpleNamespace(logits=lg)

class PassThroughStub:
    device = "cpu"
    def __call__(self, full):
        return SimpleNamespace(logits=torch.zeros(1, full.shape[1], V))

d_faith = P.delta(FaithfulStub(), tok, tagged, P.MODE["facet_key"])
d_pass  = P.delta(PassThroughStub(), tok, tagged, P.MODE["facet_key"])
ok &= check(f"faithful model -> Delta strongly positive ({d_faith:+.3f})", d_faith > 1.0)
ok &= check(f"pass-through model -> Delta ~ 0 ({d_pass:+.3f})", abs(d_pass) < 1e-6)

print("\n" + ("ALL PASS -- wiring is correct; ready for a real model run."
              if ok else "SOME CHECKS FAILED -- fix before a model run."))
exit(0 if ok else 1)
