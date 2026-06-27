"""merge_scores.py -- build scores_v2.jsonl = base/card from v1 (thinking-off) +
mrprompt-family from the thinking-on retest, for a fair combined analysis."""
import json, os
base = os.path.expanduser("~/mdrp-repro")
v1 = [json.loads(l) for l in open(f"{base}/scores.jsonl")]
th = [json.loads(l) for l in open(f"{base}/scores_think.jsonl")]
keep = [r for r in v1 if r["condition"] in ("base", "card")]
out = keep + th
with open(f"{base}/scores_v2.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote scores_v2.jsonl: {len(out)} rows "
      f"(base/card thinking-off + mrprompt-family thinking-on)")
