#!/bin/bash
cd ~/mdrp-repro/CharacterEval/data
python3 - <<'PY'
import json
cp = json.load(open("character_profiles.json"))
print("=== character_profiles.json ===")
print("type:", type(cp).__name__, "| n:", len(cp))
if isinstance(cp, dict):
    k = list(cp)[0]; v = cp[k]
    print("first key:", k, "| value type:", type(v).__name__)
    print(json.dumps({k: v}, ensure_ascii=False)[:1000])
elif isinstance(cp, list):
    print("first elem keys:", list(cp[0].keys()) if isinstance(cp[0], dict) else "n/a")
    print(json.dumps(cp[0], ensure_ascii=False)[:1000])

print("\n=== test_data.jsonl ===")
recs = [json.loads(l) for l in open("test_data.jsonl")]
print("n records:", len(recs))
r = recs[0]
print("keys:", list(r.keys()))
print(json.dumps(r, ensure_ascii=False)[:1800])

# distinct characters
chars = set()
for r in recs:
    for kk in ("role", "character", "name", "role_name"):
        if kk in r: chars.add(r[kk])
print("\ndistinct character-ish values (sample):", list(sorted(chars))[:20], "...total", len(chars))
PY
