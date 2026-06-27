"""
assemble.py -- Phase A step 2: build MS-FA instances from CharacterEval.

For each (character, STM=context): facetize the character (cached), then ask gpt-4.1
which scene_facet the STM cues, whether it is clearly cued, and the counter-facet
(M_L^anti: invert social_role/emotion/behavior/thinking, keep situation). Keep only
clearly-cued instances -- the paper's MS selection ("STM where facet differences induce
measurably different continuations").

Usage:  python assemble.py <N_instances> [max_per_char]
Writes ~/mdrp-repro/instances.jsonl
"""
import os, json, sys, random
from facetize import facetize, client, MODEL

BASE = os.path.expanduser("~/mdrp-repro")
random.seed(0)

SELECT_SYS = """给定一个角色的 scene_facets 列表，以及一段对话上下文(STM，最后一句是对话者发言，接下来该角色要回应)。请完成：
1. 判断该 STM 最可能"检索/激活"哪一个 facet（cued_index 从0开始；cued_title）。
2. clearly_cued: 该上下文是否明确指向某个 facet，且该 facet 会显著影响角色的回应（true/false）。只有 true 的样本才有诊断价值。
3. counter_facet: 该 facet 的反事实版本——保持 situation 不变，但把 social_role / emotional_state / behavior_pattern / thinking_pattern 反转为相反倾向，并给出相应的 cue_phrases。
严格输出 JSON：{"cued_index":int,"cued_title":str,"clearly_cued":bool,"counter_facet":{"title":str,"situation":str,"social_role":str,"emotional_state":str,"behavior_pattern":str,"thinking_pattern":str,"cue_phrases":[str]}}"""

def get_facets(name, profiles, cache):
    if name in cache:
        return cache[name]
    fp = f"{BASE}/facets/{name}.json"
    if os.path.exists(fp):
        sch = json.load(open(fp))
    else:
        sch, _ = facetize(name, profiles[name])
        os.makedirs(f"{BASE}/facets", exist_ok=True)
        json.dump(sch, open(fp, "w"), ensure_ascii=False, indent=2)
    cache[name] = sch
    return sch

def select_invert(schema, stm):
    facets = schema.get("scene_facets", [])
    brief = [{"index": i, "title": f.get("title"), "situation": f.get("situation"),
              "social_role": f.get("social_role"), "emotional_state": f.get("emotional_state"),
              "behavior_pattern": f.get("behavior_pattern"), "thinking_pattern": f.get("thinking_pattern")}
             for i, f in enumerate(facets)]
    user = f"scene_facets:\n{json.dumps(brief, ensure_ascii=False)}\n\nSTM(对话上下文):\n{stm}"
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SELECT_SYS}, {"role": "user", "content": user}],
        response_format={"type": "json_object"}, temperature=0.4,
    )
    return json.loads(r.choices[0].message.content)

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    max_per = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    profiles = json.load(open(f"{BASE}/CharacterEval/data/character_profiles.json"))
    data = json.load(open(f"{BASE}/CharacterEval/data/test_data.jsonl"))

    by_role = {}
    for rec in data:
        if rec["role"] in profiles:
            by_role.setdefault(rec["role"], []).append(rec)
    usable = sorted(by_role, key=lambda r: -len(by_role[r]))
    print(f"usable chars (profile+context): {len(usable)}; total contexts: {sum(len(v) for v in by_role.values())}")

    cache = {}
    out = open(f"{BASE}/instances.jsonl", "w")
    n, per = 0, {}
    # round-robin across characters for diversity
    pools = {r: random.sample(by_role[r], min(len(by_role[r]), 12)) for r in usable}
    order = [r for r in usable]
    idx = {r: 0 for r in usable}
    while n < N and any(idx[r] < len(pools[r]) for r in order):
        for r in order:
            if n >= N or per.get(r, 0) >= max_per or idx[r] >= len(pools[r]):
                continue
            rec = pools[r][idx[r]]; idx[r] += 1
            try:
                sch = get_facets(r, profiles, cache)
                sel = select_invert(sch, rec["context"])
            except Exception as e:
                print("  skip (err):", r, repr(e)[:80]); continue
            if not sel.get("clearly_cued"):
                continue
            ci = sel.get("cued_index", -1)
            facets = sch.get("scene_facets", [])
            if not (0 <= ci < len(facets)):
                continue
            inst = {
                "id": rec["id"], "role": r, "novel_name": rec.get("novel_name"),
                "stm": rec["context"], "schema": sch,
                "cued_index": ci, "cued_title": sel.get("cued_title"),
                "facet_anti": sel["counter_facet"],
            }
            out.write(json.dumps(inst, ensure_ascii=False) + "\n")
            n += 1; per[r] = per.get(r, 0) + 1
            print(f"  [{n}/{N}] {r} <- facet '{sel.get('cued_title')}'")
    out.close()
    print(f"wrote {n} instances to {BASE}/instances.jsonl across {len(per)} chars: {per}")

if __name__ == "__main__":
    main()
