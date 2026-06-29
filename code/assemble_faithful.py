# -*- coding: utf-8 -*-
"""
assemble_faithful.py -- Stage B: build MS-FA instances from CharacterEval using the
faithful facet-LTMs (facets_faithful/, built via Fig.15).

For each (character, STM): pick which scene_facet the STM cues, whether it is clearly
cued, and the counter-facet (invert social_role/emotion/behavior/thinking, keep
situation). Keep only clearly-cued instances. This selection is experimental apparatus
(the paper's MS instance-selection), not a method-under-test prompt, so it is unchanged
from assemble.py -- only the facet source switched to the faithful LTMs.

Usage:  python assemble_faithful.py <N> [max_per_char]
Writes ~/mdrp-repro/instances_faithful.jsonl
"""
import os, json, sys, random
from openai import OpenAI

BASE = os.path.expanduser("~/mdrp-repro")
MODEL = "gpt-4.1-2025-04-14"
client = OpenAI()
random.seed(0)

SELECT_SYS = """给定一个角色的 scene_facets 列表，以及一段对话上下文(STM，最后一句是对话者发言，接下来该角色要回应)。请完成：
1. 判断该 STM 最可能"检索/激活"哪一个 facet（cued_index 从0开始；cued_title）。
2. clearly_cued: 该上下文是否明确指向某个 facet，且该 facet 会显著影响角色的回应（true/false）。只有 true 的样本才有诊断价值。
3. counter_facet: 该 facet 的反事实版本——保持 situation 不变，但把 social_role / emotional_state / behavior_pattern / thinking_pattern 反转为相反倾向，并给出相应的 cue_phrases。
严格输出 JSON：{"cued_index":int,"cued_title":str,"clearly_cued":bool,"counter_facet":{"title":str,"situation":str,"social_role":str,"emotional_state":str,"behavior_pattern":str,"thinking_pattern":str,"cue_phrases":[str]}}"""

def load_facets(name):
    fp = f"{BASE}/facets_faithful/{name}.json"
    return json.load(open(fp)) if os.path.exists(fp) else None

def scene_facets(schema):
    return schema.get("scene_facets") or schema.get("Personality", {}).get("scene_facets", [])

def select_invert(schema, stm):
    facets = scene_facets(schema)
    brief = [{"index": i, "title": f.get("title"), "situation": f.get("situation"),
              "social_role": f.get("social_role"), "emotional_state": f.get("emotional_state"),
              "behavior_pattern": f.get("behavior_pattern"), "thinking_pattern": f.get("thinking_pattern")}
             for i, f in enumerate(facets)]
    user = f"scene_facets:\n{json.dumps(brief, ensure_ascii=False)}\n\nSTM(对话上下文):\n{stm}"
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SELECT_SYS}, {"role": "user", "content": user}],
        response_format={"type": "json_object"}, temperature=0.4)
    return json.loads(r.choices[0].message.content)

def normalize_schema(schema):
    """Flatten Personality.* up to top level so downstream render/judge see scene_facets."""
    if "scene_facets" in schema:
        return schema
    p = schema.get("Personality", {})
    return {"name": schema.get("name", ""),
            "Nickname": schema.get("Nickname", ""),
            "relationships": schema.get("Relationships", []),
            "global_summary": schema.get("global_summary", ""),
            "core_traits": p.get("core_traits", []),
            "scene_facets": p.get("scene_facets", [])}

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_per = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    profiles = json.load(open(f"{BASE}/CharacterEval/data/character_profiles.json"))
    data = json.load(open(f"{BASE}/CharacterEval/data/test_data.jsonl"))

    by_role = {}
    for rec in data:
        if rec["role"] in profiles and load_facets(rec["role"]):
            by_role.setdefault(rec["role"], []).append(rec)
    usable = sorted(by_role, key=lambda r: -len(by_role[r]))
    print(f"usable chars (faithful facet-LTM + context): {len(usable)}; "
          f"contexts: {sum(len(v) for v in by_role.values())}")

    out = open(f"{BASE}/instances_faithful.jsonl", "w")
    n, per = 0, {}
    pools = {r: random.sample(by_role[r], min(len(by_role[r]), 12)) for r in usable}
    idx = {r: 0 for r in usable}
    while n < N and any(idx[r] < len(pools[r]) for r in usable):
        for r in usable:
            if n >= N or per.get(r, 0) >= max_per or idx[r] >= len(pools[r]):
                continue
            rec = pools[r][idx[r]]; idx[r] += 1
            try:
                sch = normalize_schema(load_facets(r))
                sel = select_invert(sch, rec["context"])
            except Exception as e:
                print("  skip (err):", r, repr(e)[:80]); continue
            if not sel.get("clearly_cued"):
                continue
            ci = sel.get("cued_index", -1)
            if not (0 <= ci < len(sch["scene_facets"])):
                continue
            out.write(json.dumps({
                "id": rec["id"], "role": r, "novel_name": rec.get("novel_name"),
                "stm": rec["context"], "schema": sch,
                "cued_index": ci, "cued_title": sel.get("cued_title"),
                "facet_anti": sel["counter_facet"]}, ensure_ascii=False) + "\n")
            n += 1; per[r] = per.get(r, 0) + 1
            print(f"  [{n}/{N}] {r} <- facet '{sel.get('cued_title')}'")
    out.close()
    print(f"wrote {n} instances across {len(per)} chars")

if __name__ == "__main__":
    main()
