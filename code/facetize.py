"""
facetize.py -- Phase A step 1: turn a CharacterEval profile into the paper's
Narrative Schema (Table 7 / Appendix C), via gpt-4.1. Reconstructed construction
prompt (the paper's Appendix O.1 verbatim is not public); fields follow Table 7 exactly.

Usage:  python facetize.py <character_name>
Saves ~/mdrp-repro/facets/<name>.json
"""
import os, json, sys
from openai import OpenAI

MODEL = "gpt-4.1-2025-04-14"               # the snapshot the paper used
client = OpenAI()                          # OPENAI_API_KEY from env

SYS = """你是角色人设构建助手。给定一个角色的资料，请把它组织成结构化的"叙事图式"(Narrative Schema)，作为角色扮演用的长期记忆(LTM)。严格输出 JSON 对象，顶层键如下：

- name: 角色名
- relationships: 数组，每项 {"name":人物, "relationship":关系}
- global_summary: 一段背景概述
- core_traits: 数组，每项 {"trait":核心性格特质, "desc":基于典型行为的简短解释}（约4-6条）
- scene_facets: 数组（约8-10条），每项覆盖一种"反复出现的互动情境"，字段：
    - title: 简洁的场景标签
    - time_scope: 该模式典型的故事/人生阶段
    - situation: 激活该facet的反复出现的互动情境（定义检索目标）
    - social_role: 典型社会姿态（如挑战者/保护者）
    - emotional_state: 该情境下的特征情绪
    - behavior_pattern: 对话中典型的行动/策略
    - thinking_pattern: 驱动行为的优先级/信念
    - conflict_with_core: 该facet如何延展或张力于核心特质
    - source_scenes: 可追溯的原文证据/场景
    - cue_phrases: 数组，能从对话线索(STM)匹配到该facet的词汇/概念触发器

只输出 JSON，不要额外文字。所有内容用中文。"""

def facetize(name, profile):
    user = f"角色名：{name}\n角色资料(JSON)：\n{json.dumps(profile, ensure_ascii=False, indent=2)}"
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    return json.loads(r.choices[0].message.content), r.usage

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "老默"
    base = os.path.expanduser("~/mdrp-repro")
    profs = json.load(open(f"{base}/CharacterEval/data/character_profiles.json"))
    if name not in profs:
        print("not in profiles. sample:", list(profs)[:10]); sys.exit(1)
    schema, usage = facetize(name, profs[name])
    os.makedirs(f"{base}/facets", exist_ok=True)
    out = f"{base}/facets/{name}.json"
    json.dump(schema, open(out, "w"), ensure_ascii=False, indent=2)
    print(f"saved {out}  | tokens in/out: {usage.prompt_tokens}/{usage.completion_tokens}")
    print(f"core_traits: {len(schema.get('core_traits', []))}  scene_facets: {len(schema.get('scene_facets', []))}")
    sf = schema.get("scene_facets", [])
    if sf:
        print("--- first scene_facet ---")
        print(json.dumps(sf[0], ensure_ascii=False, indent=2))
        print("--- cue_phrases across facets ---")
        for f in sf:
            print(f"  [{f.get('title','?')}] {f.get('cue_phrases', [])}")
