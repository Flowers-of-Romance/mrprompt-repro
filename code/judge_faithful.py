# -*- coding: utf-8 -*-
"""
judge_faithful.py -- Stage D scoring with the paper's REAL metric.

Primary: MS-FA (Table 21) = CONTRASTIVE facet separability. For one instance (same STM),
the judge sees the response under the TRUE facet-LTM and under the INVERTED (anti)
facet-LTM, plus both facet descriptions, and rates 1/5/10 separability+own-facet
faithfulness. This replaces the retracted single-response FA judge.

Computed:
  - MS-FA(OFF) : pair (mrprompt, mrprompt_anti)
  - MS-FA(ON)  : pair (mrprompt_think, mrprompt_anti_think)   -> isolates CoT's effect
Secondary (single-response facet adherence, for OUR cue-key ablation + a card descriptor):
  - adh[cond] : how well cond's response enacts the CUED facet (base/card/mrprompt/
    mrprompt_noscene/mrprompt_nokey/mrprompt_wrongkey).

Writes ~/mdrp-repro/scores_faithful.jsonl  (one record per instance id).
"""
import os, json, sys
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import faithful_prompts as F

client = OpenAI()
JUDGE = "gpt-4.1-mini-2025-04-14"
BASE = os.path.expanduser("~/mdrp-repro")

ADH_SYS = """你是严格的角色扮演评测员。给定一个"目标情境facet"（其 social_role/emotional_state/behavior_pattern/thinking_pattern 代表角色在该情境下应有的表现）、对话上下文(STM)、以及角色生成的一句回应。
评估：该回应在多大程度上体现并契合"目标facet"所描述的社会姿态、情绪、行为方式与思维倾向？
打分 1-10：10=高度契合该facet；5=中性/不明显；1=完全不契合甚至体现相反倾向。
只依据是否契合该facet评分，不评价文采或流畅度。只输出 JSON：{"score":1到10的整数,"reason":"简短理由"}"""

def _facet_desc(f):
    return {k: f.get(k) for k in ("title", "situation", "social_role",
            "emotional_state", "behavior_pattern", "thinking_pattern")}

def msfa_pair(true_facet, anti_facet, stm, resp_true, resp_anti):
    """Table 21 MS-FA: contrastive separability of true vs anti facet responses (1/5/10)."""
    user = (f"对话上下文(STM):\n{stm}\n\n"
            f"【真实情境切面】(回应A应体现):\n{json.dumps(_facet_desc(true_facet), ensure_ascii=False)}\n"
            f"回应A:\n{resp_true}\n\n"
            f"【反转(对照)情境切面】(回应B应体现):\n{json.dumps(_facet_desc(anti_facet), ensure_ascii=False)}\n"
            f"回应B:\n{resp_anti}")
    r = client.chat.completions.create(
        model=JUDGE, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": F.MSFA_JUDGE_SYS}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content).get("score")

def adherence(facet, stm, response):
    if not response or not response.strip():
        return None
    user = f"目标facet:\n{json.dumps(_facet_desc(facet), ensure_ascii=False)}\n\nSTM:\n{stm}\n\n角色回应:\n{response}"
    r = client.chat.completions.create(
        model=JUDGE, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": ADH_SYS}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content).get("score")

ADH_CONDS = ["base", "card", "mrprompt", "mrprompt_noscene",
             "mrprompt_nokey", "mrprompt_wrongkey",
             "card_think", "mrprompt_think"]   # OFF-vs-ON pairs isolate the CoT effect

def score_instance(args):
    it, gens_by_cond = args
    iid = it["id"]
    true_facet = it["schema"]["scene_facets"][it["cued_index"]]
    anti_facet = it["facet_anti"]
    stm = it["stm"]
    g = lambda c: (gens_by_cond.get((str(iid), c)) or {}).get("response", "")
    rec = {"id": iid, "role": it["role"], "cued_title": it["cued_title"]}

    # MS-FA contrastive, both thinking modes
    if g("mrprompt").strip() and g("mrprompt_anti").strip():
        rec["msfa_off"] = msfa_pair(true_facet, anti_facet, stm, g("mrprompt"), g("mrprompt_anti"))
    if g("mrprompt_think").strip() and g("mrprompt_anti_think").strip():
        rec["msfa_on"] = msfa_pair(true_facet, anti_facet, stm, g("mrprompt_think"), g("mrprompt_anti_think"))

    # single-response adherence to the CUED facet (cue-key ablation + card descriptor)
    adh = {}
    for c in ADH_CONDS:
        s = adherence(true_facet, stm, g(c))
        if s is not None:
            adh[c] = s
    rec["adh"] = adh
    return rec

def main():
    gen_path = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/generations_faithful.jsonl"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/scores_faithful.jsonl"
    insts = [json.loads(l) for l in open(f"{BASE}/instances_faithful.jsonl")]
    gens_by_cond = {}
    for l in open(gen_path):
        d = json.loads(l)
        gens_by_cond[(str(d["id"]), d["condition"])] = d
    print(f"scoring {len(insts)} instances ({gen_path}) with {JUDGE}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(score_instance, [(it, gens_by_cond) for it in insts]))
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # quick summary
    def mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(sum(xs) / len(xs), 3) if xs else None
    print("wrote", out_path)
    print("MS-FA(OFF):", mean([r.get("msfa_off") for r in recs]),
          " MS-FA(ON):", mean([r.get("msfa_on") for r in recs]))
    for c in ADH_CONDS:
        print(f"  adh[{c}]:", mean([r.get("adh", {}).get(c) for r in recs]))

if __name__ == "__main__":
    main()
