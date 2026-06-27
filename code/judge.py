"""
judge.py -- Phase B step: MREval FA scoring with gpt-4.1-mini.
Reconstructed FA rubric (Table 1): does the response adhere to / enact the cued facet?
For mrprompt_anti we additionally score adherence to the ANTI facet (swap sensitivity).

Writes ~/mdrp-repro/scores.jsonl
"""
import os, json
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

client = OpenAI()
JUDGE = "gpt-4.1-mini-2025-04-14"
BASE = os.path.expanduser("~/mdrp-repro")

FA_SYS = """你是严格的角色扮演评测员。给定一个"目标情境facet"（其 social_role/emotional_state/behavior_pattern/thinking_pattern 代表角色在该情境下应有的表现）、对话上下文(STM)、以及角色生成的一句回应。
评估：该回应在多大程度上体现并契合"目标facet"所描述的社会姿态、情绪、行为方式与思维倾向？
打分 1-10：10=高度契合该facet；5=中性/不明显；1=完全不契合甚至体现相反倾向。
只依据是否契合该facet评分，不评价文采或流畅度。只输出 JSON：{"score":1到10的整数,"reason":"简短理由"}"""

def judge_fa(facet, stm, response):
    fdesc = {k: facet.get(k) for k in ("title", "situation", "social_role",
             "emotional_state", "behavior_pattern", "thinking_pattern")}
    user = f"目标facet:\n{json.dumps(fdesc, ensure_ascii=False)}\n\nSTM:\n{stm}\n\n角色回应:\n{response}"
    r = client.chat.completions.create(
        model=JUDGE, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": FA_SYS}, {"role": "user", "content": user}])
    return json.loads(r.choices[0].message.content).get("score")

def score_one(args):
    g, insts = args
    it = insts[g["id"]]
    true_facet = it["schema"]["scene_facets"][it["cued_index"]]
    if not g["response"].strip():            # truncated/empty -> missing, don't penalize
        return {"id": g["id"], "role": g["role"], "condition": g["condition"], "fa_true": None}
    rec = {"id": g["id"], "role": g["role"], "condition": g["condition"],
           "fa_true": judge_fa(true_facet, it["stm"], g["response"])}
    # swap-sensitivity: how much does the anti-condition response follow the anti facet
    if g["condition"] == "mrprompt_anti":
        rec["fa_anti"] = judge_fa(it["facet_anti"], it["stm"], g["response"])
    return rec

def main():
    import sys
    gen_path = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/generations.jsonl"
    out_path = sys.argv[2] if len(sys.argv) > 2 else f"{BASE}/scores.jsonl"
    insts = {it["id"]: it for it in (json.loads(l) for l in open(f"{BASE}/instances.jsonl"))}
    gens = [json.loads(l) for l in open(gen_path)]
    print(f"judging {len(gens)} generations ({gen_path}) with {JUDGE}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(score_one, [(g, insts) for g in gens]))
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", out_path)

if __name__ == "__main__":
    main()
