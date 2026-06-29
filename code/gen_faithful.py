# -*- coding: utf-8 -*-
"""
gen_faithful.py -- Stage C generation, confound-free.

Vs the retracted gen_clean.py:
  - system prompt = paper's Fig.18/Fig.19 (faithful_render), not the self-made MAGIC_IF;
  - UNIFORM max_new_tokens for every condition (thinking flag is the ONLY manipulated
    variable -> no thinking/token confound). OFF conditions stop early at EOS.
  - main axis = 7 conditions all thinking-OFF; ablation arm = card_think / mrprompt_think
    (thinking-ON) to measure the CoT contribution directly.

Writes ~/mdrp-repro/generations_faithful.jsonl  (resumable).
"""
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import faithful_render as R

BASE = os.path.expanduser("~/mdrp-repro")
MODEL = "Qwen/Qwen3-8B"
MAX_NEW = 1024                      # uniform for ALL conditions

# condition -> (thinking, card_ltm?)
CFG = {
    "base":              (False, False),
    "card":              (False, True),
    "mrprompt":          (False, False),
    "mrprompt_noscene":  (False, False),
    "mrprompt_anti":     (False, False),
    "mrprompt_nokey":    (False, False),
    "mrprompt_wrongkey": (False, False),
    # --- ablation arm: thinking-ON, same prompts/budget (lets MS-FA be scored ON vs OFF) ---
    "card_think":         (True,  True),
    "mrprompt_think":     (True,  False),
    "mrprompt_anti_think":(True,  False),
}
CARD_FOR = {"card": "card", "card_think": "card"}            # which conds need Card-LTM
THINK_AS_MR = {"mrprompt_think": "mrprompt",                 # think-arm reuses base cond render
               "mrprompt_anti_think": "mrprompt_anti"}

def render_cond(cond, it, card_cache):
    """Map a CFG condition to (schema, card_schema, render-condition)."""
    schema = it["schema"]
    card_schema = None
    if cond in CARD_FOR:
        cpath = f"{BASE}/cards_faithful/{it['role']}.json"
        if it["role"] not in card_cache:
            card_cache[it["role"]] = json.load(open(cpath)) if os.path.exists(cpath) else None
        card_schema = card_cache[it["role"]]
        rcond = "card"
    elif cond in THINK_AS_MR:
        rcond = THINK_AS_MR[cond]
    else:
        rcond = cond
    return R.system_for(rcond, schema, card_schema, it["cued_index"], it["facet_anti"], role=it["role"])

def extract(raw, thinking):
    if thinking:
        return raw.split("</think>")[-1].strip() if "</think>" in raw else ""
    return raw.strip()

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    insts = [json.loads(l) for l in open(f"{BASE}/instances_faithful.jsonl")]
    print(f"{len(insts)} instances x {len(CFG)} conditions = {len(insts)*len(CFG)} generations")
    out_path = f"{BASE}/generations_faithful.jsonl"
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                d = json.loads(l); done.add((str(d["id"]), d["condition"]))
            except Exception:
                pass
    print(f"resume: {len(done)} already done")
    out = open(out_path, "a")
    card_cache, trunc = {}, 0
    for n, it in enumerate(insts):
        for cond, (think, _) in CFG.items():
            if (str(it["id"]), cond) in done:
                continue
            system_text = render_cond(cond, it, card_cache)
            msgs = R.build_messages(system_text, it["stm"], it["role"])
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=think)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True,
                                   temperature=0.7, top_p=0.8, pad_token_id=tok.eos_token_id)
            resp = extract(tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True), think)
            if think and not resp:
                trunc += 1
            out.write(json.dumps({"id": it["id"], "role": it["role"], "condition": cond,
                                  "cued_index": it["cued_index"], "cued_title": it["cued_title"],
                                  "think": think, "response": resp}, ensure_ascii=False) + "\n")
            out.flush()
        print(f"[{n+1}/{len(insts)}] {it['role']}  (think-trunc so far={trunc})")
    out.close()
    print(f"wrote {out_path} | thinking-arm truncations: {trunc}")

if __name__ == "__main__":
    main()
