"""
gen.py -- Phase B step: generate in-character responses with Qwen3-8B (ROCm) for
each instance x condition. Run in the comfy-rocm venv.

Conditions:
  base / card / mrprompt          -> RQ2/RQ3 replication (does MRPrompt > Card > Base?)
  mrprompt_noscene                -> FU (facet utility)
  mrprompt_anti                   -> FA swap-sensitivity
  mrprompt_nokey / mrprompt_wrongkey -> our cue-key ablation

Writes ~/mdrp-repro/generations.jsonl
"""
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import renderers as R

BASE = os.path.expanduser("~/mdrp-repro")
MODEL = "Qwen/Qwen3-8B"
CONDITIONS = ["base", "card", "mrprompt", "mrprompt_noscene",
              "mrprompt_anti", "mrprompt_nokey", "mrprompt_wrongkey"]

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    insts = [json.loads(l) for l in open(f"{BASE}/instances.jsonl")]
    print(f"{len(insts)} instances x {len(CONDITIONS)} conditions = {len(insts)*len(CONDITIONS)} generations")
    out = open(f"{BASE}/generations.jsonl", "w")
    for n, it in enumerate(insts):
        for cond in CONDITIONS:
            persona = R.persona_for(it["schema"], cond, it["cued_index"], it["facet_anti"])
            msgs = R.build_messages(persona, it["stm"], it["role"])
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=128, do_sample=True,
                                   temperature=0.7, top_p=0.8, pad_token_id=tok.eos_token_id)
            resp = tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip()
            out.write(json.dumps({"id": it["id"], "role": it["role"], "condition": cond,
                                  "cued_index": it["cued_index"], "cued_title": it["cued_title"],
                                  "response": resp}, ensure_ascii=False) + "\n")
            out.flush()
        print(f"[{n+1}/{len(insts)}] {it['role']}")
    out.close()
    print("wrote", f"{BASE}/generations.jsonl")

if __name__ == "__main__":
    main()
