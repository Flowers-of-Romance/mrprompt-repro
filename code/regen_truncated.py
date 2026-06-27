"""
regen_truncated.py -- fill the cells where the thinking CoT overran 640 tokens
(empty response). Re-generate only those with a much larger budget, update in place.
"""
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import renderers as R

BASE = os.path.expanduser("~/mdrp-repro")
MODEL = "Qwen/Qwen3-8B"
F = f"{BASE}/generations_think.jsonl"

def strip_think(s):
    return s.split("</think>")[-1].strip() if "</think>" in s else ""

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    insts = {it["id"]: it for it in (json.loads(l) for l in open(f"{BASE}/instances.jsonl"))}
    gens = [json.loads(l) for l in open(F)]
    todo = [i for i, g in enumerate(gens) if not g["response"].strip()]
    print(f"{len(todo)} truncated cells to regenerate (max_new_tokens=1536)")
    still = 0
    for k, i in enumerate(todo):
        g = gens[i]; it = insts[g["id"]]
        persona = R.persona_for(it["schema"], g["condition"], it["cued_index"], it["facet_anti"])
        msgs = R.build_messages(persona, it["stm"], it["role"])
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=1536, do_sample=True,
                                 temperature=0.7, top_p=0.8, pad_token_id=tok.eos_token_id)
        resp = strip_think(tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True))
        gens[i]["response"] = resp
        if not resp:
            still += 1
        print(f"[{k+1}/{len(todo)}] {g['id']} {g['condition']} -> {len(resp)} chars" + (" STILL EMPTY" if not resp else ""))
    with open(F, "w") as f:
        for g in gens:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"rewrote {F} | still empty after regen: {still}")

if __name__ == "__main__":
    main()
