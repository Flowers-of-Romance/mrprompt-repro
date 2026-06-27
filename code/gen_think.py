"""
gen_think.py -- fair retest: MRPrompt with Qwen3 thinking ENABLED, so the Magic-If
chain-of-thought ("extract cues -> select facet -> derive signals -> respond")
actually executes. Conditions chosen to make the cue-key test decisive under the CoT:
  mrprompt (full) / mrprompt_nokey / mrprompt_wrongkey
If the CoT's "select facet via cues" step really uses cue keys, full should beat
nokey/wrongkey here. Compare mrprompt(full,think) vs card/base(v1) for replication.

Writes ~/mdrp-repro/generations_think.jsonl
"""
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import renderers as R

BASE = os.path.expanduser("~/mdrp-repro")
MODEL = "Qwen/Qwen3-8B"
CONDS = ["mrprompt", "mrprompt_nokey", "mrprompt_wrongkey"]

def strip_think(s):
    if "</think>" in s:
        return s.split("</think>")[-1].strip()
    return ""   # ran out of budget inside the think block -> no answer emitted

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    insts = [json.loads(l) for l in open(f"{BASE}/instances.jsonl")]
    print(f"{len(insts)} x {len(CONDS)} = {len(insts)*len(CONDS)} thinking-on generations")
    out = open(f"{BASE}/generations_think.jsonl", "w")
    trunc = 0
    for n, it in enumerate(insts):
        for cond in CONDS:
            persona = R.persona_for(it["schema"], cond, it["cued_index"], it["facet_anti"])
            msgs = R.build_messages(persona, it["stm"], it["role"])
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=640, do_sample=True,
                                   temperature=0.7, top_p=0.8, pad_token_id=tok.eos_token_id)
            raw = tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
            resp = strip_think(raw)
            if not resp:
                trunc += 1
            out.write(json.dumps({"id": it["id"], "role": it["role"], "condition": cond,
                                  "cued_index": it["cued_index"], "cued_title": it["cued_title"],
                                  "response": resp}, ensure_ascii=False) + "\n")
            out.flush()
        print(f"[{n+1}/{len(insts)}] {it['role']}  (truncated-so-far={trunc})")
    out.close()
    print(f"wrote {BASE}/generations_think.jsonl  | truncated(no answer): {trunc}")

if __name__ == "__main__":
    main()
