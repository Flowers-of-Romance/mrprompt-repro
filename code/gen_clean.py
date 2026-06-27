"""
gen_clean.py -- definitive run: all 7 conditions, uniform per-family budget, no regen.
  base / card           : thinking OFF, 128 tok (paper baselines, no protocol)
  mrprompt-family (x5)   : thinking ON, 1024 tok uniform (Magic-If CoT runs)
Writes ~/mdrp-repro/generations_clean.jsonl
"""
import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import renderers as R

BASE = os.path.expanduser("~/mdrp-repro")
MODEL = "Qwen/Qwen3-8B"
# condition -> (enable_thinking, max_new_tokens)
CFG = {
    "base": (False, 128), "card": (False, 128),
    "mrprompt": (True, 1024), "mrprompt_noscene": (True, 1024),
    "mrprompt_anti": (True, 1024), "mrprompt_nokey": (True, 1024),
    "mrprompt_wrongkey": (True, 1024),
}

def extract(raw, thinking):
    if thinking:
        return raw.split("</think>")[-1].strip() if "</think>" in raw else ""
    return raw.strip()

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    insts = [json.loads(l) for l in open(f"{BASE}/instances.jsonl")]
    print(f"{len(insts)} instances x {len(CFG)} conditions = {len(insts)*len(CFG)} generations")
    # resume: skip (id, condition) pairs already present in the output file
    out_path = f"{BASE}/generations_clean.jsonl"
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                d = json.loads(l)
                done.add((str(d["id"]), d["condition"]))
            except Exception:
                pass
    print(f"resume: {len(done)} generations already done, skipping those")
    out = open(out_path, "a")
    trunc = 0
    for n, it in enumerate(insts):
        for cond, (think, mnt) in CFG.items():
            if (str(it["id"]), cond) in done:
                continue
            persona = R.persona_for(it["schema"], cond, it["cued_index"], it["facet_anti"])
            msgs = R.build_messages(persona, it["stm"], it["role"])
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=think)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g = model.generate(**ids, max_new_tokens=mnt, do_sample=True,
                                   temperature=0.7, top_p=0.8, pad_token_id=tok.eos_token_id)
            resp = extract(tok.decode(g[0, ids.input_ids.shape[1]:], skip_special_tokens=True), think)
            if think and not resp:
                trunc += 1
            out.write(json.dumps({"id": it["id"], "role": it["role"], "condition": cond,
                                  "cued_index": it["cued_index"], "cued_title": it["cued_title"],
                                  "response": resp}, ensure_ascii=False) + "\n")
            out.flush()
        print(f"[{n+1}/{len(insts)}] {it['role']}  (trunc so far={trunc})")
    out.close()
    print(f"wrote {BASE}/generations_clean.jsonl | truncated: {trunc}")

if __name__ == "__main__":
    main()
