# -*- coding: utf-8 -*-
"""
build_ltms.py -- Stage A (faithful): construct both LTMs from the paper's VERBATIM
prompts, replacing the earlier self-made facetize.SYS / programmatic card.

  - facet-LTM  via Fig.15 (MRPrompt-LTM)   -> facets_faithful/<name>.json
  - Card-LTM   via Fig.14 (Card persona)   -> cards_faithful/<name>.json

Input "summary" = the CharacterEval profile dict serialized to readable text (the only
available source material per character). gpt-4.1, cached (skip if file exists).

Usage:
  python build_ltms.py <name>          # one character (test)
  python build_ltms.py --all [limit]   # all characters in profiles
"""
import os, json, sys
from openai import OpenAI
import faithful_prompts as F

MODEL = "gpt-4.1-2025-04-14"
BASE = os.path.expanduser("~/mdrp-repro")
client = OpenAI()

def profile_to_summary(name, prof):
    """Serialize the CharacterEval profile dict into a natural-language summary block."""
    if isinstance(prof, str):
        return prof
    lines = [f"角色：{name}"]
    for k, v in prof.items():
        if v in (None, "", [], {}):
            continue
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(f"{k}：{v}")
    return "\n".join(lines)

def _construct(sys_prompt, user_prompt):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    return json.loads(r.choices[0].message.content), r.usage

def _unwrap(obj, name):
    """Fig.14/15 schemas nest under the character name key; unwrap to the inner object."""
    if isinstance(obj, dict) and len(obj) == 1 and name in obj and isinstance(obj[name], dict):
        return obj[name]
    if isinstance(obj, dict) and "name" in obj:
        return obj
    # single-key wrap under some other label
    if isinstance(obj, dict) and len(obj) == 1:
        inner = next(iter(obj.values()))
        if isinstance(inner, dict):
            return inner
    return obj

def build_one(name, prof):
    summary = profile_to_summary(name, prof)
    os.makedirs(f"{BASE}/facets_faithful", exist_ok=True)
    os.makedirs(f"{BASE}/cards_faithful", exist_ok=True)
    fpath = f"{BASE}/facets_faithful/{name}.json"
    cpath = f"{BASE}/cards_faithful/{name}.json"
    usage = {}
    if not os.path.exists(fpath):
        obj, u = _construct(F.MRPROMPT_CONSTRUCT_SYS, F.mrprompt_construct_user(name, summary))
        json.dump(_unwrap(obj, name), open(fpath, "w"), ensure_ascii=False, indent=2)
        usage["facet"] = (u.prompt_tokens, u.completion_tokens)
    if not os.path.exists(cpath):
        obj, u = _construct(F.CARD_CONSTRUCT_SYS, F.card_construct_user(name, summary))
        json.dump(_unwrap(obj, name), open(cpath, "w"), ensure_ascii=False, indent=2)
        usage["card"] = (u.prompt_tokens, u.completion_tokens)
    return usage

def main():
    profs = json.load(open(f"{BASE}/CharacterEval/data/character_profiles.json"))
    if not sys.argv[1:] or sys.argv[1] != "--all":
        name = sys.argv[1] if sys.argv[1:] else list(profs)[0]
        if name not in profs:
            print("not in profiles. sample:", list(profs)[:10]); sys.exit(1)
        u = build_one(name, profs[name])
        print(f"built {name}: {u or 'cached'}")
        fac = json.load(open(f"{BASE}/facets_faithful/{name}.json"))
        card = json.load(open(f"{BASE}/cards_faithful/{name}.json"))
        sf = fac.get("Personality", fac).get("scene_facets") or fac.get("scene_facets", [])
        print(f"  facet-LTM scene_facets: {len(sf)}")
        cf = card.get("Personality", {}).get("scene_facets") or card.get("scene_facets", [])
        print(f"  Card-LTM  scene_facets: {len(cf)}")
        return
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else len(profs)
    names = list(profs)[:limit]
    tin = tout = 0
    for i, name in enumerate(names):
        try:
            u = build_one(name, profs[name])
            for k, (pi, po) in u.items():
                tin += pi; tout += po
            print(f"[{i+1}/{len(names)}] {name}  {u or 'cached'}")
        except Exception as e:
            print(f"[{i+1}/{len(names)}] {name}  ERROR {repr(e)[:100]}")
    print(f"done. tokens in/out: {tin}/{tout}")

if __name__ == "__main__":
    main()
