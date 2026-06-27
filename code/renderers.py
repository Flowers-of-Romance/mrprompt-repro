"""
renderers.py -- Phase A step 3: render the three persona-prompt formats from a
Narrative Schema, reconstructed from the paper (Base / Card / MRPrompt + Magic-If,
§3.3 / Appendix B / Fig.3). Also the M_L variants used by the splits/ablations.

Used by the generation step. All Chinese (CharacterEval subset).
"""
import copy, json

MAGIC_IF = """【行动准则·Magic-If】你就是{name}本人。请在心中按以下步骤推理后，只输出角色的一句回应：
1) 从对话(STM)中提取线索；2) 选择最相关的情境facet；3) 由该facet推导社会姿态/情绪/行为/思维；4) 以角色口吻生成回应。
【严格规则】以当下视角说话、不剧透未来；对不确定或范围外的事表达不确定或拒绝、不编造；不得暴露你是AI；只回应一轮，自然简洁。"""

FACET_FIELDS = ["situation", "social_role", "emotional_state",
                "behavior_pattern", "thinking_pattern", "cue_phrases"]

# ----------------------------------------------------------------- Base (narrative)
def render_base(schema):
    parts = [f"【人物】{schema.get('name','')}"]
    if schema.get("global_summary"):
        parts.append(schema["global_summary"])
    traits = "、".join(t.get("trait", "") for t in schema.get("core_traits", []))
    if traits:
        parts.append(f"性格上，他/她{traits}。")
    # interleave a few representative episodes as flowing prose
    eps = []
    for f in schema.get("scene_facets", [])[:5]:
        eps.append(f"在{f.get('situation','某些情境')}中，{f.get('behavior_pattern','')}")
    if eps:
        parts.append("。".join(eps) + "。")
    return "\n".join(parts)

# ----------------------------------------------------------------- Card (profile card)
def render_card(schema):
    card = {
        "name": schema.get("name", ""),
        "global_summary": schema.get("global_summary", ""),
        "personality": [t.get("trait", "") for t in schema.get("core_traits", [])],
        "relationships": schema.get("relationships", []),
    }
    return "【人物卡】\n" + json.dumps(card, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------- MRPrompt (facet LTM + protocol)
def _facet_lines(f, drop_keys=False):
    out = []
    for k in FACET_FIELDS:
        if drop_keys and k in ("situation", "cue_phrases"):
            continue
        v = f.get(k)
        if v:
            out.append(f"    {k}: {v if not isinstance(v, list) else '、'.join(v)}")
    return "\n".join(out)

def _variant_facets(schema, mode, cued_index=None, anti=None):
    """returns the scene_facets list for the requested M_L variant."""
    facets = copy.deepcopy(schema.get("scene_facets", []))
    if mode == "no_scene":
        return []
    if mode == "anti" and cued_index is not None and anti is not None:
        facets[cued_index] = {**facets[cued_index], **anti}        # replace cued facet
    if mode == "wrongkey" and cued_index is not None and len(facets) > 1:
        other = (cued_index + 1) % len(facets)
        facets[cued_index]["situation"] = facets[other].get("situation", "")
        facets[cued_index]["cue_phrases"] = facets[other].get("cue_phrases", [])
    return facets

def render_mrprompt(schema, mode="full", cued_index=None, anti=None, use_protocol=True):
    """mode in {full, anti, no_scene, nokey, wrongkey}."""
    name = schema.get("name", "")
    parts = [f"【长期记忆·叙事图式】人物：{name}"]
    if schema.get("global_summary"):
        parts.append("概述：" + schema["global_summary"])
    ct = schema.get("core_traits", [])
    if ct:
        parts.append("核心特质：")
        for t in ct:
            parts.append(f"  - {t.get('trait','')}：{t.get('desc','')}")
    facets = _variant_facets(schema, mode, cued_index, anti)
    if facets:
        parts.append("情境facet（可按对话线索检索）：")
        drop = (mode == "nokey")
        for i, f in enumerate(facets):
            parts.append(f"  [{f.get('title','facet'+str(i))}]")
            parts.append(_facet_lines(f, drop_keys=drop))
    text = "\n".join(parts)
    if use_protocol:
        text += "\n\n" + MAGIC_IF.format(name=name)
    return text

# ----------------------------------------------------------------- generation messages
def build_messages(persona_text, stm, role):
    user = (f"以下是对话上下文(STM)，最后一句是对话者的发言，接下来轮到你（{role}）回应：\n\n"
            f"{stm}\n\n请只输出{role}接下来要说的一句话（角色口吻，自然简洁，不要旁白或解释）。")
    return [{"role": "system", "content": persona_text},
            {"role": "user", "content": user}]

# conditions used in the experiment
def persona_for(schema, condition, cued_index=None, anti=None):
    """condition -> persona text.
    base / card / mrprompt (full) -- the paper's 3 formats (RQ2/RQ3 replication)
    mrprompt_anti / mrprompt_noscene -- MS-FA / MS-FU variants
    mrprompt_nokey / mrprompt_wrongkey -- our cue-key ablation
    """
    if condition == "base":             return render_base(schema)
    if condition == "card":             return render_card(schema)
    if condition == "mrprompt":         return render_mrprompt(schema, "full")
    if condition == "mrprompt_anti":    return render_mrprompt(schema, "anti", cued_index, anti)
    if condition == "mrprompt_noscene": return render_mrprompt(schema, "no_scene")
    if condition == "mrprompt_nokey":   return render_mrprompt(schema, "nokey", cued_index)
    if condition == "mrprompt_wrongkey":return render_mrprompt(schema, "wrongkey", cued_index)
    raise ValueError(condition)

if __name__ == "__main__":
    import os
    base = os.path.expanduser("~/mdrp-repro")
    insts = [json.loads(l) for l in open(f"{base}/instances.jsonl")]
    it = insts[0]
    print("=== instance:", it["role"], "| cued:", it["cued_title"], "===")
    for c in ("base", "card", "mrprompt", "mrprompt_anti", "mrprompt_nokey"):
        p = persona_for(it["schema"], c, it["cued_index"], it["facet_anti"])
        print(f"\n########## {c}  ({len(p)} chars) ##########")
        print(p[:700])
