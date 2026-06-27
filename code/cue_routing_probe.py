"""
cue_routing_probe.py
--------------------
Fills the key-only ablation that "Memory-Driven Role-Playing" (Table 3) lacks.

Question: do cue keys (situation / cue_phrases) do DISTINGUISHING work, or is the
Schema gain just facet decomposition?
  -> author's "cue-addressable" mechanism (A)  vs  "non-distinguishing" (B)

Metric: judge-free faithfulness contrast via forced-choice logprob.
  pref(M) = logP(ans_orig | ctx, M) - logP(ans_anti | ctx, M)        [nats]
  Delta(cond) = pref(M=body_orig) - pref(M=body_anti)
  Delta >> 0 : model reads & conditions on the facet.  Delta ~ 0 : pass-through.

4 conditions, body text byte-identical wherever present:
  flat            persona prose, no structure, no keys
  facet_nokey     structured facet, body only (keys deleted)
  facet_wrongkey  body + keys borrowed from a MISMATCHED facet  (length / "key-shaped" control)
  facet_key       body + matching keys                          (full Schema)

Contrasts:
  nokey - flat        -> does decomposition help?
  key   - nokey       -> do keys help at all?
  key   - wrongkey    -> is it the MATCHING cue (routing) or just key-shaped tokens?

Run:  python cue_routing_probe.py --attn      (--attn adds Tier-2 attention measurement)
"""

import argparse, itertools
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"

# ---------------------------------------------------------------- model
def load(model_id=MODEL_ID, want_attn=False):
    tok = AutoTokenizer.from_pretrained(model_id)
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if cuda else torch.float32,   # fp32 on CPU
        device_map="auto" if cuda else None,
        # eager is REQUIRED for output_attentions / attn hooks (sdpa & flash hide weights)
        attn_implementation="eager" if want_attn else "sdpa",
    )
    if not cuda:
        model = model.to("cpu")
    model.eval()
    return tok, model

# ---------------------------------------------------------------- prompt builder
# Manual chat-ish format so token spans stay exact. Swap to tok.apply_chat_template
# if you need the model's real template -- the metric is template-agnostic as long
# as every condition shares everything except the manipulated facet block.
SYS = "You are the following character. Reply in character, in one short line, to the latest utterance.\n"

def _tids(tok, text):
    return tok(text, add_special_tokens=False).input_ids

def render_key(situation, cues):
    return f"Situation: {situation}\nCues: {', '.join(cues)}\n"

def build(tok, item, mode, body_key):
    """body_key in {orig, anti}. Returns (input_ids[1,T], spans dict, ctx_len).
    For mode in {key, wrongkey} the cue-key SEGMENT is token-length-matched (padded
    with a neutral filler id) so that key vs wrongkey differ only in cue CONTENT, not
    length -- isolating the matching-cue effect from any 'more key-shaped tokens' effect.
    body text is byte-identical across key/nokey/wrongkey: only the keys vary."""
    body = item[f"body_{body_key}"]
    seq = [("sys", _tids(tok, SYS))]
    if mode == "flat":
        seq.append(("body", _tids(tok, f"Persona: {body}\n")))
    else:
        seq.append(("pre", _tids(tok, "## Character traits\n")))
        if mode in ("key", "wrongkey"):
            m = _tids(tok, render_key(item["situation"], item["cue_phrases"]))
            w = _tids(tok, render_key(item["wrong_situation"], item["wrong_cue_phrases"]))
            chosen = (m if mode == "key" else w)[:]
            filler = _tids(tok, " .")[-1:] or [0]            # one neutral token
            chosen += filler * (max(len(m), len(w)) - len(chosen))
            seq.append(("cuekey", chosen))
        seq.append(("body", _tids(tok, f"Content: {body}\n")))
    seq.append(("stm", _tids(tok, f'\nLatest utterance:\nOther: "{item["stm"]}"\n')))
    seq.append(("asst_head", _tids(tok, 'You: "')))

    ids, spans = [], {}
    for name, t in seq:
        s = len(ids); ids += t; spans[name] = (s, len(ids))
    return torch.tensor([ids]), spans, len(ids)

# ---------------------------------------------------------------- metric
@torch.no_grad()
def answer_logprob(model, prompt_ids, answer_ids):
    """sum logP(answer | prompt) under teacher forcing."""
    full = torch.cat([prompt_ids, torch.tensor([answer_ids])], dim=1).to(model.device)
    logits = model(full).logits[0]                      # (T, V)
    P = prompt_ids.shape[1]
    lp = 0.0
    for i, a in enumerate(answer_ids):
        lp += F.log_softmax(logits[P - 1 + i].float(), dim=-1)[a].item()
    return lp

def pref(model, tok, item, mode, body_key):
    pids, _, _ = build(tok, item, mode, body_key)
    a_orig = tok(item["ans_orig"], add_special_tokens=False).input_ids
    a_anti = tok(item["ans_anti"], add_special_tokens=False).input_ids
    return answer_logprob(model, pids, a_orig) - answer_logprob(model, pids, a_anti)

def delta(model, tok, item, mode):
    """faithfulness: pref(body_orig) - pref(body_anti)."""
    return pref(model, tok, item, mode, "orig") - pref(model, tok, item, mode, "anti")

# ---------------------------------------------------------------- Tier-2: attention mass
@torch.no_grad()
def attn_mass(model, tok, item, mode="key", body_key="orig"):
    """mean attention mass from the generation query (last prompt token) to the
    cuekey span and the body span, per layer (averaged over heads). Needs eager attn."""
    pids, spans, _ = build(tok, item, mode, body_key)
    out = model(pids.to(model.device), output_attentions=True)
    attns = out.attentions                              # tuple[L] of (1, H, T, T)
    q = -1                                              # last prompt token = predicts ans[0]
    res = {}
    for name in ("cuekey", "body"):
        if name not in spans:
            continue
        s, e = spans[name]
        per_layer = [attns[l][0, :, q, s:e].sum(-1).mean().item() for l in range(len(attns))]
        res[name] = per_layer
    return res

# ---------------------------------------------------------------- Tier-2b: residual-delta sweep
@torch.no_grad()
def residual_delta_sweep(model, tok, items):
    """per layer: mean over items of ||h(orig) - h(anti)|| at the gen position (mode=key),
    raw and normalized by ||h(orig)||. Localizes where the facet distinction lives in the
    residual stream -- directly comparable to the activation-steering layer band."""
    L = model.config.num_hidden_layers
    raw, norm = np.zeros(L), np.zeros(L)
    for it in items:
        po, _, _ = build(tok, it, "key", "orig")
        pa, _, _ = build(tok, it, "key", "anti")
        ho = model(po.to(model.device), output_hidden_states=True).hidden_states
        ha = model(pa.to(model.device), output_hidden_states=True).hidden_states
        for i in range(L):                       # hidden_states[i+1] = output of layer i
            do, da = ho[i + 1][0, -1].float(), ha[i + 1][0, -1].float()
            d = (do - da).norm().item()
            raw[i] += d
            norm[i] += d / (do.norm().item() + 1e-6)
    return raw / len(items), norm / len(items)

def print_profile(name, vals, band=(16, 23)):
    """compact per-layer profile + band[16-22] concentration vs overall."""
    vals = np.asarray(vals, float)
    L = len(vals); arg = int(np.argmax(vals))
    bl, bh = band[0], min(band[1], L)
    bmean = float(vals[bl:bh].mean()) if bl < L else float("nan")
    omean = float(vals.mean())
    ratio = bmean / omean if omean else float("nan")
    print(f"  {name}: argmax=L{arg} ({vals[arg]:.3f}); "
          f"band[{bl}-{bh-1}] mean={bmean:.3f} vs overall {omean:.3f} (ratio {ratio:.2f}x)")
    print("    L:  " + " ".join(f"{i:>5d}" for i in range(L)))
    print("       " + " ".join(f"{v:5.3f}" for v in vals))

# ---------------------------------------------------------------- stats
def bootstrap_ci(diffs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    boot = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return d.mean(), tuple(np.percentile(boot, [2.5, 97.5]))

# ---------------------------------------------------------------- data (plug MRBench items here)
# Built from the paper's real persona (Tom Sawyer, "Whitewashing Aunt Polly's Fence"
# facet, Fig.7 / Table 7). body_orig = matching facet enactment; body_anti = the
# paper's counter-facet M_L^anti (inverted). STM is the anchor (Appendix D); the
# forced choice reveals which facet the model conditions on. Twain's actual reframe
# is ans_orig. cue keys = situation + cue_phrases (the fields Table 7 says "enable
# retrieval"); wrong_* borrow the Graveyard-scene facet's keys (mismatched control).
DATA = [
    dict(
        # Tom Sawyer / "Whitewashing" facet, verbatim fields from Fig.3 MRPrompt LTM.
        situation="Whitewashing the fence (early life); a chore Tom is set to do.",
        cue_phrases=["work", "exclusive opportunity"],
        wrong_situation="Confronting danger in the graveyard at midnight.",
        wrong_cue_phrases=["graveyard", "murder", "fear"],
        # body = enactment signals (social_role / emotional_state / behavior / thinking)
        body_orig=("social_role: trickster. emotion: sly, playful. behavior: feign "
                   "enjoyment, make the chore seem an 'exclusive' privilege. "
                   "thinking: inventiveness and fun over obedience."),
        body_anti=("social_role: dutiful drudge. emotion: glum, resentful. behavior: "
                   "admit the chore is miserable and plainly wish to escape it. "
                   "thinking: obedience and honesty over scheming."),
        # STM final turn u_K = Ben's taunt (Fig.3 dialogue context)
        stm=("Say—I'm going in a-swimming, I am. Don't you wish you could? But of "
             "course you'd druther work—wouldn't you? Course you would!"),
        # ans_orig = Twain's feigned-enjoyment reframe (matches facet); ans_anti = honest misery
        ans_orig="Work? Maybe it is and maybe it ain't. All I know is, it suits Tom Sawyer.",
        ans_anti="I'd a heap rather be swimming—this is awful, I'm stuck here all day.",
    ),
    dict(  # Elizabeth Bennet -- Pride and Prejudice (paper's EN source list)
        situation="A ball at Meryton; a proud gentleman has just slighted her as merely 'tolerable'.",
        cue_phrases=["dance", "proud", "tolerable"],
        wrong_situation="Walking three miles through mud to tend her sick sister at Netherfield.",
        wrong_cue_phrases=["mud", "walk", "sister"],
        body_orig=("social_role: spirited equal. emotion: amused, affronted. behavior: meets "
                   "slights with arch wit and refuses to be cowed by rank. thinking: self-respect over flattery."),
        body_anti=("social_role: anxious inferior. emotion: hurt, eager. behavior: tries to please "
                   "the gentleman and win his approval. thinking: rank and approval over self-respect."),
        stm="Come, Miss Bennet -- a man of his consequence calling you only 'tolerable.' Surely you'll try to win him round?",
        ans_orig="I could easily forgive his pride if he had not mortified mine. Win him round? I'd not give him the satisfaction.",
        ans_anti="Perhaps if I am gentler he will think better of me. I should so like to have his good opinion.",
    ),
    dict(  # Long John Silver -- Treasure Island (paper's EN source list)
        situation="Aboard the Hispaniola, winning young Jim's trust before the mutiny.",
        cue_phrases=["lad", "trust", "ship"],
        wrong_situation="Cornered after the mutiny fails, bargaining for his own neck.",
        wrong_cue_phrases=["bargain", "pardon", "neck"],
        body_orig=("social_role: genial mentor masking a pirate. emotion: warm on the surface, calculating "
                   "beneath. behavior: flatters and befriends to disarm, turns ruthless when it serves him. "
                   "thinking: charm is a tool; loyalty is to himself."),
        body_anti=("social_role: plain honest seaman. emotion: sincere, steady. behavior: speaks bluntly and "
                   "keeps his word. thinking: loyalty and honesty above gain."),
        stm="John, the men whisper you mean to turn on the captain. Tell me straight -- are you my friend or no?",
        ans_orig="Friend? Why, Jim lad, you're the very picture of what I was at your age. Trust old John -- I'd never see ye come to harm.",
        ans_anti="I'll not lie to you, Jim. There's mischief afoot and I want no part of it. Stand by the captain and so will I.",
    ),
    dict(  # Yossarian -- Catch-22 (paper's EN source list)
        situation="At the airbase, ordered to fly yet more bombing missions.",
        cue_phrases=["missions", "killed", "crazy"],
        wrong_situation="Lounging in the hospital, faking a liver complaint to avoid duty.",
        wrong_cue_phrases=["hospital", "liver", "censor"],
        body_orig=("social_role: reluctant flier. emotion: terrified, indignant. behavior: schemes to be "
                   "grounded and insists everyone is trying to get him killed. thinking: survival over glory; the war is absurd."),
        body_anti=("social_role: eager patriot. emotion: proud, gung-ho. behavior: volunteers for missions, "
                   "hungry for medals. thinking: duty and glory over self."),
        stm="Yossarian, they've raised the missions to seventy. You're not going to make a fuss again, are you?",
        ans_orig="Seventy? They're trying to kill me -- of course they are, everyone is. I won't go up there to be murdered for Cathcart's promotion.",
        ans_anti="Seventy? Good -- more chances at the enemy. Sign me up, I want every mission I can get.",
    ),
    dict(  # Hermione Granger -- Harry Potter (paper's EN source list)
        situation="First year at Hogwarts; the trio considering breaking school rules after curfew.",
        cue_phrases=["rules", "expelled", "library"],
        wrong_situation="Punching Draco Malfoy in the face in third year.",
        wrong_cue_phrases=["punch", "Malfoy", "fist"],
        body_orig=("social_role: anxious rule-keeper and scholar. emotion: alarmed, earnest. behavior: cites "
                   "rules and urges caution and study. thinking: knowledge and order prevent disaster."),
        body_anti=("social_role: reckless thrill-seeker. emotion: gleeful, careless. behavior: dismisses rules "
                   "and books and charges in. thinking: rules are for cowards."),
        stm="Hermione, we're going out after curfew to catch him -- are you in, or are you going to tattle?",
        ans_orig="You'll get us all expelled -- or worse! Have you even read what happens if you're caught? We should fetch a teacher, not go sneaking about.",
        ans_anti="Rules? Who cares about rules -- let's go! Forget the library, this'll be far more fun than studying.",
    ),
    dict(  # Jo March -- Little Women (paper's EN source list)
        situation="At a party, refusing to behave like a proper young lady.",
        cue_phrases=["proper", "marry", "writing"],
        wrong_situation="Sitting at Beth's bedside through her illness.",
        wrong_cue_phrases=["Beth", "sick", "bedside"],
        body_orig=("social_role: headstrong tomboy and aspiring writer. emotion: restless, fierce. behavior: "
                   "scorns ladylike convention and talks of independence and her stories. thinking: ambition and freedom over marriage."),
        body_anti=("social_role: demure conventional girl. emotion: placid, content. behavior: longs for a good "
                   "marriage and a quiet home and defers to propriety. thinking: domestic respectability over ambition."),
        stm="Jo, you're nearly grown -- surely it's time you stopped your scribbling and thought about finding a husband?",
        ans_orig="I'd rather be a free spinster and paddle my own canoe! I mean to make my own way with my pen, not marry to be kept.",
        ans_anti="Yes, I suppose you're right. I should put away my silly stories and set my heart on a good marriage and a home.",
    ),
    dict(  # Sydney Carton -- A Tale of Two Cities (paper's EN source list)
        situation="Dissipated and idle in London, dismissive of his own worth.",
        cue_phrases=["wasted", "drink", "care"],
        wrong_situation="Quietly resolving to take Darnay's place at the guillotine.",
        wrong_cue_phrases=["guillotine", "place", "sacrifice"],
        body_orig=("social_role: brilliant wastrel. emotion: weary, self-mocking. behavior: belittles himself, "
                   "drinks, professes to care for nothing. thinking: his life is worth little."),
        body_anti=("social_role: striving careerist. emotion: confident, ambitious. behavior: guards his prospects "
                   "and speaks of his bright future. thinking: his life and success are paramount."),
        stm="Carton, a man of your gifts -- don't you care at all what becomes of you?",
        ans_orig="Care? I care for no man on earth, and no man on earth cares for me. I am a disappointed drudge, and the wine's as good a friend as any.",
        ans_anti="Care? Of course -- I've a fine career ahead and mean to make the most of it. A man must look to his own advancement.",
    ),
    dict(  # Ebenezer Scrooge -- A Christmas Carol
        situation="At his counting-house on Christmas Eve, asked for charity.",
        cue_phrases=["money", "Christmas", "humbug"],
        wrong_situation="Weeping at his own neglected grave shown by the last spirit.",
        wrong_cue_phrases=["grave", "spirit", "weep"],
        body_orig=("social_role: miser. emotion: cold, irritable. behavior: refuses every appeal for money and "
                   "sneers at goodwill. thinking: profit over people; charity is folly."),
        body_anti=("social_role: open-handed benefactor. emotion: warm, jovial. behavior: gives freely and wishes "
                   "all well. thinking: generosity and fellowship over gold."),
        stm="At this festive season, Mr. Scrooge, may we put you down for something for the poor and destitute?",
        ans_orig="Are there no prisons? Are there no workhouses? I wish to be left alone. Christmas a humbug -- good afternoon!",
        ans_anti="The poor, you say? Put me down for a generous sum -- and a merry Christmas to you! It is the season for open hands.",
    ),
    dict(  # Don Quixote -- Don Quixote
        situation="On the plain of Montiel, spotting windmills on the horizon.",
        cue_phrases=["giants", "battle", "lance"],
        wrong_situation="On his deathbed, sane at last, renouncing the books of chivalry.",
        wrong_cue_phrases=["deathbed", "renounce", "sane"],
        body_orig=("social_role: deluded knight-errant. emotion: exalted, valiant. behavior: mistakes the ordinary "
                   "for the marvelous and charges it as adventure. thinking: chivalric romance is reality."),
        body_anti=("social_role: clear-eyed pragmatist. emotion: calm, sober. behavior: names things plainly and "
                   "avoids needless danger. thinking: the world is exactly as it appears."),
        stm="Look there, master -- on the plain ahead. What do you make of them?",
        ans_orig="Fortune guides us! Look, Sancho -- thirty monstrous giants. I shall do battle and slay them all. Spur on -- this is noble adventure!",
        ans_anti="Those? Windmills, plainly -- their sails turning in the wind, nothing more. We'll keep our distance and ride on.",
    ),
    dict(  # Sherlock Holmes -- Sherlock Holmes
        situation="A new client enters 221B; Holmes observes him before a word is spoken.",
        cue_phrases=["observe", "deduce", "client"],
        wrong_situation="Slumped in his chair between cases, reaching for the cocaine bottle.",
        wrong_cue_phrases=["cocaine", "boredom", "violin"],
        body_orig=("social_role: observing detective. emotion: cool, intent. behavior: reads strangers from minute "
                   "details and states deductions flatly. thinking: data and inference over guesswork."),
        body_anti=("social_role: incurious bystander. emotion: indifferent, vague. behavior: notices nothing and "
                   "asks the obvious. thinking: such things cannot be known by looking."),
        stm="You've never met this gentleman before, Holmes. You can't possibly know anything about him.",
        ans_orig="On the contrary. A retired sergeant of Marines -- the tattoo, the carriage of the shoulders, the chalk on his cuff. Quite plain to one who observes.",
        ans_anti="You're quite right, Watson, I haven't the faintest idea. We shall have to wait until he tells us who he is.",
    ),
    dict(  # Captain Ahab -- Moby-Dick
        situation="On the quarterdeck, nailing the doubloon to the mast for the white whale.",
        cue_phrases=["whale", "hunt", "vengeance"],
        wrong_situation="A rare quiet moment, gazing at the sea and thinking of his young wife and child.",
        wrong_cue_phrases=["wife", "child", "home"],
        body_orig=("social_role: monomaniac captain. emotion: vengeful, exalted. behavior: bends the whole crew to "
                   "his hunt and scorns profit and prudence. thinking: vengeance on the whale above all."),
        body_anti=("social_role: prudent merchant captain. emotion: measured, dutiful. behavior: hunts whales for "
                   "oil and steers clear of needless danger. thinking: a safe, profitable voyage over any grudge."),
        stm="Captain, the hold's near full of oil. The men say it's time we turned for home.",
        ans_orig="Home? I'll chase him round Good Hope, and round the Horn, and round perdition's flames before I give him up! The doubloon's for whoever raises the white whale!",
        ans_anti="Aye, a full hold and a fair wind -- good sense. Make ready to come about; we'll set our course for home and a tidy profit.",
    ),
    dict(  # Raskolnikov -- Crime and Punishment (paper's EN source list)
        situation="In his garret after the murder, feverish and tormented by what he has done.",
        cue_phrases=["guilt", "blood", "confess"],
        wrong_situation="Lecturing coolly in a tavern on his theory of the extraordinary man.",
        wrong_cue_phrases=["theory", "Napoleon", "extraordinary"],
        body_orig=("social_role: guilt-haunted murderer. emotion: feverish, anguished. behavior: starts at every "
                   "knock, half-longs to confess, cannot rest. thinking: conscience crushing the deed."),
        body_anti=("social_role: untroubled idler. emotion: light, carefree. behavior: sleeps well and thinks "
                   "nothing of the past. thinking: no deed weighs on him at all."),
        stm="Rodya, you look dreadful -- pale as a corpse, jumping at shadows. What on earth is the matter with you?",
        ans_orig="Nothing -- leave me! No... do you hear someone on the stair? I can't breathe in here. Perhaps it would be better to just tell them and have done.",
        ans_anti="The matter? Nothing in the world. I slept like a child and haven't a care. You worry over shadows, my friend.",
    ),
]

CONDITIONS = ["flat", "facet_nokey", "facet_wrongkey", "facet_key"]
MODE = {"flat": "flat", "facet_nokey": "nokey", "facet_wrongkey": "wrongkey", "facet_key": "key"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--attn", action="store_true", help="also run Tier-2 attention measurement")
    args = ap.parse_args()

    tok, model = load(args.model, want_attn=args.attn)

    # Tier-1: behavioral Delta per condition
    deltas = {c: [] for c in CONDITIONS}
    for it in DATA:
        for c in CONDITIONS:
            deltas[c].append(delta(model, tok, it, MODE[c]))

    print("=== Tier-1: faithfulness Delta (nats), mean [95% CI] ===")
    for c in CONDITIONS:
        m, (lo, hi) = bootstrap_ci(deltas[c])
        print(f"  {c:16s}  {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

    print("\n=== paired contrasts (the actual hypotheses) ===")
    for a, b, label in [
        ("facet_nokey", "flat", "decomposition: nokey - flat"),
        ("facet_key", "facet_nokey", "keys at all:   key - nokey"),
        ("facet_key", "facet_wrongkey", "MATCHING cue:  key - wrongkey"),
    ]:
        d = [x - y for x, y in zip(deltas[a], deltas[b])]
        m, (lo, hi) = bootstrap_ci(d)
        sig = "" if lo <= 0 <= hi else "  *CI excludes 0*"
        print(f"  {label:28s}  {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")

    # Tier-2: attention localization (layer x {cuekey, body})
    if args.attn:
        print("\n=== Tier-2a: attention mass gen-query -> span, FULL per-layer sweep (mode=key) ===")
        acc = {}
        for it in DATA:
            m = attn_mass(model, tok, it, mode="key")
            for span, per_layer in m.items():
                acc.setdefault(span, []).append(per_layer)
        for span in ("cuekey", "body"):
            if span in acc:
                print_profile(f"attn:{span}", np.mean(acc[span], 0))

        print("\n=== Tier-2b: residual delta ||h(orig)-h(anti)|| at gen pos, FULL sweep (mode=key) ===")
        raw, nrm = residual_delta_sweep(model, tok, DATA)
        print_profile("resid:raw ", raw)
        print_profile("resid:norm", nrm)

if __name__ == "__main__":
    main()
