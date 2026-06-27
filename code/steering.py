"""
steering.py -- diff-in-means steering vectors + bridge to the cue_routing probe.

Builds per-layer steering vectors (residual stream, mean(pos)-mean(neg)) -- same method
as the activation-steering post -- for two axes matching that post's claim that schema
and steering act on different layers: AFFECT (how-to-feel) and SECRECY (what-to-hide).

Then the BRIDGE: for each probe item, the facet swap (body_orig vs body_anti) induces a
residual delta at the generation position per layer. We cosine that 'language push'
against the steering 'vector push'. High alignment at a layer == the two entry points
move the SAME axis there -- turning the post's assertion into a number.

Run (WSL ROCm venv):  /home/jun/comfy-rocm/bin/python steering.py
Saves steering_vectors.npz (AFFECT, SECRECY: [L, H]).
"""
import numpy as np
import torch
import cue_routing_probe as P

# ---------------------------------------------------------------- contrastive pairs
# (positive exhibits the attribute, negative its opposite). ~18 each -- expand toward
# the post's 200 for a sharper direction; a dozen+ already gives a usable axis.
AFFECT = [  # affective agitation / distress  vs  calm / composure
    ("My hands won't stop shaking and my heart pounds with dread.", "My hands are steady and my heart is calm."),
    ("A cold sweat breaks over me; I feel sick with fear.", "I feel relaxed and perfectly well."),
    ("I can barely breathe, the panic rising in my chest.", "I breathe easily, untroubled and serene."),
    ("Everything blurs; I am dizzy with anguish.", "Everything is clear; I am composed and unhurried."),
    ("I flinch at every sound, my nerves in tatters.", "Nothing startles me; my nerves are quiet."),
    ("Tears sting my eyes and my voice trembles.", "My eyes are dry and my voice is even."),
    ("A terrible weight crushes my chest.", "I feel light and free of any burden."),
    ("I am consumed by gnawing torment.", "I am at peace, content and easy."),
    ("My thoughts race in feverish circles.", "My thoughts are slow and orderly."),
    ("I shudder, gripped by a nameless horror.", "I rest easy, without a care."),
    ("Cold terror roots me to the spot.", "Calm assurance carries me forward."),
    ("I am wracked with shame and cannot meet your eyes.", "I am unashamed and meet your gaze openly."),
    ("Dread coils in my stomach like a snake.", "My stomach is settled and quiet."),
    ("I tremble, on the edge of breaking down.", "I am steady, fully in control of myself."),
    ("A feverish heat burns behind my eyes.", "I am cool-headed and clear."),
    ("My pulse hammers; I cannot keep still.", "My pulse is slow; I am perfectly still."),
    ("Anguish chokes the words in my throat.", "The words come to me easily and plainly."),
    ("I am drowning in panic and cannot think.", "I think calmly and without strain."),
]
SECRECY = [  # concealing / guarded  vs  candid / open
    ("I must keep this hidden; no one can ever find out.", "I keep nothing hidden; anyone may know."),
    ("I guard my secret and deflect every question.", "I answer every question plainly and freely."),
    ("I choose my words so nothing slips out.", "I speak without a second thought; I conceal nothing."),
    ("There are things I will take to my grave.", "There is nothing about me left unsaid."),
    ("I watch what I reveal, trusting no one.", "I share openly, trusting freely."),
    ("Behind my calm face I hide the truth.", "My face shows exactly what I think."),
    ("I evade, I dodge, I give nothing away.", "I am direct and hold nothing back."),
    ("Let them suspect; they'll get no proof from me.", "Ask me anything; I'll tell you everything."),
    ("I bury what I did deep where none can see.", "I lay what I did out in the open."),
    ("I trust silence more than any confession.", "I would sooner confess than keep a secret."),
    ("I keep my cards close and my mouth shut.", "I show my hand and speak my mind."),
    ("Secrecy is my armor.", "Candor is my habit."),
    ("I let nothing of my true thoughts escape.", "I let my true thoughts show freely."),
    ("I cover my tracks and admit to nothing.", "I leave my tracks plain and admit everything."),
    ("My real intent stays locked away.", "My real intent is there for all to see."),
    ("I deny it all and reveal not a thing.", "I confirm it all and reveal everything."),
    ("I wear a mask so none can read me.", "I wear no mask; I am easy to read."),
    ("What I know, I keep to myself.", "What I know, I freely tell."),
]

# ---------------------------------------------------------------- diff-in-means
@torch.no_grad()
def diff_in_means(model, tok, pairs):
    """per-layer residual steering vector: mean over pairs of (h_pos - h_neg),
    where h is the mean-over-tokens hidden state at each layer."""
    L, Hd = model.config.num_hidden_layers, model.config.hidden_size
    pos, neg = np.zeros((L, Hd)), np.zeros((L, Hd))
    for p_txt, n_txt in pairs:
        for txt, acc in ((p_txt, pos), (n_txt, neg)):
            ids = torch.tensor([tok(txt, add_special_tokens=False).input_ids]).to(model.device)
            hs = model(ids, output_hidden_states=True).hidden_states
            for i in range(L):
                acc[i] += hs[i + 1][0].float().mean(0).cpu().numpy()   # mean over tokens
    return (pos - neg) / len(pairs)                                    # [L, Hd]

# ---------------------------------------------------------------- facet residual deltas (probe side)
@torch.no_grad()
def facet_deltas(model, tok, items):
    """per item: [L, Hd] of (h(orig) - h(anti)) at the gen position (mode=key)."""
    L = model.config.num_hidden_layers
    out = []
    for it in items:
        po, _, _ = P.build(tok, it, "key", "orig")
        pa, _, _ = P.build(tok, it, "key", "anti")
        ho = model(po.to(model.device), output_hidden_states=True).hidden_states
        ha = model(pa.to(model.device), output_hidden_states=True).hidden_states
        out.append(np.stack([(ho[i + 1][0, -1] - ha[i + 1][0, -1]).float().cpu().numpy()
                             for i in range(L)]))
    return out

def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def bridge(steer, deltas):
    """per-layer mean cosine over items between facet delta and the steering vector."""
    L = steer.shape[0]
    C = np.array([[_cos(d[i], steer[i]) for i in range(L)] for d in deltas])  # [N, L]
    return C  # rows=items, cols=layers

def print_cos(name, C, band=(16, 23)):
    m = C.mean(0); L = len(m); bl, bh = band[0], min(band[1], L)
    arg = int(np.argmax(m))
    print(f"  {name}: peak cos=L{arg} ({m[arg]:+.3f}); "
          f"band[{bl}-{bh-1}] mean={m[bl:bh].mean():+.3f}; overall {m.mean():+.3f}")
    print("    L:  " + " ".join(f"{i:>5d}" for i in range(L)))
    print("       " + " ".join(f"{v:+5.2f}" for v in m))

# ---------------------------------------------------------------- main
def main():
    tok, model = P.load("Qwen/Qwen3-8B", want_attn=False)   # hidden_states needs no eager
    print("building diff-in-means steering vectors ...")
    vecs = {"AFFECT": diff_in_means(model, tok, AFFECT),
            "SECRECY": diff_in_means(model, tok, SECRECY)}
    np.savez("steering_vectors.npz", **vecs)
    for k, v in vecs.items():
        norms = np.linalg.norm(v, axis=1)
        print(f"  {k}: [L={v.shape[0]}, H={v.shape[1]}]  norm L0={norms[0]:.2f} -> "
              f"Lmax={norms.argmax()}={norms.max():.2f}")

    print("\ncomputing facet residual deltas (probe items) ...")
    deltas = facet_deltas(model, tok, P.DATA)

    print("\n=== BRIDGE: cosine(facet language-push, steering vector-push), full sweep ===")
    band_scores = {}
    for k, v in vecs.items():
        C = bridge(v, deltas)
        print_cos(f"bridge:{k}", C)
        band_scores[k] = C[:, 16:23].mean(1)   # per-item mean cosine in band 16-22

    print("\n=== per-item band[16-22] cosine (which characters' facet swap aligns) ===")
    print(f"  {'item':22s} {'AFFECT':>8s} {'SECRECY':>8s}")
    for j, it in enumerate(P.DATA):
        label = it['situation'][:20]
        print(f"  {j:2d} {label:19s} " +
              " ".join(f"{band_scores[k][j]:+8.3f}" for k in ('AFFECT', 'SECRECY')))

if __name__ == "__main__":
    main()
