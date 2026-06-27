# mrprompt-repro

An independent, same-conditions **reproduction of MRPrompt** — the prompting method from
*Memory-Driven Role-Playing: Evaluation and Enhancement of Persona Knowledge Utilization in
LLMs* (MDRP, arXiv:2603.19313) — testing its central mechanistic claim that persona facets
are **cue-addressable** (recalled by matching dialogue cues to facet cue-keys).

This is a **verification, not a takedown**: we run the paper's own recipe and ask whether the
mechanism it advertises is what actually drives the gains. Model under test: **Qwen3-8B**.

## TL;DR findings

**Behavioral (N=100, GPT-4.1-mini judge, all 7 conditions, uniform thinking-ON budget):**

| condition | mean FA (adherence to true cued facet) |
|---|---|
| base | 6.26 |
| card | 6.11 |
| **mrprompt** | **6.78** |
| mrprompt_noscene | 6.53 |
| mrprompt_nokey | 6.88 |
| mrprompt_wrongkey | 6.68 |
| mrprompt_anti | 5.78 |

- **The headline replicates and is significant:** `mrprompt − card` = +0.67 [+0.20, +1.12].
- **Cue-addressable is NOT supported:** deleting (`nokey`) or scrambling (`wrongkey`) the
  cue-keys does not reduce adherence (`full − nokey` = −0.07 ns, `full − wrongkey` = +0.15 ns).
- **The active ingredient is facet CONTENT + chain-of-thought, not cue-key routing:** swapping
  in the opposite facet's content (`anti`) drops adherence significantly (`full − anti` = +1.06 [+0.58, +1.56]).

**Mechanistic (n=29 characters, controlled forced-choice logprob probe + activations):**

- **Cue-routing null holds:** `key − wrongkey` = −0.84 [−2.98, +1.43] (CI crosses 0); cue-keys
  are if anything negative (`key − nokey` = −3.98 [−6.74, −1.25]).
- **Facet content localizes to layers 16–22** (statistically significant with bootstrap CIs):
  body-content attention concentrated 1.52× [1.45, 1.58]; residual norm-delta 1.26× [1.23, 1.30].
- **Per-character "same-axis" bridge is positive** (+0.050 [+0.020, +0.079] in the 16–22 band):
  a facet swap and a paraphrase-built disposition steering vector move the *same* axis in the
  *same* band — once the generic-axis confound (which had made an earlier bridge null) is removed.

> The paper is explicitly black-box ("we treat LTM as a black-box conditioning source", D.2).
> Everything mechanistic here is in territory the paper declines to enter.

## Caveats

- Verbatim paper prompts (Appendix N/O) are not public; construction/judge prompts are
  **recipe-faithful reconstructions**, not byte-identical. The official 200-instance MRBench is
  in an unreleased anonymized repo.
- Construction model: `gpt-4.1-2025-04-14`; judge: `gpt-4.1-mini-2025-04-14`.
- Effective n per condition 94–100 (truncated/empty thinking responses scored as missing).
- Single model (Qwen3-8B), single seed family.

## Layout

```
code/                interpretability + pipeline scripts (see below)
facets/              77 GPT-4.1-constructed persona facet schemas (LTM)
instances.jsonl      100 assembled (LTM, STM) evaluation instances
generations_clean.jsonl / scores_clean.jsonl   definitive N=100 run + judge scores
generations*.jsonl / scores*.jsonl             earlier pilot / thinking-on runs
interp_A.log         Lever A: cue-routing null + 16–22 localization (n=29, bootstrap CIs)
interp_B.log         Lever B: per-character same-axis bridge
```

Not committed (regenerable / third-party): `venv/`, `CharacterEval/` (see Attribution).

## Key scripts (`code/`)

| file | role |
|---|---|
| `facetize.py`, `assemble.py`, `renderers.py` | build facet schemas + (LTM, STM) instances + per-condition personas |
| `gen_clean.py` | definitive generation, all 7 conditions, resume-capable (append mode) |
| `judge.py`, `analyze.py` | MREval-style FA scoring + aggregation with bootstrap CIs |
| `cue_routing_probe.py` | controlled forced-choice logprob probe (flat / nokey / wrongkey / key) + attention & residual sweeps |
| `probe_ext.py` | 17 additional literary characters (+ paraphrase pairs) |
| `run_probe_ext.py` | Lever A: n=29 null + localization with per-layer bootstrap CIs |
| `steering.py`, `bridge_percharacter.py` | Lever B: diff-in-means steering + per-character same-axis bridge |
| `watch_eval_clean.sh`, `run_interp.sh` | orchestration |

Scripts assume data at `~/mdrp-repro/` and a torch/transformers/openai environment
(this work used a ROCm venv on a Strix Halo / gfx1151 APU). Paths are absolute to that
working dir — this repo is a **research record**, not a turnkey package.

## Attribution

Evaluation instances are adapted from **CharacterEval** (Tu et al., 2024), released under the
MIT License. The CharacterEval clone itself is not vendored here. Any downstream use must comply
with the original CharacterEval license/terms. MRPrompt / MDRP is the work of its original
authors (arXiv:2603.19313); this repository is an independent reproduction and is not affiliated
with them.
