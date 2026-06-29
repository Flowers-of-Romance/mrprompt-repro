# mrprompt-repro

An independent, **faithful reproduction of MRPrompt** — the prompting method from
*Memory-Driven Role-Playing: Evaluation and Enhancement of Persona Knowledge Utilization in
LLMs* (MDRP, arXiv:2603.19313) — testing its central mechanistic claim that persona facets
are **cue-addressable** (recalled by matching dialogue cues to facet cue-keys).

This is a **verification, not a takedown**: we run the paper's own recipe and ask whether the
mechanism it advertises is what actually drives the gains. Model under test: **Qwen3-8B**.

> **Correction (2026-06-29).** An earlier version of this repo used *reconstructed* prompts and a
> *reconstructed* FA metric, on the mistaken belief that the paper's prompts and scoring rubric
> were unpublished. They are in fact public: the generation/construction prompts are in the
> paper's appendix (Fig.14/15/18/19) and the scoring rubric is Table 21 (MREval). The behavioral
> results below have been **re-run with the paper's verbatim prompts and its real MS-FA metric**,
> with the thinking/token confound removed. The qualitative conclusions held; this version is the
> one that can legitimately be called a faithful reproduction.

## Method (faithful)

- **LTM construction** uses the paper's verbatim Chinese prompts: Card-LTM via **Fig.14**,
  MRPrompt facet-LTM via **Fig.15**. The two genuinely differ (Card facets carry 4 fields and no
  retrieval cues; MRPrompt facets carry 10 fields including `cue_phrases`), so the card-vs-mrprompt
  contrast is now real rather than a programmatic projection.
- **Generation** wraps each LTM in the paper's verbatim system prompts: plain role-play **Fig.18**
  (base, card) and the Magic-If memory-augmented protocol **Fig.19** (mrprompt family). The earlier
  self-made "explicit 4-step CoT" protocol is gone.
- **Confound removed.** Every condition uses an identical `max_new_tokens` budget; the *only*
  manipulated variable is Qwen3's thinking mode. The main axis is 7 conditions with thinking OFF;
  an ablation arm (`card_think`, `mrprompt_think`, `mrprompt_anti_think`) toggles thinking ON so its
  contribution can be measured directly. (The earlier run confounded format with thinking on/off and
  token budget.)
- **Scoring** is the paper's **MS-FA** (Table 21): a *contrastive* judge that sees the same
  character's responses under the true facet-LTM and the inverted (anti) facet-LTM and rates their
  separability 1/5/10. A single-response facet-adherence score (1–10) is kept as a secondary metric
  for the cue-key ablation.

## TL;DR findings

**Behavioral, faithful (N=100, all responses non-empty, GPT-4.1-mini judge; paired Δ ± SEM):**

Single-response facet adherence (1–10) by condition:

| condition | adherence | condition | adherence |
|---|---|---|---|
| base | 7.23 | mrprompt_nokey | 7.47 |
| card | 6.97 | mrprompt_wrongkey | 7.40 |
| **mrprompt** | **7.43** | card_think | 7.75 |
| mrprompt_noscene | 7.28 | mrprompt_think | 8.08 |

MS-FA contrastive separability (1/5/10): OFF = **8.20**, ON = **9.21**.

- **Cue-addressable is NOT supported (robust).** Deleting (`nokey`, Δ=−0.04±0.15), scrambling
  (`wrongkey`, Δ=+0.03±0.15) or removing scene facets entirely (`noscene`, Δ=+0.15±0.17) does not
  move facet adherence. The model is not doing cue-key facet retrieval. This survives the faithful
  prompts and the real MS-FA metric.
- **The active ingredient is chain-of-thought, not the structured LTM.** With the confound removed,
  toggling thinking ON gives a large, consistent gain — `card_think − card` = +0.78 [±0.17],
  `mrprompt_think − mrprompt` = +0.65 [±0.16], `MS-FA(ON) − MS-FA(OFF)` = +1.01 [±0.38]. The
  structural effect is small by comparison: `mrprompt − base` = +0.20±0.16 (ns), `mrprompt − card`
  = +0.46±0.16, and `card − base` = −0.26 (the lean card is worse than plain narrative).

**Mechanistic (n=29 characters, controlled forced-choice logprob probe + activations):**
*(Independent English-character interpretability probe; unchanged, see `interp_A.log` / `interp_B.log`.)*

- **Cue-routing null holds:** `key − wrongkey` = −0.84 [−2.98, +1.43] (CI crosses 0); cue-keys are
  if anything negative (`key − nokey` = −3.98 [−6.74, −1.25]).
- **Facet content localizes to layers 16–22:** body-content attention concentrated 1.52×
  [1.45, 1.58]; residual norm-delta 1.26× [1.23, 1.30].
- **Per-character "same-axis" bridge is positive** (+0.050 [+0.020, +0.079] in the 16–22 band).

> The paper is explicitly black-box ("we treat LTM as a black-box conditioning source", D.2).
> Everything mechanistic here is in territory the paper declines to enter.

## Caveats

- Single model (Qwen3-8B), single seed family; Chinese CharacterEval characters.
- Construction model: `gpt-4.1-2025-04-14`; judge: `gpt-4.1-mini-2025-04-14`.
- The instance-selection step (which STM cues which facet, and the inverted counter-facet) is our
  own apparatus, not a paper prompt; it selects, it does not implement the method under test.
- The official MRBench instances are in an unreleased anonymized repo; we use CharacterEval
  characters with the paper's verbatim construction/generation/scoring prompts.

## Layout

```
code/                    pipeline + interpretability scripts (see below)
facets_faithful/         78 facet-LTM schemas built via Fig.15 (MRPrompt-LTM)
cards_faithful/          78 Card-LTM schemas built via Fig.14
instances_faithful.jsonl 100 assembled (facet-LTM, STM, cued facet, anti-facet) instances
generations_faithful.jsonl   N=100 × 10 conditions, uniform budget, thinking the only toggle
scores_faithful.jsonl    MS-FA(off/on) + per-condition facet-adherence
interp_A.log             mechanistic Lever A: cue-routing null + 16–22 localization (n=29)
interp_B.log             mechanistic Lever B: per-character same-axis bridge
```

Not committed (regenerable / third-party): `venv/`, `CharacterEval/`, `*.log` working logs.
The earlier reconstruction's scripts (`facetize.py`, `renderers.py`, `judge.py`, `gen_clean.py`, …)
are kept under `code/` as a provenance record; the faithful pipeline supersedes them.

## Key scripts (`code/`)

| file | role |
|---|---|
| `faithful_prompts.py` | the paper's verbatim Fig.14/15/18/19 prompts + Table 21 MS-FA rubric |
| `build_ltms.py` | construct Card-LTM (Fig.14) and facet-LTM (Fig.15) per character |
| `assemble_faithful.py` | build (LTM, STM, cued/anti facet) instances from CharacterEval |
| `faithful_render.py` | wrap each condition's LTM in Fig.18 (plain) / Fig.19 (Magic-If) |
| `gen_faithful.py` | generation, 10 conditions, uniform budget, thinking-only toggle, resumable |
| `judge_faithful.py` | MS-FA contrastive scoring (off/on) + facet-adherence |
| `analyze_faithful.py` | paired Δ for the structure / cue-key / CoT contrasts |
| `cue_routing_probe.py`, `probe_ext.py`, `run_probe_ext.py` | mechanistic Lever A (logprob probe + localization) |
| `steering.py`, `bridge_percharacter.py` | mechanistic Lever B (same-axis bridge) |

Scripts assume data at `~/mdrp-repro/` and a torch/transformers/openai environment (this work used
a ROCm venv on a Strix Halo / gfx1151 APU). Paths are absolute to that working dir — this repo is a
**research record**, not a turnkey package.

## Attribution

Evaluation characters are adapted from **CharacterEval** (Tu et al., 2024), released under the MIT
License. The CharacterEval clone itself is not vendored here. Any downstream use must comply with the
original CharacterEval license/terms. MRPrompt / MDRP is the work of its original authors
(arXiv:2603.19313); this repository is an independent reproduction and is not affiliated with them.
