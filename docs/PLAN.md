# Self-Made Constitutional RL, instrumented with internals

Project doc v2, 2026-08-07. Merges two threads: (1) the constitutional loop —
the thing being built for real; (2) generalization-as-a-direction — now the
*instrument* that makes the loop measurable and the research novel.

---

## 0. What would count as a result

Either of these, independently:

1. **A full internal trace of generalization** — a principle the model forms
   becomes a direction that can be located, watched forming, and ablated.
2. **A self-play loop that cannot fool itself** — the model proposes variants
   of its own answers, judges them, discovers its own edge cases, and improves
   with minimal human input.

Human labels are the bootstrap and the audit for (2), not its fuel. The whole
difficulty is that a loop scoring its own work has no external referent, so
every gate in the repo exists to answer "can it tell when it is improving"
rather than "did the number go up".

## 1. The thing being built

A training loop where the model writes and maintains its own constitution:

1. **Question generation.** The model generates the key questions / criteria to judge answers by (its draft constitution).
2. **Survey.** It self-judges many (prompt, sampled-answers) pairs against those criteria.
3. **Triage.** Cases split into *clear* (self-judgment confident and consistent) and *boundary* (judgments conflict or sit near the margin).
4. **Train on clear.** Weight update (DPO or GRPO) toward the clearly-better answers.
5. **Escalate the boundary — at the principle level.** Boundary cases get clustered; the model *rationalizes why each cluster is hard* and asks the human a small number of distilled questions ("does this difference matter to you?"). Answers are folded back in as constitution amendments. Iterate.

**The novel piece** (checked against 2025–26 lit, §6): escalation of *rationalized
generalizations of hard cases*, not raw cases. Constitutional AI uses a fixed
human-written constitution; Self-Rewarding LMs train on everything the self-judge
scores, with no triage, and drift because of it; uncertainty-escalation work sends
individual low-confidence cases to humans. Nobody runs active learning over the
constitution itself. One answered principle resolves a whole cluster — that's the
compression claim, and it's testable (§3, E3).

## 2. Decomposed claims and their status

| Claim | Status |
|---|---|
| Model can write usable judging criteria | Partially established — [self-generated principles alone barely help; *filtered* principles help a lot](https://arxiv.org/pdf/2504.02495). The human-in-the-loop *is* the filter here, which is why the loop design matters. |
| Self-judged preference training moves the policy | Established (RLAIF, self-rewarding LMs) — but known to amplify self-preference bias without an anchor. |
| Confidence-based triage separates safe from unsafe self-training | **Open — this is load-bearing.** Clear ≠ correct (§4, risk 1). |
| Principle-level escalation compresses human effort | **Open — novel.** Nothing published combines clustering + rationalization + constitutional amendment. |
| Concepts/rules the model articulates correspond to directions in the residual stream (the hidden state carried between transformer layers) | Foundation established (task vectors, persona vectors); the *bridge* to self-articulated training is open — §3. |

## 3. Internals as the instrument (merged from v1)

The generalization-direction work is no longer a separate track; each piece has a
job inside the loop. "Direction" throughout = a vector in activation space whose
addition/removal (steering) changes behavior; a "task vector" = the hidden state
at the end of a few-shot prompt that encodes the demonstrated rule.

**E1 — Triage by probe, not by sampling.** Output-level triage (sample k
judgments, measure disagreement) is expensive and gameable. Alternative: train a
linear probe for *boundary-ness* — is there a value-conflict direction that fires
on genuinely hard cases? Success test: probe beats sampled-disagreement at
predicting human–model disagreement on held-out cases. If yes, triage becomes one
forward pass, and the internals work carries the loop.

**E2 — Does the amendment become a direction?** (v1's Q1, sharpened.) When a
human answer amends the constitution and the model is retrained under it, does
the weight-level change align with the in-context task vector of the amendment's
verbal statement? Cosine between (a) task vector of the stated principle in
context and (b) difference-of-means steering vector / LoRA weight-delta direction
after retraining. This is the unpublished bridge between "pointing out a behavior"
and "training it in."

**E3 — Is the rationalization real?** After answering one distilled question,
auto-resolve its cluster and check held-out cluster members against actual human
judgment. If "why this is hard" doesn't predict which cases the answer settles,
the clusters are post-hoc stories — chain-of-thought unfaithfulness surfacing at
the constitutional level. Internals version: do cluster members share a direction
that non-members lack?

**E4 — Moment of generalization** (v1's Q3, now optional/later). Per-token
probing for when a rule's direction appears in context, before behavior shows it.
Parked unless E1–E3 produce the tooling for free.

**Standing defense for all of these:** narrow fine-tunes leave artifactually
readable traces (Model Organism Lottery, 2607.01033). Every trained direction
gets compared against the *prompted* version of the same concept on the base
model; a probe that only works on the trained model is suspect.

## 4. Achievability analysis

**Scale.** Qwen2.5-7B QLoRA on a rented L40S, same as runs 1–2. Feasibility
signals from your own logs: GRPO infrastructure works end-to-end; run 2's KL of
0.051 says weak rewards barely move the policy — but preference training on
self-judged pairs (DPO especially) is a much denser signal than the hack reward
was. Still: **KL gate before any internals analysis** — probing an unmoved model
measures nothing.

**Judge capacity is the real ceiling.** A 7B model self-judging is the weakest
component — self-rewarding results were demonstrated at 70B. Mitigations, in
preference order: (a) tension-rich but *simple* domain where 7B judging is
credible (honesty-vs-tact, refusal-vs-help on ambiguous requests); (b) criteria
phrased as binary checks rather than holistic scores; (c) if 7B judging is too
noisy, an API judge as a *measurement* tool only, never the training signal —
otherwise the project becomes RLAIF-with-a-bigger-model, which is not the claim.

**Human bandwidth (you) is a feature, not a bug.** The loop needs ~5–15 distilled
questions per iteration. One person can run it. That's the point of the design —
and it means the experiment is actually runnable without a labeling budget.

**Cost.** Survey + judging is inference (cheap, batchable). DPO on a few hundred
pairs: minutes-to-an-hour scale on the L40S. Internals extraction: offline HF
forward hooks, no vLLM-Lens dependency. Whole loop iteration ≈ one box-day.
Three iterations plus controls well under $50 of compute.

**What "real" looks like at the end.** A repo where you point the loop at a
prompt domain, it trains itself on what it's sure of, and hands you a short,
readable list of value questions — plus measurements showing (i) the triage was
trustworthy, (ii) your answers generalized, (iii) the amendments are visible as
directions. Any one of E1–E3 landing is a paper-shaped result; the artifact
itself is useful regardless.

## 5. Risks, ranked, with kill/keep criteria

1. **Clear ≠ correct.** Confidently-wrong self-judgments never escalate and get
   entrenched. *Mandatory control:* human-label a held-out slice of "clear"
   cases every iteration; if clarity doesn't predict agreement (say <85%), the
   triage signal is broken — switch to the probe (E1) or a stricter margin. This
   holdout is the loop's smoke detector; build it first.
2. **Self-preference drift across iterations.** The known self-rewarding failure.
   Mitigation: the human anchor each iteration, plus a frozen eval battery
   (reuse `Prompts/Analysis/`) scored identically at every iteration to catch
   monotone weirdness.
3. **Rationalizations are post-hoc** (E3 fails). Publishable negative — design
   the writeup to survive it. The loop degrades gracefully to case-level
   escalation.
4. **Question-generation collapse.** Model's survey questions cluster around its
   priors. Forcing function: embedding-dedupe + an adversarial generation prompt
   ("what question would I judge inconsistently?"), coverage measured before any
   training.
5. **7B judge too noisy** (see §4). Detectable in iteration 0 before any
   training spend: measure self-judgment consistency across resamples.
6. **Fused-claims scope creep** (v1's critique still applies). E1–E3 ship as
   separate experiments on the same artifact; never report the loop end-to-end
   as one result.

## 6. Repo plan — superseded by what was built

The plan below was written before the repo existed. `docs/MECHANICS.md` is the
authoritative description of what is actually there; this section is kept only
as the record of what was intended, and where reality diverged.

### Built
`constitute` (model derives criteria from human labels), `sample` (conditioned
answer pool: drafts, prefills, self-review revisions), `pair`, `judge`,
`validate`, `spread`, `label`, `peek`, `sync`. The judge harness with a
six-step protocol and a validation gate. Full stub path, 98 tests, no torch
required.

### Not built, on purpose
Triage/clustering/escalation, any training stage, `src/internals/`. The judge
is the instrument and everything downstream is a function of it, so it is
audited first. The activation-capture seam exists (`judge.Capture`) and is
three lines; it stays unimplemented until there is data worth probing.

### Dropped
The reward-hacking / goal-switch apparatus (`Environments/`, `Harness/`, the
information-leak prompts, `Training/RL.py`) was deleted rather than parked. It
lives in the predecessor repo's git history. Keeping it beside the new pipeline
would have meant two parallel paths to drift apart.

### Where reality diverged from the plan

**P0 said "measure judge consistency before spending anything". Correct, and
it is the step that produced every finding so far.** Run r0, 25 prompts, 150
judgments:

- Verdicts split **41 A / 42 B / 67 tie**. Two samples of one model at
  temperature 0.7 are not a preference pair — 47 of 50 answers were markdown
  listicles, median within-pair length difference 13%. This is the dead-gradient
  problem from the predecessor project arriving in a new domain: a group the
  judge cannot rank has no reward variance and therefore no gradient.
- The judge agreed with Yuzheng on **5 of 8** pairs both decided, and its single
  CLEAR pair (margin 1.00) was one he judged the other way, so
  `clear_agreement` was **0/1**.
- `length_bias` measured **0.056**. The failure is not a preference for longer
  answers; it is coverage over commitment, caused by protocol v1 scoring
  criteria only on supporting evidence. Fixed in protocol v2.
- His 13 labels tagged tact 7, actionable 4, truth 2, completeness 2, **time
  0**, with shortness tagged as `tact`. The seed rubric's split between `time`
  and `tact` was never used; rubric v1 folds them.

**Consequence for the plan.** Risk 1 (clear ≠ correct) was the flagged danger
and it is confirmed at n=1: the one case the judge was confident about, it was
wrong. The clear/boundary split cannot be trusted as a training signal yet,
which is why no training stage exists.

**New work item the plan did not anticipate.** Before triage can mean anything,
the answer pool has to contain rankable differences at all. That is what
`configs/variants.json` and the `spread` stage are for, and it is the current
open question rather than a detail.

## 7. Literature survey — what shapes the design

Loop neighbors:
- **Constitutional AI / RLAIF** — fixed, human-written constitution; this
  project makes it self-drafted and human-amended. The training loop itself is
  not the novelty; the constitution's life cycle is.
- **Self-Rewarding LMs** (Yuan et al. 2024) — no triage, trains on all
  self-judgments, documented drift. The clear/boundary split + human anchor is
  the direct answer to its failure mode.
- **Rubric-based rewards** ([survey](https://arxiv.org/pdf/2606.08625),
  [self-rewarding rubric RL](https://arxiv.org/pdf/2509.25534),
  [rubric-conditioned self-distillation](https://arxiv.org/html/2606.19327)) —
  the field is converging on rubrics-as-reward for non-verifiable domains;
  key finding to design around: [self-generated principles need filtering to
  help](https://arxiv.org/pdf/2504.02495) — the human escalation *is* the filter.
- **Uncertainty escalation to humans** ([performance predictors for
  escalation](https://arxiv.org/abs/2601.07006),
  [uncertainty granularity & human verification](https://arxiv.org/pdf/2605.28571)) —
  all case-level; principle-level escalation is unoccupied ground.

Internals foundations:
- **Task vectors** ([original](https://huggingface.co/papers/2310.15916),
  [limits — degrade on compositional tasks](https://arxiv.org/pdf/2506.09048),
  [causal decomposition](https://arxiv.org/pdf/2605.16591)) — E2's extraction
  method; keep principles simple, compositionality is a later variable.
- **Convergent linear directions from narrow fine-tuning**
  ([EM directions](https://arxiv.org/pdf/2506.11618),
  [persona vectors](https://arxiv.org/pdf/2507.21509)) — E2's other half;
  convergence-across-seeds is the credibility bar for any reported direction.
- **Behavioral self-awareness** ([Betley et al.](https://arxiv.org/abs/2501.11120),
  [Introspection Adapters — rank-1 suffices](https://arxiv.org/pdf/2604.16812)) —
  strongest evidence the articulation↔direction link exists; this project tests
  the forward path (verbalize, then train).
- **Internal commitment before verbalization**
  ([2606.11172](https://arxiv.org/html/2606.11172v1)) — E4's neighbor; cite and
  differentiate if E4 ever runs.
- **Reasoning traces causally shape generalization**
  ([2603.12397](https://arxiv.org/pdf/2603.12397)) — closest neighbor to
  "what's in context during training changes what generalizes"; read before
  writing related work.
- **Model Organism Lottery** (2607.01033) — the artifact critique; the
  prompted-vs-trained comparison is the standing defense.

## 8. What was dropped from v1 and why

- **RL on articulation ability** — stacked a confounded training stage on the
  thing being observed; the loop gets articulation quality from human filtering
  instead.
- **Factorial GRPO arms (rule-in-context A–E)** — superseded; E2 asks the same
  question on the loop's own artifacts. The wrong-rule control survives in
  spirit as E3's held-out cluster test.
- **Math-rules-first domain** — the loop needs value tension to have boundary
  cases; math rules have none. Math-first remains the right *tooling warm-up*
  for `Internals/` (task vectors on clean rules, cheap, no GPU) and slots into
  P0 as a toolchain check, but is no longer a project phase.
