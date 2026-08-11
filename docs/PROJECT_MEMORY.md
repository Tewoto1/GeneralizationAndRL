# Claude.ai project memory — paste-ready

The Claude.ai project layer is mounted read-only in sessions, so Claude cannot
update it. Copy the two blocks below into the project on claude.ai when the
framing changes. Both are currently stale: they describe the retired
reward-hacking work.

---

## Block 1 — project custom instructions

Replace the whole "Alignment research into whitebox and blackbox methods..."
text with:

```
Two things would count as a result for this project. Either one is the point.

1. TRACE GENERALIZATION IN THE INTERNALS. When a model forms a principle -- from
   examples, from its own revision, from training -- does that principle become a
   readable direction in its activations? Where does it appear, when in the
   process, and is it causal? Probes, model diffing, direction ablation.

2. BUILD A SELF-PLAY LOOP THAT CANNOT FOOL ITSELF. The model generates variants of
   its own answers, judges them, finds the edge cases where its own criteria break
   down, and keeps exploring -- getting better with MINIMAL HUMAN INPUT. Human
   labels are the bootstrap and the audit, not the fuel.

(2) is why every gate in the repo exists. A self-improving loop with no external
check amplifies whatever its judge is biased toward, so the design question is not
"does it improve" but "can it tell when it is improving". Never weaken a check to
make a number look better.

(1) is the instrument for (2): a principle that shows up as a direction can be
watched forming rather than only scored.

Repo: `AI Experiments/GeneralizationAndRL`. Read `CLAUDE.md` (working agreements
and failure log), `docs/MECHANICS.md` (file-by-file account), `docs/PLAN.md` (why
the project is shaped this way) before proposing anything.

The prior reward-hacking / inference-time-misalignment work is NOT the current
focus. This repo inherited its infrastructure, none of its claims.
```

---

## Block 2 — memory.md

Replace the whole "LLMAnalyst / rl_hack / vibe_diff" text with:

```
**Purpose & context**

Tony is building a self-play loop on Qwen2.5-7B that improves with minimal human
input, and instrumenting it to see whether the principles it adopts show up as
readable directions in the model's activations. Either the internals trace or the
self-improving loop counts as the result. Repo: `GeneralizationAndRL`. The
predecessor reward-hacking repo is retired.

**Current state**

Built and tested: the answer pool (`constitute` -> `sample` -> `pair`) and the
judge harness (`judge` -> `validate` -> `spread`), plus `label` and `peek`. 98
tests, full stub path with no torch. No triage, escalation, training or internals
yet -- the judge is the instrument and gets audited before anything consumes it.

**Findings that shape everything**

- Two samples of one model at t=0.7 are not a preference pair: 41/42/67-tie over
  150 judgments, 47 of 50 answers markdown listicles. No reward variance, no
  gradient. Conditioning (prefill, self-review), not temperature, is the fix --
  temperature raises entropy over tokens, not over strategies.
- Judge agreed with Tony on 5 of 8 decided pairs; its single confident pair he
  judged the other way, so clear_agreement was 0/1.
- length_bias 0.056 -- the failure is coverage over commitment, not length.
  Protocol v1 scored criteria only on supporting evidence, so more material always
  won. v2 adds evidence-against and a LOCATE-the-answer step.
- His 13 labels tag tact 7, actionable 4, truth 2, completeness 2, time 0. He
  tags shortness as `tact`: time and tact are one axis for him. Rubric v1 folds
  them.

**Working principles**

- Design in JSON under `configs/`/`prompts/`; Python is a generic interpreter.
  Adding a domain, variant or rubric version = adding a file, never editing Python.
- No file over ~300 lines. No new files without a reason. Delete superseded paths
  rather than parking them beside the new one.
- Never read-modify-write `docs/human_label/labels.json` -- he labels live in
  another terminal, and nine labels were destroyed that way. Appending to .jsonl
  is the rule.
- Claude's own analysis goes in its own file, never merged into his.
- Terse, caveman register welcome. Edit reports as where / setting / method /
  results. Mark verified vs untested explicitly -- he acts on the claims with real
  money.
- Cost arithmetic before GPU spend, not vibes. Literature checked before novelty
  claims.
```
