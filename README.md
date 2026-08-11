# GeneralizationAndRL

Two things would count as a result. Either one is the point of the project.

**1. Trace generalization in the internals, all the way.** When a model forms a
principle — from examples, from its own revision, from training — does that
principle become a readable direction in its activations? Where does it appear,
when in the process, and is it causal? Probes, model diffing, direction
ablation.

**2. Build a self-play loop that cannot fool itself.** The model generates
variants of its own answers, judges them, finds the edge cases where its own
criteria break down, and keeps exploring — getting better with **minimal human
input**. Human labels are the bootstrap and the audit, not the fuel. The
destination is a loop that improves without someone feeding it preferences.

The second is why every gate in this repo exists. A self-improving loop with no
external check amplifies whatever its judge is biased toward, so the design
question is not "does it improve" but "can it tell when it is improving". Every
control here — both presentation orders, the clear-vs-boundary split, the
human-label gate, the controls that must come out clear — is there to make the
loop unable to lie to itself about its own progress.

The first is the instrument for the second: if a principle the loop adopts
shows up as a direction, you can watch the loop learn rather than only score it.

## Where things stand

Built and tested: the **answer pool** and the **judge harness**. Neither the
triage/escalation stages nor any training exists yet, on purpose — the judge is
the measurement instrument, and everything downstream is a function of it, so
it gets audited before anything consumes it.

Findings from run `r0`, which are why the current code looks the way it does:

- **Two samples of one model are not a preference pair.** 25 prompts, 150
  judgments, verdicts split 41 A / 42 B / 67 tie. 47 of 50 answers were
  markdown listicles; median within-pair length difference 13%. A group the
  judge cannot rank has no reward variance, and under GRPO no gradient.
- **The judge agreed with the human on 5 of 8 pairs both decided.** Its single
  confident pair was one the human judged the other way.
- **`length_bias` measured 0.056**, so the failure is not a preference for
  longer answers. It is *coverage over commitment*: protocol v1 asked only for
  evidence supporting a criterion, so adding a section could raise a score and
  never lower one. Protocol v2 requires evidence against as well.

## Try it in 30 seconds, without a GPU

```bash
./run.sh test          # 98 unit + smoke tests, ~1s
./run.sh stub demo     # the whole pipeline on a canned model
```

`stub demo` runs the real pipeline against a fake model that deliberately
misbehaves — one pair where the judge always picks whichever answer is shown
first, one where it emits no parseable verdict, one where it skips the
self-check step — and the harness should catch each and the gate should refuse
to pass.

## The pipeline

```bash
./run.sh constitute r1   # model derives criteria from your labelled examples
./run.sh sample     r1   # answer pool: drafts + prefills + self-review revisions
./run.sh pair       r1   # choose which pairs to judge                    no GPU
./run.sh judge      r1   # k judgments per pair, in BOTH presentation orders
./run.sh validate   r1   # audit the judge against your labels -- THE GATE
./run.sh spread     r1   # per-variant: did any conditioning produce signal? no GPU
./run.sh label      r1   # add your own verdicts, blind to the judge
```

`./run.sh pool r1` chains constitute → sample → pair → judge → spread.
`./run.sh night r1 --kill` runs it detached and destroys the box afterwards.

Stages resume. An interrupted `sample` skips the (prompt, variant) pairs it
already wrote and an interrupted `judge` skips the pair_ids it already judged,
so a multi-hour run continues instead of appending a second copy of everything.

## Design commitments

**The answer pool is conditioned, not just resampled.** Temperature raises
entropy over tokens, not over strategies — RLHF collapsed the output
distribution onto one mode, so hotter sampling returns the same listicle with
worse word choices. `configs/variants.json` reaches other modes by conditioning:
a prefill that seeds the assistant turn, or a self-review against a
constitution. Two plain drafts are kept as the control, and they cost nothing
because self-review needs a draft anyway.

**The judge reasons in explicit steps and checks its own work.** Six steps:
locate the answer → commitments → consequences → criteria one at a time →
adversarial self-check → verdict. Step 4 asks for evidence *against* each
criterion, because scoring only on supporting text means more material always
wins. Step 5 names what to hunt for, since "check your work" alone produces
rubber-stamping.

**Every pair is judged in both orders, always.** Not only as a validity check:
a pair whose verdict depends on which answer came first *is* a boundary case.

**No system prompt is injected, ever.** Qwen's template silently adds "You are
Qwen, a helpful assistant" when the caller supplies none, and a judge scoring
helpfulness with that in context is not impartial. `src/common/chat.py` makes
the system turn three-valued, defaults to none, and records the choice in the
run manifest. There is a test asserting this; do not delete it.

**Nothing is silently discarded.** Unparseable output is kept with its raw
text, and truncation is recorded as truncation rather than inferred from a
parse failure.

**`validate` gates the pipeline.** A judge that fails its gates stops the run.
Producing data from a broken instrument is worse than producing none.

## The constitution

`prompts/judge/rubric_v1.json` is derived from 13 hand labels rather than
written from scratch; each criterion carries a `_derivation` field naming the
labels it came from. `rubric_seed.json` (the hand-written v0) is kept unedited
so the two can be run as arms.

Current criteria: `answers_first`, `specific_or_asks`, `why_with_what`,
`assumes_competence`, `true_and_real`.

`docs/human_label/labels.json` is the only ground truth in the project. Entries
tagged `_annotator: claude` are provisional and are overridden by labelling the
pair by hand.

## Layout

```
configs/         design: model, judge thresholds, variants, hub repos, domains
prompts/judge/   the judging protocol and the constitutions
prompts/sample/  constitution-writer and self-review prompts
src/common/      config, chat templating, run directories, hub sync
src/judge/       parsing, aggregation, validation, model backends
src/sample.py    constitutions, variants, pair selection
src/cli/         one entrypoint; stages split across pool / score / review
tests/           unit (pure logic) + smoke (whole pipeline on the stub)
docs/            MECHANICS.md, PLAN.md, and your labels
runs/            outputs, gitignored
```

Rule: code in `src/`, design in `configs/`, prompt text in `prompts/`, outputs
in `runs/`. Adding a domain, a variant or a rubric version means adding a JSON
file, never editing Python.

## Hugging Face

Repo ids live in `configs/hub.json`: logs → `tewoto/Remote_logging_RL`
(dataset, `runs/<experiment>/`), adapters → `tewoto/LoRA_Adapters` (model,
`<experiment>/`).

```bash
./run.sh whoami          # check the token before renting a box
./run.sh push r1
./run.sh pull r1
```

`HF_TOKEN` is found by walking **up** from the repo root for a `.env`, so it
can live outside the git tree. A token already in the environment wins over the
file.

## Setup

```bash
pip install -r requirements.txt
```

The stub path needs only `pytest` — no torch, no GPU, no network. The real path
defaults to `Qwen/Qwen2.5-7B-Instruct`; `MODEL=... ./run.sh sample r1`
overrides it without editing a tracked file.

Prior reward-hacking / model-organism work, which this repo inherits its
infrastructure from but none of its claims: `<old repo url>`.
