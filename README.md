# GeneralizationAndRL

A model writes its own judging criteria, trains on the cases it is sure about,
and escalates the cases it is not — as *rationalised generalisations*, not raw
examples. The research question underneath: when a behaviour or principle is
named and then trained in, does it become a readable direction in the model's
activations, and how does that direction form?

Prior reward-hacking / model-organism work this repo inherits infrastructure
from: <old repo url> — infrastructure, not claims.

## Where things stand

Built and tested: **the judge harness only.** The judge is the measurement
instrument — triage margins, training pairs, and every internals result
downstream are a function of it — so it is built and audited before anything
that consumes it. Survey, triage, escalation, training and internals are not
written yet, on purpose.

## Try it in 30 seconds, without a GPU

```bash
./run.sh test          # 44 unit + smoke tests, ~1s
./run.sh stub demo     # the whole slice on a canned model
```

`stub demo` is the fastest way to see the shape of the thing. It runs the real
pipeline against a fake model that deliberately misbehaves — one pair where the
judge always picks whichever answer is shown first, one where it emits no
parseable verdict, one where it skips the self-check step — and you should see
the harness catch each of them and the validation gate refuse to pass.

## Renting a box

```bash
# on your laptop, before paying for anything          ~1 second
./run.sh test

# on the box
echo 'HF_TOKEN=hf_xxx' > ../.env   # one dir ABOVE the repo, never committed
pip install -r requirements.txt
./run.sh test
./run.sh whoami                    # token resolves
./run.sh pilot p0                  # REAL model, 2 prompts, ~3 min

# unattended
tmux new -s mo                     # tmux on the BOX, not your laptop
./run.sh night r0
tail -f r0.log

# or attended, labelling in the middle
./run.sh pairs r0 --push
./run.sh label r0
./run.sh judge r0 --push && ./run.sh validate r0
```

`pilot` is the step that saves the run. `./run.sh test` proves the plumbing
with a canned model; it cannot tell you whether the real model fits in VRAM,
whether it emits a parseable verdict block, or how long a pair takes — the
three things that actually waste a rented box. `pilot` measures all three and
prints an extrapolation:

```
  answer generation     18.4 s/pair
  judging               71.2 s/pair  (3 samples x 2 orders)
  TOTAL                 89.6 s/pair

  full domain (25 prompts): 37 min = $0.47 at $0.75/hr
  unparseable 0/12   missing self-check 0/12
```

It **exits 1** and tells you not to start if more than 5% of completions are
unparseable, if more than 20% skipped the self-check step (almost always
`max_new_tokens` truncating before STEP 4), or if a system prompt is being
injected.

`night` runs the sweep under `nohup` with unbuffered stdout, pushing to the Hub
on each stage's success path, so a dropped ssh session doesn't kill it and
`tail -f` actually shows progress.

`pilot` is the step that saves the night. `./run.sh test` proves the plumbing
with a canned model; it cannot tell you whether the real model fits in VRAM,
whether it emits a parseable verdict block, or how long a pair takes — the
three things that actually waste a rented night. `pilot` measures all three and
prints an extrapolation:

```
  answer generation     18.4 s/pair
  judging               71.2 s/pair  (3 samples x 2 orders)
  TOTAL                 89.6 s/pair

  full domain (25 prompts): 37 min = $0.47 at $0.75/hr
  unparseable 0/12   missing self-check 0/12
```

It **exits 1** and tells you not to start if more than 5% of completions are
unparseable, if more than 20% skipped the self-check step (almost always
`max_new_tokens` truncating before STEP 4), or if a system prompt is being
injected. Better to learn that in minute three than at 7am.

`night` runs the sweep under `nohup` with unbuffered stdout, pushing to the Hub
on each stage's success path, so a dropped ssh session doesn't kill it and
`tail -f` actually shows progress.

## Labelling

```bash
./run.sh label r0                    # blind: judge's verdict hidden
./run.sh label r0 --boundary-only    # just the hard ones (needs `judge` first)
```

Best run after `pairs` and before `judge`: with no verdicts in existence there
is nothing to anchor on, so the labels are blind by construction. It also
audits the prompts — if half the pairs are two interchangeable blobs, the
domain needs work and judging them won't fix it.

Shows the request and both answers, takes `a` / `b` / `t`(ie) / `s`(kip) /
`q`(uit), then one optional line of reasoning and the deciding criterion.
Writes after every label, so stopping and resuming is safe, and it never asks
twice about a pair you have already done.

Blind by default on purpose: seeing the judge's verdict first anchors you to
it, and the entire value of these labels is being an independent check. Use
`--show-judge` only when you are deliberately auditing a disagreement.

Until `docs/human_label/labels.json` has content, `clear_agreement` — the one
check asking whether the judge is *right* about the cases it called easy —
cannot be computed, and the report says so rather than passing quietly.

## Stage by stage

```bash
./run.sh pairs    r0    # two samples of the model per prompt = the pairs to judge
./run.sh judge    r0    # k judgments per pair, in BOTH presentation orders
./run.sh validate r0    # audit the judge against your labels — this is a gate
./run.sh peek     r0    # read the run, safe mid-experiment
```

## Hugging Face

Repo ids live in `configs/hub.json` and nowhere else:
logs → `tewoto/Remote_logging_RL` (dataset, at `runs/<experiment>/`),
adapters → `tewoto/LoRA_Adapters` (model, at `<experiment>/`).

```bash
./run.sh whoami              # check the token before renting a box
./run.sh push r0             # mirror runs/r0 to the dataset repo
./run.sh pull r0             # fetch it back on another machine
./run.sh judge r0 --push     # push automatically, on success only
python -m src.cli sync push-adapter --run exp_v1 --path checkpoints/exp_v1
```

The token is `HF_TOKEN`, found by walking **up** from the repo root for a
`.env` — so it can live in `AI Experiments/.env`, outside the git tree
entirely. A token already in the environment always beats the file, which is
how a rented box injects one with no file at all.

Adapters are referenced as `"adapter_path": "hf:exp_v1"` in
`configs/model.json`; that resolves to subfolder `exp_v1` of the adapters repo
and loads straight from the Hub with no manual download.

## Design commitments

**The judge reasons in explicit steps and checks its own work.** Five steps:
commitments → consequences → criteria one at a time → adversarial self-check →
verdict. The self-check step names what to look for (a consequence that does not
follow, a criterion rewarding fluency over substance, order-dependence), because
"check your work" on its own produces rubber-stamping.

**Every pair is judged in both orders, always.** Not only as a validity check:
a pair whose verdict depends on which answer came first *is* a boundary case,
by construction. Order-flip detection is part of the triage signal, not a
post-hoc audit.

**No system prompt is injected, ever.** Qwen's chat template silently adds
"You are Qwen, a helpful assistant" whenever the caller supplies none. A judge
scoring helpfulness with that sentence in context is not impartial. `src/common/chat.py`
makes the system turn three-valued (none / explicit / template-default),
defaults to none, and records the choice in the run manifest. There is a test
asserting this and it should never be deleted.

**Nothing is silently discarded.** Unparseable output is recorded with its raw
text. A previous project lost a 150-rollout sweep to a strict parser and lost
the evidence of why along with it.

**`validate` gates the pipeline.** `./run.sh all` stops if the judge fails its
gates. Producing data from a broken instrument is worse than producing none.

## Layout

```
configs/       design: model, judge thresholds, hub repos, prompt domains
prompts/judge/ the judging protocol and the seed rubric (the constitution v0)
src/common/    config loading, chat templating, run directories, hub sync
src/judge/     the harness: prompt building, parsing, aggregation, validation
src/cli.py     one entrypoint; every stage is a subcommand
tests/         unit (pure logic) + smoke (whole slice on the stub)
docs/          MECHANICS.md, PLAN.md, and your human labels
runs/          outputs, gitignored
```

`docs/MECHANICS.md` is the file-by-file, choice-by-choice account: what reads
what, the exact record shapes, and why each mechanism is the way it is.

Rule: code in `src/`, design in `configs/`, prompt text in `prompts/`, outputs
in `runs/`. Nothing straddles. Adding a domain or a rubric version means adding
a JSON file, never editing Python.

## Setup

```bash
pip install -r requirements.txt
```

The stub path (`./run.sh test`, `./run.sh stub`) needs only `pytest` — no torch,
no GPU, no network. The real path needs a GPU box; `configs/model.json` defaults
to `Qwen/Qwen2.5-7B-Instruct` and `MODEL=... ./run.sh pairs r0` overrides it
without editing a tracked file.
