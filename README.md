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

## The real loop

```bash
./run.sh pairs    r0    # two samples of the model per prompt = the pairs to judge
./run.sh judge    r0    # k judgments per pair, in BOTH presentation orders
./run.sh validate r0    # audit the judge against your labels — this is a gate
./run.sh peek     r0    # read the run
```

Then fill in `docs/human_label/labels.json` — your own verdicts on ~100 pairs —
and run `validate` again. Until that file has content, `clear_agreement` (the
one check that asks whether the judge is *right* about the cases it called easy)
cannot be computed, and the report says so rather than passing quietly.

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
configs/       design: model, judge thresholds, prompt domains
prompts/judge/ the judging protocol and the seed rubric (the constitution v0)
src/common/    config loading, chat templating, run directories
src/judge/     the harness: prompt building, parsing, aggregation, validation
src/cli.py     one entrypoint; every stage is a subcommand
tests/         unit (pure logic) + smoke (whole slice on the stub)
docs/          plan, prior-project state, and your human labels
runs/          outputs, gitignored
```

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
