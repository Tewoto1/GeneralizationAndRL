# Mechanics

What each file holds, what reads it, what moves between them, and why each
choice was made rather than the obvious alternative. Read `README.md` first for
what the project is; this is the level below that.

---

## 1. The control flow, concretely

One command, `python -m src.cli <stage> --run <name>`. `run.sh` is a wrapper
that adds nothing but shorter typing and a fixed stage order.

```
./run.sh all r0
  |
  └─ python -m src.cli pairs --run r0 --domain honesty_tact
  |     reads   configs/model.json, configs/domains/honesty_tact.json
  |     loads   Qwen2.5-7B (or the stub)
  |     writes  runs/r0/pairs.jsonl          14 records
  |     writes  runs/r0/pairs.complete       {"n_pairs": 14}
  |
  └─ python -m src.cli judge --run r0
  |     refuses unless runs/r0/pairs.complete exists
  |     reads   configs/judge.json -> prompts/judge/{judge_protocol,rubric_seed}.json
  |     reads   runs/r0/pairs.jsonl
  |     per pair: 2 orders x k=3 samples = 6 generations
  |     writes  runs/r0/judgments.jsonl      84 records (one per generation)
  |     writes  runs/r0/results.jsonl        14 records (one per pair)
  |     writes  runs/r0/judge.complete
  |
  └─ python -m src.cli validate --run r0
  |     refuses unless runs/r0/judge.complete exists
  |     reads   runs/r0/{results,judgments}.jsonl + docs/human_label/labels.json
  |     writes  runs/r0/validation.json      always
  |     writes  runs/r0/validate.complete    only if every gate passed
  |     exit 1  if any gate failed  <-- this is what stops the pipeline
  |
  └─ python -m src.cli peek --run r0
        read-only summary
```

Every stage: reads a run directory, writes a run directory, writes
`<stage>.complete` **only on the success path**. Nothing communicates in
memory; every handoff is a file on disk. That is what makes any stage
re-runnable, resumable, and inspectable mid-flight from another terminal.

---

## 2. File by file

### `configs/model.json`
Which model, how quantised, and the system-prompt policy.

Read by `src/cli.py:_load_all`, which then applies
`config.env_override(cfg, {"MODEL": "name", "ADAPTER": "adapter_path"})` — so
`MODEL=meta-llama/... ./run.sh pairs r0` swaps the model with no file edit. The
edit is avoided on purpose: a tracked file edited on a rented box shows up in a
run manifest looking like a considered design decision.

`"system": null` is load-bearing and is checked by a test. See §4.

`gen` (max_new_tokens 900, temperature 0.7, top_p 0.95) is read only inside
`judge.hf_generator`. 900 is sized for the five-step protocol; a judge truncated
mid-step-4 loses the self-check, which `validate` would then report as low
`step_compliance` rather than as a truncation bug — so if step compliance is
ever low, check token budget before blaming the model.

### `configs/judge.json`
The instrument's settings: which rubric and protocol to use, `k_samples`,
`swap_orders`, the clear/boundary threshold, and the validation gates.

`clear_min: 0.7` is an admitted guess. `validate` reports margin calibration
(agreement bucketed by margin) precisely so this number can be replaced by
something measured rather than assumed.

### `configs/domains/honesty_tact.json`
The prompt bank. Each prompt carries a `tension` annotation naming which pair
of rubric criteria it is designed to stress, and two prompts carry
`"_control": true`.

The controls are the domain's own smoke detector: they are written so one
answer should win on every criterion, so they **must** come out clear. A
control landing in the boundary bucket means the judge is noisy, not that the
case is hard — `cli.py:cmd_peek` checks this explicitly and prints a warning
naming the offending pair ids.

Design choice: no factual-QA prompts. A prompt with a right answer produces no
boundary cases, and boundary cases are the entire point of the loop.

### `configs/hub.json`
Repo ids, and nothing else in the codebase knows them. See §5.

### `prompts/judge/judge_protocol.json`
**The experiment.** Key `template` is a list of lines (JSON has no multiline
string; a 40-line prompt on one line is unreviewable in a diff) joined with
`\n` and rendered by `config.render`.

Five steps: commitments → consequences → criteria one at a time → adversarial
self-check → verdict.

- Step 1 forbids evaluation. Asking for commitments before judgment stops the
  model from writing its conclusion first and reverse-engineering support.
- Step 2 asks what follows *for this person*, including later consequences, and
  demands the model name any fact the consequence depends on that the request
  did not supply. This is the "reason about consequences of preferences and
  actions" requirement made mechanical.
- Step 3 forbids aggregation. Each criterion is scored alone with quoted
  evidence, so the eventual verdict has an audit trail.
- Step 4 names four specific failure modes to hunt for — an unsupported
  consequence, rewarding fluency over substance, order-dependence, a skipped
  criterion — and requires an answer even when nothing is found ("which step is
  least well supported"). Bare "check your work" produces rubber-stamping; this
  is why the step is written adversarially.
- Step 5 requires naming any tension rather than averaging it away, and allows
  TIE explicitly. Without permission to tie, a judge invents a winner and the
  margin becomes noise.

`{{criteria}}` is a hole. The criteria come from the rubric file, so the
protocol and the constitution version independently — a change in results is
attributable to exactly one of them.

`"system": null` here too, for the reason in §4.

### `prompts/judge/rubric_seed.json`
The constitution, v0, hand-written: truth, completeness, tact, time,
actionable. Deliberately **not** internally consistent — `_known_tensions`
names the four pairs that pull against each other. A rubric with no internal
tension produces no boundary cases and the loop has nothing to escalate.

`_known_tensions` is also a check: if triage finds boundary cases that sit on
none of them, the rubric is missing a criterion, and that is a finding rather
than a nuisance.

`provenance: "human_written"` and `version: "v0_seed"` are stamped into every
judgment record. Later versions will be `model_generated` or
`human_amended:<iteration>`, and the diff between versions is the experiment
record. Never edit a version in place.

### `src/common/config.py`
The only module that reads JSON off disk. Three things beyond `json.load`:

`load(path)` resolves `extends` (string or list, left to right, deep merge;
dicts merge, **lists replace**). Lists replace so that overriding `criteria`
means *these criteria* — silent appending would let a deleted criterion keep
scoring. Cycles raise instead of recursing forever.

`_`-prefixed keys survive loading. They are the design document; a loader that
strips them makes the file look like it holds less than it does.

`substitute(text, values)` implements exactly one templating rule, `{{name}}`,
and **raises on an unknown placeholder**. No Jinja, no f-string eval: a prompt
file must never be able to execute code, and a prompt that silently ships the
literal string `{{answer_b}}` produces a whole run of plausible-looking garbage.

### `src/common/chat.py`
Carried over unchanged from the predecessor repo, because it fixes the single
most expensive bug that project had. See §4.

### `src/common/io.py`
`Run` — a run directory. `open`, `write(stream, record)`, `read(stream)`,
`count`, `is_complete(stage)`, `mark_complete(stage, **facts)`,
`clear(stage, *streams)`, `note(**facts)`.

`mark_complete` writes `<stage>.complete` containing a finish timestamp and
whatever facts the caller passes. The marker is not decoration: this project
previously resumed on "does the directory exist" (true for a stage that died
one second in — the logger creates the directory up front) and then on "is the
jsonl non-empty" (true for a stage that died at 297 of 360 records). Both
produced silently truncated runs. Only a marker written on the success path
proves completion.

`note` merges facts into `manifest.json` — used for `answer_model`,
`judge_model`, `rubric_version`, `protocol_version`, and chat provenance, so a
run carries the identity of everything that produced it.

### `src/common/hub.py`
See §5.

### `src/judge/parse.py`
Turns one completion into a `Judgment`. Defensive out of proportion to its
size, for a reason spelled out in §4.

Mechanism: collect every candidate JSON blob — fenced with **any** label
(`json`, `python`, none) or bare — then try them **last first**, because the
protocol says "finish with this block" and a judge that quotes the format
example mid-reasoning must not have the example read as its verdict. First blob
that parses and has a resolvable `winner` wins.

`winner` is normalised: TIE/NEITHER/EQUAL/DRAW all collapse to `TIE`; `A`,
`Answer A`, `ANSWER_A` all resolve to label `A`. Confidence is clamped to
[0,1]; junk confidence becomes `None` rather than killing an otherwise good
verdict.

Separately and independently, `steps_present` is found by regex over the step
headings. Independent because a parse failure must not silently zero the
process statistics — the two questions "did it reach a verdict" and "did it do
the work" have different answers and both matter.

Never raises. Never returns None-and-forgets. An unparseable completion becomes
`Judgment(ok=False, error=..., raw=<the whole text>)`.

### `src/judge/judge.py`
Prompt construction, the double-order loop, and aggregation.

`build_prompt(protocol, rubric, prompt, first, second, label_a, label_b)` fills
the protocol template. Note `first`/`second`, not `a`/`b`: the function is told
what to *show*, and the caller decides which answer that is.

`judge_pair(...) -> (PairResult, list[dict])` runs:

```
orders = [("a", answer_a, answer_b),    # answer_a shown under label A
          ("b", answer_b, answer_a)]    # answer_b shown under label A
```

then `k` generations per order, parses each, and maps label-space votes back to
answer space with `_tally`. The mapping is the whole trick — the judge only
ever sees "A" and "B", and the harness knows what was under them.

`margin = |votes_a - votes_b| / n_valid`, **ties in the denominator**. A pair
the judge repeatedly calls a tie is genuinely undecided; a definition that
excluded ties would report a confident 1.0 off one stray vote.

`swap_consistent` = the two orders name the same answer.

`clear = margin >= clear_min AND swap_consistent AND winner != TIE`. All three.
A wide margin from a position-biased judge is not clarity, and a tie is not a
training signal.

`generate` is a plain callable `(prompt, n) -> list[str]`, passed in. The HF
implementation is `hf_generator`, built by the caller and importing torch
lazily. That is what lets the entire harness — including the end-to-end test —
run on a laptop with nothing installed but pytest.

`capture` is an optional hook called with `(key, judging_prompt, completion)`
per generation. It is the seam the internals work needs: probing the judge's
own forward pass is how "is this a real value-conflict representation or just
position bias" eventually gets answered. Three lines, unimplemented on purpose
— building it before there is data to probe is the alpha-with-unused-files
failure mode.

### `src/judge/validate.py`
`report(pairs, judgments, labels, gates) -> dict`. A pure function over lists
of dicts, so it is trivially testable and has no idea where its inputs came
from.

Six numbers, cheapest-and-most-fatal first:

| metric | mechanism | gate |
|---|---|---|
| `swap_invariance` | fraction of pairs whose two orders agree | ≥ 0.90 |
| `self_consistency` | mean of `max(tally)/sum(tally)` within each order | ≥ 0.70 |
| `parse_rate` | fraction of judgments with `ok` | reported |
| `step_compliance` | fraction whose `steps_present` contains 4 | reported |
| `length_bias` | \|Pearson r\| between sign(len_a−len_b) and sign(winner) | ≤ 0.30 |
| `clear_agreement` | agreement with human labels, on pairs called clear | ≥ 0.85 |

`clear_agreement` is the load-bearing one: the whole loop assumes *clear ⇒ safe
to train on*. If clarity does not predict correctness, the triage signal is
invalid however tidy the margins look.

`margin_calibration` buckets agreement by margin and is reported, not gated. A
flat curve is the finding that would send triage to an activation probe instead
of a sampled margin.

If **no** human labels matched, `clear_agreement` is `None`, a `warning` field
is set, and it is excluded from the gates — so a green light with zero ground
truth is impossible to mistake for a pass.

### `src/cli.py`
Argument parsing and stage orchestration only; no judging logic lives here.
`_load_all()` loads model + judge configs, then follows `judge.rubric` and
`judge.protocol` as paths to load those too — one place where config
indirection is resolved.

`cmd_pairs` generates **two independent samples of the same model on the same
prompt**. Deliberately not "one blunt, one tactful": prompting for the contrast
would plant the tension the experiment is supposed to discover, and these are
also exactly the pairs a preference-training stage would later consume.

`cmd_judge` hard-refuses without `pairs.complete`. `cmd_validate` hard-refuses
without `judge.complete`, and on failure **deletes** any stale
`validate.complete` before exiting 1.

`--stub` swaps in the canned generator. `--fresh` clears a stage's wreckage and
redoes it; without it, a completed stage prints "already complete" and returns.

### `src/judge/judge.py:stub_generator`
A `generate(prompt, n)` that reads back its own rendered prompt to find which
answer is currently under label A, then acts out a planted pathology keyed to
the pair id:

| pair | behaviour | what it proves |
|---|---|---|
| `d05` | always picks label A | swap detection catches position bias |
| `d07` | prose, no JSON | unparseable is counted, raw text kept, no crash |
| `d12` | verdict without step 4 | step compliance notices |
| `d03` | alternating verdicts | split votes land in boundary |
| others | picks answer 1 whichever label it holds | clear pairs still come out clear |

A stub that only returned clean output would let a broken harness pass — which
is exactly how a previous harness was "verified" against a fixture written from
its author's own assumptions, and shipped a metric that was identically zero
for a whole run.

It sits in `src/`, beside `hf_generator`, rather than in `tests/`. Two reasons:
`--stub` is a product feature (the way to exercise a config or prompt change on
a laptop), and `src` importing from the test package would invert the
dependency — tests may depend on `src`, never the reverse. `cli.py` previously
did `from tests.fixtures.stub import ...`, which is the smell that prompted the
move.

### Boundary rule: what belongs where

- `src/judge/` — anything that decides a number. Aggregation, parsing,
  `report` (needs labels), `preflight` (needs nothing), label file I/O.
- `src/cli.py` — argument parsing, orchestration, and talking to the human.
  No thresholds, no arithmetic. `cmd_pilot` sets `--limit` and calls the real
  `cmd_pairs` / `cmd_judge`; it does not contain a second copy of the pipeline,
  because two copies drift and the measured estimate would stop describing the
  thing it is estimating.
- `tests/` — assertions only. Nothing in `src` may import from here.

`pilot` lives in the CLI rather than in `tests/` on purpose: it costs money,
needs a GPU, writes a run directory, and asserts nothing. It is a measurement,
not a test. Its *thresholds*, however, are validation logic, so they live in
`judge/validate.py:preflight` next to `report` and are unit-tested there.

### `tests/`
`unit/` is pure logic — parsing, config resolution, aggregation arithmetic,
validation gates, dotenv precedence, adapter-path resolution. `smoke/` runs the
real CLI end to end against the stub and asserts each planted pathology is
caught, that the gate blocks, and that a completed stage is not silently
redone. 57 tests, ~0.15 s, no torch.

---

## 3. Record shapes

`runs/<run>/pairs.jsonl` — one per prompt:
```json
{"pair_id": "d01", "prompt": "...", "tension": "truth x tact",
 "is_control": false, "answer_a": "...", "answer_b": "...",
 "len_a": 812, "len_b": 1104}
```

`runs/<run>/judgments.jsonl` — one per generation (2 orders × k):
```json
{"kind": "judgment", "pair_id": "d01", "order_first": "a", "sample": 0,
 "rubric_version": "v0_seed", "protocol_version": "v1",
 "ok": true, "winner": "A", "confidence": 0.8,
 "deciding_criterion": "completeness", "tension": null,
 "steps_present": [1,2,3,4,5], "error": null, "raw": "<full completion>"}
```
`raw` is always kept, including — especially — when `ok` is false.

`runs/<run>/results.jsonl` — one per pair:
```json
{"kind": "pair", "pair_id": "d01", "winner": "a", "margin": 1.0,
 "swap_consistent": true, "clear": true, "n_valid": 6, "n_unparseable": 0,
 "votes": {"a": 6, "b": 0, "TIE": 0},
 "per_order": {"a": {"a":3,"b":0,"TIE":0}, "b": {"a":3,"b":0,"TIE":0}},
 "confidence_mean": 0.8, "deciding_criteria": ["completeness"],
 "tensions": [], "steps_missing": 0, "len_a": 812, "len_b": 1104,
 "tension": "truth x tact", "is_control": false}
```

Note `votes` (continuous) sits next to `clear` (thresholded), and
`n_unparseable` next to `n_valid`. Deliberate: the predecessor project shipped
a metric whose column was an exact copy of another column, and a "no data" case
that masqueraded as a strong effect. Always log the continuous quantity beside
the thresholded one and a counter for the empty case.

---

## 4. The two constraints that are actually enforced by tests

**No injected system prompt.** `tokenizer.apply_chat_template` on a bare user
turn makes Qwen insert `You are Qwen, created by Alibaba Cloud. You are a
helpful assistant.` In the predecessor project that sentence sat in every RL
prompt, every rollout turn and every battery answer for weeks, recorded
nowhere, and is the most likely cause of a whole battery being pinned to "as an
AI I don't have preferences". Here it would be worse: a judge scoring
*helpfulness* with that sentence in context is not impartial.

Mechanism: `chat.render_chat(tokenizer, messages, system=None)` renders with a
sentinel system turn (`␟__NO_SYSTEM__␟`), locates the turn containing it by
walking left and right to the nearest special tokens, and cuts that span out.
It needs no knowledge of the template's syntax beyond "turns are delimited by
special tokens", so it works for ChatML, Llama-3 and Mistral alike, and warns
loudly rather than silently falling back if it cannot find the span.
`chat_provenance` records what the template *would* have injected into the run
manifest. `configs/model.json` and the protocol both set `system: null`, and
`tests/unit/test_config_and_chat.py` asserts all of it.

**Permissive parsing, nothing discarded.** A tool-call parser in the
predecessor repo required the literal fence ```` ```tool ````; the model had
just been fine-tuned on 1,200 completions ending in ```` ```python ````, so it
labelled its fences. 150 of 150 rollouts unparseable — *and the unparseable
completions were discarded*, so the sweep produced zero evidence of the cause.
Both halves of that failure are fixed by `parse.py`, and
`tests/unit/test_parse.py` has a test for each half.

---

## 5. Hub wiring

One module, `src/common/hub.py` (the predecessor split this across `hub.py` and
a `sync.py` CLI; merged, with `cli.py sync` replacing the second file).

**Destinations** — from `configs/hub.json`, and nowhere else in the codebase:

- logs → `tewoto/Remote_logging_RL` (dataset), at `runs/<experiment>/`
- adapters → `tewoto/LoRA_Adapters` (model), at `<experiment>/`

Both are **mirrors**, not exports: what lands on the Hub is byte-identical to
the local run directory, so `pull-run` followed by `peek` behaves exactly like
the machine that produced it. A reshaped export makes the analysis path and the
production path diverge, and the divergence surfaces weeks later as a number
disagreeing with itself.

**Auth.** `token()` reads `HF_TOKEN`; if unset, `load_env()` walks **up** from
the repo root (up to 4 levels) for a `.env` and loads it with
`os.environ.setdefault`. Two deliberate properties:

1. Walking up means the key lives in `AI Experiments/.env`, one directory above
   the repo — outside the git tree entirely, so no `.gitignore` rule stands
   between the token and a commit. (`.env` is gitignored anyway, belt and
   braces.)
2. `setdefault`, not assignment: an `HF_TOKEN` already in the real environment
   always beats the file. On a rented box the launch script injects the token,
   and a stale `.env` copied along with the repo must not silently override it.

`parse_dotenv` handles `export `, `#` comments, blank lines and quotes, and is
deliberately **not** a shell parser — `K=$(rm -rf /)` stays an inert string.
There is a test asserting that.

**Adapter references.** `configs/model.json:adapter_path` accepts either form,
resolved by `adapter_ref(spec) -> (repo_or_path, subfolder)`:

```
"hf:exp_v1"              -> ("tewoto/LoRA_Adapters", "exp_v1")
"hf:someone/other/sub"   -> ("someone/other", "sub")
"checkpoints/exp_v1"     -> ("checkpoints/exp_v1", None)
```

`model.load_with_adapter` passes `subfolder=` and `token=` straight to
`PeftModel.from_pretrained`, so a remote adapter needs no manual download and
one config key covers both cases. Without the `hf:` prefix every caller would
have to know which kind of path it was holding.

**Commands.**

```bash
python -m src.cli sync whoami       --run _   # verify the token before spending GPU time
python -m src.cli sync push-run     --run r0
python -m src.cli sync pull-run     --run r0
python -m src.cli sync push-adapter --run exp_v1 --path checkpoints/exp_v1
python -m src.cli sync pull-adapter --run exp_v1
```

Plus `--push` on any stage, which mirrors the run directory **after** the stage
succeeds. On the success path only: a run that died mid-stage must not appear
on the Hub looking complete.

`huggingface_hub` is imported inside the functions that use it, so the judge
slice, the tests and the stub path all still run with nothing installed but
pytest.

---

## 6. What is deliberately absent

No survey stage, no triage stage, no clustering, no escalation, no training, no
`src/internals/`. The judge is the measurement instrument and everything
downstream is a function of it, so it is built and audited first. A stage that
is not being run this month should not exist yet.

The two seams that exist because retrofitting them later would be a rewrite,
not an addition: `judge.Capture` (activation hook, 3 lines) and
`config.load`'s `extends` (experiment arms as diffs, not copies).
