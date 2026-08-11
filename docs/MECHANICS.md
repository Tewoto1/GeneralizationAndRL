# Mechanics

What each file holds, what reads it, what moves between them, and why each
choice was made rather than the obvious alternative. `README.md` says what the
project is; this is the level below that.

---

## 1. Control flow

One command, `python -m src.cli <stage> --run <name>`. `run.sh` adds shorter
typing and a fixed stage order, nothing else.

```
./run.sh pool r1
  |
  └─ constitute   reads  docs/human_label/labels.json + runs/<labels-from>/pairs.jsonl
  |               reads  prompts/sample/constitution_writer.json
  |               writes runs/r1/constitutions/c*.json     one file each, with raw text
  |               writes runs/r1/constitutions.jsonl       same minus raw
  |               writes runs/r1/constitute.complete
  |
  └─ sample       reads  configs/variants.json, configs/domains/<domain>.json
  |               reads  runs/r1/constitutions/*.json
  |               writes runs/r1/answers.jsonl    one per (prompt, variant)
  |               writes runs/r1/reviews.jsonl    raw self-review text, for reading
  |               writes runs/r1/sample.complete
  |
  └─ pair         reads  runs/r1/answers.jsonl              no GPU
  |               writes runs/r1/pairs.jsonl                 same schema as ever
  |               writes runs/r1/pairs.complete
  |
  └─ judge        refuses without pairs.complete
  |               reads  configs/judge.json -> prompts/judge/{judge_protocol,rubric_v1}
  |               per pair: 2 orders x k samples
  |               writes runs/r1/judgments.jsonl   one per generation
  |               writes runs/r1/results.jsonl     one per pair
  |               writes runs/r1/judge.complete
  |
  └─ spread       reads  runs/r1/results.jsonl               no GPU
                  writes runs/r1/spread.json

validate   refuses without judge.complete; writes validation.json always and
           validate.complete only when every gate passed; exit 1 otherwise
label      appends to docs/human_label/labels.jsonl
peek       read-only, safe mid-run
```

Every stage reads a run directory and writes a run directory. Nothing
communicates in memory.

**`.complete` is not decoration.** This project previously resumed on "does the
directory exist" (true for a stage that died one second in) and then on "is the
jsonl non-empty" (true for a stage that died at 297 of 360). Both produced
silently truncated runs.

**The marker says a stage FINISHED; it does not say what it got through.**
That is a separate mechanism: `Run.done(stream, *keys)` returns the keys already
written, and `sample` skips `(prompt_id, variant)` pairs while `judge` skips
`pair_id`s. Without it, a stage that died at item 100 of 150 and was re-run
would append 100 duplicates on top of the originals.

---

## 2. File by file

### `configs/model.json`
Model, quantisation, system-prompt policy, and per-role generation profiles.

Read by `src/cli/base.py:load_all`, which then applies `env_override` for
`MODEL`, `ADAPTER` and `LABELS`. Editing a tracked file on a rented box would
show up in a run manifest looking like a considered design decision; an env var
does not.

`"system": null` is load-bearing and is checked by a test. See §4.

`gen` has a `judge` sub-block merged over the base by `generators.hf_generator`.
The two roles want opposite things: an answer capped short is not concise, it
is broken, and the judge would then score `assumes_competence` against an
artefact the harness created. Caps are ceilings, not budgets — generation stops
at EOS, so a generous cap costs wall-clock only on runs where it binds. 900 was
an arbitrary number that truncated most judgments mid-STEP-3, and a truncated
completion has no verdict block, so it was recorded as `unparseable`, pointing
at the parser instead of the cap.

### `configs/judge.json`
Which rubric and protocol to use, `k_samples`, `swap_orders`, the clear
threshold, and the validation gates.

`clear_min: 0.7` is a guess. `validate` reports margin calibration — agreement
bucketed by margin — so it can be replaced with something measured.

### `configs/variants.json`
How the answer pool is built, and the reason it exists at all.

r0 sampled the same model twice at temperature 0.7. The judge split 41/42/67
across 150 judgments, which is a coin flip. Temperature does not fix that: it
raises entropy over tokens, not over strategies, and RLHF collapsed the output
distribution onto one mode. Reaching the others takes conditioning.

- `draft` ×2 — no conditioning. The control, and free, because self-review
  needs a draft anyway. Two of them reproduce r0's setup exactly, which is the
  baseline every other variant is measured against.
- `pre_*` — seeds the assistant turn and lets the model continue. Weaker than
  an instruction but not neutral, so every prefill answer is tagged
  `conditioning: prefill` and must never be read as unconditioned.
- `self_review` — expanded at runtime to one variant per constitution, since
  the constitutions do not exist until a run has produced them.

`pairing.strategy: anchor_on_draft` — every pair is `draft_0` vs one other
variant. N variants cost N pairs rather than N-choose-2, and every variant is
measured against the *same* control, so the numbers are comparable.

### `configs/domains/honesty_tact.json`
25 prompts where the honest, the kind and the time-respecting answer are not
the same answer. Every prompt carries `trap` (the failure mode it baits),
`tension` (which criteria pull apart), and `clarify_expected`.

`clarify_expected` is a contrast, not a blanket: three prompts where one
unstated fact decides the answer and asking is the best move, against prompts
that look under-specified but are not. A judge that rewards clarification
everywhere is as broken as one that never does, and only having both kinds can
tell them apart.

`_control` prompts should win on every criterion for one answer, so they must
come out clear. A control in the boundary bucket means the judge is noisy, not
that the case is hard — `cmd_peek` names the offenders.

### `prompts/judge/judge_protocol.json` (v2)
**The experiment.** `template` is a list of lines (JSON has no multiline string
and a 40-line prompt on one line is unreviewable in a diff), joined and
rendered by `config.render`.

Six steps: locate → commitments → consequences → criteria → self-check →
verdict.

- **LOCATE** is new in v2. Quote the sentence that answers the question, say
  how many words precede it, say whether a heading points to it. It exists
  because r0's d06 turned on exactly this: both answers gave a number, but one
  buried it in section 5 of 5 and the other put the question in a heading. A
  number the reader cannot find has not been given.
- COMMITMENTS forbids evaluation, so the judge cannot write its conclusion
  first and reverse-engineer support.
- CONSEQUENCES asks what follows *for this person*, and names any fact the
  consequence depends on that the request did not supply.
- **CRITERIA** requires evidence FOR **and AGAINST** each criterion. v1 asked
  only for supporting text, so adding a section could raise a score and never
  lower one, and the answer covering more ground won on every criterion at
  once. Evidence against is defined as text deletable without changing what the
  person does next.
- SELF-CHECK names five specific failure modes to hunt for, including (e), the
  coverage-over-commitment one this version was written to fix.
- VERDICT permits TIE, and carries one narrow tie-break: shorter wins only when
  the answers lead to the same action, for the same reasons, given the same
  facts. Without permission to tie, a judge invents a winner and the margin
  becomes noise.

`{{criteria}}` is a hole filled from the rubric file, so the constitution and
the protocol version independently and a change in results is attributable to
exactly one of them.

`"system": null` here too.

### `prompts/judge/rubric_v1.json`
The constitution, derived from 13 hand labels rather than written from scratch.
Each criterion has a `_derivation` field naming the labels it came from.

`v0_seed` (`rubric_seed.json`) is kept unedited so the hand-written and
label-derived constitutions can run as two arms.

The tags across those 13 labels were tact 7, actionable 4, truth 2,
completeness 2, **time 0** — and shortness was tagged `tact`. v0_seed's split
between `time` and `tact` was never used, so v1 folds them into
`assumes_competence`.

`_known_tensions` is a check: boundary cases sitting on none of them mean a
criterion is missing.

### `prompts/sample/constitution_writer.json`
Labelled examples in, criteria out. The model is **not** shown the seed rubric,
or it would paraphrase it back and the exercise would measure copying. It is
asked for the principle behind each preference before writing criteria, for the
same reason the judge is asked for commitments before evaluating. It must name
a tension it cannot resolve, because a constitution whose criteria never
conflict produces no boundary cases.

Every constitution in a run is written from the **identical** examples and
differs only by sampling seed: the question is how far the model diverges from
a shared start, so the seed is the independent variable and is in the record.

### `prompts/sample/self_review.json`
Draft → review → revision, in **one** call, so the revision is conditioned on
the review just written rather than on one re-read from context. The model must
name one thing it will not change, or every review becomes a rewrite and
revision-helps cannot be distinguished from revision-churns.

The revision is emitted between `<<<REVISED>>>` markers. On failure the
fallback is **the draft, never the review text** — review notes in the answer
pool would have the judge scoring the model's own case for its answer.

### `src/common/config.py`
The only module that reads JSON off disk. `load` resolves `extends` (dicts
merge, **lists replace**, cycles raise). `_`-prefixed keys survive loading —
they are the design document. `substitute` implements exactly one rule,
`{{name}}`, and **raises on an unknown placeholder**: a prompt file must never
be able to execute code, and one that silently ships the literal `{{answer_b}}`
produces a whole run of plausible garbage.

### `src/common/chat.py`
Carried from the predecessor repo unchanged except for `prefill`. See §4.

### `src/common/io.py`
`Run` — `open`, `write`, `read`, `count`, `done`, `is_complete`,
`mark_complete`, `clear`, `note`. Plus the terminal helpers `c`, `box`,
`Progress`.

Colour is decided once, at import, by `sys.stdout.isatty()`. `night` redirects
to a log file and escape codes in that file would make every later `grep` a
fight with `\x1b[32m`. `Progress` reports measured rate and ETA because a stage
printing nothing for forty minutes is indistinguishable from a hung one.

### `src/common/hub.py`
Run directories mirror to a dataset repo, adapters to a model repo. Both are
**mirrors, not exports**, so `pull` then `peek` behaves like the machine that
produced the run.

`token()` reads `HF_TOKEN`; if unset, `load_env()` walks **up** from the repo
root for a `.env` and applies it with `setdefault`. Walking up keeps the key
outside the git tree; `setdefault` means a token injected by a launch script
beats a stale `.env` copied along with the repo. `.env` is loaded at CLI
**startup**, not lazily — `huggingface_hub` reads `HF_TOKEN` from the real
environment when `from_pretrained` downloads weights, long before any sync code
runs, which is why a lazily-loaded token produced "sending unauthenticated
requests" while `whoami` worked.

### `src/judge/parse.py`
Turns one completion into a `Judgment`. Collects every candidate JSON blob —
any fence label or none — and tries them **last first**, because the protocol
says "finish with this block" and a judge quoting the format example
mid-reasoning must not have the example read as its verdict.

`steps_present` is found separately by regex, keyed **by name**: the protocol
gained a LOCATE step and anything asking "did it do step 4" would silently have
started reading the criteria step. `did_selfcheck` accepts both names and the
old integers so previous runs stay readable.

Never raises, never returns None-and-forgets. This file is defensive out of
proportion to its size because a rigid parser in the predecessor repo made 150
of 150 rollouts unparseable *and discarded them*, so the sweep produced zero
evidence of the cause.

### `src/judge/judge.py` / `generators.py`
Split: what a verdict means, versus where completions come from. Both backends
satisfy `(prompt, n, prefill=None) -> list[str]`, so the judging logic never
learns which one it has, and the harness runs on a laptop with no torch.

`judge_pair` runs both orders and maps label-space votes back to answer space.
`margin = |votes_a - votes_b| / n_valid` with **ties in the denominator** — a
pair repeatedly called a tie is genuinely undecided, and a definition excluding
ties would report a confident 1.0 off one stray vote.
`clear = margin >= clear_min AND swap_consistent AND winner != TIE`. All three.

`capture` is the activation hook the internals work needs — probing the judge's
own forward pass is how "real value-conflict representation or position bias"
eventually gets answered. Three lines, unimplemented until there is data.

### `src/judge/validate.py`
`preflight`, `spread`, `report`, `add_label`, `load_labels`. All pure functions
over lists of dicts except the two label helpers.

`preflight` checks truncation **first** and reports it as itself, because it is
the upstream cause of both other symptoms: a completion cut before the verdict
block has no JSON (reads as unparseable) and no self-check step (reads as
skipped).

`add_label` is **append-only** to a sibling `.jsonl`. It used to read the whole
JSON, append in memory and write back, which is a lost-update race the moment
anything else touches the file — and it was not hypothetical. See CLAUDE.md.

### `src/sample.py`
Constitution parsing (ids normalised, since r0's judgments returned
`Completeness`, `completeness` and the literal `"null"` as three criteria),
variant expansion, revision extraction, pair selection.

`select_pairs` drops pairs whose two answers are the same text. A self-review
that emitted no marker falls back to the draft, so the pair would be one answer
against itself: a guaranteed tie, four judgments bought for nothing, and a fake
tie dragging that variant's `decided` rate down as though the judge had failed
to rank two real alternatives.

### `src/cli/`
`base.py` config and generators · `pool.py` constitute, sample, pair ·
`score.py` judge, validate, spread, pilot · `review.py` label, peek, sync ·
`__init__.py` parser assembly only.

`cmd_pilot` calls the ordinary stages with `--limit`, so there is no second
copy of the pipeline to drift. `pick_prompts` samples **randomly** when
limited: taking the head of the list made the first pilot draw the two
longest-answer prompts and overestimate cost 2.4×.

### `tests/`
`unit/` pure logic; `smoke/` the real CLI end to end against the stub,
including two resume tests that truncate a stream mid-way and assert the re-run
neither duplicates nor loses work. 98 tests, no torch.

The stub does not fake success. It acts out, per pair id, the failure modes the
harness must survive: position bias, unparseable output, a skipped self-check,
split votes. A stub that only returned clean output would let a broken harness
pass — which is how a previous harness was "verified" against a fixture written
from its author's own assumptions and shipped a metric identically zero for a
whole run.

---

## 3. Record shapes

`answers.jsonl` — one per (prompt, variant):
```json
{"prompt_id": "d01", "prompt": "...", "tension": "...", "is_control": false,
 "variant": "review_c0", "conditioning": "self_review", "constitution": "c0",
 "revision_ok": true, "text": "..."}
```

`pairs.jsonl` — schema unchanged since r0, so `judge`/`label`/`validate` were
untouched by the pool feature:
```json
{"pair_id": "d01::review_c0", "prompt_id": "d01", "prompt": "...",
 "variant_a": "draft_0", "variant_b": "review_c0",
 "conditioning_b": "self_review", "constitution_b": "c0",
 "answer_a": "...", "answer_b": "...", "len_a": 812, "len_b": 1104}
```

`judgments.jsonl` — one per generation. `raw` is always kept, especially when
`ok` is false:
```json
{"kind": "judgment", "pair_id": "d01::review_c0", "order_first": "a",
 "sample": 0, "rubric_version": "v1_from_labels", "protocol_version": "v2",
 "truncated": false, "ok": true, "winner": "A", "confidence": 0.8,
 "deciding_criterion": "answers_first", "tension": null,
 "steps_present": ["locate","commitments","consequences","criteria",
                   "selfcheck","verdict"], "error": null, "raw": "..."}
```

`results.jsonl` — one per pair, carrying the pair's provenance so `spread` can
group by `variant_b`. Dropping those fields once collapsed every variant into a
single row labelled `?`.

Note `votes` (continuous) sits beside `clear` (thresholded), and
`n_unparseable` beside `n_valid`. The predecessor project shipped a metric whose
column was an exact copy of another, and a "no data" case that masqueraded as a
strong effect.

---

## 4. The two constraints enforced by tests

**No injected system prompt.** `apply_chat_template` on a bare user turn makes
Qwen insert `You are Qwen, created by Alibaba Cloud. You are a helpful
assistant.` In the predecessor project that sentence sat in every prompt of
every run for weeks, recorded nowhere. Here it would be worse: a judge scoring
*helpfulness* with it in context is not impartial.

`render_chat(..., system=None)` renders with a sentinel system turn, locates
the turn containing it by walking to the nearest special tokens, and cuts that
span. It needs no knowledge of the template beyond "turns are delimited by
special tokens", so it works for ChatML, Llama-3 and Mistral alike, and warns
rather than silently falling back. `chat_provenance` records what the template
*would* have injected. `prefill` appends after the generation prompt, so the
model continues from it — and since only new tokens are decoded, callers
prepend it themselves.

**Permissive parsing, nothing discarded.** See `parse.py` above.
`tests/unit/test_parse.py` has a test for each half of that failure.

---

## 5. What is deliberately absent

No triage stage, no clustering, no escalation, no training, no
`src/internals/`. The judge is the instrument; everything downstream is a
function of it, so it is built and audited first.

Two seams exist because retrofitting them would be a rewrite rather than an
addition: `judge.Capture` (the activation hook) and `config.load`'s `extends`
(experiment arms as diffs, not copies).
