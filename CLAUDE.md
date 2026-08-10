# Working agreements for this repo

## How to report edits (Yuzheng's standing preference)

When making any edit, **describe the exact setting and the method of implementation** —
not just a description of the intent, and not just the results.

Every edit report must state:

1. **Where** — the concrete file and the concrete symbol/key being changed.
2. **The setting** — what the data/control flow actually looks like at that point
   (what is loaded from where, what is hardcoded, what the caller passes in).
3. **The method** — the mechanism used, in enough detail that the change could be
   re-derived without reading the diff. Name the schema keys, the function
   signatures, the substitution syntax.
4. **Then** results/verification.

"Added a validation gate" is not sufficient; "`src/cli.py:cmd_validate` calls
`judge.validate.report(results, judgments, labels, gates)`, writes
`runs/<run>/validation.json`, and writes `validate.complete` only when
`rep['passed']`; `cmd_survey` will refuse to start without that marker" is.

Rationale: the user is not a heavy SWE and needs to be able to parse and audit
what changed. A description-only report hides wrong implementation choices.

## Design preference: data, not code

Design lives in JSON under `configs/` and `prompts/`. Code under `src/` is a
generic interpreter of those files. Adding a prompt domain, a rubric version, or
a judge threshold must never require editing Python. Arithmetic that genuinely
must be code is exposed as a named function referenced from JSON.

The `_`-prefixed keys in those JSON files are the design document. `_role`,
`_note`, `_design`, `_why` are written deliberately, for the next reader. A plan
that contradicts them is wrong, not creative.

## Hard constraints

- **No system prompt is ever injected.** `render_chat(..., system=None)` is the
  default and it must stay the default. Qwen's template adds "You are Qwen, a
  helpful assistant" when given none; that contaminated every run of the
  predecessor project for weeks. `tests/unit/test_config_and_chat.py` asserts
  this — do not delete those tests.
- **Every pair is judged in both presentation orders.** Order-flip disagreement
  is part of the triage signal, not an optional audit.
- **Nothing is silently discarded.** Unparseable model output is recorded with
  its raw text.
- **`validate` gates everything downstream.** No stage that consumes judgments
  may run against a judge version with no passing validation report.
- **Human labels are the only ground truth.** A validation pass with zero
  matched labels is a warning, not a pass.

## Read before writing

1. `README.md` — what exists and what deliberately does not.
2. The JSON under `configs/` and `prompts/` — including the `_` keys.
3. The module you are about to touch.
4. `docs/PLAN.md` for why the project is shaped this way.

**A spec file with no interpreter is a bug, and it is the finding.** If JSON
describes behaviour that no Python reads, say so in the first sentence. For every
config file, grep for something that loads it.

**Never assume a module exists because a similar project had one.** Verify with
`ls`/`grep` in *this* repo.

## File discipline

**Default: do not add new files.** Fold into the module that owns the concern.
Layout is `src/common/`, `src/judge/`, `configs/`, `prompts/`, `tests/`, `docs/`.
A new file needs an explicit reason, and in a borderline case, a question first.

**Code budget is a real constraint.** Before adding a block, ask what it
duplicates. Two sources of truth for one default is a defect even when both are
currently correct.

Prefer deleting a superseded path over keeping it beside the new one.

Working beta over unused alpha: a stage that is not being run this month should
not exist yet.

## Communication

Terse by default; caveman register is welcome and is the standing preference.
Lead with the decisive fact. No narration of tool calls, no restating the
question. Do not invent notation or jargon; if a new term is needed, define it
the first time it appears.

Ask a clarifying question only when the answer genuinely changes the work.

## Failure log — mistakes made in the predecessor repo, do not repeat

- A rigid tool-call parser required one literal fence label; the model had just
  been trained to emit a different one. Result: 150 of 150 rollouts unparseable —
  and the unparseable completions were *discarded*, so the sweep produced zero
  evidence of the cause. Parsers are permissive and always keep raw text.
- `hack_rate` was defined so that its column was an exact copy of another
  column. A metric that exactly tracks another metric is broken, not confirmed.
  Always log the continuous quantity next to any thresholded one, plus a counter
  for the "no data" case, so a broken join cannot masquerade as a strong effect.
- Resume guards checked directory existence (true for a stage that died one
  second in), then non-empty JSONL (true for a stage that died at 297 of 360).
  Only a `.complete` marker written on the success path proves completion.
- A harness was verified against a hand-written fixture in the shape the author
  had *assumed*. The real data defined its own shape and the metric was
  identically zero for a whole run. A fixture written from your own assumption
  tests nothing — pull one real record and run against that.
- A config key was renamed in one place and missed in another; cost was one
  overnight training run. After any rename, grep for the old name in all
  consuming code before committing.
- Prompts, environments and configs were created at the repo root instead of
  inside the package that owned them, twice, after being told where they go.
