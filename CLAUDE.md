# Working agreements for this repo

**What this project is now.** Two possible results, either of which is the
point:

1. **Trace generalization in the internals.** When a model forms a principle,
   does it become a readable direction in its activations — where, when, and
   is it causal.
2. **A self-play loop that cannot fool itself.** The model generates variants
   of its own answers, judges them, finds the edge cases where its criteria
   break, and keeps improving with **minimal human input**. Human labels are
   the bootstrap and the audit, not the fuel.

(2) is why every gate here exists. A self-improving loop with no external check
amplifies whatever its judge is biased toward, so the design question is not
"does it improve" but "can it tell when it is improving". Both presentation
orders, the clear/boundary split, the human-label gate, the controls that must
come out clear — all of it is there so the loop cannot lie to itself about its
own progress. Do not weaken a check to make a number look better.

(1) is the instrument for (2): a principle that shows up as a direction can be
watched rather than only scored. See `README.md`.

The reward-hacking / inference-time-misalignment work this repo inherited its
infrastructure from is **not** the current focus. It survives only in the
predecessor repo's history and in the failure log at the bottom of this file.

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

"Added a validation gate" is not sufficient; "`src/cli/score.py:cmd_validate` calls
`judge.validate.report(results, judgments, labels, gates)`, writes
`runs/<run>/validation.json`, and writes `validate.complete` only when
`rep['passed']`; `cmd_spread` and every later stage read it, and `judge` refuses to start without a completed `pairs` stage" is.

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
Layout is `src/cli/`, `src/common/`, `src/judge/`, `src/sample.py`, `configs/`,
`prompts/`, `tests/`, `docs/`. No file over ~300 lines; if one grows past that,
it is holding two concerns.
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

## Working on files a human is holding

`docs/human_label/labels.json` is written by a person, live, in another
terminal. Never read-modify-write it, and never write it at all while a
labelling session could be open. Nine labels were destroyed that way: the
snapshot was taken before they landed and the write-back erased them.

The rule generalises to any file the person may be editing right now: re-read
immediately before writing, and prefer appending to a `.jsonl` over rewriting
a `.json`. `judge.validate.add_label` appends for this reason.

Claude's own analysis output goes in its own file, never merged into a
human-authored one without being asked.

## Judge findings so far (r0, Qwen2.5-7B, 25 prompts)

- Two independent samples of the same model at temperature 0.7 are not a
  preference pair: 41 / 42 / 67-tie across 150 judgments, 47 of 50 answers
  markdown listicles, median within-pair length difference 13%. Under GRPO
  that group has no reward variance and therefore no gradient.
- Judge agreed with Yuzheng on 5 of 8 pairs both decided. Its one CLEAR pair
  was one he judged the other way, so `clear_agreement` was 0/1.
- `length_bias` measured 0.056, so the failure is not a preference for longer
  answers. It is coverage over commitment, caused by protocol v1 asking only
  for evidence FOR a criterion. Fixed in protocol v2.
- His tags across 13 labels: tact 7, actionable 4, truth 2, completeness 2,
  time 0. Shortness is tagged `tact`. `time` and `tact` are one axis for him.

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
- `device_map="auto"` does not fail when a model nearly fits. It strands
  modules on CPU and copies them back every forward, which reads as "the GPU is
  slow" and eventually OOMs mid-generation trying to land one. Pin the device;
  `model.check_no_offload` now raises at load time instead.
- A config comment asserted "VRAM is not the constraint" and a 24 GB card then
  OOMed on it. A `_note` that states a fact is a claim, and a wrong one costs
  the next reader the same hour it cost this time.
