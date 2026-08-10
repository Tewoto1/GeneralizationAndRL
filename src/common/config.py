"""
Config loading. JSON in `configs/` and `prompts/` is the design; Python is its
interpreter. This module is the only thing that reads those files off disk.

Two features beyond `json.load`:

  `extends`  — a config may name a sibling it inherits from, merged deeply, so
               experiment arms are diffs against a base instead of copies.
  `_`-keys   — keys starting with `_` are documentation for humans and are kept
               in the loaded dict on purpose. They are the design record; a
               loader that strips them makes the file look like it holds less
               than it does. Callers just never look them up.

Also holds `substitute`, the one templating rule used everywhere: `{{name}}`.
No Jinja, no f-string eval — a prompt file must never be able to execute code.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def _deep_merge(base: dict, over: dict) -> dict:
    """`over` wins, except that two dicts at the same key merge instead of replace.

    Lists replace wholesale: a config that overrides `criteria` means *these
    criteria*, not "these appended to the parent's".
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(path: str | Path, _seen: tuple[str, ...] = ()) -> dict:
    """Load a JSON config, resolving `extends` relative to the same directory.

    `extends` may be a string or a list (applied left to right, later wins).
    Cycles raise rather than recursing forever.
    """
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    key = str(p.resolve())
    if key in _seen:
        raise ValueError(f"config `extends` cycle: {' -> '.join(_seen + (key,))}")
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")

    cfg = json.loads(p.read_text())
    parents = cfg.pop("extends", None)
    if not parents:
        return cfg
    if isinstance(parents, str):
        parents = [parents]

    merged: dict = {}
    for parent in parents:
        merged = _deep_merge(merged, load(p.parent / parent, _seen + (key,)))
    return _deep_merge(merged, cfg)


def env_override(cfg: dict, mapping: dict[str, str]) -> dict:
    """Apply `{ENV_VAR: "dotted.key"}` overrides in place, returning cfg.

    Exists so a rented box can be pointed at a different model without editing a
    tracked file (and without the edit silently ending up in a run manifest as
    though it were the design).
    """
    for var, dotted in mapping.items():
        val = os.environ.get(var)
        if val is None:
            continue
        *parents, leaf = dotted.split(".")
        node = cfg
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = val
    return cfg


def substitute(text: str, values: dict[str, Any]) -> str:
    """Replace `{{name}}` with `values['name']`.

    Unknown placeholders raise. A prompt that silently ships the literal string
    `{{answer_b}}` to a model is the kind of bug that produces a whole run of
    plausible-looking garbage.
    """
    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in values:
            raise KeyError(f"no value for placeholder {{{{{name}}}}}")
        return str(values[name])

    return _PLACEHOLDER.sub(repl, text)


def render(template: str | list[str], values: dict[str, Any]) -> str:
    """Render a prompt template. Lists are joined with newlines.

    Templates live as lists of lines in JSON because JSON has no multiline
    string and a 40-line prompt on one line is unreviewable in a diff.
    """
    text = "\n".join(template) if isinstance(template, list) else template
    return substitute(text, values)
