"""
Config resolution and prompt rendering.

Two failure classes these guard against, both of which have cost this project
real time before: a renamed config key that nothing notices until a run is
underway, and a prompt template that silently ships an unsubstituted
placeholder to the model.

The chat tests do not load a model. They use a fake tokenizer that behaves like
ChatML, which is enough to prove the system-turn policy is enforced.
"""
import json

import pytest

from src.common import config as C


# --------------------------------------------------------------- config -------
def test_extends_merges_deeply(tmp_path):
    """An arm inherits its parent and overrides one nested key, without having
    to restate the rest. Copy-paste configs are how two sources of truth for one
    default get created."""
    (tmp_path / "base.json").write_text(json.dumps(
        {"margin": {"clear_min": 0.7, "other": 1}, "k": 3}))
    (tmp_path / "arm.json").write_text(json.dumps(
        {"extends": "base.json", "margin": {"clear_min": 0.9}}))
    cfg = C.load(tmp_path / "arm.json")
    assert cfg["margin"] == {"clear_min": 0.9, "other": 1}
    assert cfg["k"] == 3


def test_lists_replace_rather_than_append(tmp_path):
    """Overriding `criteria` means *these* criteria. Silent appending would let
    a deleted criterion keep scoring."""
    (tmp_path / "b.json").write_text(json.dumps({"criteria": [1, 2, 3]}))
    (tmp_path / "a.json").write_text(json.dumps({"extends": "b.json", "criteria": [9]}))
    assert C.load(tmp_path / "a.json")["criteria"] == [9]


def test_extends_cycle_raises(tmp_path):
    """A cycle must fail loudly, not hang."""
    (tmp_path / "a.json").write_text(json.dumps({"extends": "b.json"}))
    (tmp_path / "b.json").write_text(json.dumps({"extends": "a.json"}))
    with pytest.raises(ValueError, match="cycle"):
        C.load(tmp_path / "a.json")


def test_underscore_keys_survive_loading(tmp_path):
    """`_role` / `_note` are the design document and must reach the reader of
    the loaded dict, not be stripped as noise."""
    (tmp_path / "x.json").write_text(json.dumps({"_role": "why", "v": 1}))
    assert C.load(tmp_path / "x.json")["_role"] == "why"


def test_env_override(monkeypatch):
    """A rented box can be pointed at another model without editing a tracked
    file (which would otherwise land in a manifest looking like design)."""
    monkeypatch.setenv("MODEL", "some/other-model")
    cfg = C.env_override({"name": "Qwen/Qwen2.5-7B-Instruct"}, {"MODEL": "name"})
    assert cfg["name"] == "some/other-model"


def test_missing_placeholder_raises_not_silently_ships():
    """A prompt containing a literal '{{answer_b}}' would produce a whole run of
    plausible-looking garbage. Fail at render time instead."""
    with pytest.raises(KeyError, match="answer_b"):
        C.render(["A: {{answer_a}}", "B: {{answer_b}}"], {"answer_a": "x"})


def test_render_joins_list_templates():
    """Templates are lists of lines in JSON so diffs are reviewable."""
    assert C.render(["one {{x}}", "two"], {"x": "1"}) == "one 1\ntwo"


def test_real_configs_load_and_render():
    """The shipped configs actually parse, and the judge protocol renders with
    the real rubric. Catches a broken JSON edit before a GPU run does."""
    judge = C.load("configs/judge.json")
    rubric = C.load(judge["rubric"])
    protocol = C.load(judge["protocol"])
    domain = C.load("configs/domains/honesty_tact.json")

    from src.judge.judge import build_prompt
    p = build_prompt(protocol, rubric, domain["prompts"][0]["text"], "AAA", "BBB")
    assert "{{" not in p, "unsubstituted placeholder left in the judging prompt"
    assert "STEP 4" in p and "ANSWER A:" in p
    for c in rubric["criteria"]:
        assert c["id"] in p, f"criterion {c['id']} missing from rendered prompt"


def test_domain_prompts_are_well_formed():
    """Unique ids, every prompt annotated, and enough controls to detect a noisy
    judge. `trap` and `tension` are design annotations, never shown to the
    model — cmd_sample reads only `text` — so this checks the design record is
    complete rather than checking anything the model sees."""
    prompts = C.load("configs/domains/honesty_tact.json")["prompts"]
    ids = [p["id"] for p in prompts]
    assert len(ids) == len(set(ids)), "duplicate prompt id"

    controls = [p for p in prompts if p.get("_control")]
    assert len(controls) >= 2, "too few controls to detect a noisy judge"

    for p in prompts:
        assert p["text"].strip() and p.get("tension")
        if not p.get("_control"):
            assert p.get("trap"), f"{p['id']} has no trap annotation"


def test_flip_pair_is_present_and_actually_mirrored():
    """One prompt pair tells the same story from opposite sides. A calibrated
    judge should reach OPPOSITE verdicts on the two halves; the same verdict
    twice is evidence it is scoring warmth toward whoever is speaking rather
    than the conduct described. The pair is useless if one half goes missing,
    so find them by trap rather than by id and assert they are mirrored."""
    prompts = C.load("configs/domains/honesty_tact.json")["prompts"]
    a = next(p for p in prompts if p.get("trap") == "sycophancy_moral")
    b = next(p for p in prompts if p.get("trap") == "sycophancy_moral_flip")
    for k in ("cofounder", "oversensitive", "round"):
        assert k in a["text"] and k in b["text"], f"{k!r} missing from the flip pair"


def test_clarification_is_tested_in_both_directions():
    """The domain must contain prompts where asking a clarifying question is
    the right move AND prompts where it is a stall. A judge that rewards
    clarification everywhere is as broken as one that never rewards it, and
    only having both kinds can tell those apart."""
    prompts = C.load("configs/domains/honesty_tact.json")["prompts"]
    ask = [p for p in prompts if p.get("clarify_expected") is True]
    dont = [p for p in prompts if p.get("clarify_expected") is False]
    assert len(ask) >= 2, "no prompts where clarifying is the right move"
    assert len(dont) >= 2, "no prompts where clarifying would be a stall"
    assert all("clarify_expected" in p for p in prompts), "unannotated prompt"


def test_protocol_declares_no_system_turn():
    """The judge protocol must not carry an assistant persona. An injected
    'You are a helpful assistant' puts a thumb on the scale of every
    helpfulness judgment, and it contaminated every run of the predecessor
    project for weeks before anyone noticed."""
    assert C.load("configs/judge.json")["protocol"]
    assert C.load("prompts/judge/judge_protocol.json")["system"] is None
    assert C.load("configs/model.json")["system"] is None


# ----------------------------------------------------------------- chat -------
class FakeTok:
    """Minimal ChatML-ish tokenizer: enough to exercise the system-turn logic."""
    all_special_tokens = ["<|im_start|>", "<|im_end|>"]
    additional_special_tokens: list = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        msgs = list(messages)
        if not msgs or msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": "You are Qwen, a helpful assistant."}] + msgs
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def test_no_system_turn_is_actually_removed():
    """system=None must yield a prompt with no system turn at all — not the
    template's default persona."""
    from src.common.chat import render_chat
    out = render_chat(FakeTok(), [{"role": "user", "content": "hi"}], system=None)
    assert "system" not in out and "helpful assistant" not in out
    assert "hi" in out


def test_explicit_system_is_used_verbatim():
    from src.common.chat import render_chat
    out = render_chat(FakeTok(), [{"role": "user", "content": "hi"}], system="RULES")
    assert "RULES" in out and "helpful assistant" not in out


def test_template_default_is_reported_for_the_manifest():
    """What the template *would* inject is recorded, so a later reader can tell
    which regime produced a set of judgments."""
    from src.common.chat import chat_provenance
    prov = chat_provenance(FakeTok(), None)
    assert prov["system_policy"] == "none"
    assert "helpful assistant" in prov["template_would_inject"]


def test_prefill_seeds_the_assistant_turn():
    """The prefill must land AFTER the generation prompt, so the model
    continues from it rather than being shown it as user text."""
    from src.common.chat import render_chat
    out = render_chat(FakeTok(), [{"role": "user", "content": "hi"}],
                      system=None, prefill="The short answer is")
    assert out.endswith("The short answer is")
    assert out.index("assistant") < out.index("The short answer is")


def test_prefill_is_optional_and_changes_nothing_when_absent():
    from src.common.chat import render_chat
    msgs = [{"role": "user", "content": "hi"}]
    assert render_chat(FakeTok(), msgs) == render_chat(FakeTok(), msgs, prefill=None)
