"""
Shared plumbing for the stage modules: config loading and generator building.

Lives here rather than in `__init__` so the stage modules can import it without
importing the parser, and the parser can import the stages without a cycle.
"""
from __future__ import annotations

from ..common import config as cfg_mod
from ..judge import judge as J

DEFAULTS = {"model": "configs/model.json", "judge": "configs/judge.json"}

# One place. Repeating this string in each subparser is two sources of truth for
# one default, which is a defect even while both copies happen to agree.
DEFAULT_DOMAIN = "honesty_tact"


def load_all(domain: str | None = None) -> dict:
    """Model + judge config, with the rubric and protocol they point at resolved."""
    out = {k: cfg_mod.load(v) for k, v in DEFAULTS.items()}
    cfg_mod.env_override(out["model"], {"MODEL": "name", "ADAPTER": "adapter_path"})
    # LABELS points the judge at a different ground-truth file. Needed because
    # the label set is not a property of the code: a held-out split for
    # measuring a tuned judge must never be the split it was tuned on, and
    # swapping it by env var keeps both out of the tracked config.
    cfg_mod.env_override(out["judge"], {"LABELS": "validate.labels"})
    out["rubric"] = cfg_mod.load(out["judge"]["rubric"])
    out["protocol"] = cfg_mod.load(out["judge"]["protocol"])
    if domain:
        out["domain"] = cfg_mod.load(f"configs/domains/{domain}.json")
    return out


def generator(cfg: dict, stub: bool, role: str = "answer"):
    """`role` picks the generation profile — see judge.hf_generator."""
    if stub:
        return J.stub_generator(), {"model": "STUB", "role": role,
                                    "chat": {"system_policy": "none"}}
    return J.hf_generator(cfg["model"], role=role)


def judgments_per_pair(cfg: dict) -> int:
    """How many forward passes one pair costs, read from config rather than
    written into a print statement where it would drift."""
    jc = cfg["judge"]
    return jc.get("k_samples", 3) * (2 if jc.get("swap_orders", True) else 1)


def pick_prompts(domain: dict, limit: int | None):
    """Prompts to run, sampled RANDOMLY when limited.

    Taking the head of the list made the pilot's cost estimate hostage to
    prompt order: the first pilot drew d01/d02, which happen to produce the
    longest answers in the set (~3300 chars against a ~2100 mean), and the
    extrapolation came out 2.4x too high. Seeded, so two pilots stay comparable.
    """
    prompts = domain["prompts"]
    if not limit or limit >= len(prompts):
        return prompts
    import random
    return random.Random(0).sample(prompts, limit)
