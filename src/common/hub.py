"""
Hugging Face Hub sync: run logs to a dataset repo, LoRA adapters to a model
repo. Repo ids come from `configs/hub.json` and appear nowhere else.

Two shapes of thing move:

  runs/<experiment>/      -> dataset repo, under runs/<experiment>/
  a trained adapter dir   -> model repo,   under <experiment>/

Both are *mirrors*, not exports. What lands on the Hub is byte-identical to
what sat on disk, so `pull` followed by `peek` behaves exactly like the machine
that produced the run. A reshaped export would mean the analysis path and the
production path diverge, and the divergence would only show up when a number
disagreed with itself weeks later.

Auth: `HF_TOKEN`, found by walking UP from the repo root for a `.env` file.
That is deliberate — the key lives one directory above the repo, so no rule and
no .gitignore entry stands between the token and a commit. An HF_TOKEN already
in the environment always beats the file, which is how a rented box injects a
token with no file at all.

`huggingface_hub` is imported lazily inside the functions that need it. The
judge slice, the tests and the stub path must all run on a laptop with nothing
installed but pytest.
"""
from __future__ import annotations

import os
from pathlib import Path

from .config import ROOT, load

# Files never worth uploading, whatever directory they are in.
_SKIP = ("*.pyc", "__pycache__/*", ".DS_Store")


# ------------------------------------------------------------------- auth ----
def find_dotenv(start: Path | None = None, max_up: int = 4) -> Path | None:
    """Nearest `.env` at or above `start` (default: repo root). None if absent."""
    here = (start or ROOT).resolve()
    for d in [here, *here.parents][:max_up + 1]:
        p = d / ".env"
        if p.is_file():
            return p
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """Minimal .env parser: KEY=VALUE per line.

    Handles `export ` prefixes, `#` comments, blank lines, and surrounding
    single or double quotes. Deliberately not a general shell parser — a config
    file that can execute code is a config file that can surprise you.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k:
            out[k] = v
    return out


def load_env(start: Path | None = None) -> Path | None:
    """Load the nearest `.env` into os.environ WITHOUT overwriting what is set.

    Precedence — real environment beats file — matters on a rented box, where
    the token is injected by the launch script and a stale `.env` copied along
    with the repo would otherwise silently win.
    """
    p = find_dotenv(start)
    if p is None:
        return None
    for k, v in parse_dotenv(p.read_text()).items():
        os.environ.setdefault(k, v)
    return p


def token(required: bool = True) -> str | None:
    """The HF token, loading `.env` on first need."""
    cfg = load("configs/hub.json")
    var = cfg["auth"]["token_env"]
    if not os.environ.get(var):
        load_env()
    tok = os.environ.get(var)
    if not tok and required:
        where = find_dotenv()
        raise RuntimeError(
            f"{var} not set and no value found in {where or 'any .env at or above ' + str(ROOT)}. "
            f"Either export {var}=... or put it in a .env one directory above the repo.")
    return tok


# ------------------------------------------------------------------ paths ----
def adapter_ref(spec: str) -> tuple[str, str | None]:
    """Resolve `configs/model.json:adapter_path` to (repo_or_path, subfolder).

        "hf:exp_v1"      -> ("tewoto/LoRA_Adapters", "exp_v1")   # generic repo
        "hf:org/r/sub"   -> ("org/r", "sub")                     # explicit repo
        "Checkpoints/x"  -> ("Checkpoints/x", None)              # local dir

    The `hf:` prefix exists so one config key covers both local and remote
    adapters. Without it every caller would need to know which kind it has.
    """
    if not spec.startswith("hf:"):
        return spec, None
    rest = spec[3:].strip("/")
    parts = rest.split("/")
    if len(parts) == 1:                       # bare experiment name
        return load("configs/hub.json")["adapters"]["repo"], parts[0]
    if len(parts) == 2:                       # already org/repo, no subfolder
        return rest, None
    return "/".join(parts[:2]), "/".join(parts[2:])


# ------------------------------------------------------------------ client ---
def _api():
    from huggingface_hub import HfApi
    return HfApi(token=token())


def _ensure(repo_id: str, repo_type: str, private: bool) -> None:
    from huggingface_hub.utils import HfHubHTTPError
    try:
        _api().create_repo(repo_id, repo_type=repo_type, private=private,
                           exist_ok=True)
    except HfHubHTTPError as e:                # already exists, or no create right
        print(f"[hub] create_repo({repo_id}) said: {e}")


# -------------------------------------------------------------------- runs ---
def push_run(experiment: str, run_dir: Path | None = None,
             message: str | None = None) -> str:
    """Mirror `runs/<experiment>/` to the dataset repo under the same path."""
    from .io import RUNS
    src = Path(run_dir or RUNS / experiment)
    if not src.is_dir():
        raise FileNotFoundError(f"no run directory at {src}")

    cfg = load("configs/hub.json")["logs"]
    _ensure(cfg["repo"], "dataset", cfg.get("private", True))
    _api().upload_folder(
        folder_path=str(src), repo_id=cfg["repo"], repo_type="dataset",
        path_in_repo=f"runs/{experiment}", ignore_patterns=list(_SKIP),
        commit_message=message or f"run {experiment}",
    )
    url = f"https://huggingface.co/datasets/{cfg['repo']}/tree/main/runs/{experiment}"
    print(f"[hub] pushed {src} -> {url}")
    return url


def pull_run(experiment: str, into: Path | None = None) -> Path:
    """Fetch `runs/<experiment>/` from the dataset repo into the local runs/."""
    from huggingface_hub import snapshot_download

    from .io import RUNS
    cfg = load("configs/hub.json")["logs"]
    dest = Path(into or RUNS)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=cfg["repo"], repo_type="dataset", token=token(),
        allow_patterns=[f"runs/{experiment}/*"], local_dir=str(dest.parent),
    )
    out = dest / experiment
    print(f"[hub] pulled {experiment} -> {out}")
    return out


# ---------------------------------------------------------------- adapters ---
def push_adapter(experiment: str, local_dir: Path | str,
                 message: str | None = None) -> str:
    """Upload a trained adapter to the model repo under `<experiment>/`."""
    src = Path(local_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"no adapter directory at {src}")

    cfg = load("configs/hub.json")["adapters"]
    _ensure(cfg["repo"], "model", cfg.get("private", True))
    _api().upload_folder(
        folder_path=str(src), repo_id=cfg["repo"], repo_type="model",
        path_in_repo=experiment, ignore_patterns=list(_SKIP),
        commit_message=message or f"adapter {experiment}",
    )
    url = f"https://huggingface.co/{cfg['repo']}/tree/main/{experiment}"
    print(f"[hub] pushed {src} -> {url}")
    return url


def pull_adapter(experiment: str, into: Path | str = "checkpoints") -> Path:
    """Download `<experiment>/` from the adapter repo. Rarely needed directly —
    `load_with_adapter` can stream straight from the Hub."""
    from huggingface_hub import snapshot_download
    cfg = load("configs/hub.json")["adapters"]
    dest = ROOT / into if not Path(into).is_absolute() else Path(into)
    snapshot_download(repo_id=cfg["repo"], repo_type="model", token=token(),
                      allow_patterns=[f"{experiment}/*"], local_dir=str(dest))
    out = dest / experiment
    print(f"[hub] pulled adapter {experiment} -> {out}")
    return out


def whoami() -> dict:
    """Check the token works, and against which account, before spending GPU time."""
    return _api().whoami()
