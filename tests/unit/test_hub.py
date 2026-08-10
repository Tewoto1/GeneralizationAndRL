"""
Hub wiring — the parts that can be tested without a network or a token.

Everything here is pure logic: where the token is found, what precedence it
has, and how an adapter reference resolves to (repo, subfolder). The upload
calls themselves are one-liners over `huggingface_hub`; what has historically
gone wrong is not the upload but the path and the credential.
"""
import pytest

from src.common import hub


# ------------------------------------------------------------------ dotenv ---
def test_parse_dotenv_handles_the_usual_shapes():
    """Comments, blanks, `export`, and quotes all appear in real .env files."""
    env = hub.parse_dotenv(
        "# a comment\n"
        "\n"
        "HF_TOKEN=hf_plain\n"
        "export OTHER='single'\n"
        'QUOTED="double"\n'
        "SPACED = spaced \n"
        "no_equals_line\n"
    )
    assert env == {"HF_TOKEN": "hf_plain", "OTHER": "single",
                   "QUOTED": "double", "SPACED": "spaced"}


def test_dotenv_is_not_a_shell():
    """A config file that can execute code is a config file that can surprise
    you. `$(...)` stays an inert string."""
    assert hub.parse_dotenv("K=$(rm -rf /)")["K"] == "$(rm -rf /)"


def test_find_dotenv_walks_up_to_the_parent_directory(tmp_path):
    """The token lives one directory ABOVE the repo, so it cannot be committed
    even by accident. The finder must therefore look upwards, not just in cwd."""
    parent = tmp_path / "AI Experiments"
    repo = parent / "GeneralizationAndRL"
    repo.mkdir(parents=True)
    (parent / ".env").write_text("HF_TOKEN=hf_from_parent\n")

    found = hub.find_dotenv(repo)
    assert found == parent / ".env"


def test_find_dotenv_prefers_the_nearest(tmp_path):
    """A repo-local .env beats one further up."""
    parent = tmp_path / "p"
    repo = parent / "r"
    repo.mkdir(parents=True)
    (parent / ".env").write_text("HF_TOKEN=far\n")
    (repo / ".env").write_text("HF_TOKEN=near\n")
    assert hub.find_dotenv(repo) == repo / ".env"


def test_find_dotenv_returns_none_when_absent(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert hub.find_dotenv(d, max_up=0) is None


def test_real_environment_beats_the_file(tmp_path, monkeypatch):
    """On a rented box the launch script injects HF_TOKEN. A stale .env copied
    along with the repo must not silently override it."""
    (tmp_path / ".env").write_text("HF_TOKEN=from_file\n")
    monkeypatch.setenv("HF_TOKEN", "from_environment")
    hub.load_env(tmp_path)
    import os
    assert os.environ["HF_TOKEN"] == "from_environment"


def test_dotenv_fills_an_unset_variable(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("HF_TOKEN=from_file\n")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    hub.load_env(tmp_path)
    import os
    assert os.environ["HF_TOKEN"] == "from_file"


def test_missing_token_names_where_it_looked(monkeypatch, tmp_path):
    """The error has to be actionable — a bare KeyError at 3am on a rented box
    is not."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hub, "find_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        hub.token(required=True)


# ---------------------------------------------------------------- adapters ---
def test_bare_experiment_name_resolves_against_the_generic_repo():
    """`hf:exp_v1` means subfolder exp_v1 of the adapters repo in
    configs/hub.json — one repo, one folder per experiment, so organisms stay
    comparable instead of scattering across near-duplicate repos."""
    repo, sub = hub.adapter_ref("hf:exp_v1")
    assert repo == "tewoto/LoRA_Adapters" and sub == "exp_v1"


def test_explicit_org_repo_passes_through():
    assert hub.adapter_ref("hf:someone/other_repo") == ("someone/other_repo", None)


def test_explicit_repo_with_subfolder():
    assert hub.adapter_ref("hf:someone/other/sub/dir") == ("someone/other", "sub/dir")


def test_local_path_is_left_alone():
    """One config key covers both local and remote adapters; without the `hf:`
    prefix every caller would need to know which kind it holds."""
    assert hub.adapter_ref("checkpoints/exp_v1") == ("checkpoints/exp_v1", None)


def test_hub_config_matches_what_the_code_expects():
    """Guards the rename-in-one-place-miss-it-in-another failure: a config key
    renamed without updating its reader cost this project an overnight run."""
    from src.common.config import load
    cfg = load("configs/hub.json")
    assert cfg["logs"]["repo"] and cfg["logs"]["type"] == "dataset"
    assert cfg["adapters"]["repo"] and cfg["adapters"]["type"] == "model"
    assert cfg["auth"]["token_env"] == "HF_TOKEN"
