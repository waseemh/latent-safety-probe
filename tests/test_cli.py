import json

import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

from latent_safety_probe.cli import app  # noqa: E402
from latent_safety_probe.concepts import load_default  # noqa: E402

runner = CliRunner()


def test_concepts_validate():
    cs = load_default()
    assert {c.name for c in cs} >= {"harmful_intent", "jailbreak_framing", "control_topic"}
    assert all(c.n_pairs >= 30 for c in cs)
    assert cs["control_topic"].role == "control"
    r = runner.invoke(app, ["concepts"])
    assert r.exit_code == 0


def test_scan_random_model_is_not_pass(tiny_model_dir, tmp_path):
    """A randomly initialised model has no safety structure; it must not PASS, and the
    JSON report must be well-formed with CIs."""
    out = tmp_path / "r.json"
    r = runner.invoke(app, ["scan", tiny_model_dir, "--device", "cpu", "--dtype", "float32",
                            "--json", str(out), "--bootstrap", "30", "--batch-size", "16"])
    assert r.exit_code in (1, 2), r.output
    rep = json.loads(out.read_text())
    assert rep["overall"] in ("WARNING", "CRITICAL")
    for c in rep["concepts"].values():
        assert len(c["paired_accuracy_ci"]) == 2
        assert 0 <= c["paired_accuracy"] <= 1
    assert rep["extraction"]["chat_template"] is True


def test_baseline_roundtrip_and_compare(tiny_model_dir, tmp_path):
    bdir = tmp_path / "b"
    r = runner.invoke(app, ["baseline", "create", tiny_model_dir, "--model-id", "org/tiny",
                            "--baselines-dir", str(bdir), "--device", "cpu", "--batch-size", "16"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["baseline", "list", "--baselines-dir", str(bdir)])
    assert "org/tiny" in r.output
    out = tmp_path / "c.json"
    r = runner.invoke(app, ["scan", tiny_model_dir, "--compare", "org/tiny", "--baselines-dir",
                            str(bdir), "--device", "cpu", "--json", str(out), "--bootstrap", "20",
                            "--batch-size", "16"])
    rep = json.loads(out.read_text())
    assert rep["compared_to"] == "org/tiny"
    # Same model vs its own baseline: directions identical => cosine ≈ 1
    for c in rep["concepts"].values():
        assert c["baseline_cosine"] > 0.99
