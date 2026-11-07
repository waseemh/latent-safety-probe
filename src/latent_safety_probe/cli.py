"""lsprobe command-line interface."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .extract import LoadOptions
from .report import ScanReport, Thresholds

app = typer.Typer(help="Latent-space tamper detector for LLM safety training.", no_args_is_help=True)
baseline_app = typer.Typer(help="Manage baselines for drift comparison.", no_args_is_help=True)
app.add_typer(baseline_app, name="baseline")
console = Console(stderr=True)

EXIT = {"PASS": 0, "CRITICAL": 1, "WARNING": 2}
EXIT_DRIFT = 3


def _load_opts(device, dtype, trust_remote_code, allow_pickle, load_8bit, load_4bit,
               no_chat_template, max_length) -> LoadOptions:
    return LoadOptions(device=device, dtype=dtype, trust_remote_code=trust_remote_code,
                       allow_pickle=allow_pickle, load_in_8bit=load_8bit, load_in_4bit=load_4bit,
                       chat_template=not no_chat_template, max_length=max_length)


def _setup_logging(verbose: bool):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)


def render(report: ScanReport, out: Console):
    color = {"PASS": "green", "WARNING": "yellow", "CRITICAL": "red"}
    t = Table(title=f"lsprobe · {report.model}", show_lines=False)
    for col in ("concept", "role", "pairs", "layer", "held-out sep", "paired acc", "AUROC",
                "refusal gap", "steer Δ", "verdict"):
        t.add_column(col, justify="right" if col not in ("concept", "role", "verdict") else "left")
    for c in report.concepts.values():
        v = c.verdict or "—"
        t.add_row(
            c.concept, c.role, str(c.n_pairs), str(c.layer),
            f"{c.loo_separation:.2f} [{c.loo_separation_ci[0]:.2f}, {c.loo_separation_ci[1]:.2f}]",
            f"{c.paired_accuracy:.2f} [{c.paired_accuracy_ci[0]:.2f}, {c.paired_accuracy_ci[1]:.2f}]",
            f"{c.cv_auroc:.2f}",
            "—" if c.refusal_gap is None else
            f"{c.refusal_gap:+.2f} [{c.refusal_gap_ci[0]:+.2f}, {c.refusal_gap_ci[1]:+.2f}]",
            "—" if c.steering_delta is None else
            f"{c.steering_delta:+.2f} [{c.steering_delta_ci[0]:+.2f}, {c.steering_delta_ci[1]:+.2f}]",
            f"[{color.get(v, 'white')}]{v}[/]",
        )
    out.print(t)
    if report.compared_to:
        d = Table(title=f"Drift vs baseline '{report.compared_to}'")
        d.add_column("concept"); d.add_column("cosine", justify="right")
        d.add_column("baseline sep", justify="right"); d.add_column("now", justify="right")
        for c in report.concepts.values():
            if c.baseline_cosine is not None:
                d.add_row(c.concept, f"{c.baseline_cosine:.3f}",
                          f"{c.baseline_loo_separation:.2f}", f"{c.loo_separation:.2f}")
        out.print(d)
    extra = ""
    if report.safety_ratio is not None:
        extra = f"\nSafety/control ratio: {report.safety_ratio:.2f}  (control ref {report.control_reference:.2f})"
    out.print(Panel(f"[bold {color[report.overall]}]{report.overall}[/]\n{report.recommendation}{extra}"
                    f"\n\n{report.scan_seconds:.1f}s · chat_template={report.extraction.get('chat_template')}"
                    f" · {report.extraction.get('device')}/{report.extraction.get('dtype')}",
                    title="Result"))


@app.command()
def scan(
    model: str = typer.Argument(..., help="HF model id or local path"),
    concepts: Optional[Path] = typer.Option(None, "--concepts", help="Custom concepts YAML"),
    compare: Optional[str] = typer.Option(None, "--compare", help="Baseline id to compare against"),
    baselines_dir: Path = typer.Option("./baselines", "--baselines-dir"),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Write JSON report here ('-' for stdout)"),
    batch_size: int = typer.Option(8, "--batch-size"),
    n_boot: int = typer.Option(300, "--bootstrap", help="Bootstrap resamples for CIs"),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
    allow_pickle: bool = typer.Option(False, "--allow-pickle"),
    load_8bit: bool = typer.Option(False, "--load-8bit"),
    load_4bit: bool = typer.Option(False, "--load-4bit"),
    no_chat_template: bool = typer.Option(False, "--no-chat-template", help="Feed raw prompts"),
    max_length: int = typer.Option(256, "--max-length"),
    pass_min_accuracy: float = typer.Option(Thresholds().pass_min_accuracy, "--pass-min-acc"),
    critical_max_accuracy: float = typer.Option(Thresholds().critical_max_accuracy, "--critical-max-acc"),
    pass_min_refusal_gap: float = typer.Option(Thresholds().pass_min_refusal_gap, "--pass-min-gap",
                                               help="Lower CI bound of refusal gap (nats) for PASS"),
    critical_max_refusal_gap: float = typer.Option(Thresholds().critical_max_refusal_gap, "--critical-max-gap",
                                                   help="Upper CI bound of refusal gap below this => CRITICAL"),
    pass_min_steering: float = typer.Option(Thresholds().pass_min_steering, "--pass-min-steer",
                                            help="Lower CI bound of steering rise (nats) for PASS"),
    critical_max_steering: float = typer.Option(Thresholds().critical_max_steering, "--critical-max-steer",
                                                help="Upper CI bound of steering rise below this => CRITICAL"),
    no_steering: bool = typer.Option(False, "--no-steering", help="Skip the steering pass"),
    min_cosine: float = typer.Option(0.8, "--min-cosine", help="Drift check: min cosine vs baseline"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Run a held-out safety-structure scan (and optional baseline drift check)."""
    from .scanner import (BaselineStore, compare_to_baseline, load_extractor, resolve_concepts,
                          run_scan)

    _setup_logging(verbose)
    opts = _load_opts(device, dtype, trust_remote_code, allow_pickle, load_8bit, load_4bit,
                      no_chat_template, max_length)
    thresholds = Thresholds(pass_min_accuracy=pass_min_accuracy,
                            critical_max_accuracy=critical_max_accuracy,
                            pass_min_refusal_gap=pass_min_refusal_gap,
                            critical_max_refusal_gap=critical_max_refusal_gap,
                            pass_min_steering=pass_min_steering,
                            critical_max_steering=critical_max_steering)
    with console.status(f"Loading {model}…"):
        ex, model_info, extraction = load_extractor(model, opts)
    with console.status("Extracting activations and scoring…"):
        report, scores = run_scan(ex, resolve_concepts(str(concepts) if concepts else None), model,
                                  batch_size, n_boot, thresholds,
                                  extraction_meta=extraction, model_info=model_info,
                                  steering=not no_steering)
    drift_failed = False
    if compare:
        store = BaselineStore(baselines_dir)
        if not store.exists(compare):
            console.print(f"[red]No baseline '{compare}'. Create one with: lsprobe baseline create {compare}[/]")
            raise typer.Exit(1)
        report = compare_to_baseline(report, scores, store, compare)
        drift_failed = any(c.baseline_cosine is not None and c.baseline_cosine < min_cosine
                           for c in report.concepts.values())

    if json_out is not None:
        payload = report.model_dump_json(indent=2)
        if str(json_out) == "-":
            print(payload)
        else:
            json_out.write_text(payload)
    if json_out is None or str(json_out) != "-":
        render(report, console)

    code = EXIT[report.overall]
    if code == 0 and drift_failed:
        code = EXIT_DRIFT
    raise typer.Exit(code)


@baseline_app.command("create")
def baseline_create(
    model: str = typer.Argument(...),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Store under this id"),
    concepts: Optional[Path] = typer.Option(None, "--concepts"),
    baselines_dir: Path = typer.Option("./baselines", "--baselines-dir"),
    batch_size: int = typer.Option(8, "--batch-size"),
    device: str = typer.Option("auto", "--device"),
    dtype: str = typer.Option("auto", "--dtype"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
    allow_pickle: bool = typer.Option(False, "--allow-pickle"),
    load_8bit: bool = typer.Option(False, "--load-8bit"),
    load_4bit: bool = typer.Option(False, "--load-4bit"),
    no_chat_template: bool = typer.Option(False, "--no-chat-template"),
    max_length: int = typer.Option(256, "--max-length"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Scan a trusted model and store its directions for later drift checks."""
    from .scanner import BaselineStore, load_extractor, resolve_concepts, run_scan

    _setup_logging(verbose)
    opts = _load_opts(device, dtype, trust_remote_code, allow_pickle, load_8bit, load_4bit,
                      no_chat_template, max_length)
    with console.status(f"Loading {model}…"):
        ex, model_info, extraction = load_extractor(model, opts)
    with console.status("Scanning…"):
        report, scores = run_scan(ex, resolve_concepts(str(concepts) if concepts else None), model,
                                  batch_size, n_boot=100, extraction_meta=extraction,
                                  model_info=model_info, steering=False)
    path = BaselineStore(baselines_dir).save(model_id or model, report, scores,
                                             model_info.get("fingerprint"))
    render(report, console)
    console.print(f"Baseline saved: {path}")


@baseline_app.command("list")
def baseline_list(baselines_dir: Path = typer.Option("./baselines", "--baselines-dir")):
    from .scanner import BaselineStore

    for b in BaselineStore(baselines_dir).list():
        print(b)


@baseline_app.command("show")
def baseline_show(model_id: str, baselines_dir: Path = typer.Option("./baselines", "--baselines-dir")):
    from .scanner import BaselineStore

    meta, dirs = BaselineStore(baselines_dir).load(model_id)
    meta["directions"] = {k: list(v.shape) for k, v in dirs.items()}
    print(json.dumps(meta, indent=2))


@app.command()
def concepts(path: Optional[Path] = typer.Argument(None, help="YAML to validate (default: built-in)")):
    """List (and validate) the concept set."""
    from .scanner import resolve_concepts

    cs = resolve_concepts(str(path) if path else None)
    t = Table(title="Concepts")
    t.add_column("name"); t.add_column("role"); t.add_column("pairs", justify="right"); t.add_column("description")
    for c in cs:
        t.add_row(c.name, c.role, str(c.n_pairs), c.description)
    Console().print(t)


if __name__ == "__main__":
    app()
