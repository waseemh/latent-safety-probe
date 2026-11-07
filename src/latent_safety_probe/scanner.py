"""Scan orchestration and baseline storage."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .concepts import ConceptSet, load_default, load_file
from .extract import Extractor, HFExtractor, LoadOptions
from .probe import (ConceptScore, bootstrap_ci, refusal_gap, refusal_gap_accuracy,
                    score_concept, search_layers)
from .report import ConceptReport, ScanReport, Thresholds, concept_verdict, overall_verdict

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- baselines
def weights_fingerprint(model_path: str) -> Optional[str]:
    """SHA-256 over local safetensors files (sorted). None for hub IDs."""
    p = Path(model_path)
    if not p.is_dir():
        return None
    files = sorted(p.glob("*.safetensors"))
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


class BaselineStore:
    def __init__(self, root: str | Path = "./baselines"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, model_id: str) -> Path:
        return self.root / (model_id.replace("/", "__") + ".npz")

    def exists(self, model_id: str) -> bool:
        return self._path(model_id).exists()

    def list(self) -> list[str]:
        return sorted(p.stem.replace("__", "/") for p in self.root.glob("*.npz"))

    def save(self, model_id: str, report: ScanReport, scores: dict[str, ConceptScore],
             fingerprint: Optional[str]) -> Path:
        meta = {
            "model_id": model_id,
            "fingerprint": fingerprint,
            "model_info": report.model_info,
            "extraction": report.extraction,
            "created_at": report.created_at,
            "layers": {k: int(v.layer) for k, v in scores.items()},
            "loo_separation": {k: float(v.loo_separation) for k, v in scores.items()},
        }
        arrays = {f"dir__{k}": v.direction.astype(np.float32) for k, v in scores.items()}
        path = self._path(model_id)
        np.savez_compressed(path, meta=json.dumps(meta), **arrays)
        return path

    def load(self, model_id: str) -> tuple[dict, dict[str, np.ndarray]]:
        with np.load(self._path(model_id)) as z:
            meta = json.loads(str(z["meta"]))
            dirs = {k[5:]: z[k] for k in z.files if k.startswith("dir__")}
        return meta, dirs


# ----------------------------------------------------------------------------- scanning
def run_scan(
    extractor: Extractor,
    concepts: ConceptSet,
    model_name: str,
    batch_size: int = 8,
    n_boot: int = 300,
    thresholds: Thresholds | None = None,
    layer_band: tuple[float, float] = (0.25, 0.8),
    seed: int = 0,
    extraction_meta: dict | None = None,
    model_info: dict | None = None,
    steering: bool = True,
) -> tuple[ScanReport, dict[str, ConceptScore]]:
    t0 = time.time()
    thresholds = thresholds or Thresholds()
    layers = search_layers(extractor.n_layers, *layer_band)

    scores: dict[str, ConceptScore] = {}
    for c in concepts:
        logger.info("Extracting %s (%d pairs)", c.name, c.n_pairs)
        acts_pos, gap_pos = extractor.extract_with_refusal(c.positive, batch_size)
        acts_neg, gap_neg = extractor.extract_with_refusal(c.negative, batch_size)
        s = score_concept(c.name, c.role, acts_pos, acts_neg, layers, n_boot, seed)
        # Refusal link: does the model lean toward refusing the positive twin...
        s.refusal_gap = refusal_gap(gap_pos, gap_neg)
        s.refusal_gap_ci = bootstrap_ci(refusal_gap, gap_pos, gap_neg, n_boot, seed=seed)
        s.refusal_gap_accuracy = refusal_gap_accuracy(gap_pos, gap_neg)
        if steering:
            # ...and does pushing the *negative* prompts along the concept direction, by the
            # positive-negative mean gap, raise their refusal log-odds? Causal, not correlational.
            pos_L, neg_L = acts_pos[s.layer], acts_neg[s.layer]
            shift = float((pos_L @ s.direction).mean() - (neg_L @ s.direction).mean())
            _, gap_steered = extractor.extract_with_refusal(
                c.negative, batch_size, steer=(s.layer, s.direction * shift))
            s.steering_delta = refusal_gap(gap_steered, gap_neg)
            s.steering_delta_ci = bootstrap_ci(refusal_gap, gap_steered, gap_neg, n_boot, seed=seed)
        scores[c.name] = s
        logger.info("  %s: layer %d  loo_sep=%.2f  acc=%.2f  auroc=%.2f  refusal_gap=%+.2f  "
                    "steer=%s  (in-sample %.1f)", c.name, s.layer, s.loo_separation,
                    s.paired_accuracy, s.cv_auroc, s.refusal_gap,
                    "n/a" if s.steering_delta is None else f"{s.steering_delta:+.2f}",
                    s.insample_separation)

    reports: dict[str, ConceptReport] = {}
    for name, s in scores.items():
        reports[name] = ConceptReport(
            concept=name, role=s.role, n_pairs=s.n_pairs, layer=s.layer,
            loo_separation=s.loo_separation, loo_separation_ci=s.loo_separation_ci,
            paired_accuracy=s.paired_accuracy, paired_accuracy_ci=s.paired_accuracy_ci,
            cv_auroc=s.cv_auroc, insample_separation=s.insample_separation,
            refusal_gap=s.refusal_gap, refusal_gap_ci=s.refusal_gap_ci,
            refusal_gap_accuracy=s.refusal_gap_accuracy, steering_delta=s.steering_delta,
            steering_delta_ci=s.steering_delta_ci,
            verdict=(concept_verdict(s.paired_accuracy_ci, s.refusal_gap_ci, s.steering_delta_ci,
                                     thresholds) if s.role == "safety" else None),
            per_layer={ls.layer: round(ls.loo_separation, 3) for ls in s.per_layer},
        )

    controls = [s.loo_separation for s in scores.values() if s.role == "control"]
    safety = [s.loo_separation for s in scores.values() if s.role == "safety"]
    control_ref = max(controls) if controls else None
    ratio = (min(safety) / control_ref) if (control_ref and safety and control_ref > 0) else None

    overall, rec = overall_verdict(reports, thresholds)
    report = ScanReport(
        model=model_name,
        model_info=model_info or {},
        extraction=extraction_meta or {},
        thresholds=thresholds,
        concepts=reports,
        control_reference=control_ref,
        safety_ratio=ratio,
        overall=overall,
        recommendation=rec,
        scan_seconds=time.time() - t0,
    )
    return report, scores


def compare_to_baseline(report: ScanReport, scores: dict[str, ConceptScore],
                        store: BaselineStore, baseline_id: str) -> ScanReport:
    meta, dirs = store.load(baseline_id)
    mismatch = []
    for k in ("num_hidden_layers", "hidden_size", "model_type"):
        if meta["model_info"].get(k) != report.model_info.get(k):
            mismatch.append(k)
    if mismatch:
        raise ValueError(f"Baseline '{baseline_id}' architecture differs in {mismatch}; "
                         "direction comparison is meaningless across architectures.")
    for name, s in scores.items():
        if name not in dirs:
            continue
        cos = float(np.dot(s.direction, dirs[name]))
        report.concepts[name].baseline_cosine = cos
        report.concepts[name].baseline_loo_separation = meta["loo_separation"].get(name)
    report.compared_to = baseline_id
    return report


def load_extractor(model_path: str, opts: LoadOptions) -> tuple[HFExtractor, dict, dict]:
    ex = HFExtractor.from_pretrained(model_path, opts)
    cfg = ex.model.config
    model_info = {
        "model_type": cfg.model_type,
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "hidden_size": int(cfg.hidden_size),
        "architecture": type(ex.model).__name__,
        "fingerprint": weights_fingerprint(model_path),
    }
    extraction = {
        "chat_template": ex.use_chat_template,
        "device": ex.device,
        "dtype": str(next(ex.model.parameters()).dtype),
        "token": "last_real_token",
        "max_length": ex.max_length,
    }
    return ex, model_info, extraction


def resolve_concepts(path: Optional[str]) -> ConceptSet:
    return load_file(path) if path else load_default()
