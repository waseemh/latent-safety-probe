"""latent-safety-probe — latent-space tamper detector for LLM safety training.

The probe statistics (``latent_safety_probe.probe``) and concept sets are importable without
torch; model loading lives in ``latent_safety_probe.extract`` / ``latent_safety_probe.scanner`` and is
imported lazily.
"""

from __future__ import annotations

import importlib

from .concepts import Concept, ConceptSet, load_default, load_file
from .probe import (cv_auroc, insample_separation, loo_separation, mean_diff_direction,
                    paired_accuracy, score_concept)
from .report import ScanReport, Thresholds

__version__ = "0.1.0"

_LAZY = {
    "HFExtractor": "latent_safety_probe.extract",
    "LoadOptions": "latent_safety_probe.extract",
    "BaselineStore": "latent_safety_probe.scanner",
    "compare_to_baseline": "latent_safety_probe.scanner",
    "load_extractor": "latent_safety_probe.scanner",
    "run_scan": "latent_safety_probe.scanner",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(name)


__all__ = [
    "Concept", "ConceptSet", "load_default", "load_file",
    "cv_auroc", "insample_separation", "loo_separation", "mean_diff_direction",
    "paired_accuracy", "score_concept", "ScanReport", "Thresholds", *_LAZY,
]
