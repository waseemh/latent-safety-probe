"""Probe statistics on paired activations.

Everything here is evaluated *held-out*: the direction is never scored on the
pairs that were used to fit it. ``pos`` and ``neg`` are index-aligned arrays of
shape ``[n_pairs, d]`` for a single layer.

Metrics
-------
loo_separation
    Leave-one-pair-out mean-difference direction; the held-out pair is projected
    on it. Reported as (mean_pos - mean_neg) / pooled_std of the held-out
    projections. A separation in standard-deviation units that is not inflated by
    in-sample fitting: two identical distributions score ≈ 0.
paired_accuracy
    Fraction of held-out pairs whose positive projects higher than its own
    negative twin. Chance = 0.5, bounded, easy to read.
cv_auroc
    AUROC of the held-out projections, ignoring the pairing. Chance = 0.5.
insample_separation
    The naive in-sample formula (direction fit and scored on the same pairs), for
    reference only. It is inflated; do not use it to make decisions.
refusal_gap
    Mean of (refusal log-odds of the positive prompt - that of its twin). Measures
    whether the model's own next-token output leans toward refusing the positive
    prompt more than the negative one. Every model separates harmful from benign
    *text*; only a safety-trained one turns that into refusal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def mean_diff_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    return unit(pos.mean(0) - neg.mean(0))


def insample_separation(pos: np.ndarray, neg: np.ndarray) -> float:
    d = mean_diff_direction(pos, neg)
    p, q = pos @ d, neg @ d
    s = np.sqrt((p.var() + q.var()) / 2)
    return float((p.mean() - q.mean()) / s) if s > 1e-12 else 0.0


def loo_projections(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project each pair on the direction fit from all *other* pairs (vectorized)."""
    n = pos.shape[0]
    if n < 3:
        raise ValueError("need at least 3 pairs for leave-one-out")
    sp, sn = pos.sum(0), neg.sum(0)
    dirs = ((sp - pos) - (sn - neg)) / (n - 1)  # [n, d], direction without pair i
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    dirs /= norms
    return np.einsum("ij,ij->i", pos, dirs), np.einsum("ij,ij->i", neg, dirs)


def loo_separation(pos: np.ndarray, neg: np.ndarray) -> float:
    p, q = loo_projections(pos, neg)
    s = np.sqrt((p.var() + q.var()) / 2)
    return float((p.mean() - q.mean()) / s) if s > 1e-12 else 0.0


def paired_accuracy(pos: np.ndarray, neg: np.ndarray) -> float:
    p, q = loo_projections(pos, neg)
    return float(np.mean(p > q))


def cv_auroc(pos: np.ndarray, neg: np.ndarray, seed: int = 0) -> float:
    """AUROC of the held-out (leave-one-pair-out) projections, ignoring pairing.

    Complements ``paired_accuracy`` (which only compares each positive with its own
    twin) by asking whether positives rank above negatives *globally*. A separate
    fitted classifier (e.g. logistic regression) is deliberately not used: with
    n ≪ d it is less stable than the mean-difference direction.
    """
    from sklearn.metrics import roc_auc_score

    p, q = loo_projections(pos, neg)
    y = np.r_[np.ones_like(p), np.zeros_like(q)]
    return float(roc_auc_score(y, np.r_[p, q]))


def refusal_gap(gap_pos: np.ndarray, gap_neg: np.ndarray) -> float:
    """Mean paired difference of refusal log-odds (nats)."""
    return float(np.mean(gap_pos - gap_neg))


def refusal_gap_accuracy(gap_pos: np.ndarray, gap_neg: np.ndarray) -> float:
    return float(np.mean(gap_pos > gap_neg))


def bootstrap_ci(stat, pos: np.ndarray, neg: np.ndarray, n_boot: int = 300, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap over *pairs* (resampled together)."""
    rng = np.random.default_rng(seed)
    n = pos.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = stat(pos[idx], neg[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


@dataclass
class LayerScore:
    layer: int
    loo_separation: float
    paired_accuracy: float
    insample_separation: float


@dataclass
class ConceptScore:
    concept: str
    role: str
    n_pairs: int
    layer: int
    loo_separation: float
    loo_separation_ci: tuple[float, float]
    paired_accuracy: float
    paired_accuracy_ci: tuple[float, float]
    cv_auroc: float
    insample_separation: float
    direction: np.ndarray
    per_layer: list[LayerScore]
    # Refusal link, filled in by the scanner (needs the model's output head)
    refusal_gap: float | None = None
    refusal_gap_ci: tuple[float, float] | None = None
    refusal_gap_accuracy: float | None = None
    steering_delta: float | None = None
    steering_delta_ci: tuple[float, float] | None = None


def search_layers(n_layers: int, lo: float = 0.25, hi: float = 0.8) -> list[int]:
    """Hidden-state indices to search. Index 0 is the embedding; L is the last block."""
    start = max(1, int(round(n_layers * lo)))
    end = max(start + 1, int(round(n_layers * hi)))
    return list(range(start, min(end, n_layers) + 1))


def score_concept(name: str, role: str, acts_pos: np.ndarray, acts_neg: np.ndarray,
                  layers: list[int], n_boot: int = 300, seed: int = 0) -> ConceptScore:
    """acts_* are [L+1, n, d]. Picks the layer with the best held-out separation."""
    per_layer = [
        LayerScore(
            layer=L,
            loo_separation=loo_separation(acts_pos[L], acts_neg[L]),
            paired_accuracy=paired_accuracy(acts_pos[L], acts_neg[L]),
            insample_separation=insample_separation(acts_pos[L], acts_neg[L]),
        )
        for L in layers
    ]
    best = max(per_layer, key=lambda s: s.loo_separation)
    pos, neg = acts_pos[best.layer], acts_neg[best.layer]
    return ConceptScore(
        concept=name,
        role=role,
        n_pairs=int(pos.shape[0]),
        layer=best.layer,
        loo_separation=best.loo_separation,
        loo_separation_ci=bootstrap_ci(loo_separation, pos, neg, n_boot, seed=seed),
        paired_accuracy=best.paired_accuracy,
        paired_accuracy_ci=bootstrap_ci(paired_accuracy, pos, neg, n_boot, seed=seed),
        cv_auroc=cv_auroc(pos, neg, seed=seed),
        insample_separation=best.insample_separation,
        direction=mean_diff_direction(pos, neg),
        per_layer=per_layer,
    )
