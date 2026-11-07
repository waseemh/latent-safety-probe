import numpy as np
import pytest

from latent_safety_probe.probe import (bootstrap_ci, cv_auroc, insample_separation, loo_separation,
                            paired_accuracy, score_concept, search_layers)


def make(n, d, shift, seed=0):
    rng = np.random.default_rng(seed)
    s = np.zeros(d); s[0] = shift
    return rng.standard_normal((n, d)) + s, rng.standard_normal((n, d))


def test_identical_distributions_score_near_zero_heldout():
    """No real signal => held-out separation ≈ 0 and accuracy ≈ 0.5."""
    pos, neg = make(32, 2048, shift=0.0)
    assert abs(loo_separation(pos, neg)) < 0.5
    assert 0.3 <= paired_accuracy(pos, neg) <= 0.7
    assert 0.3 <= cv_auroc(pos, neg) <= 0.7
    # ...while the in-sample formula is wildly inflated on the same data
    assert insample_separation(pos, neg) > 10


def test_real_signal_is_detected():
    pos, neg = make(32, 512, shift=6.0)
    assert loo_separation(pos, neg) > 2.0
    assert paired_accuracy(pos, neg) > 0.9
    assert cv_auroc(pos, neg) > 0.9


def test_loo_matches_explicit_loop():
    pos, neg = make(12, 64, shift=2.0)
    from latent_safety_probe.probe import loo_projections, mean_diff_direction
    p, q = loo_projections(pos, neg)
    for i in range(12):
        m = np.ones(12, bool); m[i] = False
        d = mean_diff_direction(pos[m], neg[m])
        assert np.isclose(p[i], pos[i] @ d) and np.isclose(q[i], neg[i] @ d)


def test_bootstrap_ci_brackets_estimate():
    pos, neg = make(40, 128, shift=3.0)
    est = paired_accuracy(pos, neg)
    lo, hi = bootstrap_ci(paired_accuracy, pos, neg, n_boot=100)
    assert lo <= est <= hi and 0 <= lo and hi <= 1


def test_search_layers_band():
    assert search_layers(32) == list(range(8, 27))
    assert search_layers(4) and all(1 <= L <= 4 for L in search_layers(4))


def test_score_concept_picks_best_layer():
    L, n, d = 6, 20, 64
    rng = np.random.default_rng(1)
    acts_pos = rng.standard_normal((L, n, d)); acts_neg = rng.standard_normal((L, n, d))
    acts_pos[3, :, 0] += 5.0  # only layer 3 carries signal
    s = score_concept("x", "safety", acts_pos, acts_neg, layers=list(range(1, L)), n_boot=50)
    assert s.layer == 3
    assert s.paired_accuracy > 0.9
    assert np.isclose(np.linalg.norm(s.direction), 1.0)
