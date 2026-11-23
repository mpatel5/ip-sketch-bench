"""Checks for InnerProductAwareASQ. They skip until fit() is implemented."""

import itertools

import numpy as np
import pytest

from ipsketch.data import outlier_channels
from ipsketch.methods import AdaptiveStochasticQuant, InnerProductAwareASQ


def _implemented():
    try:
        InnerProductAwareASQ(1).fit(np.ones((2, 4), dtype=np.float32))
        return True
    except NotImplementedError:
        return False


pytestmark = pytest.mark.skipif(not _implemented(), reason="InnerProductAwareASQ.fit not implemented yet")


def weighted_cost(x, lam, levels):
    """ADV objective for one vector: sum_i lam_i * (up - x_i) * (x_i - down)."""
    levels = np.sort(levels)
    c = 0.0
    for v, l in zip(x, lam):
        j = np.searchsorted(levels, v)
        if j == 0 or j == len(levels) or levels[j - 1] == v:
            continue
        c += l * (levels[j] - v) * (v - levels[j - 1])
    return c


def test_levels_are_optimal_vs_brute_force():
    """Your per-vector levels must match brute-force search on small inputs.

    Expose a helper `weighted_asq_levels(x, lam, s)` from ipsketch.methods that
    returns the chosen levels (sorted, length s, including min and max).
    """
    from ipsketch.methods import weighted_asq_levels

    rng = np.random.default_rng(0)
    for _ in range(30):
        x = rng.standard_normal(9)
        lam = rng.exponential(1.0, 9)
        lam[rng.integers(9)] *= 50  # one heavily weighted coordinate
        s = 4
        xs = np.sort(x)
        best = min(
            weighted_cost(x, lam, np.array([xs[0], *[xs[i] for i in mid], xs[-1]]))
            for mid in itertools.combinations(range(1, 8), s - 2)
        )
        assert weighted_cost(x, lam, weighted_asq_levels(x, lam, s)) == pytest.approx(best, abs=1e-9)


def test_uniform_weights_reduce_to_plain_asq():
    from ipsketch.methods import _asq_levels, weighted_asq_levels

    rng = np.random.default_rng(1)
    x = rng.standard_normal(40)
    ones = np.ones_like(x)
    a = weighted_cost(x, ones, weighted_asq_levels(x, ones, 8))
    b = weighted_cost(x, ones, _asq_levels(np.sort(x), 8))
    assert a == pytest.approx(b)


def test_unbiased():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((3, 24)).astype(np.float32)
    q = rng.standard_normal((1, 24)).astype(np.float32)
    est = np.mean([InnerProductAwareASQ(2, seed=s).fit(X).estimate_ip(q)[0] for s in range(3000)], axis=0)
    scale = np.linalg.norm(q) * np.linalg.norm(X, axis=1)
    assert np.all(np.abs(est - (q @ X.T)[0]) / scale < 0.05)


def test_beats_plain_asq_on_inner_products_with_outlier_queries():
    """The point of the paper: weight by E[x_i^2] and outlier channels stop dominating the error."""
    X, Q = outlier_channels(1000, 64, 200, seed=4)
    err = lambda m: np.mean([np.abs(m(s).fit(X).estimate_ip(Q) - Q @ X.T).mean() for s in range(3)])
    ip = err(lambda s: InnerProductAwareASQ(2, seed=s))
    plain = err(lambda s: AdaptiveStochasticQuant(2, seed=s))
    assert ip < 0.9 * plain


def test_bit_budget_counts_level_table():
    from ipsketch.data import isotropic

    X, _ = isotropic(100, 64, 1)
    assert InnerProductAwareASQ(2).fit(X).bits_per_vector == 2 * 64 + 4 * 32
