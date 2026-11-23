import itertools

import numpy as np
import pytest

from ipsketch.bench import evaluate, exact_topk
from ipsketch.data import isotropic, outlier_channels
from ipsketch.methods import (
    METHODS,
    QJL,
    AdaptiveStochasticQuant,
    CountSketchIP,
    GaussianJL,
    RotatedQuant,
    UniformQuant,
    _asq_levels,
)

UNBIASED = [
    lambda s: UniformQuant(2, stochastic=True, seed=s),
    lambda s: RotatedQuant(2, seed=s),
    lambda s: AdaptiveStochasticQuant(2, seed=s),
    lambda s: GaussianJL(4, seed=s),
    lambda s: CountSketchIP(4, seed=s),
    lambda s: QJL(1, seed=s),
]


@pytest.mark.parametrize("make", UNBIASED)
def test_unbiased_inner_products(make):
    """Averaged over random seeds, unbiased estimators converge to the true inner product."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 32)).astype(np.float32) * np.array([1, 2, 5, 10], dtype=np.float32)[:, None]
    q = rng.standard_normal((1, 32)).astype(np.float32)
    true = (q @ X.T)[0]
    trials = 3000
    est = np.mean([make(s).fit(X).estimate_ip(q)[0] for s in range(trials)], axis=0)
    scale = np.linalg.norm(q) * np.linalg.norm(X, axis=1)
    assert np.all(np.abs(est - true) / scale < 0.05), (est, true)


def test_nearest_rounding_is_deterministic_and_close():
    X, Q = isotropic(500, 64, 20)
    m = UniformQuant(8).fit(X)
    assert np.allclose(m.estimate_ip(Q), Q @ X.T, rtol=0, atol=0.5)


def test_asq_dp_matches_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(25):
        x = np.sort(rng.standard_normal(9))
        s = 4

        def cost(levels):
            c = 0.0
            for v in x:
                j = np.searchsorted(levels, v)
                if j == 0 or levels[j - 1] == v or j == len(levels):
                    continue
                a, b = levels[j - 1], levels[j]
                c += (b - v) * (v - a)
            return c

        best = min(
            cost(np.array([x[0], *[x[i] for i in mid], x[-1]]))
            for mid in itertools.combinations(range(1, len(x) - 1), s - 2)
        )
        assert cost(_asq_levels(x, s)) == pytest.approx(best, abs=1e-9)


def test_asq_has_lower_reconstruction_error_than_uniform():
    """ASQ minimizes squared reconstruction error (not inner-product error)."""
    X, _ = outlier_channels(400, 64, 1, seed=2)
    mse = lambda m: np.mean([((m(s).fit(X).Xq - X) ** 2).mean() for s in range(5)])
    asq = mse(lambda s: AdaptiveStochasticQuant(2, seed=s))
    uni = mse(lambda s: UniformQuant(2, stochastic=True, seed=s))
    assert asq < 0.5 * uni


def test_rotation_helps_with_outlier_channels():
    X, Q = outlier_channels(2000, 64, 50, seed=3)
    err = lambda m: np.abs(m.fit(X).estimate_ip(Q) - Q @ X.T).mean()
    assert err(RotatedQuant(2)) < err(UniformQuant(2, stochastic=True))


@pytest.mark.parametrize("key", list(METHODS))
@pytest.mark.parametrize("bits", [1, 4])
def test_bit_budget_respected(key, bits):
    X, _ = isotropic(300, 64, 1)
    m = METHODS[key](bits, 0).fit(X)
    if key == "asq":
        assert m.bits_per_vector == bits * 64 + (1 << bits) * 32  # level-table overhead
    elif key in ("uniform", "uniform_stoch", "rotated"):
        assert m.bits_per_vector == bits * 64 + 64
    else:
        assert m.bits_per_vector <= bits * 64


def test_evaluate_perfect_estimator_has_full_recall():
    X, Q = isotropic(1000, 32, 20)

    class Exact:
        def estimate_ip(self, q):
            return q @ X.T

    res = evaluate(Exact(), Q, X, exact_topk(Q, X, 10))
    assert res["recall@10"] == 1.0 and res["rel_ip_error"] == 0.0
