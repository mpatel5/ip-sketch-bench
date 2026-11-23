"""Vector compression methods compared on how well they preserve inner products.

Every method follows the same protocol:

    m = Method(bits=b, seed=...)
    m.fit(X)                 # compress the database X (n x d)
    est = m.estimate_ip(Q)   # (nq x n) estimates of Q @ X.T
    m.bits_per_vector        # storage per database vector, overheads included

Queries are kept at full precision (the asymmetric setting used in vector search
and KV-cache attention). Storage counts per-vector metadata such as scales and
norms, but not shared, dataset-level state such as codebooks or projection
matrices, which is regenerated from a seed or amortized over the database.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

FLOAT_BITS = 32


class Compressor:
    name = "base"
    unbiased = False  # whether E[estimate] = true inner product

    def __init__(self, bits: int, seed: int = 0):
        self.bits = bits
        self.seed = seed
        self.d: int | None = None
        self.n: int | None = None

    def fit(self, X: np.ndarray) -> "Compressor":
        raise NotImplementedError

    def estimate_ip(self, Q: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def bits_per_vector(self) -> float:
        raise NotImplementedError

    @property
    def bits_per_dim(self) -> float:
        return self.bits_per_vector / self.d

    def label(self) -> str:
        return f"{self.name} ({self.bits}b)"


# --------------------------------------------------------------------------- #
# Scalar quantization
# --------------------------------------------------------------------------- #

def _uniform_quantize(X: np.ndarray, bits: int, stochastic: bool, rng: np.random.Generator) -> np.ndarray:
    """Per-vector uniform quantization onto 2^bits evenly spaced levels in [min, max]."""
    levels = (1 << bits) - 1
    lo = X.min(axis=1, keepdims=True)
    hi = X.max(axis=1, keepdims=True)
    step = np.where(hi > lo, (hi - lo) / levels, 1.0)
    t = (X - lo) / step
    if stochastic:
        f = np.floor(t)
        t = f + (rng.random(X.shape) < (t - f))  # round up w.p. frac -> unbiased
    else:
        t = np.rint(t)
    return (lo + np.clip(t, 0, levels) * step).astype(np.float32)


class UniformQuant(Compressor):
    """b-bit uniform scalar quantization with a per-vector min/max range."""

    def __init__(self, bits: int, stochastic: bool = False, seed: int = 0):
        super().__init__(bits, seed)
        self.stochastic = stochastic
        self.unbiased = stochastic
        self.name = "Uniform (stochastic)" if stochastic else "Uniform (nearest)"

    def fit(self, X):
        self.n, self.d = X.shape
        self.Xq = _uniform_quantize(X, self.bits, self.stochastic, np.random.default_rng(self.seed))
        return self

    def estimate_ip(self, Q):
        return Q @ self.Xq.T

    @property
    def bits_per_vector(self):
        return self.bits * self.d + 2 * FLOAT_BITS  # codes + min and max


def random_rotation(d: int, seed: int) -> np.ndarray:
    """Haar-random orthogonal matrix via QR of a Gaussian matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Qm, R = np.linalg.qr(A)
    return (Qm * np.sign(np.diag(R))).astype(np.float32)


class RotatedQuant(Compressor):
    """Random rotation, then unbiased uniform quantization.

    Rotating spreads outlier coordinates across all dimensions, shrinking each
    vector's dynamic range, which is the idea behind QuIP/TurboQuant-style
    schemes. Inner products are rotation-invariant, so queries are rotated too.
    """

    name = "Rotation + uniform (stochastic)"
    unbiased = True

    def fit(self, X):
        self.n, self.d = X.shape
        self.R = random_rotation(self.d, self.seed + 1)
        Y = X @ self.R.T
        self.Yq = _uniform_quantize(Y, self.bits, True, np.random.default_rng(self.seed))
        return self

    def estimate_ip(self, Q):
        return (Q @ self.R.T) @ self.Yq.T

    @property
    def bits_per_vector(self):
        return self.bits * self.d + 2 * FLOAT_BITS


@njit(cache=True)
def _asq_levels(x_sorted: np.ndarray, s: int) -> np.ndarray:
    """Exact adaptive stochastic quantization levels for one vector (O(s d^2) DP).

    Chooses s levels from the input values, always including min and max,
    minimizing the total variance of unbiased stochastic rounding:
        sum over x in [a, b] between adjacent levels of (b - x)(x - a).
    An optimal level set always exists among the input values, so searching
    over indices is exact.
    """
    d = x_sorted.shape[0]
    if s >= d:
        return x_sorted.copy()
    p1 = np.zeros(d + 1)
    p2 = np.zeros(d + 1)
    for i in range(d):
        p1[i + 1] = p1[i] + x_sorted[i]
        p2[i + 1] = p2[i] + x_sorted[i] * x_sorted[i]

    INF = 1e300
    dp = np.full((s, d), INF)
    parent = np.zeros((s, d), dtype=np.int64)
    dp[0, 0] = 0.0
    for k in range(1, s):
        for j in range(k, d):
            best = INF
            arg = 0
            xj = x_sorted[j]
            for i in range(k - 1, j):
                if dp[k - 1, i] >= INF:
                    continue
                xi = x_sorted[i]
                cnt = j - i - 1
                if cnt > 0:
                    s1 = p1[j] - p1[i + 1]
                    s2 = p2[j] - p2[i + 1]
                    cost = -s2 + (xi + xj) * s1 - xi * xj * cnt
                else:
                    cost = 0.0
                v = dp[k - 1, i] + cost
                if v < best:
                    best = v
                    arg = i
            dp[k, j] = best
            parent[k, j] = arg

    idx = np.empty(s, dtype=np.int64)
    j = d - 1
    for k in range(s - 1, -1, -1):
        idx[k] = j
        j = parent[k, j]
    return x_sorted[idx]


def _stochastic_round_to_levels(x: np.ndarray, levels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    hi_idx = np.clip(np.searchsorted(levels, x, side="left"), 1, len(levels) - 1)
    lo, hi = levels[hi_idx - 1], levels[hi_idx]
    gap = hi - lo
    p_up = np.where(gap > 0, (x - lo) / np.where(gap > 0, gap, 1.0), 0.0)
    return np.where(rng.random(x.shape) < p_up, hi, lo)


class AdaptiveStochasticQuant(Compressor):
    """Adaptive stochastic quantization (ASQ): per-vector optimal levels, unbiased rounding.

    Uses the exact O(s d^2) dynamic program for the variance-optimal s = 2^b
    levels. Each vector stores its own levels, which is real overhead at higher
    bit widths and is counted in bits_per_vector.
    """

    name = "ASQ (exact DP)"
    unbiased = True

    def fit(self, X):
        self.n, self.d = X.shape
        s = 1 << self.bits
        rng = np.random.default_rng(self.seed)
        Xq = np.empty_like(X, dtype=np.float32)
        for r in range(self.n):
            xs = np.sort(X[r].astype(np.float64))
            lv = _asq_levels(xs, s)
            Xq[r] = _stochastic_round_to_levels(X[r], lv, rng)
        self.Xq = Xq
        return self

    def estimate_ip(self, Q):
        return Q @ self.Xq.T

    @property
    def bits_per_vector(self):
        return self.bits * self.d + (1 << self.bits) * FLOAT_BITS  # codes + level table


@njit(cache=True)
def _weighted_asq_levels_sorted(x_sorted: np.ndarray, lam_sorted: np.ndarray, s: int) -> np.ndarray:
    """Exact weighted ASQ levels for one vector whose values are sorted ascending.

    Minimizes sum_i lam_i * (up_i - x_i) * (x_i - down_i) over s levels chosen
    from the input values, always including min and max. For consecutive
    levels at sorted positions j < k, the points strictly between them cost
        C[j, k] = -L2 + (x_j + x_k) * L1 - x_j * x_k * L0,
    where L0, L1 and L2 are the sums of lam, lam*x and lam*x^2 over (j, k),
    read in O(1) from prefix sums. With lam = 1 this is plain ASQ.
    """
    d = x_sorted.shape[0]
    if s >= d:
        return x_sorted.copy()
    p0 = np.zeros(d + 1)
    p1 = np.zeros(d + 1)
    p2 = np.zeros(d + 1)
    for i in range(d):
        w = lam_sorted[i]
        xi = x_sorted[i]
        p0[i + 1] = p0[i] + w
        p1[i + 1] = p1[i] + w * xi
        p2[i + 1] = p2[i] + w * xi * xi

    INF = 1e300
    dp = np.full((s, d), INF)
    parent = np.zeros((s, d), dtype=np.int64)
    dp[0, 0] = 0.0
    for k in range(1, s):
        for j in range(k, d):
            best = INF
            arg = 0
            xj = x_sorted[j]
            for i in range(k - 1, j):
                if dp[k - 1, i] >= INF:
                    continue
                xi = x_sorted[i]
                l0 = p0[j] - p0[i + 1]
                if l0 > 0.0:
                    l1 = p1[j] - p1[i + 1]
                    l2 = p2[j] - p2[i + 1]
                    cost = -l2 + (xi + xj) * l1 - xi * xj * l0
                else:
                    cost = 0.0
                v = dp[k - 1, i] + cost
                if v < best:
                    best = v
                    arg = i
            dp[k, j] = best
            parent[k, j] = arg

    idx = np.empty(s, dtype=np.int64)
    j = d - 1
    for k in range(s - 1, -1, -1):
        idx[k] = j
        j = parent[k, j]
    return x_sorted[idx]


def weighted_asq_levels(x: np.ndarray, lam: np.ndarray, s: int) -> np.ndarray:
    """Optimal weighted-ASQ levels for one vector x with per-coordinate weights lam.

    Sorts x and carries each weight along with its coordinate before running the
    exact DP. Returns s sorted levels (fewer if x has fewer than s entries).
    """
    order = np.argsort(x, kind="stable")
    xs = np.ascontiguousarray(x[order], dtype=np.float64)
    ls = np.ascontiguousarray(np.asarray(lam, dtype=np.float64)[order])
    return _weighted_asq_levels_sorted(xs, ls, s)


class InnerProductAwareASQ(Compressor):
    """Inner-product-aware ASQ: the average-case objective (ADV) from White & Singal (NeurIPS 2026).

    Plain ASQ minimizes sum_i VarSSQ(w_i, Q), which treats every coordinate
    equally. With unbiased stochastic rounding,
        Var[<w_hat, x>] = sum_i x_i^2 * VarSSQ(w_i, Q),
    so averaging over queries x gives weights lambda_i = E[x_i^2]. ADV minimizes
        sum_i lambda_i * (w_i_up - w_i) * (w_i - w_i_down)
    over level sets Q with |Q| <= s = 2^bits. This is weighted ASQ.

    lambda is estimated from the database X, as a proxy for the query
    distribution (in this benchmark, queries are drawn from the same distribution).

    Uses an exact O(s d^2) dynamic program over weighted prefix sums; see
    docs/ipasq.md for the faster algorithms in the paper.
    """

    name = "IP-aware ASQ (ADV)"
    unbiased = True

    def fit(self, X):
        self.n, self.d = X.shape
        self.lam = (X.astype(np.float64) ** 2).mean(axis=0)  # lambda_i = E[x_i^2]
        s = 1 << self.bits
        rng = np.random.default_rng(self.seed)
        Xq = np.empty_like(X, dtype=np.float32)
        for r in range(self.n):
            lv = weighted_asq_levels(X[r], self.lam, s)
            Xq[r] = _stochastic_round_to_levels(X[r], lv, rng)
        self.Xq = Xq
        return self

    def estimate_ip(self, Q):
        return Q @ self.Xq.T

    @property
    def bits_per_vector(self):
        return self.bits * self.d + (1 << self.bits) * FLOAT_BITS


# --------------------------------------------------------------------------- #
# Sketches
# --------------------------------------------------------------------------- #

def _sketch_dim(bits: int, d: int, per_coord_bits: int, overhead: int = 0) -> int:
    """Sketch length that fits a budget of `bits` per original dimension."""
    return max(1, (bits * d - overhead) // per_coord_bits)


class GaussianJL(Compressor):
    """Johnson-Lindenstrauss Gaussian projection to m float32 coordinates.

    <Sq, Sx> with S having N(0, 1/m) entries is an unbiased inner-product
    estimator. At a fixed bit budget, storing full floats leaves only
    m = b*d/32 dimensions, which is the point of comparison.
    """

    name = "Gaussian JL (float32)"
    unbiased = True

    def fit(self, X):
        self.n, self.d = X.shape
        self.m = _sketch_dim(self.bits, self.d, FLOAT_BITS)
        rng = np.random.default_rng(self.seed + 2)
        self.S = (rng.standard_normal((self.m, self.d)) / math.sqrt(self.m)).astype(np.float32)
        self.SX = X @ self.S.T
        return self

    def estimate_ip(self, Q):
        return (Q @ self.S.T) @ self.SX.T

    @property
    def bits_per_vector(self):
        return self.m * FLOAT_BITS


class CountSketchIP(Compressor):
    """CountSketch: each coordinate hashed to one of m buckets with a random sign.

    Unbiased for inner products, and applying it costs O(d) per vector instead
    of O(md) for a dense projection.
    """

    name = "CountSketch (float32)"
    unbiased = True

    def fit(self, X):
        self.n, self.d = X.shape
        self.m = _sketch_dim(self.bits, self.d, FLOAT_BITS)
        rng = np.random.default_rng(self.seed + 3)
        self.h = rng.integers(0, self.m, self.d)
        self.sgn = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), self.d)
        # One nonzero (+-1) per column. Stored densely so the projection runs on
        # BLAS; a production version would use the O(d) scatter-add form.
        self.S = np.zeros((self.m, self.d), dtype=np.float32)
        self.S[self.h, np.arange(self.d)] = self.sgn
        self.SX = self._apply(X)
        return self

    def _apply(self, A):
        return A @ self.S.T

    def estimate_ip(self, Q):
        return self._apply(Q) @ self.SX.T

    @property
    def bits_per_vector(self):
        return self.m * FLOAT_BITS


class QJL(Compressor):
    """1-bit quantized JL: store sign(Sx) and ||x||.

    For Gaussian rows s, E[<s, q> sign(<s, x>)] = sqrt(2/pi) <q, x> / ||x||, so
        est = sqrt(pi/2) / m * ||x|| * <Sq, sign(Sx)>
    is unbiased, with only the query projected at full precision.
    """

    name = "QJL (1-bit signs)"
    unbiased = True

    def fit(self, X):
        self.n, self.d = X.shape
        self.m = _sketch_dim(self.bits, self.d, 1, overhead=FLOAT_BITS)
        rng = np.random.default_rng(self.seed + 4)
        self.S = rng.standard_normal((self.m, self.d)).astype(np.float32)
        self.signs = np.sign(X @ self.S.T).astype(np.float32)
        self.signs[self.signs == 0] = 1.0
        self.norms = np.linalg.norm(X, axis=1).astype(np.float32)
        return self

    def estimate_ip(self, Q):
        return math.sqrt(math.pi / 2) / self.m * ((Q @ self.S.T) @ self.signs.T) * self.norms

    @property
    def bits_per_vector(self):
        return self.m + FLOAT_BITS


# --------------------------------------------------------------------------- #
# Learned codebooks
# --------------------------------------------------------------------------- #

class ProductQuant(Compressor):
    """Product quantization: split dimensions into M blocks, k-means with 256 codes each.

    b bits per dimension -> M = b*d/8 one-byte codes per vector. Codebooks are
    learned from the data and shared, so they aren't counted per vector.
    """

    name = "Product quantization"

    def __init__(self, bits: int, seed: int = 0, train_size: int = 10_000):
        super().__init__(bits, seed)
        self.train_size = train_size

    def fit(self, X):
        from sklearn.cluster import KMeans

        self.n, self.d = X.shape
        self.M = max(1, min(self.d, (self.bits * self.d) // 8))
        self.blocks = np.array_split(np.arange(self.d), self.M)
        rng = np.random.default_rng(self.seed)
        train = X[rng.choice(self.n, min(self.train_size, self.n), replace=False)]
        Xq = np.empty_like(X, dtype=np.float32)
        for b, cols in enumerate(self.blocks):
            k = min(256, len(train))
            km = KMeans(n_clusters=k, n_init=1, max_iter=25, random_state=self.seed + b).fit(train[:, cols])
            codes = km.predict(X[:, cols])
            Xq[:, cols] = km.cluster_centers_[codes]
        self.Xq = Xq
        return self

    def estimate_ip(self, Q):
        return Q @ self.Xq.T

    @property
    def bits_per_vector(self):
        return 8 * self.M


METHODS = {
    "uniform": lambda b, s: UniformQuant(b, stochastic=False, seed=s),
    "uniform_stoch": lambda b, s: UniformQuant(b, stochastic=True, seed=s),
    "rotated": lambda b, s: RotatedQuant(b, seed=s),
    "asq": lambda b, s: AdaptiveStochasticQuant(b, seed=s),
    "ipasq": lambda b, s: InnerProductAwareASQ(b, seed=s),
    "jl": lambda b, s: GaussianJL(b, seed=s),
    "countsketch": lambda b, s: CountSketchIP(b, seed=s),
    "qjl": lambda b, s: QJL(b, seed=s),
    "pq": lambda b, s: ProductQuant(b, seed=s),
}
