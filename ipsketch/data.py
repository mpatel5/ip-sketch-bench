"""Datasets: synthetic embeddings for quick runs, GloVe for real ones."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def isotropic(n: int, d: int = 128, nq: int = 500, seed: int = 0):
    """Standard Gaussian vectors: the easy case, with no structure to exploit."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n + nq, d)).astype(np.float32)
    return X[:n], X[n:]


def outlier_channels(n: int, d: int = 128, nq: int = 500, n_outliers: int = 4,
                     outlier_scale: float = 20.0, seed: int = 0):
    """Embeddings with a few high-magnitude channels.

    Per-dimension scales are log-normal, and a handful of channels are 20x
    larger, mimicking the outlier features seen in transformer activations and
    KV caches. Per-vector min/max quantizers waste most of their range on these.
    """
    rng = np.random.default_rng(seed)
    scales = np.exp(rng.normal(0.0, 0.5, d))
    scales[rng.choice(d, n_outliers, replace=False)] *= outlier_scale
    X = (rng.standard_normal((n + nq, d)) * scales).astype(np.float32)
    return X[:n], X[n:]


def glove(path: str | Path, n: int | None = None, nq: int = 1000, seed: int = 0):
    """GloVe vectors from the Stanford .txt release (one word + d floats per line).

    Queries are a random sample of held-out vectors, as in standard ANN benchmarks.
    """
    rows = []
    with open(path, encoding="utf8") as f:
        for i, line in enumerate(f):
            if n is not None and i >= n + nq:
                break
            parts = line.rstrip().split(" ")
            rows.append(np.asarray(parts[1:], dtype=np.float32))
    X = np.stack(rows)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    return X[perm[nq:]], X[perm[:nq]]


DATASETS = {"isotropic": isotropic, "outliers": outlier_channels}
