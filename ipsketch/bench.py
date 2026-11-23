"""Benchmark harness: compress, estimate inner products, measure accuracy and recall.

Example:
    python -m ipsketch.bench --dataset outliers --n 20000 --bits 1 2 4 8
    python -m ipsketch.bench --glove data/glove.6B.300d.txt --n 100000 --bits 1 2 4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATASETS, glove
from .methods import METHODS


def exact_topk(Q: np.ndarray, X: np.ndarray, k: int, batch: int = 64) -> np.ndarray:
    out = np.empty((len(Q), k), dtype=np.int64)
    for i in range(0, len(Q), batch):
        S = Q[i:i + batch] @ X.T
        out[i:i + batch] = np.argpartition(-S, k, axis=1)[:, :k]
    return out


def evaluate(method, Q: np.ndarray, X: np.ndarray, true_top10: np.ndarray, batch: int = 64) -> dict:
    """Inner-product error, bias, and maximum-inner-product-search recall."""
    abs_err = bias = abs_true = 0.0
    r10 = r10_100 = 0.0
    for i in range(0, len(Q), batch):
        qb = Q[i:i + batch]
        true = qb @ X.T
        est = method.estimate_ip(qb)
        diff = est - true
        abs_err += np.abs(diff).sum()
        bias += diff.sum()
        abs_true += np.abs(true).sum()
        top10 = np.argpartition(-est, 10, axis=1)[:, :10]
        top100 = np.argpartition(-est, 100, axis=1)[:, :100]
        for j, t in enumerate(true_top10[i:i + batch]):
            ts = set(t.tolist())
            r10 += len(ts & set(top10[j].tolist())) / 10
            r10_100 += len(ts & set(top100[j].tolist())) / 10
    nq = len(Q)
    return {
        "rel_ip_error": abs_err / abs_true,
        "rel_bias": bias / abs_true,
        "recall@10": r10 / nq,
        "recall10@100": r10_100 / nq,
    }


def run(X, Q, methods, bits_list, seed, dataset_name, max_asq_bits=4, asq_n=None):
    true_top10 = exact_topk(Q, X, 10)
    rows = []
    for name in methods:
        for b in bits_list:
            if name in ("asq", "ipasq") and b > max_asq_bits:
                continue  # 2^b levels per vector: DP and level-table overhead blow up
            Xm = X
            tt = true_top10
            if name in ("asq", "ipasq") and asq_n is not None and asq_n < len(X):
                # ASQ's exact DP is O(4^b d^2) per vector; evaluate on a prefix if asked.
                Xm = X[:asq_n]
                tt = exact_topk(Q, Xm, 10)
            m = METHODS[name](b, seed)
            t0 = time.perf_counter()
            m.fit(Xm)
            enc = time.perf_counter() - t0
            res = evaluate(m, Q, Xm, tt)
            rows.append({
                "dataset": dataset_name, "method": m.name, "key": name, "bits": b,
                "bits_per_dim": m.bits_per_dim, "n": len(Xm), "d": X.shape[1],
                "encode_s": enc, **res,
            })
            print(f"{m.name:34s} b={b}  bits/dim={m.bits_per_dim:5.2f}  "
                  f"ip_err={res['rel_ip_error']:.3f}  bias={res['rel_bias']:+.4f}  "
                  f"R@10={res['recall@10']:.3f}  R10@100={res['recall10@100']:.3f}  "
                  f"enc={enc:.2f}s", flush=True)
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for ds, g in df.groupby("dataset"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for name, h in g.groupby("method"):
            h = h.sort_values("bits_per_dim")
            axes[0].plot(h["bits_per_dim"], h["rel_ip_error"], marker="o", label=name)
            axes[1].plot(h["bits_per_dim"], h["recall10@100"], marker="o", label=name)
        axes[0].set_yscale("log")
        axes[0].set_xlabel("bits per dimension (incl. overhead)")
        axes[0].set_ylabel("relative inner-product error")
        axes[1].set_xlabel("bits per dimension (incl. overhead)")
        axes[1].set_ylabel("MIPS recall 10@100")
        axes[1].set_ylim(0, 1.02)
        for a in axes:
            a.grid(alpha=0.3)
        axes[1].legend(fontsize=8, loc="lower right")
        fig.suptitle(f"Inner-product preservation vs. storage: {ds} (n={g['n'].max():,}, d={g['d'].iloc[0]})")
        fig.tight_layout()
        fig.savefig(out / f"{ds}.png", dpi=140)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=list(DATASETS), default="outliers")
    p.add_argument("--glove", help="path to a GloVe .txt file (overrides --dataset)")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--d", type=int, default=128, help="dimension for synthetic data")
    p.add_argument("--nq", type=int, default=500)
    p.add_argument("--bits", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--methods", nargs="+", default=[m for m in METHODS if m != "ipasq"])
    p.add_argument("--asq-n", type=int, default=None, help="evaluate ASQ on the first N vectors only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    a = p.parse_args()

    if a.glove:
        X, Q = glove(a.glove, a.n, a.nq, a.seed)
        name = Path(a.glove).stem
    else:
        X, Q = DATASETS[a.dataset](a.n, a.d, a.nq, seed=a.seed)
        name = a.dataset
    print(f"{name}: database {X.shape}, queries {Q.shape}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = run(X, Q, a.methods, a.bits, a.seed, name, asq_n=a.asq_n)
    csv = out / f"{name}.csv"
    df.to_csv(csv, index=False)
    plot(df, out)
    print(f"wrote {csv} and {out / (name + '.png')}")


if __name__ == "__main__":
    main()
