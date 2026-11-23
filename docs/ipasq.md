# Implementing inner-product-aware ASQ

This is a guide to adding the average-case objective (ADV) from White & Singal, *Inner Product Aware Quantization: Provably Fast, Accurate, and Adaptive Algorithms* (NeurIPS 2026, arXiv:2606.00289), to this benchmark.

## Why this is the next step

The benchmark exposed the problem this paper solves. On the outlier-channel data, plain ASQ cut quantization variance about 6× but barely improved inner-product error. The reason is that ASQ weights every coordinate equally, while the queries put almost all their mass on a few channels.

## The math

With unbiased stochastic rounding (SSQ), the coordinates are rounded independently, so for any query x:

```
Var[<w_hat, x>] = sum_i x_i^2 * VarSSQ(w_i, Q),   VarSSQ(w_i, Q) = (w_i_up - w_i)(w_i - w_i_down)
```

Averaging over queries gives the ADV objective, which is weighted ASQ with weights λ_i = E[x_i²]:

```
ADV(w, s) = min over |Q| <= s of  sum_i λ_i * (w_i_up - w_i)(w_i - w_i_down)
```

Plain ASQ is the special case λ_i = 1. Only the diagonal of E[xxᵀ] matters, because SSQ rounds coordinates independently.

## Step 1: weighted DP (required)

Start from `_asq_levels` in `ipsketch/methods.py`, which is the unweighted O(s·d²) dynamic program.

1. Sort w, and **carry each λ_i along with its w_i** through the sort. This is the easy place to introduce a bug.
2. The interval cost for consecutive levels at sorted indices j < k is
   `C[j,k] = sum_{j<i<k} λ_i (w_k - w_i)(w_i - w_j)`.
   Expand it into the weighted prefix sums `Λ0 = Σλ`, `Λ1 = Σλw`, `Λ2 = Σλw²` so each cost is O(1):
   `C[j,k] = -Λ2 + (w_j + w_k)·Λ1 - w_j·w_k·Λ0`, with every sum taken over the open interval (j, k).
3. Keep the same DP recurrence and backtracking.
4. Expose `weighted_asq_levels(x, lam, s)` as a module-level function. The tests call it directly.
5. In `InnerProductAwareASQ.fit`, compute levels per vector with `lam = self.lam`, then reuse `_stochastic_round_to_levels`.

When `pytest tests/test_ipasq.py` passes (brute-force optimality, reduction to plain ASQ, unbiasedness, and beating ASQ on outlier queries), run the benchmark with `--methods asq ipasq uniform_stoch rotated` on `outliers` and on GloVe.

## Step 2: make it fast (the paper's algorithmic contribution)

The cost matrix C is concave Monge and sorted (Lemmas D.3 and D.4), which is what lets the exact problem be solved in O(d log d + ds) time instead of O(s·d²). Try these in order:

- **MixDV s-approximation** (Theorem 3): binary-search a threshold t, then greedily check whether s levels can keep every interval cost ≤ t. It runs in O(d log d) and is within a factor of s of optimal. It's short to write, and makes a good warm start.
- **Exact speedup**: the Lagrangian / shortest-path search (Wilber-style) with a penalty τ per level, using the warm-started range [τ'/2, 2τ'] from the approximate solver.

Benchmark encode time against the current DP, which took 13.8 s for 20,000 GloVe vectors at 4 bits on an M5 Max.

## Step 3: questions worth answering (research)

- How much of the gain survives when λ is estimated from a **small sample** of queries, rather than the whole database?
- On GloVe, nearest rounding beat unbiased rounding for search. Does a **biased, inner-product-weighted** rounding beat both for recall?
- **Worst-case objective (MDV):** minimize max_j VarSSQ(w_j, Q). Does it reduce the worst 1% of query errors, as the paper reports?
- **KV caches:** the λ_i are exactly the per-channel energy of the attention queries, which is the bridge to `sketchkv`.
