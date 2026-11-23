# ip-sketch-bench

**How well do vector compression schemes preserve inner products?**

Inner products are the core operation behind attention, retrieval and maximum-inner-product search (MIPS). Every KV-cache compressor, embedding quantizer and vector index trades memory for inner-product accuracy. This repo compares eight such schemes on the same footing: the same bit budget (with per-vector overhead counted), the same data, and the same metrics.

| Family | Methods |
|---|---|
| Scalar quantization | uniform with nearest rounding; uniform with stochastic (unbiased) rounding; random rotation + stochastic uniform; **adaptive stochastic quantization (ASQ)** with an exact O(s·d²) dynamic program |
| Sketches | Gaussian Johnson–Lindenstrauss; CountSketch; **QJL**, 1-bit signs of a JL projection plus the norm |
| Learned codebooks | product quantization (k-means, 256 codes per block) |

Queries stay at full precision. That's the asymmetric setting used in vector search and in KV-cache attention, where the query is fresh and the keys are stored compressed.

## Results: embeddings with outlier channels

The dataset is 20,000 database vectors and 500 queries in d = 128. Per-dimension scales are log-normal, and four channels are 20× larger, similar to the outlier features in transformer activations and KV caches. Reproduce it with `python -m ipsketch.bench --dataset outliers --asq-n 20000`.

![results](results/outliers.png)

| Method | Bits/dim (incl. overhead) | Relative IP error | MIPS recall 10@100 | Encode time |
|---|---:|---:|---:|---:|
| Uniform, nearest rounding | 1.5 | 0.355 | 0.738 | 0.02 s |
| Uniform, stochastic | 1.5 | 0.501 | 0.576 | 0.03 s |
| Rotation + uniform (stochastic) | 2.5 | 0.106 | 0.989 | 0.03 s |
| ASQ, exact DP | 3.0 | 0.161 | 0.938 | 1.6 s |
| Gaussian JL (float32) | 2.0 | 0.660 | 0.674 | < 0.01 s |
| CountSketch (float32) | 2.0 | 0.394 | 0.855 | < 0.01 s |
| **QJL (1-bit)** | **1.0** | **0.194** | **0.975** | 0.01 s |
| **Product quantization** | **1.0** | **0.049** | **0.996** | 4.2 s |

Full results for every method at 1, 2, 4 and 8 bits are in [`results/outliers.csv`](results/outliers.csv), and the isotropic Gaussian control is in [`results/isotropic.csv`](results/isotropic.csv).

### What the numbers show

1. **Learned codebooks win on accuracy, at a cost.** Product quantization is the most accurate at every budget, but it has to be trained on the data, and it encodes hundreds to thousands of times slower than the data-oblivious methods (4–18 s vs. under 0.15 s here). That cost matters for KV caches, which are written once per generated token.
2. **QJL is the best data-oblivious method at 1 bit per dimension.** Signs of a Gaussian projection plus one norm reach 0.975 recall 10@100 at 1.0 bits/dim. Uniform quantization reaches 0.74 even with 1.5 bits/dim. Its error then shrinks slowly with more bits, because each sign carries at most one bit of information.
3. **Spending the budget on full floats is wasteful.** At a fixed bit budget, JL and CountSketch that store float32 coordinates keep only b·d/32 dimensions, and they're the worst methods at every budget. Spending bits on quantization beats spending them on extra projection dimensions.
4. **Rotation fixes outlier channels.** A random orthogonal rotation before stochastic rounding spreads the outliers across all dimensions. At 2 bits it cuts inner-product error from 0.165 to 0.106 and raises recall 10@100 from 0.90 to 0.99, for one extra matrix multiply.
5. **Minimizing squared error isn't the same as preserving inner products.** ASQ picks per-vector levels that minimize quantization variance, and it does cut variance about 6× compared with uniform levels on this data (verified in the tests). Yet its inner-product error barely moves (0.161 vs. 0.165 at 2 bits). The queries share the same outlier channels, so the inner-product error is dominated by a few coordinates that the variance objective weights no more heavily than the rest. ASQ also pays for its level table, so 2 bits becomes 3.0 bits/dim and 4 bits becomes 8.0. This gap is what *inner-product-aware* quantization targets (see the roadmap below).

All the stochastic methods and sketches are unbiased: their measured bias is within ±0.002 of the true inner product, and the tests check this.

## Run it on GloVe

The real-data run uses GloVe 300-d word vectors, an 822 MB download:

```bash
./scripts/download_glove.sh                       # downloads data/glove.6B.300d.txt
python -m ipsketch.bench --glove data/glove.6B.300d.txt --n 100000 --nq 1000 \
    --bits 1 2 4 8 --asq-n 20000
```

ASQ's exact DP is O(4ᵇ·d²) per vector, so `--asq-n` evaluates it on a subset.

## Install and test

```bash
pip install -e ".[dev]"
pytest
```

The tests check that:
- every stochastic quantizer and sketch is unbiased, averaged over 3,000 seeds;
- the ASQ dynamic program matches brute-force search over all level sets;
- ASQ beats uniform levels on squared error, and rotation beats plain uniform rounding on outlier data;
- every method's reported bits per vector matches the budget it claims.

## Roadmap

- [ ] GloVe-300 results at 100K+ vectors
- [ ] Inner-product-aware adaptive quantization: quantization objectives that preserve inner products with unseen query vectors, including fast exact and approximate algorithms (White & Singal, NeurIPS 2026)
- [ ] Two-stage schemes: MSE quantizer + 1-bit QJL on the residual (TurboQuant-style)
- [ ] Apply the best methods to a real LLM's KV cache (next project: `sketchkv`)

## License

Copyright (c) 2026 Mitesh Patel. All rights reserved.
