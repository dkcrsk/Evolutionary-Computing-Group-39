# Roulette Selection: Windowing vs Sigma Scaling

Comparison of two fitness-proportionate (roulette-wheel) selection weightings
for Variant B, evaluated over 5 seeds (42, 43, 44, 45, 46). All other EA
settings (population init, crossover, mutation, population size, module
budget, evaluation budget) are held identical between the two runs — only
the selection weighting formula differs.

## Implementations

| Variant | Weighting formula | How it's implemented | Key parameter |
|---|---|---|---|
| **Baseline (windowing)** | f′(x) = β − f(x), β = worst fitness in current population | `roulette_selection`: each generation, finds the max (worst) finite fitness in the population and subtracts every individual's fitness from it. Individuals closer to 0 (better) get bigger slices; the current worst gets a slice of ~0. | none beyond `k` (controls `n // k` parents selected) |
| **Sigma scaling** | f′(x) = max(f̄ − f(x) + c·σ, 0), f̄ = mean, σ = std of current population's fitness | `roulette_selection_sigma`: computes the population's mean and standard deviation each generation, rescales every individual relative to that spread instead of the single worst member, clips negative results to 0 | `c` (multiplier on σ, set to 2.0 — the standard value from Eiben & Smith) |

Both formulas are adapted from Eiben & Smith's maximisation form for our
minimisation problem (lower tree-edit distance = better fitness).

## Results (5 seeds, final best fitness — lower is better)

| Seed | Baseline | Sigma | Diff (baseline − sigma) |
|---|---|---|---|
| 42 | 13.2770 | 13.6647 | −0.3877 |
| 43 | 12.8000 | 13.0124 | −0.2124 |
| 44 | 13.0811 | 12.7576 | +0.3236 |
| 45 | 13.0152 | 12.2296 | +0.7856 |
| 46 | 13.2889 | 13.3296 | −0.0407 |

| Metric | Baseline | Sigma |
|---|---|---|
| Mean final best fitness | 13.09 | 13.00 |
| Std across seeds | **0.20** | 0.55 |
| Per-seed wins | **3/5** | 2/5 |

## Why we chose windowing (baseline) for robustness

The mean difference between the two variants (0.09) is small relative to
either variant's own run-to-run spread, and sigma's slightly better mean is
driven almost entirely by a single seed (45, +0.79 in sigma's favour) — drop
that one seed and baseline is ahead on the remaining four. Win-count (3/5
baseline) and mean-of-means (sigma lower) point in different directions,
which is typical of noise dominating a real effect rather than a genuine
advantage for either variant.

Since the two variants are statistically indistinguishable on final fitness
with n=5, we selected the metric that does differentiate them cleanly:
**run-to-run stability**. Baseline's standard deviation (0.20) is more than
2.5× lower than sigma's (0.55), meaning its outcome is far less sensitive to
the random seed. A selection operator that produces more consistent results
across independent runs is the more defensible choice when performance alone
can't separate the two.

**Limitation:** n=5 seeds gives limited statistical power. This conclusion
is descriptive (mean, std, per-seed table), not backed by a significance
test. More seeds would be needed to confirm the variance difference is a
real property of the operator rather than sampling noise.
