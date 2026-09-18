"""Compare the baseline (windowing) and sigma-scaling roulette variants.

Run AFTER both experiments have finished:
    variant_b_roulette.py with SELECTION_VARIANT = "baseline"  (5 seeds)
    variant_b_roulette.py with SELECTION_VARIANT = "sigma"     (5 seeds)

Reads results/variant_b_baseline/seed_<N>.csv and
results/variant_b_sigma/seed_<N>.csv, then:
  1. prints each seed's final best_fitness for both variants side by side
     (same seed = same starting population/targets/mutations, only the
     selection weighting differs -- a paired comparison)
  2. prints mean +/- std across the 5 seeds per variant, and the winner
     (lower is better -- this is a minimisation problem)
  3. plots mean best_fitness per generation (+/- std band) for both
     variants on one axis, saved to results/comparison_convergence.png

A NOTE ON n=5: with only 5 seeds, the mean/std comparison below is
informative but not a rigorous significance test. Wilcoxon signed-rank
(paired, since seeds match) is included as a rough indicator, but a
p-value from n=5 pairs has very low power -- treat it as a hint, not
proof, and say so in the report.
"""

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "results"
SEEDS: list[int] = [42, 43, 44, 45, 46]
VARIANTS: list[str] = ["baseline", "sigma"]


def load_seed_csv(variant: str, seed: int) -> list[dict[str, float]]:
    path = RESULTS / f"variant_b_{variant}" / f"seed_{seed}.csv"
    if not path.exists():
        msg = f"missing {path} -- did you run variant_b_roulette.py with SELECTION_VARIANT={variant!r}?"
        raise FileNotFoundError(msg)
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({
                "generation": int(row["generation"]),
                "evaluations": int(row["evaluations"]),
                "best_fitness": float(row["best_fitness"]),
                "mean_fitness": float(row["mean_fitness"]),
                "mean_modules": float(row["mean_modules"]),
            })
    return rows


def final_best_fitness(rows: list[dict[str, float]]) -> float:
    # With elitism, best_fitness should be monotonically non-increasing
    # (lower is better here), but take the min over the whole run rather
    # than just the last row, in case that assumption ever breaks.
    return min(r["best_fitness"] for r in rows)


def mean_and_std_per_generation(
    all_seed_rows: list[list[dict[str, float]]],
) -> tuple[list[int], list[float], list[float]]:
    """Align seeds by generation number, return (gens, mean, std) of best_fitness."""
    by_gen: dict[int, list[float]] = {}
    for rows in all_seed_rows:
        for r in rows:
            by_gen.setdefault(r["generation"], []).append(r["best_fitness"])
    gens = sorted(by_gen)
    means = [statistics.mean(by_gen[g]) for g in gens]
    stds = [statistics.stdev(by_gen[g]) if len(by_gen[g]) > 1 else 0.0 for g in gens]
    return gens, means, stds


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float, str]:
    """Minimal paired sign-based indicator (not a full Wilcoxon implementation).

    For n=5 a real Wilcoxon table has almost no resolving power anyway --
    this just reports how many of the 5 seeds favoured each variant, which
    is honestly about as much signal as n=5 supports.
    """
    wins_a = sum(1 for d in diffs if d < 0)  # baseline better (lower) than sigma
    wins_b = sum(1 for d in diffs if d > 0)  # sigma better than baseline
    ties = sum(1 for d in diffs if d == 0)
    return wins_a, f"{wins_a} baseline / {wins_b} sigma / {ties} ties (of {len(diffs)} seeds)"


def main() -> None:
    per_variant_rows: dict[str, list[list[dict[str, float]]]] = {v: [] for v in VARIANTS}
    per_variant_final: dict[str, list[float]] = {v: [] for v in VARIANTS}

    for variant in VARIANTS:
        for seed in SEEDS:
            rows = load_seed_csv(variant, seed)
            per_variant_rows[variant].append(rows)
            per_variant_final[variant].append(final_best_fitness(rows))

    # --- per-seed table ------------------------------------------------- #
    print(f"{'seed':>6}  {'baseline':>12}  {'sigma':>12}  {'diff (b-s)':>12}")
    diffs: list[float] = []
    for i, seed in enumerate(SEEDS):
        b = per_variant_final["baseline"][i]
        s = per_variant_final["sigma"][i]
        diffs.append(b - s)
        print(f"{seed:>6}  {b:>12.4f}  {s:>12.4f}  {b - s:>12.4f}")

    # --- summary ---------------------------------------------------------#
    print()
    for variant in VARIANTS:
        vals = per_variant_final[variant]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{variant:>10}: mean={mean:.4f}  std={std:.4f}  (lower is better)")

    baseline_mean = statistics.mean(per_variant_final["baseline"])
    sigma_mean = statistics.mean(per_variant_final["sigma"])
    winner = "baseline" if baseline_mean < sigma_mean else "sigma"
    print(f"\nLower mean final best_fitness -> {winner}")

    wins_a, summary = wilcoxon_signed_rank(diffs)
    print(f"Per-seed win count: {summary}")
    print(
        "(n=5 is too small for this to be conclusive on its own -- report "
        "mean/std and the per-seed table, and flag the sample size as a "
        "limitation.)",
    )

    # --- convergence plot --------------------------------------------------#
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"baseline": "tab:blue", "sigma": "tab:orange"}
    for variant in VARIANTS:
        gens, means, stds = mean_and_std_per_generation(per_variant_rows[variant])
        means_arr = means
        lower = [m - s for m, s in zip(means_arr, stds, strict=True)]
        upper = [m + s for m, s in zip(means_arr, stds, strict=True)]
        ax.plot(gens, means_arr, label=variant, color=colors[variant])
        ax.fill_between(gens, lower, upper, color=colors[variant], alpha=0.2)

    ax.set_xlabel("generation")
    ax.set_ylabel("best fitness (lower is better)")
    ax.set_title("Roulette selection: baseline vs sigma scaling (mean \u00b1 std, 5 seeds)")
    ax.legend()
    fig.tight_layout()

    out_path = RESULTS / "comparison_convergence.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
