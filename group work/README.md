# Group 39 - Assignment 1: Tournament vs. Roulette-Wheel Selection

Evolving robot morphologies (tree genomes) toward five target bodies with ARIEL
(commit 3c62f63). Two EA variants differ only in parent selection; a
random-search baseline uses the same evaluation budget.

## Layout

- `variant_a_tournament.py` - Variant A: partitioned tournament selection (k=5)
- `variant_b_roulette.py` - Variant B: windowed roulette without replacement
- `variant_random_baseline.py` - random-search baseline (same 10,100-eval budget)
- `analyze_results.py` - sanity checks, all plots, statistics, report tables
- `tournament_selection.py`, `roulette_selection.py`, `random_search.py` - operators (each has a self-test)
- `tree_edit_distance.py`, `target_bodies/` - fitness function and the five targets (course-provided)
- `compare_roulette_variants_5_seeds.py`, `roulette_variant_comparison.md` - paired pilot: windowing vs sigma scaling
- `results/<condition>/seed_N.csv` - per-run logs (generation, evaluations, best_fitness, mean_fitness, mean_modules)
- `results/<condition>/seed_N_selection_pressure.csv` - per-generation mean fitness of the chosen parents
- `analysis_output/` - figures, summary_stats.txt, report_tables.tex, combined CSVs

## Reproduce everything

From the repository root (uv installs dependencies on first run):

```
uv run "group work/variant_a_tournament.py"
uv run "group work/variant_b_roulette.py"
uv run "group work/variant_random_baseline.py"
uv run "group work/analyze_results.py"
```

Each runner executes 10 independent runs (seeds 42-51), ~15-20 min total for all
three. `analyze_results.py` first prints PASS/FAIL sanity checks (10 runs per
condition, 10,100-evaluation budget, 20-module cap, monotone best-so-far), then
regenerates every figure and table in `analysis_output/`.

## Headline result (10 runs per condition)

Tournament 12.46 +/- 0.31, windowed roulette 12.89 +/- 0.29, random 16.87 +/- 0.26
(lower is better). Tournament vs roulette: Mann-Whitney U = 13.5, p = 0.0064.
