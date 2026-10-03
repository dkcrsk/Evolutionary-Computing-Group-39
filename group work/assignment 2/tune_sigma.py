"""Tune mutation sigma with Optuna for one variant.

Runs a compact copy of the EA at half scale (pop 50 x 40 generations) on
tuning seeds 101-102, lets Optuna's TPE sampler propose sigma values, and
minimizes the mean final best fitness. The EA itself is entirely ours;
Optuna only chooses the sigma values to try.

Usage:  uv run "group work/assignment 2/tune_sigma.py"
Ori: change the 'import variant_uniform as v' line to variant_arithmetic.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import optuna

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import a2_core as core  # noqa: E402
import variant_uniform as v  # noqa: E402  <-- Ori: variant_arithmetic

from ariel.ec import set_seed  # noqa: E402

POP_SIZE = 50        # half scale, identical for both variants' tuning
GENERATIONS = 40
TUNE_SEEDS = [101, 102]   # separate from grid seeds 42-46 on purpose
N_TRIALS = 10


def one_run(sigma, seed, n_in, n_out, n):
    rng = np.random.default_rng(seed)
    set_seed(seed)
    pop = [rng.normal(0.0, v.INIT_SIGMA, n) for _ in range(POP_SIZE)]
    fits = [core.evaluate(g, n_in, n_out) for g in pop]
    best = min(fits)
    for _ in range(GENERATIONS):
        order = list(np.argsort(fits))
        elites = [pop[i].copy() for i in order[:v.ELITE_COUNT]]
        elite_fits = [fits[i] for i in order[:v.ELITE_COUNT]]
        children = []
        while len(children) < POP_SIZE - v.ELITE_COUNT:
            child = v.crossover(v.tournament_pick(pop, fits, rng),
                                v.tournament_pick(pop, fits, rng), rng)
            children.append(child + rng.normal(0.0, sigma, n))
        child_fits = [core.evaluate(c, n_in, n_out) for c in children]
        pop, fits = elites + children, elite_fits + child_fits
        best = min(best, min(fits))
    return best


def main():
    n_in, n_out, n = core.get_sizes()
    print(f"tuning sigma for {v.CONDITION}: inputs {n_in} outputs {n_out} genotype {n}")
    rows = []

    def objective(trial):
        sigma = trial.suggest_float("sigma", 0.01, 0.3, log=True)
        t0 = time.time()
        bests = [one_run(sigma, s, n_in, n_out, n) for s in TUNE_SEEDS]
        mean_best = float(np.mean(bests))
        rows.append([trial.number, round(sigma, 5)] + [round(b, 4) for b in bests]
                    + [round(mean_best, 4), int(time.time() - t0)])
        print(f"trial {trial.number:2d}  sigma {sigma:.4f}  "
              f"bests {bests[0]:.4f}/{bests[1]:.4f}  mean {mean_best:.4f}")
        return mean_best

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    out = HERE / "results_tuning"
    out.mkdir(exist_ok=True)
    with open(out / f"{v.CONDITION}_sigma_trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "sigma", "best_seed101", "best_seed102",
                    "mean_best", "seconds"])
        w.writerows(rows)
    print(f"\nBEST sigma for {v.CONDITION}: {study.best_params['sigma']:.4f} "
          f"(mean best {study.best_value:.4f})")
    print(f"trials -> results_tuning/{v.CONDITION}_sigma_trials.csv")


if __name__ == "__main__":
    main()