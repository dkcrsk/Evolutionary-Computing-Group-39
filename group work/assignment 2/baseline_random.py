"""Baseline: pure random search (no evolution) at the SAME evaluation budget.

Every genotype is drawn fresh from N(0, INIT_SIGMA) - the same distribution
the EAs use for their initial population - and evaluated with the shared
core.evaluate(). Nothing is selected, crossed or mutated.

Budget matches the EAs exactly: POP_SIZE evaluations for "generation 0",
then (POP_SIZE - ELITE_COUNT) per generation (the EAs do not re-evaluate
their elites), i.e. 100 + 100 * 98 = 9,900 evaluations per seed.

CSV columns match the variants, so the same plotting code works:
  best_fitness  = best-so-far over ALL evaluations up to this generation
                  (the EAs' elitism makes their best_fitness monotone too)
  mean_fitness / std_fitness / diversity = over this generation's batch only

Run a seed:      uv run "group work/assignment 2/baseline_random.py" 42
Watch the best:  uv run "group work/assignment 2/baseline_random.py" watch
Video the best:  uv run "group work/assignment 2/baseline_random.py" video

Final grid (team decision): seeds 42-46, 5 per condition.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import a2_core as core  # noqa: E402
import variant_uniform as v  # noqa: E402  (shared frozen budget settings)

from ariel.ec import set_seed  # noqa: E402

# Budget settings are imported, not retyped, so they cannot drift from the EAs.
POP_SIZE = v.POP_SIZE            # 100
GENERATIONS = v.GENERATIONS      # 100
ELITE_COUNT = v.ELITE_COUNT      # 2  (only used to match the per-gen eval count)
INIT_SIGMA = v.INIT_SIGMA        # 0.5
BATCH = POP_SIZE - ELITE_COUNT   # 98 evals per generation after gen 0

CONDITION = "random"


def diversity(pop):
    """Same definition as the variants: mean over genes of the per-gene std."""
    return float(np.mean(np.std(np.array(pop), axis=0, ddof=1)))


def main(seed):
    rng = np.random.default_rng(seed)
    set_seed(seed)
    n_in, n_out, n = core.get_sizes()
    print(f"{CONDITION}  seed {seed}  inputs {n_in}  outputs {n_out}  genotype {n}")

    ckpt_dir = HERE / "checkpoints" / f"{CONDITION}_seed_{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    improvements = []

    t0 = time.time()
    # generation 0: same size and same sampling as the EAs' initial population
    batch = [rng.normal(0.0, INIT_SIGMA, n) for _ in range(POP_SIZE)]
    fits = [core.evaluate(g, n_in, n_out) for g in batch]
    evaluations = POP_SIZE
    best_f = min(fits)
    best_g = batch[int(np.argmin(fits))].copy()
    np.savez(ckpt_dir / "gen_0.npz", flat=best_g)
    improvements.append([0, evaluations, best_f])
    print(f"gen   0  best {best_f:.4f}  mean {np.mean(fits):.4f}")

    rows = []
    for gen in range(1, GENERATIONS + 1):
        batch = [rng.normal(0.0, INIT_SIGMA, n) for _ in range(BATCH)]
        fits = [core.evaluate(g, n_in, n_out) for g in batch]
        evaluations += len(batch)

        if min(fits) < best_f:  # new best-so-far -> checkpoint
            best_f = min(fits)
            best_g = batch[int(np.argmin(fits))].copy()
            np.savez(ckpt_dir / f"gen_{gen}.npz", flat=best_g)
            improvements.append([gen, evaluations, best_f])

        div = diversity(batch)
        rows.append([gen, evaluations, best_f, float(np.mean(fits)),
                     float(np.std(fits, ddof=1)), div])
        print(f"gen {gen:3d}  best {best_f:.4f}  batch mean {np.mean(fits):.4f}")

    elapsed = time.time() - t0
    print(f"\n{evaluations} evals in {elapsed:.0f} s -> {elapsed/evaluations:.2f} s/eval")

    out_dir = HERE / "results" / CONDITION
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"seed_{seed}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "evaluations", "best_fitness",
                    "mean_fitness", "std_fitness", "diversity"])
        w.writerows(rows)
    with open(ckpt_dir / "checkpoints.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "evaluations", "best_fitness"])
        w.writerows(improvements)

    np.savez(HERE / f"best_{CONDITION}.npz", flat=best_g)
    print(f"csv -> results/{CONDITION}/seed_{seed}.csv   "
          f"checkpoints -> {ckpt_dir.name}/")


def replay(mode, path=None):
    flat = np.load(path if path else HERE / f"best_{CONDITION}.npz")["flat"]
    n_in, n_out, _ = core.get_sizes()
    f = core.evaluate(flat, n_in, n_out, mode=mode)
    print(f"replayed, fitness {f:.4f}"
          + ("  (video saved in videos/)" if mode == "video" else ""))


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "42"
    if a in ("watch", "video"):
        replay("launcher" if a == "watch" else "video",
               sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        main(int(a))
