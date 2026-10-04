"""Variant A: EA with uniform crossover on controller weights.

Run a seed:      uv run "group work/assignment 2/variant_uniform.py" 42
Watch the best:  uv run "group work/assignment 2/variant_uniform.py" watch
Video the best:  uv run "group work/assignment 2/variant_uniform.py" video
Replay a checkpoint:  ... watch checkpoints/uniform_seed_42/gen_11.npz

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

from ariel.ec import set_seed  # noqa: E402

# EA parameters (sigma and population must be IDENTICAL across variants)
POP_SIZE = 100        # toy for building; real values come from sigma sweep + pilot
GENERATIONS = 100
TOURNAMENT_K = 3
ELITE_COUNT = 2
CROSSOVER_P = 0.5    # per-gene chance to take parent 1's gene
MUTATION_SIGMA = 0.0208
INIT_SIGMA = 0.5

CONDITION = "uniform"


def tournament_pick(pop, fits, rng):
    idx = rng.integers(0, len(pop), TOURNAMENT_K)
    return pop[min(idx, key=lambda i: fits[i])]


def crossover(p1, p2, rng):
    mask = rng.random(p1.size) < CROSSOVER_P
    return np.where(mask, p1, p2)


def diversity(pop):
    """Population diversity: std of each gene across the population, averaged over genes."""
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
    pop = [rng.normal(0.0, INIT_SIGMA, n) for _ in range(POP_SIZE)]
    fits = [core.evaluate(g, n_in, n_out) for g in pop]
    evaluations = POP_SIZE
    best_f = min(fits)
    np.savez(ckpt_dir / "gen_0.npz", flat=pop[int(np.argmin(fits))])
    improvements.append([0, evaluations, best_f])
    print(f"gen   0  best {best_f:.4f}  mean {np.mean(fits):.4f}")

    rows = []
    for gen in range(1, GENERATIONS + 1):
        order = list(np.argsort(fits))
        elites = [pop[i].copy() for i in order[:ELITE_COUNT]]
        elite_fits = [fits[i] for i in order[:ELITE_COUNT]]

        children = []
        while len(children) < POP_SIZE - ELITE_COUNT:
            child = crossover(tournament_pick(pop, fits, rng),
                              tournament_pick(pop, fits, rng), rng)
            children.append(child + rng.normal(0.0, MUTATION_SIGMA, n))

        child_fits = [core.evaluate(c, n_in, n_out) for c in children]
        evaluations += len(children)
        pop, fits = elites + children, elite_fits + child_fits

        gen_best = min(fits)
        if gen_best < best_f:  # new best -> checkpoint for the annotated figure
            best_f = gen_best
            np.savez(ckpt_dir / f"gen_{gen}.npz", flat=pop[int(np.argmin(fits))])
            improvements.append([gen, evaluations, best_f])

        div = diversity(pop)
        rows.append([gen, evaluations, gen_best, float(np.mean(fits)),
                     float(np.std(fits, ddof=1)), div])
        print(f"gen {gen:3d}  best {gen_best:.4f}  mean {np.mean(fits):.4f}  div {div:.4f}")

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

    np.savez(HERE / f"best_{CONDITION}.npz", flat=pop[int(np.argmin(fits))])
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