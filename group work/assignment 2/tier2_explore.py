"""Tier 2 exploration: other John Set bodies and the OlympicArena.

Uses the SAME EA as variant_uniform (its crossover, selection, frozen
sigma) so only body/world change. Writes to results_tier2/ - never mixes
with the main grid.

Probe a combo:   edit BODY / WORLD below, then
                 uv run "group work/assignment 2/tier2_explore.py" 42
Watch / video:   ... tier2_explore.py watch   (or: video)
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import a2_core as core  # noqa: E402
import variant_uniform as v  # noqa: E402  (shared frozen EA settings)

from ariel.ec import set_seed  # noqa: E402

BODY = "spider"        # "gecko" or "spider"
WORLD = "flat"     # "flat" or "olympic"
POP_SIZE = 50         # probe scale; bump to 100 x 100 for real Tier 2 runs
GENERATIONS = 40

CONDITION = f"tier2_{BODY}_{WORLD}"


def main(seed):
    rng = np.random.default_rng(seed)
    set_seed(seed)
    n_in, n_out, n = core.get_sizes(body=BODY, world=WORLD)
    print(f"{CONDITION}  seed {seed}  inputs {n_in}  outputs {n_out}  "
          f"genotype {n}  sigma {v.MUTATION_SIGMA}")

    ckpt = HERE / "checkpoints_tier2" / f"{CONDITION}_seed_{seed}"
    ckpt.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pop = [rng.normal(0.0, v.INIT_SIGMA, n) for _ in range(POP_SIZE)]
    fits = [core.evaluate(g, n_in, n_out, body=BODY, world=WORLD) for g in pop]
    evaluations = POP_SIZE
    best_f = min(fits)
    np.savez(ckpt / "gen_0.npz", flat=pop[int(np.argmin(fits))])
    print(f"gen   0  best {best_f:.4f}  mean {np.mean(fits):.4f}")

    rows = []
    for gen in range(1, GENERATIONS + 1):
        order = list(np.argsort(fits))
        elites = [pop[i].copy() for i in order[:v.ELITE_COUNT]]
        elite_fits = [fits[i] for i in order[:v.ELITE_COUNT]]
        children = []
        while len(children) < POP_SIZE - v.ELITE_COUNT:
            child = v.crossover(v.tournament_pick(pop, fits, rng),
                                v.tournament_pick(pop, fits, rng), rng)
            children.append(child + rng.normal(0.0, v.MUTATION_SIGMA, n))
        child_fits = [core.evaluate(c, n_in, n_out, body=BODY, world=WORLD)
                      for c in children]
        evaluations += len(children)
        pop, fits = elites + children, elite_fits + child_fits
        gen_best = min(fits)
        if gen_best < best_f:
            best_f = gen_best
            np.savez(ckpt / f"gen_{gen}.npz", flat=pop[int(np.argmin(fits))])
        rows.append([gen, evaluations, gen_best, float(np.mean(fits)),
                     float(np.std(fits, ddof=1))])
        print(f"gen {gen:3d}  best {gen_best:.4f}  mean {np.mean(fits):.4f}")

    print(f"\n{evaluations} evals in {time.time()-t0:.0f} s")
    out = HERE / "results_tier2" / CONDITION
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"seed_{seed}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "evaluations", "best_fitness",
                    "mean_fitness", "std_fitness"])
        w.writerows(rows)
    np.savez(HERE / f"best_{CONDITION}.npz", flat=pop[int(np.argmin(fits))])
    print(f"csv -> results_tier2/{CONDITION}/seed_{seed}.csv")


def replay(mode, path=None):
    flat = np.load(path if path else HERE / f"best_{CONDITION}.npz")["flat"]
    n_in, n_out, _ = core.get_sizes(body=BODY, world=WORLD)
    f = core.evaluate(flat, n_in, n_out, mode=mode, body=BODY, world=WORLD)
    print(f"replayed, fitness {f:.4f}")


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "42"
    if a in ("watch", "video"):
        replay("launcher" if a == "watch" else "video",
               sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        main(int(a))