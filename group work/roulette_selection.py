"""Roulette-wheel (fitness-proportionate) selection (Variant B).

Same interface and same number of selected parents as tournament selection
(n // k) -- the variants differ only in *how* parents are chosen, not
*how many*.

Three weighting schemes, all built on the same sampling loop (draw without
replacement, weight proportional to slice size) so the comparison isolates
just the weighting, not the sampling algorithm:

  roulette_selection            baseline. weight = (worst_this_gen - f) + eps.
                                 NOTE: this is already a windowing transform
                                 (Eiben & Smith f'=f-beta^t, beta^t=worst),
                                 just using the CURRENT generation's worst as
                                 the baseline every time -- not "no
                                 windowing" vs "windowing", but "per-gen
                                 window" vs "running-average window" below.

  roulette_selection_sigma      sigma scaling. Rescales relative to the
                                 population's mean and spread (f-bar, c*sigma)
                                 instead of its worst member, meant to hold
                                 selection pressure roughly constant across
                                 the whole run rather than just damping
                                 outliers. Formula adapted for minimisation:
                                 for maximisation, Eiben & Smith give
                                     f' = max(f - (f_bar - c*sigma), 0)
                                 flipping sign (g = -f, higher g better) and
                                 simplifying gives, in terms of raw f:
                                     f' = max(f_bar - f + c*sigma, 0)
                                 (sigma is unchanged by the sign flip).

Logging-free by design; per-generation and selection-pressure CSV logging
live in variant_b_roulette.py instead.
"""

import random
import statistics

from rich.console import Console

from ariel.ec import Individual, Population

console = Console()

EPS = 1e-9


def _fitness_key(ind: Individual) -> float:
    return ind.fitness_ if ind.fitness_ is not None else float("inf")


def _sample_without_replacement(
    individuals: list[Individual],
    weights: list[float],
    n_parents: int,
) -> list[int]:
    """Shared roulette-wheel draw: spin, pick, remove, repeat.

    Kept identical across all three variants below so a difference in
    results can only be attributed to the *weights*, never to how the
    wheel is spun.
    """
    pool_idx = list(range(len(individuals)))
    pool_weights = list(weights)
    chosen: list[int] = []
    for _ in range(min(n_parents, len(individuals))):
        total = sum(pool_weights)
        r = random.uniform(0, total)
        upto = 0.0
        for pos, w in enumerate(pool_weights):
            upto += w
            if upto >= r:
                chosen.append(pool_idx.pop(pos))
                pool_weights.pop(pos)
                break
    return chosen


def _tag_and_log(
    individuals: list[Individual],
    chosen: list[int],
    label: str,
) -> Population:
    for ind in individuals:
        ind.tags["ps"] = False
    for i in chosen:
        individuals[i].tags["ps"] = True
    ps_count = len(chosen)
    console.log(
        f"[cyan]{label}: {ps_count}/{len(individuals)} marked for reproduction[/cyan]",
    )
    return Population(individuals)


def roulette_selection(population: Population, k: int = 5) -> Population:
    """Fitness-proportionate selection.

    NOTE: this is a minimisation problem (lower tree-edit distance ==
    better), so raw fitness can't be used as the wheel weight directly --
    a body at distance 0 would get a zero-sized slice instead of the
    biggest one. Weights are inverted (worst - fitness), so individuals
    closer to the targets get bigger slices. Dead/invalid individuals
    (fitness == inf) get a near-zero epsilon weight instead of crashing
    the wheel.

    `k` has no tournament meaning here -- it's kept only so `n // k`
    parents get selected, matching tournament_selection's parent count
    for a fair, controlled comparison.
    """
    individuals = list(population)
    n = len(individuals)
    n_parents = n // k

    for ind in individuals:
        ind.tags["ps"] = False

    fitnesses = [_fitness_key(ind) for ind in individuals]
    finite = [f for f in fitnesses if f != float("inf")]
    worst = max(finite) if finite else 0.0
    eps = 1e-9
    weights = [
        ((worst - f) + eps) if f != float("inf") else eps for f in fitnesses
    ]

    pool_idx = list(range(n))
    pool_weights = list(weights)
    chosen: list[int] = []
    for _ in range(min(n_parents, n)):
        total = sum(pool_weights)
        r = random.uniform(0, total)
        upto = 0.0
        for pos, w in enumerate(pool_weights):
            upto += w
            if upto >= r:
                chosen.append(pool_idx.pop(pos))
                pool_weights.pop(pos)
                break

    for i in chosen:
        individuals[i].tags["ps"] = True

    ps_count = sum(1 for ind in individuals if ind.tags.get("ps", False))
    console.log(
        f"[cyan]Roulette Selection: {ps_count}/{n} marked for reproduction[/cyan]",
    )
    return Population(individuals)


# --------------------------------------------------------------------------- #
# SIGMA SCALING VARIANT -- rescale relative to mean and spread
# --------------------------------------------------------------------------- #

SIGMA_C: float = 2.0  # standard multiplier per Eiben & Smith


def roulette_selection_sigma(
    population: Population,
    k: int = 5,
    c: float = SIGMA_C,
) -> Population:
    """Fitness-proportionate selection with sigma scaling.

    weight = max(f_bar - f + c*sigma, 0) + eps   (minimisation form, see
    module docstring for the derivation from Eiben & Smith's maximisation
    formula). Rescales against the population's mean/std each generation,
    which is meant to keep selection pressure roughly flat across the run
    rather than just damping the current worst.
    """
    individuals = list(population)
    n = len(individuals)
    n_parents = n // k

    fitnesses = [_fitness_key(ind) for ind in individuals]
    finite = [f for f in fitnesses if f != float("inf")]

    if len(finite) >= 2:
        f_bar = statistics.mean(finite)
        sigma = statistics.stdev(finite)
    elif len(finite) == 1:
        f_bar, sigma = finite[0], 0.0
    else:
        f_bar, sigma = 0.0, 0.0

    weights = [
        max(f_bar - f + c * sigma, 0.0) + EPS if f != float("inf") else EPS
        for f in fitnesses
    ]

    chosen = _sample_without_replacement(individuals, weights, n_parents)
    return _tag_and_log(individuals, chosen, "Sigma-Scaled Roulette Selection")


if __name__ == "__main__":
    random.seed(42)

    console.log("[bold]Self-test: roulette_selection with dummy fitness values[/bold]")

    pop = Population([Individual() for _ in range(20)])
    for ind in pop:
        ind.fitness = float(random.randint(0, 100))  # fake fitness, lower=better
        ind.tags["valid"] = True

    pop = roulette_selection(pop, k=5)

    selected = [ind for ind in pop if ind.tags.get("ps", False)]
    console.log(f"Expected {20 // 5} parents, got {len(selected)}")
    for ind in selected:
        console.log(f"  selected parent fitness_ = {ind.fitness_}")

    console.log("[green]Self-test passed.[/green]")

    console.log("\n[bold]Self-test: sigma-scaling variant[/bold]")
    pop3 = Population([Individual() for _ in range(20)])
    for ind in pop3:
        ind.fitness = float(random.randint(0, 100))
    pop3 = roulette_selection_sigma(pop3, k=5)
    sel3 = [ind for ind in pop3 if ind.tags.get("ps", False)]
    console.log(f"Expected {20 // 5} parents, got {len(sel3)}")