"""Roulette-wheel (fitness-proportionate) selection (Variant B).

Same interface and same number of selected parents as tournament selection
(n // k) -- the variants differ only in *how* parents are chosen, not
*how many*.

Logging-free by design; per-generation and selection-pressure CSV logging
live in variant_b_roulette.py instead.
"""

import random

from rich.console import Console

from ariel.ec import Individual, Population

console = Console()


def _fitness_key(ind: Individual) -> float:
    return ind.fitness_ if ind.fitness_ is not None else float("inf")


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