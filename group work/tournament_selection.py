"""Tournament selection (Variant A) for the group's tree-based EA.

Research question: tournament vs. roulette-wheel selection.
This file implements the tournament-selection side of that comparison.

Drops into the group's shared `ops` list (see new_EC_engine_example.py /
1_body_evolution_tree.py) as a direct replacement for a truncation- or
roulette-based parent_selection step:

    ops = [
        EAOperation(tournament_selection),   # <- this file
        EAOperation(reproduction),           # crossover + mutation (teammate)
        EAOperation(evaluate),               # fitness (shared, from A1 template)
        EAOperation(survivor_selection),     # (teammate)
    ]
"""

import random

from rich.console import Console

from ariel.ec import Individual, Population

console = Console()


def _fitness_key(ind: Individual) -> float:
    
    return ind.fitness_ if ind.fitness_ is not None else float("inf")


def tournament_selection(population: Population, k: int = 5) -> Population:
    
    shuffled = population.shuffle()
    n = len(shuffled)

    for ind in shuffled:
        ind.tags["ps"] = False

    for start in range(0, n - k + 1, k):
        group = [shuffled[start + j] for j in range(k)]
        winner = min(group, key=_fitness_key)
        winner.tags["ps"] = True

    ps_count = sum(1 for ind in shuffled if ind.tags.get("ps", False))
    console.log(
        f"[cyan]Tournament Selection: {ps_count}/{n} marked for reproduction[/cyan]",
    )
    return shuffled




if __name__ == "__main__":
    random.seed(42)

    console.log("[bold]Self-test: tournament_selection with dummy fitness values[/bold]")

    pop = Population([Individual() for _ in range(20)])
    for ind in pop:
        ind.fitness = float(random.randint(0, 100))  # fake fitness, lower=better
        ind.tags["valid"] = True

    pop = tournament_selection(pop, k=5)

    selected = [ind for ind in pop if ind.tags.get("ps", False)]
    console.log(f"Expected {20 // 5} parents, got {len(selected)}")
    for ind in selected:
        console.log(f"  selected parent fitness_ = {ind.fitness_}")

    
    unselected = [ind for ind in pop if not ind.tags.get("ps", False)]
    if selected and unselected:
        assert min(ind.fitness_ for ind in selected) <= min(
            ind.fitness_ for ind in unselected
        ), "a selected parent should never be worse than every unselected individual"
    console.log("[green]Self-test passed.[/green]")
