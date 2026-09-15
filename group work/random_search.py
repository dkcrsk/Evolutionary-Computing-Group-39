"""Random-search baseline for the group's Assignment 1 experiment.

Samples random tree bodies with the SAME generator the EA uses for its
initial population, scores each with the SAME fitness (mean + std tree
edit distance to the 5 targets), and tracks the best-so-far.
No selection, no variation -- this is the "no evolution" control.

Logs one row every `log_every` evaluations so it lines up with the EA's
generations (population 100 -> log every 100).

Assumes this file lives alongside tree_edit_distance.py and target_bodies/
(same folder as the variant A / B files).
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec.genotypes.tree.operators import random_tree
from tree_edit_distance import mean_plus_std_tree_edit_distance

console = Console()

NUM_MODULES: int = 20  # same module budget as the EA
HERE = Path(__file__).parent
TARGET_DIR = HERE / "target_bodies"


def load_targets(target_dir: Path = TARGET_DIR) -> list[Any]:
    """Load the fixed target bodies (identical for every run/seed/variant)."""
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


def random_body_fitness(targets: list[Any]) -> tuple[float, int]:
    """Make ONE random body and return (fitness, number_of_modules).

    Uses random_tree with the module budget, exactly like the EA's
    create_individual(). Empty trees are re-sampled.
    """
    while True:
        genome = random_tree(max_modules=NUM_MODULES)
        if len(genome.nodes) > 0:
            break
    body = genome.to_networkx()
    fitness = mean_plus_std_tree_edit_distance(body, targets)
    return fitness, body.number_of_nodes()


def random_search(
    targets: list[Any],
    evaluations: int = 10_100,
    log_every: int = 100,
) -> list[dict[str, float]]:
    """Run random search and return one log row per `log_every` evaluations.

    Row keys (the group's shared CSV schema):
        generation, evaluations, best_fitness, mean_fitness, mean_modules
    """
    rows: list[dict[str, float]] = []
    best = float("inf")
    block_fits: list[float] = []
    block_mods: list[int] = []

    for i in range(1, evaluations + 1):
        fit, mods = random_body_fitness(targets)
        if fit < best:
            best = fit  # best-so-far, never goes up
        block_fits.append(fit)
        block_mods.append(mods)

        if i % log_every == 0:
            rows.append({
                "generation": i // log_every,
                "evaluations": i,
                "best_fitness": best,
                "mean_fitness": float(np.mean(block_fits)),
                "mean_modules": float(np.mean(block_mods)),
            })
            block_fits.clear()
            block_mods.clear()

    return rows


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    console.log("[bold]Self-test: 300 random bodies, log every 100[/bold]")
    t = load_targets()
    out = random_search(t, evaluations=300, log_every=100)

    assert len(out) == 3, "expected 3 log rows"
    bests = [r["best_fitness"] for r in out]
    assert bests == sorted(bests, reverse=True), "best_fitness must never go up"
    for r in out:
        console.log(r)
    console.log("[green]Self-test passed.[/green]")
