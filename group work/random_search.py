"""Random-search baseline: samples random tree bodies with the same generator
and fitness as the EA, tracks the best-so-far. No selection, no variation."""

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

NUM_MODULES: int = 20
HERE = Path(__file__).parent
TARGET_DIR = HERE / "target_bodies"


def load_targets(target_dir: Path = TARGET_DIR) -> list[Any]:
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


def random_body_fitness(targets: list[Any]) -> tuple[float, int]:
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
    rows: list[dict[str, float]] = []
    best = float("inf")
    block_fits: list[float] = []
    block_mods: list[int] = []

    for i in range(1, evaluations + 1):
        fit, mods = random_body_fitness(targets)
        if fit < best:
            best = fit
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

    assert len(out) == 3
    bests = [r["best_fitness"] for r in out]
    assert bests == sorted(bests, reverse=True)
    for r in out:
        console.log(r)
    console.log("[green]Self-test passed.[/green]")
