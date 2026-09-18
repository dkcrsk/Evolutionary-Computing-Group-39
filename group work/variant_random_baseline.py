"""Random-search baseline runner: 10 seeds, same evaluation budget as the EA,
one CSV per seed in results/random/seed_<N>.csv."""

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from random_search import load_targets, random_search

console = Console()

EVALUATIONS: int = 10_100
LOG_EVERY: int = 100
SEEDS: list[int] = [42, 43, 44, 45, 46, 47, 48, 49, 50] 

HERE = Path(__file__).parent
RESULTS = HERE / "results" / "random"
RESULTS.mkdir(parents=True, exist_ok=True)

FIELDS = ["generation", "evaluations", "best_fitness", "mean_fitness", "mean_modules"]


def run_one(seed: int, targets: list[Any]) -> float:
    random.seed(seed)
    np.random.seed(seed)
    rows = random_search(targets, evaluations=EVALUATIONS, log_every=LOG_EVERY)

    out = RESULTS / f"seed_{seed}.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    console.log(f"saved {out}")
    return rows[-1]["best_fitness"]


def main() -> None:
    console.rule("[bold purple]Random-search baseline[/bold purple]")
    console.log(
        f"Evaluations per run: {EVALUATIONS}, log every: {LOG_EVERY}, seeds: {SEEDS}",
    )
    targets = load_targets()

    finals: list[float] = []
    for seed in SEEDS:
        console.rule(f"Run seed={seed}")
        best = run_one(seed, targets)
        finals.append(best)
        console.log(f"Best fitness (seed {seed}): {best:.4f}")

    console.log(
        f"[green]Final best over seeds: mean={np.mean(finals):.3f} "
        f"std={np.std(finals):.3f} min={min(finals):.3f}[/green]",
    )


if __name__ == "__main__":
    main()
