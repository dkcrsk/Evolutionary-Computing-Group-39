"""Variant A: Tree-based EA using TOURNAMENT selection.

Research question: tournament vs. roulette-wheel selection (this group's
Variant A vs Variant B). This file implements Variant A end-to-end:
population initialisation, tournament-based parent selection, crossover,
mutation, evaluation against the fixed 5-body target set, and survivor
selection -- wired into ariel.ec's EA engine.

Variant B (roulette-wheel) should share IDENTICAL crossover, mutation,
population size, module budget, and evaluation budget -- only the parent
selection step differs. Who builds Variant B can copy this file and
swap only `tournament_selection` for a `roulette_selection` function with
the same signature (population in, population out, tags["ps"] set) --
everything else stays the same so the comparison isolates the one aspect
you're studying.

Assumes this file lives alongside A1_template_2026.py, tree_edit_distance.py,
target_bodies/, and tournament_selection.py in the same directory.
"""

# Standard library
import copy
import csv
import random
from pathlib import Path
from typing import Any

# Third-party
import numpy as np
from rich.console import Console
from rich.progress import track
from rich.traceback import install

# ARIEL: EA engine
from ariel.ec import EA, EAOperation, EASettings, Individual, Population

# ARIEL: genome + tree operators
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
    validate_tree_depth,
)
from ariel.ec.genotypes.tree.operators import (
    _prune_invalid_edges,  # pyright: ignore[reportPrivateUsage]  -- used this way in the course's own tree example
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

# ARIEL: target loading (same helper as A1_template_2026.py)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)

# Assignment fitness (from tree_edit_distance.py, same as A1_template_2026.py)
from tree_edit_distance import mean_plus_std_tree_edit_distance

# Your already-tested selection function -- reused here rather than
# duplicated, so this file and tournament_selection.py never drift apart.
from tournament_selection import tournament_selection

install()
console = Console()


POP_SIZE: int = 100
BUDGET: int = 100  # generations
NUM_MODULES: int = 20  # module budget per body, matches template
MAX_DEPTH: int = 12  # cap tree depth to control bloat
TOURNAMENT_K: int = 5  # fixed hyperparameter
SEXUAL_REPRODUCTION_RATE: float = 0.5  # chance of crossover

SEEDS: list[int] = [42, 43, 44, 45, 46]

HERE = Path(__file__).parent
TARGET_DIR = HERE / "target_bodies"
DATA = Path.cwd() / "__data__" / Path(__file__).stem
DATA.mkdir(parents=True, exist_ok=True)

RESULTS = HERE / "results" / "variant_a"
RESULTS.mkdir(parents=True, exist_ok=True)
FIELDS = ["generation", "evaluations", "best_fitness", "mean_fitness", "mean_modules"]
SELECTION_FIELDS = ["generation", "evaluations", "mean_parent_fitness"]

rng = np.random.default_rng(SEEDS[0])

_state: dict[str, int] = {"generation": 0, "evaluations": 0}
_rows: list[dict[str, float]] = []
_selection_rows: list[dict[str, float]] = []


def load_targets(target_dir: Path = TARGET_DIR) -> list[Any]:
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


TARGETS: list[Any] = load_targets()





def is_connected_tree(genome: TreeGenome) -> bool:
    if len(genome.nodes) == 0:
        return False
    graph = genome.to_networkx()
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
    if len(roots) != 1:
        return False
    root = roots[0]
    reachable = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for succ in graph.successors(node):
            if succ not in reachable:
                reachable.add(succ)
                stack.append(succ)
    return len(reachable) == graph.number_of_nodes()


def body_fitness(genome: TreeGenome) -> tuple[float, int]:
    """Fitness function for a single body genome. Returns the mean+std"""
    if not is_connected_tree(genome):
        return float("inf"), 0
    body = genome.to_networkx()
    if body.number_of_nodes() == 0:
        return float("inf"), 0
    fitness = mean_plus_std_tree_edit_distance(body, TARGETS)
    return fitness, body.number_of_nodes()





def create_individual() -> Individual:
    """Create a single random individual with a valid genome and no fitness."""
    while True:
        genome = random_tree(max_modules=NUM_MODULES)
        if len(genome.nodes) > 0:
            break
    ind = Individual()
    ind.genotype = genome.to_dict()
    ind.tags["ps"] = False
    return ind


def evaluate(population: Population) -> Population:
    """Evaluate all individuals in the population that require evaluation."""
    to_eval = [ind for ind in population if ind.alive and ind.requires_eval]
    for ind in track(to_eval, description="Evaluating..."):
        genome = TreeGenome.from_dict(ind.genotype)
        fitness, num_modules = body_fitness(genome)
        ind.fitness = fitness
        ind.tags["modules"] = num_modules
        ind.requires_eval = False
        _state["evaluations"] += 1
    return population

def tick_generation(population: Population) -> Population:
    _state["generation"] += 1
    return population

def log_selection_pressure(population: Population) -> Population:
    parents = [
        ind for ind in population
        if ind.tags.get("ps", False) and ind.fitness_ is not None
    ]
    if parents:
        mean_parent_fitness = float(np.mean([ind.fitness_ for ind in parents]))
        _selection_rows.append({
            "generation": _state["generation"],
            "evaluations": _state["evaluations"],
            "mean_parent_fitness": mean_parent_fitness,
        })
    return population


def crossover_bodies(parent1: Individual, parent2: Individual) -> TreeGenome:
    t1 = TreeGenome.from_dict(parent1.genotype)
    t2 = TreeGenome.from_dict(parent2.genotype)
    child1, child2 = crossover_subtree(t1, t2)
    chosen = child1 if rng.random() < 0.5 else child2
    if not is_connected_tree(chosen):
        return TreeGenome.from_dict(random.choice([t1, t2]).to_dict())
    return chosen


def mutate_body(genome: TreeGenome) -> TreeGenome:
    new = copy.deepcopy(genome)
    mutation_type = rng.choice(
        ["point", "subtree", "shrink", "hoist"],
        p=[0.4, 0.4, 0.1, 0.1],
    )
    if mutation_type == "point":
        mutate_replace_node(new)
    elif mutation_type == "subtree":
        mutate_subtree_replacement(new, max_modules=NUM_MODULES)
    elif mutation_type == "shrink":
        mutate_shrink(new)
    elif mutation_type == "hoist":
        mutate_hoist(new)
    _prune_invalid_edges(new)
    return new


def reproduction(population: Population) -> Population:
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    if not parents:
        console.log("[yellow]No parents tagged -- using entire population[/yellow]")
        parents = list(population)

    offspring: list[Individual] = []
    target_pool = POP_SIZE * 2
    while len(population) + len(offspring) < target_pool:
        if len(parents) >= 2 and rng.random() < SEXUAL_REPRODUCTION_RATE:
            p1, p2 = random.sample(parents, 2)
            child_genome = crossover_bodies(p1, p2)
        else:
            parent = random.choice(parents)
            child_genome = TreeGenome.from_dict(parent.genotype)

        child_genome = mutate_body(child_genome)

        """Repair loop: keep mutating until valid, or give up and use a fresh 
        random genome after 20 attempts (rare in practice)."""
        attempts = 0
        while not (
            len(child_genome.nodes) > 0
            and validate_tree_depth(child_genome, MAX_DEPTH)
        ):
            child_genome = mutate_body(child_genome)
            attempts += 1
            if attempts >= 20:
                child_genome = random_tree(max_modules=NUM_MODULES)
                break

        child = Individual()
        child.genotype = child_genome.to_dict()
        child.tags["ps"] = False
        offspring.append(child)

    population.extend(offspring)
    return population


def survivor_selection(population: Population) -> Population:
    population = population.sort(sort="min", attribute="fitness_")
    survivors = population[:POP_SIZE]
    for ind in population:
        if ind not in survivors:
            ind.alive = False
 
    fits = [
        ind.fitness_
        for ind in survivors
        if ind.fitness_ is not None and ind.fitness_ != float("inf")
    ]
    mods = [
        ind.tags.get("modules", 0)
        for ind in survivors
        if ind.fitness_ is not None and ind.fitness_ != float("inf")
    ]
    if fits:
        best_fitness = min(fits)
        mean_fitness = float(np.mean(fits))
        mean_modules = float(np.mean(mods)) if mods else 0.0
        console.log(
            f"[green]Gen {_state['generation']} stats -- best={best_fitness:.4f} "
            f"mean={mean_fitness:.4f} mean_modules={mean_modules:.2f}[/green]",
        )
        _rows.append({
            "generation": _state["generation"],
            "evaluations": _state["evaluations"],
            "best_fitness": best_fitness,
            "mean_fitness": mean_fitness,
            "mean_modules": mean_modules,
        })
    return population


def select_parents(population: Population) -> Population:
    return tournament_selection(population, k=TOURNAMENT_K)

def run_one(seed: int) -> Individual | None:
    
    global rng
    random.seed(seed)
    rng = np.random.default_rng(seed)
 
    
    _state["generation"] = 0
    _state["evaluations"] = 0
    _rows.clear()
    _selection_rows.clear()
 
    settings = EASettings(
        is_maximisation=False,
        num_steps=BUDGET,
        target_population_size=POP_SIZE,
        output_folder=DATA,
        db_file_name=f"variant_a_seed{seed}.db",
    )
 
    population = Population([create_individual() for _ in range(POP_SIZE)])
    population = evaluate(population)
 
    ops = [
        tick_generation,
        EAOperation(select_parents),
        log_selection_pressure,
        EAOperation(reproduction),
        EAOperation(evaluate),
        EAOperation(survivor_selection),
    ]
 
    ea = EA(
        population,
        operations=ops,
        num_steps=settings.num_steps,
        is_maximisation=settings.is_maximisation,
        db_file_path=settings.db_file_path,
        db_handling=settings.db_handling,
        quiet=settings.quiet,
    )
    ea.run()
 
    stats_path = RESULTS / f"seed_{seed}.csv"
    with stats_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(_rows)
    console.log(f"saved {stats_path}")
 
    selection_path = RESULTS / f"seed_{seed}_selection_pressure.csv"
    with selection_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        writer.writerows(_selection_rows)
    console.log(f"saved {selection_path}")
 
    return ea.get_solution("best", only_alive=False)

def main() -> None:
    """Run Variant A over 5 independent seeds, as the assignment requires."""
    console.rule("[bold purple]Variant A: Tournament Selection[/bold purple]")
    console.log(
        f"Population: {POP_SIZE}, Generations: {BUDGET}, "
        f"Module budget: {NUM_MODULES}, Tournament k: {TOURNAMENT_K}",
    )

    for seed in SEEDS:
        console.rule(f"Run seed={seed}")
        best = run_one(seed)
        if best is not None and best.fitness_ is not None:
            console.log(f"Best fitness (seed {seed}): {best.fitness_:.4f}")


if __name__ == "__main__":
    main()