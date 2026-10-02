"""Variant A prototype for A2 : EA with uniform crossover on controller weights.

Run:    uv run "group work/assignment 2/ea_uniform_a2.py" 42
Watch:  uv run "group work/assignment 2/ea_uniform_a2.py" watch
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import mujoco as mj

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import A2_template_2026 as t  # noqa: E402

POP_SIZE = 20
GENERATIONS = 15
TOURNAMENT_K = 3
ELITE_COUNT = 2
CROSSOVER_P = 0.5      # per-gene chance to inherit from parent 1
MUTATION_SIGMA = 0.1   # gaussian noise added to every child gene
INIT_SIGMA = 0.5


def get_sizes():
    mj.set_mjcb_control(None)
    world = t.build_world()
    robot = t.build_robot()
    world.spawn(robot.spec, position=t.SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    return len(data.qpos), model.nu


def to_matrices(flat, input_size, output_size):
    cut = input_size * t.HIDDEN_SIZE
    w1 = flat[:cut].reshape(input_size, t.HIDDEN_SIZE)
    w2 = flat[cut:].reshape(t.HIDDEN_SIZE, output_size)
    return [w1, w2]


def evaluate(flat, input_size, output_size, mode="simple"):
    weights = to_matrices(flat, input_size, output_size)
    mj.set_mjcb_control(None)
    world = t.build_world()
    robot = t.build_robot()
    world.spawn(robot.spec, position=t.SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    def control_callback(m, d):
        d.ctrl[:] = t.nn_controller(m, d, weights)

    initial_position = t.get_core_position(data)
    mj.set_mjcb_control(control_callback)
    if mode == "simple":
        t.simple_runner(model, data, duration=t.SIM_DURATION)
    else:
        t.viewer.launch(model=model, data=data)
    mj.set_mjcb_control(None)
    return t.fitness_function(initial_position, t.get_core_position(data))


def tournament_pick(pop, fits, rng):
    idx = rng.integers(0, len(pop), TOURNAMENT_K)
    winner = min(idx, key=lambda i: fits[i])
    return pop[winner]


def uniform_crossover(p1, p2, rng):
    mask = rng.random(p1.size) < CROSSOVER_P
    return np.where(mask, p1, p2)


def main(seed):
    rng = np.random.default_rng(seed)
    t.set_seed(seed)  # ariel's own generator, separate from ours
    input_size, output_size = get_sizes()
    n = input_size * t.HIDDEN_SIZE + t.HIDDEN_SIZE * output_size
    print(f"seed {seed}  genotype length {n}")

    t0 = time.time()
    pop = [rng.normal(0.0, INIT_SIGMA, n) for _ in range(POP_SIZE)]
    fits = [evaluate(g, input_size, output_size) for g in pop]
    evaluations = POP_SIZE
    print(f"initial population  best {min(fits):.4f}  mean {np.mean(fits):.4f}")

    rows = []
    for gen in range(1, GENERATIONS + 1):
        order = list(np.argsort(fits))
        elites = [pop[i].copy() for i in order[:ELITE_COUNT]]
        elite_fits = [fits[i] for i in order[:ELITE_COUNT]]

        children = []
        while len(children) < POP_SIZE - ELITE_COUNT:
            p1 = tournament_pick(pop, fits, rng)
            p2 = tournament_pick(pop, fits, rng)
            child = uniform_crossover(p1, p2, rng)
            child = child + rng.normal(0.0, MUTATION_SIGMA, n)
            children.append(child)

        child_fits = [evaluate(c, input_size, output_size) for c in children]
        evaluations += len(children)

        pop = elites + children
        fits = elite_fits + child_fits
        best, mean = min(fits), float(np.mean(fits))
        rows.append([gen, evaluations, best, mean])
        print(f"gen {gen:3d}  best {best:.4f}  mean {mean:.4f}  evals {evaluations}")

    elapsed = time.time() - t0
    print(f"\n{evaluations} evaluations in {elapsed:.0f} s"
          f"  ->  {elapsed / evaluations:.2f} s per evaluation")

    out_dir = HERE / "results_test" / "uniform"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"seed_{seed}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "evaluations", "best_fitness", "mean_fitness"])
        writer.writerows(rows)

    best_flat = pop[int(np.argmin(fits))]
    np.savez(HERE / "best_uniform.npz", flat=best_flat)
    print(f"csv saved to {out_dir}/seed_{seed}.csv, best genotype to best_uniform.npz")


def watch():
    flat = np.load(HERE / "best_uniform.npz")["flat"]
    input_size, output_size = get_sizes()
    fitness = evaluate(flat, input_size, output_size, mode="launcher")
    print(f"replayed best, fitness {fitness:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "watch":
        watch()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 42)