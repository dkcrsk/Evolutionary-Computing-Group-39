"""Shared core for A2: world, robot, controller, and evaluate().

Every variant and the baseline import THIS file, so the experiment has
exactly one copy of the truth. Settings frozen here apply to all runs.

Team decisions (worksheet, Oct 1): inputs = qpos + target vector + sin/cos
time; seeds = 5 per condition (42-46). Fitness: PLAIN distance for now -
pending team decision plain vs delta before the pilot.
"""

from pathlib import Path

import mujoco as mj
import numpy as np
from mujoco import viewer

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

HERE = Path(__file__).parent
VIDEO_DIR = HERE / "videos"

# --- frozen experiment settings (identical in every run of every condition) ---
SPAWN_POS = [0.0, 0.0, 0.1]
TARGET_POSITION = [2.0, 0.0, 0.1]
SIM_DURATION = 15.0
HIDDEN_SIZE = 6

# controller inputs - DECIDED by the team (all three on); genotype length 162
USE_TARGET_INPUT = True   # (dx, dy) to the target -> steering can evolve
USE_TIME_INPUT = True     # sin/cos of time -> a beat for the gait
OMEGA = 2.0 * np.pi       # beat speed: 1 cycle per second


def build_world(mark_target=True):
    world = SimpleFlatWorld()
    if mark_target:
        # purely visual red sphere at the target; no collisions, no physics
        try:
            g = world.spec.worldbody.add_geom()
            g.name = "target_marker"
            g.type = mj.mjtGeom.mjGEOM_SPHERE
            g.size[0] = 0.06
            g.pos = TARGET_POSITION
            g.rgba = [1.0, 0.0, 0.0, 0.9]
            g.contype = 0
            g.conaffinity = 0
        except Exception as err:  # marker must never break an experiment
            print(f"target marker skipped: {err}")
    return world


def build_robot():
    return gecko()


def controller_inputs(data):
    parts = [data.qpos]
    if USE_TARGET_INPUT:
        parts.append(np.array([TARGET_POSITION[0] - data.qpos[0],
                               TARGET_POSITION[1] - data.qpos[1]]))
    if USE_TIME_INPUT:
        t = data.time
        parts.append(np.array([np.sin(OMEGA * t), np.cos(OMEGA * t)]))
    return np.concatenate(parts)


def nn_controller(model, data, weights):
    inputs = controller_inputs(data)
    w1, w2 = weights
    hidden = np.tanh(inputs @ w1)
    outputs = np.tanh(hidden @ w2)
    return outputs * (np.pi / 2)


def get_sizes():
    """Input size, output size and genotype length for the current settings."""
    mj.set_mjcb_control(None)
    world = build_world(mark_target=False)
    robot = build_robot()
    world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    n_in = len(controller_inputs(data))
    n_out = model.nu
    return n_in, n_out, n_in * HIDDEN_SIZE + HIDDEN_SIZE * n_out


def to_matrices(flat, n_in, n_out):
    cut = n_in * HIDDEN_SIZE
    return [flat[:cut].reshape(n_in, HIDDEN_SIZE),
            flat[cut:].reshape(HIDDEN_SIZE, n_out)]


def evaluate(flat, n_in, n_out, mode="simple"):
    """One rollout with the given genotype; returns fitness (lower = better).

    Fitness = PLAIN distance to target in the x-y plane at the end.
    (If the team decides delta distance, only this function changes.)
    """
    weights = to_matrices(flat, n_in, n_out)
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    def callback(m, d):
        d.ctrl[:] = nn_controller(m, d, weights)

    mj.set_mjcb_control(callback)
    if mode == "simple":
        simple_runner(model, data, duration=SIM_DURATION)
    elif mode == "video":
        VIDEO_DIR.mkdir(exist_ok=True)
        recorder = VideoRecorder(output_folder=str(VIDEO_DIR))
        video_renderer(model, data, duration=SIM_DURATION, video_recorder=recorder)
    else:  # "launcher": interactive viewer, runs until you close the window
        viewer.launch(model=model, data=data)
    mj.set_mjcb_control(None)

    end = np.array(data.qpos[0:3])
    return float(np.sqrt((end[0] - TARGET_POSITION[0]) ** 2
                         + (end[1] - TARGET_POSITION[1]) ** 2))