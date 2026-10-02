# A2 notes — Group 39

Research question: **does the crossover operator (uniform vs arithmetic) affect the final solution quality of evolved walking controllers at a fixed budget?**

Status on Oct 2. Deadline: **13 Oct 2026, 09:00** (−0.5 points per day late).

---

## Meeting of Oct 2 — what was agreed

- **Hypothesis**
    - Uniform crossover reaches a *lower* final distance to the target than arithmetic crossover.
    - To support or reject it we need, per condition
        - Mean of the final best fitness over the seeds.
        - Standard deviation over the seeds.
        - A p-value for the difference between the two conditions.

- **Budget**
    - Fixed evaluation budget across all runs.
    - Fixed number of generations: **100**.

- **Body**
    - Recheck the body type: we must use the gecko from the **John Set**.
    - Needs the one-line change in `a2_core.py` (see 8.1); not applied yet.

- **Arithmetic crossover**
    - Weighted at **0.5** (child = midpoint of the two parents).
    - Sigma: find the best value for arithmetic separately, using **Optuna**.
        - Optuna is not installed in the project environment yet.
        - It is used only to tune sigma, not as the search algorithm.
    - **Optuna settings (decided)**

        | Setting | Value |
        |---|---|
        | Sigma range | 0.01 to 0.5 |
        | Trials | 3 |
        | Seeds per trial | 42 and 43 |
        | Scale of each run | pop 50 × 40 generations (1,970 evaluations) |

        - Cost: 3 trials × 2 seeds = 6 runs, about 16 min each, so about 1.6 hours.
        - Same effort as Sai's sweep for uniform (also 6 half-scale runs).
        - Watch out: with only 3 trials Optuna does not "learn" yet.
            - Its default sampler picks the first 10 trials at random.
            - So the 3 sigmas would be 3 random values in the range.
            - Alternative with the same cost: give Optuna 3 fixed values spread over the range (e.g. 0.01, 0.07, 0.5).

- **Extra measurement: population diversity**
    - Logged per generation.
    - Computed as a mean per gene.
        - Suggested definition: std of each gene across the population, then the mean over all genes.

- **Who does what**
    - Ori: works on the arithmetic variant and tests seeds **42 and 43**.

---

## 1. What the assignment asks

- **Task**
    - Evolve the weights of a neural-network controller.
    - Fixed body from the "John Set", fixed world.
    - The robot must walk from its spawn position to a fixed target.

- **Research question**
    - Investigate *one* aspect of the EA (ours: crossover).
    - Everything else stays identical between conditions.

- **Fitness**
    - Euclidean distance in the ground plane between the core's final position and the target.
    - Lower is better.

- **Hard requirements**
    - At least 5 independent runs per condition, reported as mean and spread.
    - A baseline (e.g. random search) at the *same evaluation budget*.
    - A line plot across generations with average and std over the runs.
    - Stop at the plateau of the fitness curve, not at an arbitrary generation count.
    - Seed is a command-line argument.

- **Forbidden**
    - Editing anything in `src/ariel` (counts as fraud).
    - CPG-based controllers.
    - Off-the-shelf optimizers (nevergrad, CMA-ES libraries).
    - `SimpleTiltedWorld`.

- **Hand-in**
    - `groupnumber.zip` containing a folder with the report (`groupnumber.pdf`) and the code.
    - Report: max 6 pages, GECCO19 template; intro, methods, results and discussion, conclusions, bibliography.
    - The grade is based on the report; performance itself is not graded.

---

## 2. Files in this folder

| File / folder | What it is | Who |
|---|---|---|
| `A2_template_2026.py` | Course starter: one robot, random weights, no evolution | course |
| `ea_uniform_a2.py` | First prototype of the uniform EA, built on the template | Sai |
| `a2_core.py` | Shared world, robot, controller, fitness. **Do not edit without a ping** | Sai |
| `variant_uniform.py` | Condition A: EA with uniform crossover | Sai |
| `variant_arithmetic.py` | Condition B: same EA with arithmetic crossover | me |
| `EXPERIMENT_LOG.md` | Every run tried, and every frozen decision | Sai |
| `results/<condition>/seed_N.csv` | Per-generation statistics of the full-scale runs | — |
| `checkpoints/<condition>_seed_N/` | Best genotype saved at every improvement | — |
| `results_sweep/` | The 6 half-scale runs used to choose sigma | Sai |
| `results_test/` | Output of the old prototype (not used in the report) | Sai |

---

## 3. `a2_core.py` — the shared part

One copy of the truth: every variant imports it, so all conditions run on exactly the same problem.

- **Frozen settings**
    - Body: gecko.
    - World: `SimpleFlatWorld`.
    - Spawn: `(0, 0, 0.1)`.
    - Target: `(2, 0, 0.1)`, so 2 m straight ahead.
    - Simulation: 15 s per evaluation.
    - Hidden layer: 6 neurons.

- **Controller inputs (19 in total)**
    - `qpos`, 15 values: core position (3), core orientation (4), joint angles (8).
    - Target vector, 2 values: `(dx, dy)` from the robot to the target.
        - Lets steering evolve; without it the robot walks blindly past the target.
    - Clock, 2 values: `sin` and `cos` of time at 1 cycle per second.
        - Gives the network a beat to build a gait on.
        - It is only an input signal, not a CPG (approved by the TA).

- **Network**
    - `inputs (19) → tanh hidden (6) → tanh output (8)`.
    - Outputs are scaled by π/2, giving joint targets in [−π/2, π/2].
    - No bias terms.

- **Genotype**
    - One flat vector of real numbers.
    - Length = 19 × 6 + 6 × 8 = **162**.
    - `to_matrices()` cuts it back into the two weight matrices.

- **`evaluate(flat, n_in, n_out, mode)`**
    - Builds a fresh world and robot, runs one rollout, returns the fitness.
    - Fitness = plain distance to the target in the x-y plane at the end.
    - Modes
        - `simple`: headless, used during evolution.
        - `video`: saves an mp4 in `videos/`.
        - `launcher`: interactive viewer.
    - Deterministic: the same genotype always gives the same fitness.

- **Target marker**
    - A red sphere at the target position.
    - Visual only (no collisions), so it cannot change the results.

---

## 4. `variant_uniform.py` — the EA (Sai)

- **Parameters**

    | Parameter | Value |
    |---|---|
    | Population | 100 |
    | Generations | 100 |
    | Tournament size | 3 |
    | Elites | 2 |
    | Crossover | uniform, `CROSSOVER_P = 0.5` |
    | Mutation sigma | 0.05 |
    | Initial sigma | 0.5 |

- **One run, step by step**
    1. Initialise 100 genotypes from a normal distribution N(0, 0.5).
    2. Evaluate all of them (generation 0).
    3. Each generation
        - Copy the 2 best unchanged (elitism), reusing their known fitness.
        - Create 98 children
            - Pick 2 parents, each by a tournament of 3.
            - Apply crossover to get *one* child.
            - Add Gaussian noise N(0, 0.05) to *every* gene.
        - Evaluate the 98 children.
        - New population = 2 elites + 98 children.
    4. After 100 generations, write the outputs.

- **The operators**
    - Parent selection: tournament
        - Draw 3 random individuals (with replacement), keep the best.
    - Uniform crossover
        - Each gene is taken from parent 1 with probability 0.5, otherwise from parent 2.
        - The child holds only values that already exist in the parents.
    - Mutation
        - Always applied, to all 162 genes; there is no mutation probability.
    - Survivor selection
        - Generational replacement with 2 elites.

- **Budget**
    - 100 + 100 × 98 = **9,900 evaluations** per run.
    - `EXPERIMENT_LOG.md` says 10,100; the code gives 9,900 because elites are not re-evaluated.
    - The baseline must use the number the code actually spends.

- **Outputs**
    - `results/uniform/seed_N.csv`
        - Columns: generation, evaluations, best, mean, std of the population.
        - Starts at generation 1; generation 0 is only in `checkpoints.csv`.
    - `checkpoints/uniform_seed_N/gen_G.npz`
        - Best genotype, saved each time the best fitness improves.
    - `checkpoints/uniform_seed_N/checkpoints.csv`
        - Generation, evaluations and fitness of each improvement.
    - `best_uniform.npz`
        - Best genotype of the last run; overwritten by each new run.

- **Commands** (from the repo root)
    - Run a seed: `uv run "group work/assignment 2/variant_uniform.py" 42`
    - Watch the best: `... variant_uniform.py" watch`
    - Video of the best: `... variant_uniform.py" video`
    - Replay a checkpoint: `... watch checkpoints/uniform_seed_42/gen_11.npz`

---

## 5. Decisions Sai froze (from `EXPERIMENT_LOG.md`)

- **Sigma = 0.05**
    - Sweep: 0.05 / 0.1 / 0.2, seeds 42 and 43, at half scale (pop 50 × 40 generations).

    | Sigma | Best, seed 42 | Best, seed 43 | Average |
    |---|---|---|---|
    | 0.05 | 0.54 | 0.62 | 0.58 |
    | 0.1 | 1.28 | 0.60 | 0.94 |
    | 0.2 | 1.35 | 0.99 | 1.17 |

    - Smaller was better on both the average and the consistency between seeds.
    - With 0.2 the mutation destroyed progress (seed 42 stuck at its generation-0 best for 21 generations).
    - 2 seeds are enough to pick a value; the 5 seeds are for the final result.

- **Budget = pop 100 × 100 generations**
    - Seed 42 plateaued around generation 59.
    - Seed 43 was still improving at generation 94, so 100 generations are kept.

- **Seeds = 42 to 46**, 5 per condition.

- **Fitness = plain distance**
    - The "delta distance" variant gives almost the same ranking because the spawn is fixed.

- **Gecko + flat world first**
    - Only crossover changes, so any difference is attributable to it.
    - Spider and Olympic arena come afterwards as a "does the winner generalise?" bonus.

- **Uniform results so far (full scale)**
    - Seed 42: best 0.0179 (the robot reaches the target, 1.8 cm away).
    - Seed 43: best 0.2028.
    - Seeds 44 to 46: not run yet.

---

## 6. `variant_arithmetic.py` — what was just done

- **How it was made**
    - Copy of `variant_uniform.py`.
    - Only three things differ: the crossover function, its knob, and the condition name.

- **The crossover**
    ```python
    ALPHA = 0.5

    def crossover(p1, p2, rng):
        return ALPHA * p1 + (1.0 - ALPHA) * p2
    ```
    - Whole arithmetic recombination: every gene of the child is a weighted average of the two parents.
    - With `ALPHA = 0.5` the child is the midpoint of its parents.

- **Uniform vs arithmetic, the idea to explain in the report**
    - Uniform
        - Mixes existing gene values; creates no new values.
        - Keeps the spread of the population.
    - Arithmetic
        - Creates new values that lie between the parents.
        - Pulls the population toward its centre, so diversity shrinks faster.
        - Only mutation pushes the population outward again.

- **What is identical in both variants**
    - Sigma, population, generations, tournament size, elites, initialisation.
    - The initial population for a given seed (same seed, same random draws), so both conditions start from the same generation 0.

- **Checks done**
    - Smoke test (pop 6 × 2 generations, written to a scratch folder): CSV, checkpoints and best genotype are produced correctly.
    - Evaluation is deterministic on this laptop too.
    - `a2_core.py` was not touched.

- **Speed on this laptop**
    - Power-saver profile: 2.28 s per evaluation, about 6.4 h per run.
    - Performance profile: 0.50 s per evaluation, about 85 min per run.
    - Before running: `powerprofilesctl set performance`.
    - Sai's laptop: 0.22 to 0.28 s per evaluation.

- **Status**
    - Seed 42 is running at full scale (started Oct 2).
    - Nothing is committed yet.

---

## 7. Still open

- **Settled in the Oct 2 meeting**
    - Alpha: fixed 0.5.
    - Stopping: fixed 100 generations.
    - Body: gecko from the John Set.
    - Arithmetic seeds 42 and 43: Ori.
    - Optuna settings: range 0.01 to 0.5, 3 trials, seeds 42 and 43, pop 50 × 40 generations.

- **For the group to decide**
    - Who runs arithmetic seeds 44 to 46?
    - A third condition without crossover (listed as open in the log)?
    - Sigma for uniform: keep 0.05 from the old body, or re-tune it on the John Set gecko the same way as arithmetic?
    - Optuna with 3 trials: random values (default) or 3 fixed values over the range?

- **The Optuna settings, explained** (background for the decision above)
    - How Optuna works
        - We give it a function: "run the EA with this sigma, return the final best fitness".
        - It proposes a sigma, looks at the result, and proposes a better one. Each attempt is a *trial*.
    - Choice 1: the sigma range
        - The interval Optuna is allowed to search.
        - Sai's sweep tried 0.05 / 0.1 / 0.2 and the *smallest* won, so the best value may lie below 0.05.
        - Arithmetic shrinks the population, so its best sigma may be larger.
        - Suggestion: 0.01 to 0.5, searched on a log scale.
    - Choice 2: the number of trials
        - More trials find a better sigma but cost more time.
        - Total cost = trials × seeds per trial × time of one run.
        - Suggestion: 10 trials (one parameter does not need more).
    - Choice 3: the seeds
        - One run is noisy: with the same sigma, seed 42 gave 1.28 and seed 43 gave 0.60.
        - With one seed per trial Optuna may pick a sigma that was only lucky.
        - Suggestion: 2 seeds per trial, and Optuna scores the *average* of the two.
        - Cleaner: tune on seeds that are not the final ones (e.g. 1 and 2), so the final seeds 42 to 46 stay unseen.
    - Choice 4: the scale of each run
        - Full scale (pop 100 × 100 generations) takes about 80 min per run.
            - 10 trials × 2 seeds × 80 min ≈ 27 hours: not feasible.
        - Half scale, as in Sai's sweep (pop 50 × 40 generations = 1,970 evaluations), takes about 16 min per run.
            - 10 trials × 2 seeds × 16 min ≈ 5.3 hours per condition.
            - Less if several runs go in parallel on the 4 cores.
        - Risk: the best sigma at half scale is not guaranteed to be the best at full scale.
    - Fairness
        - Uniform's sigma came from 6 runs on the old body.
        - If arithmetic gets 20 runs with Optuna on the new body, arithmetic is now the better-tuned one.
        - Suggestion: run the *same* Optuna procedure for both conditions and report the tuning effort.
    - Diversity: add the logging to *both* variants so the CSV columns match.

- **Missing for the assignment**
    - Baseline: random search with the same 9,900 evaluations, 5 seeds.
    - Uniform seeds 44, 45, 46.
    - Arithmetic seeds 43 to 46.
    - The line plot: mean and std of the best fitness over the 5 seeds, per generation, one line per condition.
    - A statistical comparison of the final best fitness between conditions.

---

## 8. To check with the TA

Ordered by how much a "no" would cost us. The first three could force re-running everything, so ask them before the remaining seeds are run.

---

### 8.1 Could invalidate the runs

- **Is our gecko a "John Set" body?**
    - The assignment says the body must come from `prebuilt_robots.john_set`.
    - `a2_core.py` imports `prebuilt_robots.gecko`, the same one the course template uses.
    - These are two different robots
        - Ours: 8 joints, 15 `qpos` values.
        - `john_set.gecko`: 6 joints, 13 `qpos` values.
    - Ask: is the template's gecko accepted, or must we switch to the `john_set` one?
    - If we must switch: genotype length changes (162 → 138) and all runs are redone.
    - **Our position (Oct 2): switch to the `john_set` one. Not done yet; Sai has to agree first because it changes `a2_core.py`.**
        - The change is one line in `a2_core.py`
            - `from ...prebuilt_robots.gecko import gecko` → `from ...prebuilt_robots.john_set import gecko`
        - Measured with the `john_set` gecko, without editing the shared file
            - 17 inputs, 6 outputs, genotype 138.
            - 0.48 s per evaluation on this laptop (the old body: 0.50).
            - Evaluation stays deterministic.
            - 30 random controllers: best 1.78, mean 1.97 (standing still scores 2.00).
        - What becomes invalid
            - Uniform seeds 42 and 43, and every arithmetic run so far.
            - The sigma sweep, since 0.05 was chosen on the old body; decide whether to redo it.
            - Old checkpoints (162 genes) cannot be replayed on the new body.

- **Is the EA "built on `ariel.ec`" enough?**
    - Our variants use only `set_seed` from it; the loop and the operators are plain numpy.
    - The template expects `EA`, `Individual`, `Population`, and the SQLite logging that comes with them.
    - `ariel.ec` has a ready `uniform` crossover but no arithmetic one.
    - Ask
        - Is a hand-written loop acceptable, or is it a grading penalty?
        - If we must use `ariel.ec`, may we add our own arithmetic operator next to their uniform one?
        - Are our CSV files an acceptable replacement for their database?

- **Is the "generalisation bonus" allowed at all?**
    - Plan: repeat the comparison on spider + Olympic arena after the main results.
    - The assignment says body and world are "fixed for your whole assignment".
    - Ask: is a small secondary experiment on another body/world allowed, or does it break that rule?
    - Remember A1: we lost 0.2 for mixing questions.

---

### 8.2 Fairness of the comparison

- **Sigma was tuned with uniform crossover only**
    - 0.05 was chosen from uniform runs, then reused for arithmetic.
    - Arithmetic shrinks the population, so it may need a *larger* sigma to do well.
    - Risk: a reviewer says "uniform won because the settings were tuned for it".
    - Ask: same sigma for both (simple, one variable changes), or tune sigma per condition (fairer, equal tuning effort for each)?
    - Cheap middle road: run the same 3-value sweep for arithmetic and report it.

- **Which arithmetic crossover?**
    - Fixed alpha 0.5 (midpoint), or alpha drawn at random per child?
    - Whole-vector blend (ours), or blending only some genes?
    - Ask: is comparing against one specific version fine, as long as we name it precisely?

- **One child per pair of parents**
    - Textbook crossover returns two children; ours returns one.
    - With alpha 0.5 the two children would be identical anyway.
    - Ask: is it worth a sentence in the methods, or irrelevant?

- **No crossover probability**
    - Crossover is applied to every child, and mutation to every gene.
    - Ask: is a third "mutation only" condition expected to show that crossover matters at all?

---

### 8.3 Budget and stopping

- **Fixed 100 generations vs "run until plateau"**
    - The assignment says the plateau is the stopping criterion, not a fixed generation count.
    - Seed 42 plateaued near generation 59; seed 43 was still improving at generation 94.
    - Ask: is "100 generations, chosen from a pilot" acceptable, or do we need a formal plateau rule (e.g. no improvement for N generations)?

- **What counts as the budget?**
    - Our runs spend 9,900 evaluations (elites are not re-evaluated).
    - Ask: is "same budget" judged in evaluations (what we assume) or in generations?

---

### 8.4 Baseline

- **Which baseline?**
    - Random search, or a fixed non-evolved controller? The assignment allows both.

- **If random search**
    - Sample from the same N(0, 0.5) as our initial population?
    - 9,900 samples per seed, 5 seeds, same seeds 42 to 46?

- **How to put it on a per-generation plot**
    - It has no generations.
    - Proposal: best-so-far after every 98 evaluations, so its x-axis lines up with the EA.
    - Ask: is that the expected way to draw it?

---

### 8.5 Statistics and the plot

- **What goes on the y-axis?**
    - Best fitness of the generation, or the population mean?
    - "Average/std over your independent runs": average over the 5 seeds of the per-run best, correct?

- **Is 5 seeds enough to claim a difference?**
    - Which test do they expect: Mann-Whitney U, or a paired test?
    - Our seeds are paired (same seed gives the same initial population in both conditions).
    - A paired Wilcoxon test with 5 pairs cannot go below p = 0.0625, so it can never be "significant" at 0.05.
    - Ask: should we run more seeds (e.g. 10), or is reporting mean, spread and effect size enough?

- **Spread**
    - Standard deviation as the assignment says, or are median and quartiles accepted (fitness is skewed: 0.018 vs 0.20 already)?

---

### 8.6 Controller and fitness

- **Clock input**
    - sin/cos of time was approved verbally as "not a CPG".
    - Ask for it in writing (mail or forum), and state it explicitly in the methods.

- **Absolute position as input**
    - `qpos` contains the robot's world position, and we also feed the vector to the target.
    - With a fixed target these carry the same information twice.
    - Ask: any objection, or just mention it?

- **Fitness is measured only at the end**
    - A robot that reaches the target and then walks past it scores badly.
    - Ask: is plain final distance fine, or do they prefer a variant from `targeted_locomotion`?

- **Deterministic evaluation**
    - Same genotype always gives the same fitness, so the only randomness is in the EA.
    - Ask: is that fine, or do they expect noise (e.g. varied spawn)?

---

### 8.7 Hand-in and report

- **What goes in the zip?**
    - Only code and report, or also result CSVs, checkpoints and videos?
    - Is there a size limit?

- **Reproducibility**
    - Is "one command per seed, per condition" enough, or do they want one script that runs everything?

- **Report**
    - Do the sigma sweep and the pilot belong in the 6 pages, or may they go in an appendix?
    - Are figures of the robot / a video link welcome, or wasted space?

- **AI tools**
    - Check the course policy on using AI assistance for code and notes, and whether it must be declared.
