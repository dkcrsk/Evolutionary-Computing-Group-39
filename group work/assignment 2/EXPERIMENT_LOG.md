# A2 Experiment Log - Group 39
Every combination tried gets a row. Data lives in CSVs; decisions live here.

## Decisions (frozen)
- inputs: qpos + target vector + sin/cos time (team, Oct 1) -> 19 inputs, genotype 162
- seeds: 5 per condition, 42-46 (team, Oct 1)
- fitness: plain distance (assignment default; delta ~identical with fixed spawn)
- body/world: gecko + flat (Tier 1)

## Open
- mutation sigma (sweep below decides)
- budget pop x gens (pilot plateau decides)
- third condition no-crossover (after pilot)

## Measured facts
- 0.22 s/eval with full inputs (Sai's laptop, Oct 1) -> full run (10,100 evals) ~ 37 min
- evaluation is deterministic: same genotype replayed 1.4741 -> 1.4743
- with target-vector input the robot visibly steers toward the target (watch replay)

## Runs tried
| date | what | settings | seeds | result (best) | decision/note |
|---|---|---|---|---|---|
| Oct 1 | toy smoke test, uniform | pop20 x 15g, sigma 0.1 | 42 | 1.4741 | pipeline works; checkpoints + video OK |
| Oct 1 | sigma sweep r1 | pop50 x 40g, sigma 0.05 | 42,43 | 0.5410 / 0.6199 | avg 0.58; s42 plateaued gen 27-40 |
| Oct 1 | sigma sweep r2 | pop50 x 40g, sigma 0.1 | 42,43 | 1.2814 / 0.5962 | avg 0.94; s42 stuck at 1.41 for 16 gens; 0.05 leading |
| Oct 1 | sigma sweep r3 | pop50 x 40g, sigma 0.2 | 42,43 | 1.3532 / 0.9865 | avg 1.17; s42 frozen at gen-0 best for 21 gens - mutation destroying progress |
| Oct 1 | SIGMA FROZEN | 0.05 -> 0.58, 0.1 -> 0.94, 0.2 -> 1.17 (avg best) | 42,43 | sigma = 0.05 | monotone: smaller better on mean AND seed-consistency; tuning effort 6 half-scale runs |
| Oct 2 | GRID uniform s42 | pop100 x 100g, sigma 0.05 | 42 | 0.0179 | plateau ~gen 59; robot reaches the target (1.8 cm) |
| Oct 2 | GRID uniform s43 | pop100 x 100g, sigma 0.05 | 43 | 0.2028 | still improving at gen 94 -> keep 100 gens |
| Oct 2 | BUDGET CONFIRMED | pop 100 x 100 generations | - | - | one full run ~46 min at 0.28 s/eval |

## BODY CORRECTION (2 Oct)
Template imported the 8-hinge gecko from gecko.py; the assignment requires the John Set
body, and john_set.py provides gecko() with 6 hinges. Core import fixed. All runs before
this point used the wrong body and are archived in archive_8hinge/ (kept as pipeline
evidence, excluded from all analysis). New dimensions: 17 inputs, 6 outputs, genotype 138.
Sigma will be re-tuned for the new search space (per-variant, Optuna, seeds 101-102).


| Oct 3 | optuna sigma study (uniform) | 10 trials x seeds 101,102, pop50x40 | - | best sigma 0.0208 (mean 0.658) | small-sigma region wins again; trials csv in results_tuning/ |
| Oct 3 | MISTAKE + archive | ran 42,43 full scale BEFORE freezing sigma (used 0.05) | 42,43 | 0.4250 / 0.3409 | archived in archive_sigma005_fullscale/, excluded from grid; rerun with 0.0208 |
| Oct 3 | GRID uniform s42 | pop100x100, sigma 0.0208, john_set body | 42 | 0.5092 | improving till gen 91; mean 0.81 |
| Oct 3 | GRID uniform s43 | pop100x100, sigma 0.0208, john_set body | 43 | 0.1966 | improving till gen 89; mean 0.77; budget 100 confirmed on new body |
| Oct 3 | note | archived sigma-0.05 runs vs tuned: mixed per seed, tuned better on avg + population means | - | - | utility flat in 0.02-0.05 region |