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