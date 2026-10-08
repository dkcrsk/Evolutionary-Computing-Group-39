"""Quick summary of the per-seed CSVs in results/arithmetic and results/uniform.

    python check_results.py            # final generation of every seed
    python check_results.py 50         # same, but at generation 50
"""
import csv
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
CONDITIONS = ["arithmetic", "uniform"]
COLS = ["best_fitness", "mean_fitness", "std_fitness", "diversity"]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    gen = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for cond in CONDITIONS:
        files = sorted((RESULTS / cond).glob("seed_*.csv"),
                       key=lambda p: int(p.stem.split("_")[1]))
        print(f"\n{cond}  ({len(files)} seeds)")
        print(f"{'seed':>6} {'gen':>4} {'evals':>6} "
              + " ".join(f"{c:>13}" for c in COLS))
        bests = []
        for path in files:
            rows = load(path)
            if gen is None:
                row = rows[-1]
            else:
                row = next((r for r in rows if int(r["generation"]) == gen), None)
                if row is None:
                    print(f"{path.stem.split('_')[1]:>6}  no generation {gen} "
                          f"(has {len(rows)})")
                    continue
            bests.append(float(row["best_fitness"]))
            vals = " ".join(f"{float(row[c]):13.4f}" if c in row else f"{'-':>13}"
                            for c in COLS)
            print(f"{path.stem.split('_')[1]:>6} {row['generation']:>4} "
                  f"{row['evaluations']:>6} {vals}")
        if bests:
            sd = statistics.stdev(bests) if len(bests) > 1 else 0.0
            print(f"  best_fitness: mean {statistics.mean(bests):.4f}  sd {sd:.4f}  "
                  f"min {min(bests):.4f}  max {max(bests):.4f}")


if __name__ == "__main__":
    main()
