"""Combine results from all three conditions and produce the report's
statistical analysis: convergence line plot, boxplot, selection-pressure
plot, descriptive stats, and a significance test comparing tournament
(Variant A) vs. roulette (Variant B).

Expects this layout (already produced by your teammates' scripts):

    results/
        random/       seed_42.csv ... seed_46.csv
        variant_a/    seed_42.csv ... seed_46.csv  (+ seed_N_selection_pressure.csv)
        variant_b/    seed_42.csv ... seed_46.csv  (+ seed_N_selection_pressure.csv)

Each seed_*.csv has: generation, evaluations, best_fitness, mean_fitness,
mean_modules. Each seed_*_selection_pressure.csv has: generation,
evaluations, mean_parent_fitness -- only variant_a/variant_b have these
(the random baseline has no selection step).

Usage
-----
    uv run analyze_results.py

Outputs (written to analysis_output/):
    combined_results.csv          -- every run, every condition, one table
    convergence_plot.png          -- fitness vs. evaluations, mean +/- std band
    final_fitness_boxplot.png     -- final best_fitness distribution per condition
    module_count_plot.png         -- bonus: mean_modules vs. evaluations per condition
    selection_pressure_plot.png   -- mean_parent_fitness vs. evaluations, tournament vs. roulette
    summary_stats.txt             -- descriptive stats + significance test
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from rich.console import Console

console = Console()

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
OUT_DIR = HERE / "analysis_output"
OUT_DIR.mkdir(exist_ok=True)

# Maps condition key -> (folder name under results/, display label).
# Order here controls plotting/legend order: tournament, roulette side by
# side, baseline last as the reference line.
CONDITIONS = {
    "tournament": ("variant_a", "Tournament selection"),
    "roulette": ("variant_b", "Roulette-wheel selection"),
    "baseline": ("random", "Random baseline"),
}

# Only these two have a selection step (and therefore a selection-pressure log).
SELECTION_CONDITIONS = {
    "tournament": ("variant_a", "Tournament selection"),
    "roulette": ("variant_b", "Roulette-wheel selection"),
}


def load_condition(folder_name: str, variant_label: str) -> pd.DataFrame | None:
    """Load every seed_<N>.csv (not the *_selection_pressure ones) under
    results/<folder_name>/, tagged with variant.
    """
    folder = RESULTS_DIR / folder_name
    if not folder.exists():
        console.log(f"[yellow]Skipping '{variant_label}': {folder} does not exist yet[/yellow]")
        return None

    # seed_42.csv etc, but NOT seed_42_selection_pressure.csv
    files = sorted(
        f for f in folder.glob("seed_*.csv")
        if "_selection_pressure" not in f.stem
    )
    if not files:
        console.log(f"[yellow]Skipping '{variant_label}': no seed_<N>.csv files found in {folder}[/yellow]")
        return None

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["variant"] = variant_label
        df["seed"] = int(f.stem.split("_")[1])
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    console.log(f"[green]Loaded {len(files)} run(s) for '{variant_label}' ({len(combined)} rows)[/green]")
    return combined


def load_selection_pressure(folder_name: str, variant_label: str) -> pd.DataFrame | None:
    """Load every seed_<N>_selection_pressure.csv under results/<folder_name>/,
    tagged with variant. Only tournament/roulette have these.
    """
    folder = RESULTS_DIR / folder_name
    if not folder.exists():
        return None

    files = sorted(folder.glob("seed_*_selection_pressure.csv"))
    if not files:
        console.log(
            f"[yellow]No selection-pressure logs found for '{variant_label}' in {folder}[/yellow]",
        )
        return None

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["variant"] = variant_label
        # filename like seed_42_selection_pressure.csv -> seed number is stem.split("_")[1]
        df["seed"] = int(f.stem.split("_")[1])
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    console.log(
        f"[green]Loaded {len(files)} selection-pressure log(s) for '{variant_label}' "
        f"({len(combined)} rows)[/green]",
    )
    return combined


def make_convergence_plot(df: pd.DataFrame) -> None:
    """Fitness vs. evaluations, mean +/- std band, one line per variant."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]
        grouped = sub.groupby("evaluations")["best_fitness"].agg(["mean", "std"]).reset_index()
        ax.plot(grouped["evaluations"], grouped["mean"], label=variant)
        ax.fill_between(
            grouped["evaluations"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15,
        )

    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best fitness (mean ± std across runs, lower is better)")
    ax.set_title("Convergence: fitness vs. evaluations")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = OUT_DIR / "convergence_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.log(f"[green]Saved {out_path}[/green]")


def make_module_count_plot(df: pd.DataFrame) -> None:
    """Bonus plot: mean_modules vs. evaluations, one line per variant.

    Shows the bloat-control mechanism, not just the fitness outcome.
    """
    if "mean_modules" not in df.columns:
        console.log("[yellow]No mean_modules column found -- skipping module count plot[/yellow]")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]
        grouped = sub.groupby("evaluations")["mean_modules"].mean().reset_index()
        ax.plot(grouped["evaluations"], grouped["mean_modules"], label=variant)

    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Mean module count")
    ax.set_title("Body size over the course of evolution")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = OUT_DIR / "module_count_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.log(f"[green]Saved {out_path}[/green]")


def make_selection_pressure_plot(df: pd.DataFrame) -> None:
    """mean_parent_fitness vs. evaluations, mean +/- std band, tournament vs.
    roulette. Shows how strongly each mechanism favours fitter parents over
    the course of the run -- lower mean parent fitness = stronger pressure
    toward the current best individuals (lower is better in this task).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]
        grouped = (
            sub.groupby("evaluations")["mean_parent_fitness"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.plot(grouped["evaluations"], grouped["mean"], label=variant)
        ax.fill_between(
            grouped["evaluations"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15,
        )

    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Mean parent fitness (mean ± std across runs, lower is better)")
    ax.set_title("Selection pressure: mean fitness of chosen parents")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path = OUT_DIR / "selection_pressure_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.log(f"[green]Saved {out_path}[/green]")


def make_final_fitness_boxplot(df: pd.DataFrame) -> pd.DataFrame:
    """Boxplot of each run's FINAL best_fitness, one box per variant, in the
    order variants first appear in df["variant"] (i.e. CONDITIONS order).

    Returns the per-run final-fitness table (variant, seed, best_fitness),
    since the significance test needs the same numbers.
    """
    finals = (
        df.sort_values("evaluations")
        .groupby(["variant", "seed"])
        .tail(1)[["variant", "seed", "best_fitness"]]
        .reset_index(drop=True)
    )

    # preserve CONDITIONS order rather than pandas' default (often alphabetical)
    order = [label for _, label in CONDITIONS.values() if label in finals["variant"].unique()]

    fig, ax = plt.subplots(figsize=(6, 5))
    data = [finals[finals["variant"] == v]["best_fitness"].to_numpy() for v in order]
    ax.boxplot(data, tick_labels=order)
    ax.set_ylabel("Final best fitness (lower is better)")
    ax.set_title("Final fitness distribution per condition")
    fig.tight_layout()
    out_path = OUT_DIR / "final_fitness_boxplot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.log(f"[green]Saved {out_path}[/green]")

    return finals


def run_significance_test(finals: pd.DataFrame, label_a: str, label_b: str) -> str:
    """Shapiro-Wilk normality check, then t-test or Mann-Whitney U.

    Returns a human-readable summary block.
    """
    a = finals[finals["variant"] == label_a]["best_fitness"].to_numpy()
    b = finals[finals["variant"] == label_b]["best_fitness"].to_numpy()

    lines = [f"Comparing '{label_a}' (n={len(a)}) vs. '{label_b}' (n={len(b)})", ""]

    if len(a) < 3 or len(b) < 3:
        lines.append("Too few runs for a normality check (need >= 3 per group). "
                      "Defaulting to the non-parametric Mann-Whitney U test.")
        stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        lines.append(f"Mann-Whitney U: U={stat:.3f}, p={p:.4f}")
    else:
        sw_a = stats.shapiro(a)
        sw_b = stats.shapiro(b)
        lines.append(f"Shapiro-Wilk normality -- {label_a}: W={sw_a.statistic:.3f}, p={sw_a.pvalue:.4f}")
        lines.append(f"Shapiro-Wilk normality -- {label_b}: W={sw_b.statistic:.3f}, p={sw_b.pvalue:.4f}")

        if sw_a.pvalue > 0.05 and sw_b.pvalue > 0.05:
            lines.append("Both groups look roughly normal (p > 0.05) -- using a two-sided t-test.")
            t_stat, p = stats.ttest_ind(a, b, equal_var=False)
            lines.append(f"Welch's t-test: t={t_stat:.3f}, p={p:.4f}")
        else:
            lines.append("At least one group deviates from normality -- using the "
                          "non-parametric Mann-Whitney U test instead.")
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            lines.append(f"Mann-Whitney U: U={stat:.3f}, p={p:.4f}")

    lines.append("")
    lines.append(f"{label_a} final fitness: mean={a.mean():.4f}, std={a.std():.4f}")
    lines.append(f"{label_b} final fitness: mean={b.mean():.4f}, std={b.std():.4f}")

    return "\n".join(lines)


def main() -> None:
    console.rule("[bold purple]Combining results[/bold purple]")

    parts = []
    for key, (folder, label) in CONDITIONS.items():
        part = load_condition(folder, label)
        if part is not None:
            parts.append(part)

    if not parts:
        console.log("[red]No results found anywhere under results/. Run the experiments first.[/red]")
        return

    combined = pd.concat(parts, ignore_index=True)
    combined_path = OUT_DIR / "combined_results.csv"
    combined.to_csv(combined_path, index=False)
    console.log(f"[green]Saved combined table: {combined_path} ({len(combined)} rows)[/green]")

    console.rule("[bold purple]Plots[/bold purple]")
    make_convergence_plot(combined)
    make_module_count_plot(combined)
    finals = make_final_fitness_boxplot(combined)

    console.rule("[bold purple]Selection pressure[/bold purple]")
    pressure_parts = []
    for key, (folder, label) in SELECTION_CONDITIONS.items():
        part = load_selection_pressure(folder, label)
        if part is not None:
            pressure_parts.append(part)

    if pressure_parts:
        pressure_combined = pd.concat(pressure_parts, ignore_index=True)
        pressure_combined.to_csv(OUT_DIR / "combined_selection_pressure.csv", index=False)
        make_selection_pressure_plot(pressure_combined)
    else:
        console.log(
            "[yellow]No selection-pressure logs found for either variant -- skipping that plot[/yellow]",
        )

    console.rule("[bold purple]Descriptive stats[/bold purple]")
    desc = finals.groupby("variant")["best_fitness"].agg(["mean", "std", "min", "max", "count"])
    console.log(desc)

    summary_lines = ["DESCRIPTIVE STATS (final best_fitness per variant)", "", desc.to_string(), ""]

    tournament_label = CONDITIONS["tournament"][1]
    roulette_label = CONDITIONS["roulette"][1]
    present = set(finals["variant"].unique())

    if tournament_label in present and roulette_label in present:
        console.rule("[bold purple]Significance test: tournament vs. roulette[/bold purple]")
        test_summary = run_significance_test(finals, tournament_label, roulette_label)
        console.log(test_summary)
        summary_lines.append("SIGNIFICANCE TEST: TOURNAMENT vs. ROULETTE")
        summary_lines.append("")
        summary_lines.append(test_summary)
    else:
        missing = {tournament_label, roulette_label} - present
        console.log(
            f"[yellow]Skipping significance test -- missing results for: {', '.join(missing)}. "
            "Run those experiments first.[/yellow]",
        )
        summary_lines.append(f"Significance test skipped -- missing results for: {', '.join(missing)}")

    summary_path = OUT_DIR / "summary_stats.txt"
    summary_path.write_text("\n".join(summary_lines))
    console.log(f"[green]Saved {summary_path}[/green]")

    console.rule("[bold purple]Done[/bold purple]")
    console.log(f"All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
