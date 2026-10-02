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
    "roulette": ("variant_b_baseline", "Windowed roulette selection"),
    "baseline": ("random", "Random baseline"),
}

# Only these two have a selection step (and therefore a selection-pressure log).
SELECTION_CONDITIONS = {
    "tournament": ("variant_a", "Tournament selection"),
    "roulette": ("variant_b_baseline", "Windowed roulette selection"),
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
        grouped = sub.groupby("generation")["best_fitness"].agg(["mean", "std"]).reset_index()
        ax.plot(grouped["generation"], grouped["mean"], label=variant)
        ax.fill_between(
            grouped["generation"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15,
        )

    ax.set_xlabel("Generation (100 evaluations each)")
    ax.set_ylabel("Best fitness (mean ± std across runs, lower is better)")
    ax.set_title("Convergence: fitness vs. generation")
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
        grouped = sub.groupby("generation")["mean_modules"].agg(["mean", "std"]).reset_index()
        ax.plot(grouped["generation"], grouped["mean"], label=variant)
        ax.fill_between(
            grouped["generation"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15,
        )

    ax.set_xlabel("Generation")
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
            sub.groupby("generation")["mean_parent_fitness"]
            .agg(["mean", "std"])
            .reset_index()
        )
        ax.plot(grouped["generation"], grouped["mean"], label=variant)
        ax.fill_between(
            grouped["generation"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15,
        )

    ax.set_xlabel("Generation")
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
    for i, d in enumerate(data, start=1):
        ax.scatter([i] * len(d), d, color="black", s=18, zorder=3, alpha=0.7)
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
            u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
            lines.append(f"Mann-Whitney U (robustness): U={u_stat:.3f}, p={u_p:.4f}")
        else:
            lines.append("At least one group deviates from normality -- using the "
                          "non-parametric Mann-Whitney U test instead.")
            stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            lines.append(f"Mann-Whitney U: U={stat:.3f}, p={p:.4f}")

    lines.append("")
    lines.append(f"{label_a} final fitness: mean={a.mean():.4f}, std={a.std(ddof=1):.4f}")
    lines.append(f"{label_b} final fitness: mean={b.mean():.4f}, std={b.std(ddof=1):.4f}")

    return "\n".join(lines)


def make_pressure_gap_plot(res: pd.DataFrame, pres: pd.DataFrame) -> None:
    """Selection intensity as a gap: population mean fitness minus chosen-parents
    mean fitness. Higher gap = parents are better than an average population
    member = stronger pressure. Unlike raw parent fitness, this controls for
    the two variants' populations sitting at different fitness levels."""
    merged = res.merge(pres, on=["variant", "seed", "generation"], suffixes=("", "_p"))
    merged["gap"] = merged["mean_fitness"] - merged["mean_parent_fitness"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in merged["variant"].unique():
        sub = merged[merged["variant"] == variant]
        grouped = sub.groupby("generation")["gap"].agg(["mean", "std"]).reset_index()
        ax.plot(grouped["generation"], grouped["mean"], label=variant)
        ax.fill_between(
            grouped["generation"],
            grouped["mean"] - grouped["std"],
            grouped["mean"] + grouped["std"],
            alpha=0.15,
        )
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Population mean - parent mean (higher = stronger pressure)")
    ax.set_title("Selection intensity of the chosen parents")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = OUT_DIR / "selection_pressure_gap_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    console.log(f"[green]Saved {out_path}[/green]")


def make_report_tables(combined: pd.DataFrame) -> None:
    """Emit report-ready LaTeX tables into analysis_output/report_tables.tex.

    Table 1: FPS transform pilot (windowing vs sigma scaling, paired seeds),
    built only if results/variant_b_sigma exists. Table 2: main comparison.
    Re-running the analysis after new seeds keeps the tables current.
    """
    L = []
    finals = (
        combined.sort_values("generation").groupby(["variant", "seed"]).tail(1)
        [["variant", "seed", "best_fitness"]]
    )

    sigma_dir = RESULTS_DIR / "variant_b_sigma"
    base_dir = RESULTS_DIR / "variant_b_baseline"
    if sigma_dir.exists() and base_dir.exists():
        def _finals(folder):
            out = {}
            for f in sorted(folder.glob("seed_*.csv")):
                if "pressure" in f.name:
                    continue
                out[int(f.stem.split("_")[1])] = float(pd.read_csv(f)["best_fitness"].iloc[-1])
            return out

        sig = _finals(sigma_dir)
        win_all = _finals(base_dir)
        seeds = sorted(k for k in sig if k in win_all)
        w = [win_all[k] for k in seeds]
        g = [sig[k] for k in seeds]
        wil = stats.wilcoxon(w, g)
        wins_w = sum(1 for a, b in zip(w, g) if a < b)
        bs = chr(92)
        L.append("% ---- Table 1: FPS transform pilot ----")
        L.append(bs + "begin{table}[t]")
        L.append(bs + "caption{Selecting the FPS transform: windowed vs. sigma-scaled roulette ($c{=}2$), paired over seeds " + f"{seeds[0]}--{seeds[-1]}" + ". All other components identical; lower fitness is better.}")
        L.append(bs + "label{tab:roulette-pick}")
        L.append(bs + "begin{tabular}{lcc}")
        L.append(bs + "toprule")
        L.append(" & Windowing & Sigma scaling " + bs + bs)
        L.append(bs + "midrule")
        L.append(f"Mean final fitness & {sum(w)/len(w):.2f} & {sum(g)/len(g):.2f} " + bs + bs)
        L.append(f"Std across seeds & {pd.Series(w).std(ddof=1):.2f} & {pd.Series(g).std(ddof=1):.2f} " + bs + bs)
        L.append(f"Per-seed wins & {wins_w}/{len(seeds)} & {len(seeds)-wins_w}/{len(seeds)} " + bs + bs)
        L.append(f"Best single run & {min(w):.2f} & {min(g):.2f} " + bs + bs)
        L.append(bs + "midrule")
        L.append(bs + "multicolumn{3}{l}{Wilcoxon signed-rank (paired): " + f"$W{{=}}{wil.statistic:.1f}$, $p{{=}}{wil.pvalue:.2f}$" + "} " + bs + bs)
        L.append(bs + "bottomrule")
        L.append(bs + "end{tabular}")
        L.append(bs + "end{table}")
        L.append("")

    bs = chr(92)
    n_max = int(finals.groupby("variant")["seed"].count().max())
    L.append("% ---- Table 2: main comparison ----")
    L.append(bs + "begin{table}[t]")
    L.append(bs + "caption{Final best fitness (mean tree-edit distance $+$ std to the five targets; lower is better) over " + str(n_max) + " independent runs per condition, equal budget of 10{,}100 evaluations.}")
    L.append(bs + "label{tab:main}")
    L.append(bs + "begin{tabular}{lcccc}")
    L.append(bs + "toprule")
    L.append("Condition & Mean & Std & Best & $n$ " + bs + bs)
    L.append(bs + "midrule")
    for _, label in CONDITIONS.values():
        vals = finals[finals["variant"] == label]["best_fitness"]
        if len(vals) == 0:
            continue
        L.append(f"{label} & {vals.mean():.2f} & {vals.std(ddof=1):.2f} & {vals.min():.2f} & {len(vals)} " + bs + bs)
    L.append(bs + "bottomrule")
    L.append(bs + "end{tabular}")
    L.append(bs + "end{table}")
    out_path = OUT_DIR / "report_tables.tex"
    out_path.write_text(chr(10).join(L) + chr(10))
    console.log(f"[green]Saved {out_path}[/green]")


def sanity_checks(df: pd.DataFrame) -> None:
    """Verify the experiment's own rules before trusting any plot or test."""
    ok = True

    def flag(cond: bool, msg: str) -> None:
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + msg)
        ok = ok and cond

    variants = list(df["variant"].unique())
    flag(len(variants) == 3, f"3 conditions present (found {len(variants)}: {variants})")

    for v, sub in df.groupby("variant"):
        seeds = list(sub["seed"].unique())
        flag(len(seeds) >= 5, f"{v}: at least 5 independent runs (found {len(seeds)})")
        for s, run in sub.groupby("seed"):
            flag(len(run) in (100, 101), f"{v} seed {s}: full run logged (found {len(run)} rows)")
            b = list(run.sort_values("generation")["best_fitness"])
            flag(all(x >= y - 1e-9 for x, y in zip(b, b[1:])), f"{v} seed {s}: best-so-far never increases")
        flag(sub["evaluations"].max() <= 10_100, f"{v}: within the 10,100-evaluation budget (max {sub['evaluations'].max()})")
        flag(sub["mean_modules"].max() <= 21.0 + 1e-9, f"{v}: body size within 20 modules + core (max {sub['mean_modules'].max():.2f})")

    print("ALL CHECKS PASSED" if ok else ">>> SOME CHECKS FAILED - do not use these results <<<")

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
    sanity_checks(combined)
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
        make_pressure_gap_plot(combined, pressure_combined)
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

    make_report_tables(combined)

    summary_path = OUT_DIR / "summary_stats.txt"
    summary_path.write_text("\n".join(summary_lines))
    console.log(f"[green]Saved {summary_path}[/green]")

    console.rule("[bold purple]Done[/bold purple]")
    console.log(f"All outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
