"""Figures for the README (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from turbotwin.thermo import SENSORS
from turbotwin.twin import DigitalTwin

COLORS = {"physics": "#1b7837", "ml": "#2166ac", "hybrid": "#b2182b", "data": "#444444"}
LABELS = {"physics": "Physics only (twin)", "ml": "ML only (GBM)", "hybrid": "Hybrid (GBM + twin)"}


def _life_profile(df: pd.DataFrame, col: str, bins: int = 20) -> tuple[np.ndarray, np.ndarray]:
    life = df.groupby("unit")["cycle"].transform("max")
    frac = (df["cycle"] / life).to_numpy()
    b = np.minimum((frac * bins).astype(int), bins - 1)
    base = df.groupby("unit")[col].transform(lambda s: s.iloc[:20].mean()).to_numpy()
    dev = df[col].to_numpy() - base
    centers = (np.arange(bins) + 0.5) / bins
    return centers, np.array([dev[b == i].mean() for i in range(bins)])


def fig_drift(twin: DigitalTwin, train: pd.DataFrame, out: Path, data_label: str = "C-MAPSS FD001") -> None:
    from turbotwin.pipeline import spec_from_twin
    from turbotwin.simulator import simulate_fleet

    assert twin.noise is not None
    sim = simulate_fleet(twin.design, 100, spec_from_twin(twin), twin.noise.sigma, twin.noise.sigma_between, seed=123)
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True)
    for ax, s in zip(axes.ravel(), SENSORS, strict=True):
        x, yr = _life_profile(train, s)
        _, ys = _life_profile(sim, s)
        ax.plot(x, yr, "o-", color=COLORS["data"], ms=3, label=f"{data_label} (observed)")
        ax.plot(x, ys, "-", color=COLORS["physics"], lw=2, label="calibrated simulator")
        ax.set_title(s, fontsize=10)
        ax.axhline(0, color="#bbbbbb", lw=0.8)
    for ax in axes[-1]:
        ax.set_xlabel("fraction of life")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Sensor drift from each engine's own new-engine level, fleet mean", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "drift_real_vs_sim.png", dpi=130)
    plt.close(fig)


def fig_tracking(twin: DigitalTwin, test: pd.DataFrame, out: Path, n: int = 3) -> None:
    lengths = test.groupby("unit")["cycle"].max().sort_values(ascending=False)
    units = lengths.index[[0, len(lengths) // 4, len(lengths) // 2]][:n]
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6.5), sharey="row")
    for j, unit in enumerate(units):
        g = test[test["unit"] == unit].sort_values("cycle")
        o = twin.observe(g)
        r = twin.track(g)
        t = o.cycles
        ax = axes[0, j]
        ax.plot(t, o.z, ".", color="#999999", ms=3, label="gas-path estimate (per cycle)")
        ax.plot(t, r.d_mean, color=COLORS["physics"], label="filtered damage")
        ax.fill_between(t, r.d_mean - 2 * r.d_sd, r.d_mean + 2 * r.d_sd, color=COLORS["physics"], alpha=0.2)
        assert twin.prior is not None
        ax.axhline(twin.prior.fail_mean, color="k", ls="--", lw=0.8, label="fleet failure threshold")
        ax.set_title(f"test engine {unit}")
        ax = axes[1, j]
        ax.plot(t, g["rul"], "k-", lw=1, label="true RUL")
        ax.plot(t, r.rul_q[:, 1], color=COLORS["physics"], label="posterior median")
        ax.fill_between(t, r.rul_q[:, 0], r.rul_q[:, 2], color=COLORS["physics"], alpha=0.2, label="80% interval")
        ax.set_xlabel("cycle")
        ax.set_ylim(0, 350)
    axes[0, 0].set_ylabel("HPC damage D")
    axes[1, 0].set_ylabel("RUL (cycles)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[1, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "twin_tracking.png", dpi=130)
    plt.close(fig)


def fig_benchmark(results: dict[str, Any], table: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    methods = ["physics", "ml", "hybrid"]
    for ax, m in zip(axes[:3], methods, strict=True):
        ax.errorbar(
            table["rul_true"],
            table[f"{m}_pred"],
            yerr=[table[f"{m}_pred"] - table[f"{m}_lo"], table[f"{m}_hi"] - table[f"{m}_pred"]],
            fmt="o",
            ms=3,
            color=COLORS[m],
            ecolor=COLORS[m],
            alpha=0.6,
            elinewidth=0.6,
        )
        ax.plot([0, 150], [0, 150], "k--", lw=0.8)
        rm = results["benchmark"][m]["rmse"]
        ax.set_title(f"{LABELS[m]}\nRMSE {rm[0]:.1f} [{rm[1]:.1f}, {rm[2]:.1f}]", fontsize=9)
        ax.set_xlabel("true RUL")
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 150)
    axes[0].set_ylabel("predicted RUL (80% interval)")
    ax = axes[3]
    vals = [results["benchmark"][m]["rmse"] for m in methods]
    ax.bar(range(3), [v[0] for v in vals], color=[COLORS[m] for m in methods])
    ax.errorbar(range(3), [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]],
                fmt="none", ecolor="k", capsize=4)  # fmt: skip
    ax.set_xticks(range(3), ["physics", "ML", "hybrid"])
    ax.set_ylabel("test RMSE (cycles), 95% CI")
    fig.tight_layout()
    fig.savefig(out / "benchmark.png", dpi=130)
    plt.close(fig)


def fig_scarcity(rows: list[dict[str, Any]], out: Path) -> None:
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for m in ["physics", "ml", "hybrid"]:
        g = df.groupby("n_train")[m]
        ax.plot(g.mean().index, g.mean(), "o-", color=COLORS[m], label=LABELS[m])
        ax.fill_between(g.mean().index, g.min(), g.max(), color=COLORS[m], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(sorted(df["n_train"].unique()), [str(n) for n in sorted(df["n_train"].unique())])
    ax.set_xlabel("run-to-failure engines available for training")
    ax.set_ylabel("test RMSE (cycles)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Data scarcity (mean, band = min-max over subsets)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "scarcity.png", dpi=130)
    plt.close(fig)


def make_figures(
    twin: DigitalTwin, train: pd.DataFrame, test: pd.DataFrame, results: dict[str, Any], out_dir: Path
) -> None:
    out = out_dir / "figures"
    out.mkdir(exist_ok=True, parents=True)
    name = results.get("dataset", {}).get("name", "FD001")
    fig_drift(twin, train, out, "C-MAPSS " + name if name.startswith("FD") else name)
    fig_tracking(twin, test, out)
    if "benchmark" in results:
        fig_benchmark(results, pd.read_csv(out_dir / "test_predictions.csv"), out)
    if "scarcity" in results:
        fig_scarcity(results["scarcity"], out)
