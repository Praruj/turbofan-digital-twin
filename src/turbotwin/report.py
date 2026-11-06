"""Render ``results.json`` as the Markdown tables quoted in the README.

python -m turbotwin.report results/fd001
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

NAMES = {"physics": "Physics only (twin)", "ml": "ML only (GBM)", "hybrid": "Hybrid (GBM + twin)"}


def _ci(v: list[float], digits: int = 1) -> str:
    return f"{v[0]:.{digits}f} [{v[1]:.{digits}f}, {v[2]:.{digits}f}]"


def render(res: dict[str, Any]) -> str:
    out: list[str] = []
    b = res.get("benchmark")
    if b:
        out.append(f"### RUL on the official held-out test engines (n = {b['n_test_engines']})\n")
        out.append(
            "| Method | RMSE (95% CI) | MAE (95% CI) | PHM08 score (95% CI) | 80% interval coverage | median width |"
        )
        out.append("|---|---|---|---|---|---|")
        for m, name in NAMES.items():
            iv = b["interval_80"][m]
            out.append(
                f"| {name} | {_ci(b[m]['rmse'])} | {_ci(b[m]['mae'])} | {_ci(b[m]['nasa_score'], 0)} "
                f"| {iv['coverage_capped_truth']:.0%} | {iv['median_width']:.0f} |"
            )
        out.append("\n| Paired difference in RMSE | estimate (95% CI) |\n|---|---|")
        for k, v in b["diff_rmse"].items():
            out.append(f"| {k} | {_ci(v, 2)} |")
    s = res.get("scarcity")
    if s:
        df = pd.DataFrame(s)
        agg = df.groupby("n_train")[list(NAMES)].agg(["mean", "min", "max"])
        out.append("\n### Data scarcity: test RMSE vs number of training engines\n")
        out.append("| training engines | subsets | " + " | ".join(NAMES.values()) + " |")
        out.append("|---|---|---|---|---|")
        for n, row in agg.iterrows():
            reps = int((df["n_train"] == n).sum())
            cells = [
                f"{row[(m, 'mean')]:.1f}" + (f" ({row[(m, 'min')]:.1f}-{row[(m, 'max')]:.1f})" if reps > 1 else "")
                for m in NAMES
            ]
            out.append(f"| {n} | {reps} | " + " | ".join(cells) + " |")
    v = res.get("validation")
    if v:
        out.append("\n### Physics validation\n")
        out.append("| Sensor | drift in data (start → end of life) | model drift at mean failure damage |")
        out.append("|---|---|---|")
        for k, d in v["drift"].items():
            out.append(f"| {k} | {d['data']:+.3f} | {d['model']:+.3f} |")
        out.append(f"\nSign agreement: {v['drift_sign_agreement']}/9 sensors.\n")
        r = v["heldout_reconstruction_r2_total"]
        out.append("| Subspace (fit to train, scored on test engines) | variance explained |\n|---|---|")
        for k, val in r.items():
            out.append(f"| {k} | {val:.3f} |")
        lt = v["lifetimes"]
        out.append(
            f"\nLifetimes: real {lt['real_mean']:.0f} ± {lt['real_sd']:.0f}, simulated {lt['sim_mean']:.0f} ± "
            f"{lt['sim_sd']:.0f} cycles; KS D = {lt['ks_stat']:.3f}, p = {lt['ks_p']:.2f}."
        )
        rc = v["recovery"]
        out.append(
            "\nParameter recovery on simulated engines: "
            + ", ".join(f"{k} {rc['true'][k]:.3f} → {rc['recovered'][k]:.3f}" for k in rc["true"])
            + f"; damage RMSE {rc['damage_rmse']:.4f} ({rc['damage_rmse_relative_to_fail']:.0%} of failure damage); "
            f"RUL RMSE {rc['rul_rmse_uncapped']:.1f}; 80% interval coverage {rc['rul_80pct_coverage']:.0%}."
        )
    return "\n".join(out) + "\n"


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results/fd001")
    res = json.loads((folder / "results.json").read_text())
    text = render(res)
    (folder / "tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
