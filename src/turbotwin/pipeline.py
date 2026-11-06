"""Config-driven, reproducible experiment pipeline.

    python -m turbotwin.pipeline --config configs/fd001.yaml

Stages (each writes into ``output_dir``):

1. ``calibrate``  - fit the twin on the training fleet, save ``twin.json``.
2. ``validate``   - physics validation: drift signs, held-out sensor reconstruction vs a
   data-driven PCA with the same degrees of freedom, simulated vs real lifetimes,
   and a parameter-recovery study on simulated engines with known damage.
3. ``benchmark``  - physics / ML / hybrid RUL on held-out engines with bootstrap CIs.
4. ``scarcity``   - the same comparison with only n training engines.
5. ``figures``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp

from turbotwin import calibration as cal
from turbotwin.data import download, load_test, load_train
from turbotwin.features import last_rows
from turbotwin.metrics import METRICS, bootstrap_ci, coverage, paired_diff_ci
from turbotwin.models import METHODS, ModelConfig, build_tables, fit_predict
from turbotwin.simulator import FleetSpec, simulate_fleet
from turbotwin.thermo import SENSORS, jacobian
from turbotwin.twin import DigitalTwin, TwinConfig


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return dict(yaml.safe_load(fh))


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def spec_from_twin(twin: DigitalTwin, max_cycles: int = 600) -> FleetSpec:
    assert twin.prior is not None and twin.direction is not None
    p = twin.prior
    return FleetSpec(
        log_ab_mean=(float(p.mean[0]), float(p.mean[1])),
        log_ab_cov=((float(p.cov[0, 0]), float(p.cov[0, 1])), (float(p.cov[1, 0]), float(p.cov[1, 1]))),
        fail_mean=p.fail_mean,
        fail_sd=p.fail_sd,
        direction=(float(twin.direction[0]), float(twin.direction[1])),
        process_sd=p.process_sd,
        max_cycles=max_cycles,
    )


def get_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Real C-MAPSS subset, or a simulated fleet (``data.source: simulated``) for CI."""
    d = cfg["data"]
    if d.get("source", "cmapss") == "cmapss":
        raw = download(d["raw_dir"])
        return load_train(raw, d["subset"]), load_test(raw, d["subset"]), d["subset"]
    # simulated: a fleet drawn from the default (FD001-like) design
    from turbotwin.thermo import EngineDesign

    design = EngineDesign()
    sigma = np.array([0.302, 4.03, 3.97, 0.405, 0.101, 0.303, 0.0201, 0.0999, 0.0613])  # FD001 noise
    spec = FleetSpec()
    tr = simulate_fleet(design, d["n_train"], spec, sigma, 0.5 * sigma, seed=d.get("seed", 0))
    te = simulate_fleet(design, d["n_test"], spec, sigma, 0.5 * sigma, seed=d.get("seed", 0) + 1, truncate=True)
    return tr, te, "simulated"


# ---------------------------------------------------------------------------- validate
def validate(twin: DigitalTwin, train: pd.DataFrame, test: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    assert twin.noise is not None and twin.prior is not None and twin.direction is not None
    out: dict[str, Any] = {}
    w = 1.0 / twin.noise.sigma
    tc = twin.cfg

    # 1. drift signs: model Jacobian along the fault direction vs mean fleet drift
    start, end = cal.engine_drift(train, tc.healthy_cycles, tc.eol_cycles)
    drift = (end - start).mean(axis=0)
    model_drift = twin.predict_sensors(np.array([twin.prior.fail_mean]))[0] - twin.predict_sensors(np.zeros(1))[0]
    out["drift"] = {
        s: {"data": float(a), "model": float(b)} for s, a, b in zip(SENSORS, drift, model_drift, strict=True)
    }
    out["drift_sign_agreement"] = int(np.sum(np.sign(drift) == np.sign(model_drift)))

    # 2. held-out reconstruction (test engines never touch calibration):
    #    physics subspace (1 and 2 health dof) vs PCA subspace learned from training drift
    def deviations(df: pd.DataFrame) -> tuple[np.ndarray, list[Any]]:
        devs, obs = [], []
        for _, g in df.groupby("unit"):
            o = twin.observe(g)
            devs.append(cal.sensor_matrix(g) - o.baseline - twin.predict_sensors(np.zeros(1))[0])
            obs.append(o)
        return np.vstack(devs), obs

    dev_tr, _ = deviations(train)
    dev_te, _obs_te = deviations(test)
    _, _, vt = np.linalg.svd(dev_tr * w, full_matrices=False)
    jac2 = jacobian(twin.design)[:, :2] * w[:, None]
    jac1 = (jacobian(twin.design)[:, :2] @ twin.direction)[:, None] * w[:, None]
    recon: dict[str, dict[str, float]] = {}
    total: dict[str, float] = {}
    dw = dev_te * w
    for name, basis in {
        "physics_1dof": jac1,
        "physics_2dof": jac2,
        "pca_1dof": vt[:1].T,
        "pca_2dof": vt[:2].T,
    }.items():
        coef, *_ = np.linalg.lstsq(basis, dw.T, rcond=None)
        fit = (basis @ coef).T
        r2 = cal.reconstruction_r2(dw, fit)
        recon[name] = dict(zip(SENSORS, map(float, r2), strict=True))
        total[name] = float(1.0 - ((dw - fit) ** 2).sum() / ((dw - dw.mean(axis=0)) ** 2).sum())
    out["heldout_reconstruction_r2"] = recon
    out["heldout_reconstruction_r2_total"] = total

    # 3. simulated vs real lifetimes (generative check of the calibrated damage model)
    spec = spec_from_twin(twin)
    sim = simulate_fleet(
        twin.design, cfg["validate"]["n_sim"], spec, twin.noise.sigma, twin.noise.sigma_between, seed=123
    )
    life_sim = sim.groupby("unit")["cycle"].max().to_numpy()
    life_real = train.groupby("unit")["cycle"].max().to_numpy()
    ks = ks_2samp(life_sim, life_real)
    out["lifetimes"] = {
        "real_mean": float(life_real.mean()),
        "real_sd": float(life_real.std(ddof=1)),
        "sim_mean": float(life_sim.mean()),
        "sim_sd": float(life_sim.std(ddof=1)),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
    }

    # 4. parameter recovery: re-calibrate a fresh twin on simulated engines with known damage
    n_rec = cfg["validate"]["n_recovery"]
    sim_tr = simulate_fleet(twin.design, n_rec, spec, twin.noise.sigma, twin.noise.sigma_between, seed=7)
    sim_te = simulate_fleet(twin.design, n_rec, spec, twin.noise.sigma, twin.noise.sigma_between, seed=8, truncate=True)
    rec = DigitalTwin(TwinConfig.from_dict(cfg["twin"])).fit(sim_tr)
    assert rec.prior is not None and rec.direction is not None
    errs, truth, med, lo, hi = [], [], [], [], []
    for _, g in sim_te.groupby("unit"):
        r = rec.track(g)
        errs.append(r.d_mean - g["damage"].to_numpy())
        truth.append(g["rul"].iloc[-1])
        lo.append(r.rul_q[-1, 0])
        med.append(r.rul_q[-1, 1])
        hi.append(r.rul_q[-1, 2])
    e = np.concatenate(errs)
    t_arr, m_arr = np.array(truth, float), np.array(med, float)
    out["recovery"] = {
        "true": {k: getattr(twin.design, k) for k in cal.COUPLING_PARAMS}
        | {"fail_mean": spec.fail_mean, "dir_eta": spec.direction[0]},
        "recovered": {k: getattr(rec.design, k) for k in cal.COUPLING_PARAMS}
        | {"fail_mean": rec.prior.fail_mean, "dir_eta": float(rec.direction[0])},
        "damage_rmse": float(np.sqrt(np.mean(e**2))),
        "damage_rmse_relative_to_fail": float(np.sqrt(np.mean(e**2)) / spec.fail_mean),
        "rul_rmse_uncapped": float(np.sqrt(np.mean((m_arr - t_arr) ** 2))),
        "rul_80pct_coverage": coverage(t_arr, np.array(lo), np.array(hi)),
    }
    return out


# ---------------------------------------------------------------------------- benchmark
def evaluate(y: np.ndarray, preds: dict[str, np.ndarray], n_boot: int, seed: int) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for m, p in preds.items():
        res[m] = {k: bootstrap_ci(y, p, f, n_boot, seed) for k, f in METRICS.items()}
    res["diff_rmse"] = {
        f"{a}-{b}": paired_diff_ci(y, preds[a], preds[b], METRICS["rmse"], n_boot, seed)
        for a, b in (("hybrid", "ml"), ("hybrid", "physics"), ("ml", "physics"))
        if a in preds and b in preds
    }
    return res


def benchmark(
    train: pd.DataFrame, test: pd.DataFrame, cfg: dict[str, Any], out_dir: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    tcfg, mcfg = TwinConfig.from_dict(cfg["twin"]), ModelConfig.from_dict(cfg["models"])
    tables = build_tables(train, test, tcfg, mcfg)
    preds = fit_predict(tables, mcfg, seed=cfg["seed"])
    idx = last_rows(test)
    pos = test.index.get_indexer(pd.Index(idx))
    y_raw = test["rul"].to_numpy(dtype=float)[pos]
    y = np.minimum(y_raw, mcfg.rul_cap)
    point = {m: preds.point[m][pos] for m in METHODS}
    res = evaluate(y_raw, point, cfg["n_boot"], cfg["seed"])
    res["against_capped_truth"] = {m: METRICS["rmse"](y, p) for m, p in point.items()}
    res["interval_80"] = {
        m: {
            "coverage_capped_truth": coverage(y, preds.lo[m][pos], preds.hi[m][pos]),
            "median_width": float(np.median(preds.hi[m][pos] - preds.lo[m][pos])),
        }
        for m in METHODS
    }
    res["conformal_halfwidth"] = preds.conformal_halfwidth
    res["n_test_engines"] = len(idx)
    table = pd.DataFrame({"unit": test.loc[idx, "unit"].to_numpy(), "rul_true": y_raw})
    for m in METHODS:
        table[f"{m}_pred"] = point[m]
        table[f"{m}_lo"] = preds.lo[m][pos]
        table[f"{m}_hi"] = preds.hi[m][pos]
    table.to_csv(out_dir / "test_predictions.csv", index=False)
    return res, table


def scarcity(train: pd.DataFrame, test: pd.DataFrame, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sc = cfg["scarcity"]
    tcfg, mcfg = TwinConfig.from_dict(cfg["twin"]), ModelConfig.from_dict(cfg["models"])
    units = np.unique(train["unit"])
    idx = last_rows(test)
    pos = test.index.get_indexer(pd.Index(idx))
    y = test["rul"].to_numpy(dtype=float)[pos]
    rows = []
    for n in sc["n_train"]:
        reps = 1 if n >= len(units) else sc["repeats"]
        for rep in range(reps):
            rng = np.random.default_rng(1000 * n + rep)
            pick = rng.choice(units, size=min(n, len(units)), replace=False)
            sub = train[train["unit"].isin(pick)]
            tables = build_tables(sub, test, tcfg, mcfg, oof_twin=n >= 2 * mcfg.n_folds)
            preds = fit_predict(tables, mcfg, seed=rep, conformal=False)
            row: dict[str, Any] = {"n_train": int(n), "rep": rep}
            for m in METHODS:
                row[m] = METRICS["rmse"](y, preds.point[m][pos])
            rows.append(row)
            print(f"  scarcity n={n} rep={rep}: " + ", ".join(f"{m}={row[m]:.2f}" for m in METHODS), flush=True)
    return rows


# ---------------------------------------------------------------------------- main
def run(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    results: dict[str, Any] = {"config": cfg}

    t0 = time.perf_counter()
    train, test, name = get_data(cfg)
    results["dataset"] = {
        "name": name,
        "n_train_engines": int(train["unit"].nunique()),
        "n_test_engines": int(test["unit"].nunique()),
        "n_train_rows": len(train),
        "n_test_rows": len(test),
    }
    twin = DigitalTwin(TwinConfig.from_dict(cfg["twin"])).fit(train)
    twin.save(out_dir / "twin.json")
    results["twin"] = twin.report
    timings["calibrate"] = time.perf_counter() - t0
    print(f"[calibrate] {timings['calibrate']:.1f}s", flush=True)

    stages = cfg.get("stages", ["validate", "benchmark", "scarcity", "figures"])
    if "validate" in stages:
        t0 = time.perf_counter()
        results["validation"] = validate(twin, train, test, cfg)
        timings["validate"] = time.perf_counter() - t0
        print(f"[validate] {timings['validate']:.1f}s", flush=True)
    if "benchmark" in stages:
        t0 = time.perf_counter()
        results["benchmark"], _ = benchmark(train, test, cfg, out_dir)
        timings["benchmark"] = time.perf_counter() - t0
        print(f"[benchmark] {timings['benchmark']:.1f}s", flush=True)
    if "scarcity" in stages:
        t0 = time.perf_counter()
        results["scarcity"] = scarcity(train, test, cfg)
        timings["scarcity"] = time.perf_counter() - t0
        print(f"[scarcity] {timings['scarcity']:.1f}s", flush=True)
    results["timings_s"] = timings
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    if "figures" in stages:
        from turbotwin.plots import make_figures

        make_figures(twin, train, test, results, out_dir)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
