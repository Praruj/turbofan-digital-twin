"""Three RUL estimators compared on the same held-out engines.

* ``physics`` - the digital twin alone: posterior median RUL from the calibrated
  gas-path model + damage filter (no learned regressor).
* ``ml`` - gradient-boosted trees on rolling sensor features (no physics).
* ``hybrid`` - the same trees given the twin's posterior summaries as extra features.

Leakage control: training rows get twin features from a twin calibrated on the *other*
folds (engine-grouped K-fold), so neither the failure threshold nor the wear prior has
seen the engine whose row is being featurised. Test rows use a twin calibrated on all
training engines. Intervals for the learned models are split-conformal, from
engine-grouped out-of-fold residuals, computed separately per band of predicted RUL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from turbotwin.features import TWIN_FEATURES, ml_features, twin_features
from turbotwin.twin import DigitalTwin, TwinConfig

METHODS: tuple[str, ...] = ("physics", "ml", "hybrid")


@dataclass
class ModelConfig:
    rul_cap: float = 125.0
    window: int = 30
    n_folds: int = 5
    interval: float = 0.8
    # Mondrian conformal: separate residual quantiles per band of *predicted* RUL, because
    # errors near end of life are much smaller than for young engines
    conformal_bins: tuple[float, ...] = (40.0, 80.0)
    gbm: dict[str, Any] = field(
        default_factory=lambda: {
            "max_iter": 300,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 40,
            "l2_regularization": 1.0,
        }
    )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        kw = dict(d)
        if "conformal_bins" in kw:
            kw["conformal_bins"] = tuple(float(v) for v in kw["conformal_bins"])
        return cls(**kw)


def conformal_halfwidths(
    resid: np.ndarray, pred_cal: np.ndarray, pred_new: np.ndarray, bins: tuple[float, ...], level: float
) -> tuple[np.ndarray, list[float]]:
    """Split-conformal half-widths, one quantile per band of predicted RUL (Mondrian).

    ``resid`` are absolute out-of-fold residuals at predictions ``pred_cal``; returns the
    half-width for every ``pred_new`` and the per-band quantiles.
    """
    band_cal = np.digitize(pred_cal, bins)
    band_new = np.digitize(pred_new, bins)
    qs = []
    for b in range(len(bins) + 1):
        r = resid[band_cal == b]
        n = len(r)
        qs.append(float(np.quantile(r, min(1.0, np.ceil((n + 1) * level) / n))) if n else float(np.max(resid)))
    return np.asarray(qs)[band_new], qs


def _gbm(cfg: ModelConfig, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=seed, **cfg.gbm)


def _folds(units: np.ndarray, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    uniq = np.unique(units)
    k = min(n_folds, len(uniq))
    return list(GroupKFold(n_splits=k).split(units, groups=units))


@dataclass
class Tables:
    x_train: pd.DataFrame
    y_train: np.ndarray
    units_train: np.ndarray
    x_test: pd.DataFrame
    ml_cols: list[str]
    twin: DigitalTwin


def build_tables(
    train: pd.DataFrame, test: pd.DataFrame, twin_cfg: TwinConfig, cfg: ModelConfig, oof_twin: bool = True
) -> Tables:
    f_train = ml_features(train, cfg.window)
    f_test = ml_features(test, cfg.window)
    ml_cols = [c for c in f_train.columns if c not in ("unit",)]

    units = train["unit"].to_numpy()
    tw_train = pd.DataFrame(index=train.index, columns=list(TWIN_FEATURES), dtype=float)
    if oof_twin and len(np.unique(units)) >= 2 * cfg.n_folds:
        for fit_idx, feat_idx in _folds(units, cfg.n_folds):
            fold_twin = DigitalTwin(twin_cfg).fit(train.iloc[fit_idx])
            part = train.iloc[feat_idx]
            tw_train.loc[part.index] = twin_features(fold_twin, part, cfg.rul_cap)
    twin = DigitalTwin(twin_cfg).fit(train)
    if not (oof_twin and len(np.unique(units)) >= 2 * cfg.n_folds):
        tw_train = twin_features(twin, train, cfg.rul_cap)
    tw_test = twin_features(twin, test, cfg.rul_cap)

    x_train = pd.concat([f_train[ml_cols], tw_train.astype(float)], axis=1)
    x_test = pd.concat([f_test[ml_cols], tw_test], axis=1)
    y = np.minimum(train["rul"].to_numpy(dtype=float), cfg.rul_cap)
    return Tables(x_train, y, units, x_test, ml_cols, twin)


def method_columns(tables: Tables, method: str) -> list[str]:
    if method == "ml":
        return tables.ml_cols
    if method == "hybrid":
        return tables.ml_cols + list(TWIN_FEATURES)
    raise ValueError(method)


@dataclass
class Predictions:
    point: dict[str, np.ndarray]
    lo: dict[str, np.ndarray]
    hi: dict[str, np.ndarray]
    conformal_halfwidth: dict[str, list[float]]


def fit_predict(tables: Tables, cfg: ModelConfig, seed: int = 0, conformal: bool = True) -> Predictions:
    """Fit ML and hybrid regressors; return test-row predictions for all methods."""
    point: dict[str, np.ndarray] = {}
    lo: dict[str, np.ndarray] = {}
    hi: dict[str, np.ndarray] = {}
    halfwidth: dict[str, list[float]] = {}

    point["physics"] = np.minimum(tables.x_test["tw_rul_q50"].to_numpy(), cfg.rul_cap)
    lo["physics"] = np.minimum(tables.x_test["tw_rul_q10"].to_numpy(), cfg.rul_cap)
    hi["physics"] = np.minimum(tables.x_test["tw_rul_q90"].to_numpy(), cfg.rul_cap)

    for method in ("ml", "hybrid"):
        cols = method_columns(tables, method)
        model = _gbm(cfg, seed).fit(tables.x_train[cols], tables.y_train)
        p = np.clip(model.predict(tables.x_test[cols]), 0.0, cfg.rul_cap)
        point[method] = p
        if conformal:
            oof = np.empty_like(tables.y_train)
            for tr_idx, va_idx in _folds(tables.units_train, cfg.n_folds):
                m = _gbm(cfg, seed).fit(tables.x_train[cols].iloc[tr_idx], tables.y_train[tr_idx])
                oof[va_idx] = m.predict(tables.x_train[cols].iloc[va_idx])
            q, per_band = conformal_halfwidths(np.abs(tables.y_train - oof), oof, p, cfg.conformal_bins, cfg.interval)
            halfwidth[method] = per_band
            lo[method] = np.clip(p - q, 0.0, cfg.rul_cap)
            hi[method] = np.clip(p + q, 0.0, cfg.rul_cap)
    return Predictions(point, lo, hi, halfwidth)
