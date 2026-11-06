"""RUL metrics with engine-level bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from turbotwin.thermo import FloatArray

Metric = Callable[[FloatArray, FloatArray], float]


def rmse(y: FloatArray, p: FloatArray) -> float:
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(y: FloatArray, p: FloatArray) -> float:
    return float(np.mean(np.abs(p - y)))


def nasa_score(y: FloatArray, p: FloatArray) -> float:
    """PHM08 asymmetric score (sum over engines): late predictions cost more."""
    d = p - y
    return float(np.sum(np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0))))


METRICS: dict[str, Metric] = {"rmse": rmse, "mae": mae, "nasa_score": nasa_score}


def bootstrap_ci(
    y: FloatArray, p: FloatArray, metric: Metric, n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling engines with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    stats = np.array([metric(y[i], p[i]) for i in idx])
    lo, hi = np.quantile(stats, [(1 - level) / 2, (1 + level) / 2])
    return metric(y, p), float(lo), float(hi)


def paired_diff_ci(
    y: FloatArray, p_a: FloatArray, p_b: FloatArray, metric: Metric, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """``metric(a) - metric(b)`` with a paired bootstrap CI (same engines in both arms)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    stats = np.array([metric(y[i], p_a[i]) - metric(y[i], p_b[i]) for i in idx])
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return metric(y, p_a) - metric(y, p_b), float(lo), float(hi)


def coverage(y: FloatArray, lo: FloatArray, hi: FloatArray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))
