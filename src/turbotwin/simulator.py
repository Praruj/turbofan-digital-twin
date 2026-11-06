"""Run-to-failure fleet simulator built on the gas-path model.

Each simulated engine gets wear parameters ``(a, b)``, a failure threshold, and a
manufacturing offset; damage is integrated with ``scipy.integrate.solve_ivp`` (plus an
optional per-cycle random walk), mapped to HPC health along the fault direction, pushed
through the cycle model and corrupted with Gaussian sensor noise. Output uses the same
columns as the C-MAPSS loader, so every downstream component runs unchanged on
simulated or real data - and on simulated data the true damage is known.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from turbotwin.thermo import SENSORS, EngineDesign, FloatArray, sensors


@dataclass(frozen=True)
class FleetSpec:
    """Distribution of engines in a simulated fleet."""

    log_ab_mean: tuple[float, float] = (-11.5, -3.95)
    log_ab_cov: tuple[tuple[float, float], tuple[float, float]] = ((0.2, 0.0), (0.0, 0.065))
    fail_mean: float = 0.027
    fail_sd: float = 0.0043
    direction: tuple[float, float] = (0.58, 0.82)  # (eta_hpc, flow_hpc) fault direction
    process_sd: float = 3e-5
    max_cycles: int = 600


def wear_ode(a: float, b: float, t_end: float, process_sd: float, rng: np.random.Generator) -> FloatArray:
    """Damage at integer cycles 1..t_end: deterministic ODE plus a random-walk term."""
    t_eval = np.arange(1, int(t_end) + 1, dtype=float)
    sol = solve_ivp(lambda _t, d: a + b * d, (0.0, float(t_end)), [0.0], t_eval=t_eval, rtol=1e-8, atol=1e-12)
    walk = np.cumsum(rng.normal(0.0, process_sd, len(t_eval))) if process_sd > 0 else 0.0
    return np.asarray(sol.y[0] + walk, dtype=float)


def simulate_fleet(
    design: EngineDesign,
    n_engines: int,
    spec: FleetSpec,
    sigma: FloatArray,
    sigma_between: FloatArray | None = None,
    seed: int = 0,
    truncate: bool = False,
) -> pd.DataFrame:
    """Simulate ``n_engines`` run-to-failure trajectories.

    Returns a long DataFrame with ``unit, cycle, <SENSORS>, rul, damage``. With
    ``truncate=True`` each trajectory is cut at a uniformly random point (a test set).
    """
    rng = np.random.default_rng(seed)
    u = np.asarray(spec.direction, dtype=float)
    u /= np.linalg.norm(u)
    between = np.zeros(len(SENSORS)) if sigma_between is None else np.asarray(sigma_between)
    frames = []
    for unit in range(1, n_engines + 1):
        log_a, log_b = rng.multivariate_normal(spec.log_ab_mean, spec.log_ab_cov)
        fail = max(rng.normal(spec.fail_mean, spec.fail_sd), 0.25 * spec.fail_mean)
        d = wear_ode(np.exp(log_a), np.exp(log_b), spec.max_cycles, spec.process_sd, rng)
        hit = np.flatnonzero(d >= fail)
        life = int(hit[0]) + 1 if len(hit) else spec.max_cycles
        d = d[:life]
        y = sensors(design, np.outer(d, u))
        y = y + rng.normal(0.0, between) + rng.normal(0.0, sigma, size=y.shape)
        df = pd.DataFrame(y, columns=list(SENSORS))
        df.insert(0, "cycle", np.arange(1, life + 1))
        df.insert(0, "unit", unit)
        df["rul"] = life - df["cycle"]
        df["damage"] = d
        if truncate:
            cut = int(rng.integers(min(31, life), life + 1))
            df = df.iloc[:cut]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
