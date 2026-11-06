"""Digital-twin calibration: fit the gas-path model to observed fleet data.

Three stages, all least squares on noise-normalised residuals:

1. :func:`fit_nominal` - design-point constants so a *new* engine reproduces the fleet's
   early-life sensor levels (9 equations, 9 unknowns).
2. :func:`fit_coupling` - fleet-level matching constants (``kappa_lpc``, ``kappa_hpc``,
   ``mach_30``) jointly with each training engine's end-of-life HPC health, from the
   start-to-end sensor drift of run-to-failure engines.
3. :func:`estimate_states` - per-cycle HPC health (efficiency and flow-capacity loss)
   by Gauss-Newton with the Laplace (linearised) covariance, i.e. classical
   gas-path analysis with uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from turbotwin.thermo import SENSORS, EngineDesign, FloatArray, jacobian, sensors

NOMINAL_PARAMS: tuple[str, ...] = (
    "pi_lpc_0", "pi_hpc_0", "eta_hpc_0", "nozzle_ratio", "gain_ps30", "gain_phi", "bpr_0", "w31_0", "w32_0",
)  # fmt: skip
COUPLING_PARAMS: tuple[str, ...] = ("kappa_lpc", "kappa_hpc", "mach_30")
_COUPLING_BOUNDS = (np.array([-1.0, 0.0, 0.05]), np.array([3.0, 3.0, 0.95]))


@dataclass(frozen=True)
class NoiseModel:
    """Per-sensor measurement noise (within-engine, cycle-to-cycle) and engine-to-engine
    spread of new-engine levels (manufacturing variation)."""

    sigma: FloatArray
    sigma_between: FloatArray

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "sigma": dict(zip(SENSORS, map(float, self.sigma), strict=True)),
            "sigma_between": dict(zip(SENSORS, map(float, self.sigma_between), strict=True)),
        }


def sensor_matrix(df: pd.DataFrame) -> FloatArray:
    return df[list(SENSORS)].to_numpy(dtype=float)


def healthy_window(df: pd.DataFrame, n_cycles: int) -> pd.DataFrame:
    return df[df["cycle"] <= n_cycles]


def estimate_noise(train: pd.DataFrame, n_cycles: int) -> NoiseModel:
    """Within-engine scatter of the first ``n_cycles`` (after removing a per-engine linear
    trend) and between-engine spread of their means."""
    early = healthy_window(train, n_cycles)
    resid, means = [], []
    for _, g in early.groupby("unit"):
        y = sensor_matrix(g)
        t = g["cycle"].to_numpy(dtype=float)
        a = np.vstack([np.ones_like(t), t]).T
        coef, *_ = np.linalg.lstsq(a, y, rcond=None)
        resid.append(y - a @ coef)
        means.append(y.mean(axis=0))
    r = np.vstack(resid)
    dof_scale = np.sqrt(len(r) / max(len(r) - 2 * len(resid), 1))
    sigma = r.std(axis=0) * dof_scale
    m = np.vstack(means)
    between = np.sqrt(np.clip(m.var(axis=0, ddof=1) - sigma**2 / n_cycles, 0.0, None))
    return NoiseModel(sigma=sigma, sigma_between=between)


def fit_nominal(train: pd.DataFrame, design: EngineDesign, n_cycles: int) -> tuple[EngineDesign, FloatArray]:
    """Design-point constants from the fleet-mean early-life sensor levels."""
    target = sensor_matrix(healthy_window(train, n_cycles)).mean(axis=0)
    x0 = np.array([getattr(design, p) for p in NOMINAL_PARAMS])
    # rough analytical start for the scale factors so the solver starts close
    probe = sensors(design, np.zeros(2))
    x0[NOMINAL_PARAMS.index("gain_phi")] *= target[SENSORS.index("phi")] / probe[SENSORS.index("phi")]
    x0[NOMINAL_PARAMS.index("gain_ps30")] *= target[SENSORS.index("Ps30")] / probe[SENSORS.index("Ps30")]

    def resid(logx: FloatArray) -> FloatArray:
        dsg = design.with_(**dict(zip(NOMINAL_PARAMS, np.exp(logx), strict=True)))
        return (sensors(dsg, np.zeros(2)) - target) / target

    sol = least_squares(resid, np.log(x0), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    fitted = design.with_(**dict(zip(NOMINAL_PARAMS, np.exp(sol.x), strict=True)))
    return fitted, resid(sol.x) * target


def engine_drift(train: pd.DataFrame, n_start: int, n_end: int) -> tuple[FloatArray, FloatArray]:
    """Per-engine mean sensor vector over the first ``n_start`` and last ``n_end`` cycles."""
    start, end = [], []
    for _, g in train.groupby("unit"):
        y = sensor_matrix(g)
        start.append(y[:n_start].mean(axis=0))
        end.append(y[-n_end:].mean(axis=0))
    return np.vstack(start), np.vstack(end)


def fit_coupling(
    train: pd.DataFrame, design: EngineDesign, noise: NoiseModel, n_start: int, n_end: int
) -> tuple[EngineDesign, FloatArray, float]:
    """Matching constants + each engine's end-of-life HPC health from start-to-end drift.

    Returns the updated design, the ``(n_engines, 2)`` end-of-life health, and the RMS
    normalised residual of the drift fit.
    """
    start, end = engine_drift(train, n_start, n_end)
    drift = end - start
    n = len(drift)
    scale = noise.sigma

    def unpack(x: FloatArray) -> tuple[EngineDesign, FloatArray]:
        dsg = design.with_(**dict(zip(COUPLING_PARAMS, x[:3], strict=True)))
        return dsg, x[3:].reshape(n, 2)

    def resid(x: FloatArray) -> FloatArray:
        dsg, h = unpack(x)
        pred = sensors(dsg, h) - sensors(dsg, np.zeros(2))
        return ((pred - drift) / scale).ravel()

    x0 = np.concatenate([[design.kappa_lpc, design.kappa_hpc, design.mach_30], np.tile([0.02, 0.02], n)])
    lb = np.concatenate([_COUPLING_BOUNDS[0], np.full(2 * n, -0.05)])
    ub = np.concatenate([_COUPLING_BOUNDS[1], np.full(2 * n, 0.5)])
    sol = least_squares(resid, x0, bounds=(lb, ub), x_scale="jac")
    dsg, h = unpack(sol.x)
    return dsg, h, float(np.sqrt(np.mean(sol.fun**2)))


def estimate_states(
    design: EngineDesign,
    y: FloatArray,
    sigma: FloatArray,
    baseline: FloatArray | None = None,
    n_iter: int = 4,
) -> tuple[FloatArray, FloatArray]:
    """Per-cycle HPC health by Gauss-Newton on noise-weighted residuals.

    ``y`` is ``(n_cycles, n_sensors)``. If ``baseline`` (the engine's own new-engine
    sensor offset relative to the model) is given it is subtracted first. Returns
    ``(states (n, 2), covariances (n, 2, 2))``; the covariance is the Laplace
    approximation ``(J^T R^-1 J)^-1`` at the solution.
    """
    yy = y - (0.0 if baseline is None else baseline)
    w = 1.0 / sigma
    h = np.zeros((len(yy), 2))
    for _ in range(n_iter):
        # model is mildly non-linear: one shared Jacobian per iteration at the mean state
        jac = jacobian(design, h.mean(axis=0))[:, :2]
        jw = jac * w[:, None]
        r = (yy - sensors(design, h)) * w
        step, *_ = np.linalg.lstsq(jw, r.T, rcond=None)
        h = h + step.T
    jac = jacobian(design, h.mean(axis=0))[:, :2]
    jw = jac * w[:, None]
    cov = np.linalg.inv(jw.T @ jw)
    return h, np.broadcast_to(cov, (len(h), 2, 2)).copy()


def engine_baseline(design: EngineDesign, y: FloatArray, n_cycles: int) -> FloatArray:
    """An engine's own early-life offset from the nominal model (manufacturing variation
    plus initial wear, which the data cannot separate)."""
    return y[:n_cycles].mean(axis=0) - sensors(design, np.zeros(2))


def reconstruction_r2(y: FloatArray, y_hat: FloatArray) -> FloatArray:
    """Per-sensor R^2 of a reconstruction (1 - SSE/SST around each sensor's mean)."""
    sse = ((y - y_hat) ** 2).sum(axis=0)
    sst = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - sse / sst
