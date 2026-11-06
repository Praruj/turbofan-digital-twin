"""Damage dynamics and Bayesian state/parameter estimation.

Damage ``D`` is the magnitude of HPC health loss along the fleet fault direction ``u``
(``health = D * u``). It follows self-accelerating wear,

    dD/dt = a + b * D,          D(0) = 0          =>  D(t) = (a/b) (exp(b t) - 1),

the continuous analogue of the exponential damage law used to generate C-MAPSS
(Saxena et al., 2008): a slow baseline wear rate ``a`` and a feedback ``b`` by which
existing damage speeds up further damage. ``a`` and ``b`` are unknown per engine.

Estimation (:class:`KalmanBank`) is a Rao-Blackwellised grid filter: on a grid of
``(log a, log b)`` hypotheses weighted by a fleet prior, damage is linear-Gaussian, so
each hypothesis runs an exact scalar Kalman filter and its marginal likelihood updates
the grid weights. The posterior over damage, wear parameters and remaining useful life
is therefore exact up to grid resolution, with no particle degeneracy and no random
numbers (fully reproducible).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from turbotwin.thermo import FloatArray


def damage_curve(t: FloatArray, a: float, b: float) -> FloatArray:
    return (a / b) * np.expm1(b * t)


def fit_wear(t: FloatArray, d: FloatArray) -> tuple[float, float]:
    """Least-squares ``(a, b)`` of the wear law for one engine's damage series."""

    def resid(p: FloatArray) -> FloatArray:
        return damage_curve(t, np.exp(p[0]), np.exp(p[1])) - d

    sol = least_squares(resid, np.log([2e-5, 0.015]), bounds=(np.log([1e-8, 1e-4]), np.log([1e-2, 0.2])))
    return float(np.exp(sol.x[0])), float(np.exp(sol.x[1]))


@dataclass(frozen=True)
class WearPrior:
    """Gaussian fleet prior over ``(log a, log b)`` and failure threshold ``D_fail``."""

    mean: FloatArray  # (2,)
    cov: FloatArray  # (2, 2)
    fail_mean: float
    fail_sd: float
    process_sd: float  # per-cycle random-walk noise on D
    meas_sd_extra: float  # model-form error added to the gas-path measurement sd

    @classmethod
    def fit(
        cls,
        wear: FloatArray,
        fail: FloatArray,
        process_sd: float,
        meas_sd_extra: float,
        shrink: float = 0.0,
    ) -> WearPrior:
        logw = np.log(wear)
        cov = np.cov(logw.T) + shrink * np.eye(2)
        return cls(logw.mean(axis=0), cov, float(fail.mean()), float(fail.std(ddof=1)), process_sd, meas_sd_extra)

    def to_dict(self) -> dict[str, object]:
        return {
            "log_ab_mean": self.mean.tolist(),
            "log_ab_cov": self.cov.tolist(),
            "fail_mean": self.fail_mean,
            "fail_sd": self.fail_sd,
            "process_sd": self.process_sd,
            "meas_sd_extra": self.meas_sd_extra,
        }


@dataclass
class FilterResult:
    """Filtered (causal) posterior summaries, one row per observed cycle."""

    d_mean: FloatArray
    d_sd: FloatArray
    log_a_mean: FloatArray
    log_b_mean: FloatArray
    rul_q: FloatArray  # (n_cycles, len(quantiles))
    quantiles: tuple[float, ...]
    log_evidence: float


class KalmanBank:
    """Grid-of-Kalman-filters posterior for the wear law, see module docstring."""

    def __init__(self, prior: WearPrior, n_grid: int = 25, span: float = 3.5, n_fail_nodes: int = 9) -> None:
        self.prior = prior
        sd = np.sqrt(np.diag(prior.cov))
        axes = [np.linspace(prior.mean[i] - span * sd[i], prior.mean[i] + span * sd[i], n_grid) for i in range(2)]
        la, lb = np.meshgrid(*axes, indexing="ij")
        self.log_a = la.ravel()
        self.log_b = lb.ravel()
        diff = np.stack([self.log_a, self.log_b], axis=1) - prior.mean
        maha = np.einsum("ni,ij,nj->n", diff, np.linalg.inv(prior.cov), diff)
        self.log_prior = -0.5 * maha
        self.a = np.exp(self.log_a)
        self.b = np.exp(self.log_b)
        # threshold uncertainty: equal-mass nodes of the (truncated-positive) Normal
        probs = (np.arange(n_fail_nodes) + 0.5) / n_fail_nodes
        self.fail_nodes = np.clip(norm.ppf(probs, prior.fail_mean, prior.fail_sd), 1e-4, None)

    def run(
        self,
        z: FloatArray,
        z_sd: FloatArray,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        rul_cap: float = 1000.0,
        compute_rul: bool = True,
    ) -> FilterResult:
        """Filter damage observations ``z`` (one per cycle, starting at cycle 1)."""
        g = len(self.a)
        m = np.zeros(g)
        p = np.full(g, 1e-8)
        logw = self.log_prior.copy()
        f = 1.0 + self.b
        q2 = self.prior.process_sd**2
        n = len(z)
        out_m, out_s, out_la, out_lb = (np.empty(n) for _ in range(4))
        out_q = np.empty((n, len(quantiles)))
        log_ev = 0.0
        for t in range(n):
            m = f * m + self.a
            p = f * f * p + q2
            r = z_sd[t] ** 2 + self.prior.meas_sd_extra**2
            s = p + r
            innov = z[t] - m
            ll = -0.5 * (np.log(2 * np.pi * s) + innov**2 / s)
            k = p / s
            m = m + k * innov
            p = (1.0 - k) * p
            # evidence increment = log sum_g w_g lik_g (weights normalised before)
            mx = (logw + ll).max()
            lse_new = mx + np.log(np.exp(logw + ll - mx).sum())
            mx0 = logw.max()
            lse_old = mx0 + np.log(np.exp(logw - mx0).sum())
            log_ev += lse_new - lse_old
            logw = logw + ll
            w = np.exp(logw - logw.max())
            w /= w.sum()
            out_m[t] = w @ m
            out_s[t] = np.sqrt(w @ (p + (m - out_m[t]) ** 2))
            out_la[t] = w @ self.log_a
            out_lb[t] = w @ self.log_b
            out_q[t] = self._rul_quantiles(m, w, quantiles, rul_cap) if compute_rul else np.nan
        return FilterResult(out_m, out_s, out_la, out_lb, out_q, quantiles, float(log_ev))

    def _rul_quantiles(self, m: FloatArray, w: FloatArray, qs: tuple[float, ...], cap: float) -> FloatArray:
        # time for D to grow from m to each threshold node under each (a, b) hypothesis
        c = self.a / self.b
        ratio = (self.fail_nodes[None, :] + c[:, None]) / (np.maximum(m, 0.0)[:, None] + c[:, None])
        tau = np.log(np.maximum(ratio, 1.0)) / self.b[:, None]
        tau = np.minimum(tau, cap).ravel()
        ww = np.repeat(w / len(self.fail_nodes), len(self.fail_nodes))
        keep = ww > 1e-12
        tau, ww = tau[keep], ww[keep]
        order = np.argsort(tau)
        cdf = np.cumsum(ww[order])
        cdf /= cdf[-1]
        return np.interp(qs, cdf, tau[order])
