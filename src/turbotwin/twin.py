"""The hybrid digital twin: calibrated gas-path model + damage filter, fitted to a fleet."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from turbotwin import calibration as cal
from turbotwin.degradation import FilterResult, KalmanBank, WearPrior, damage_curve, fit_wear
from turbotwin.thermo import SENSORS, EngineDesign, FloatArray, sensors


@dataclass
class TwinConfig:
    healthy_cycles: int = 30  # fleet-level nominal calibration window
    baseline_cycles: int = 20  # per-engine new-engine offset window
    eol_cycles: int = 10  # end-of-life window for the drift fit
    n_grid: int = 25
    process_sd_grid: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4)
    meas_extra_grid: tuple[float, ...] = (0.0, 1e-3, 2e-3)
    prior_shrink: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TwinConfig:
        kw = dict(d)
        for k in ("process_sd_grid", "meas_extra_grid"):
            if k in kw:
                kw[k] = tuple(float(v) for v in kw[k])
        return cls(**kw)


@dataclass
class Observation:
    """Stage-1 output for one engine: per-cycle damage estimate and its sd."""

    cycles: FloatArray
    health: FloatArray  # (n, 2) HPC efficiency / flow-capacity loss
    z: FloatArray  # damage along fault direction
    z_sd: FloatArray
    baseline: FloatArray


@dataclass
class DigitalTwin:
    cfg: TwinConfig
    design: EngineDesign = field(default_factory=EngineDesign)
    noise: cal.NoiseModel | None = None
    direction: FloatArray | None = None
    prior: WearPrior | None = None
    report: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ calibration
    def fit(self, train: pd.DataFrame) -> DigitalTwin:
        cfg = self.cfg
        self.noise = cal.estimate_noise(train, cfg.healthy_cycles)
        # nominal levels and matching constants interact (e.g. Mach sets the Ps30 level):
        # alternate the two fits; two passes converge to < 1e-6 relative level error
        self.design = EngineDesign()
        for _ in range(2):
            self.design, _ = cal.fit_nominal(train, self.design, cfg.healthy_cycles)
            self.design, eol_health, drift_rms = cal.fit_coupling(
                train, self.design, self.noise, cfg.healthy_cycles, cfg.eol_cycles
            )
        self.design, nominal_resid = cal.fit_nominal(train, self.design, cfg.healthy_cycles)
        u = eol_health.mean(axis=0)
        self.direction = u / np.linalg.norm(u)

        obs = {unit: self.observe(g) for unit, g in train.groupby("unit")}
        wear, fail = [], []
        for o in obs.values():
            a, b = fit_wear(o.cycles, o.z)
            wear.append((a, b))
            fail.append(damage_curve(np.array([o.cycles[-1]]), a, b)[0])
        wear_arr, fail_arr = np.array(wear), np.array(fail)

        # noise hyper-parameters by maximum marginal likelihood over the training fleet
        best: tuple[float, float, float] | None = None
        for q in cfg.process_sd_grid:
            for e in cfg.meas_extra_grid:
                prior = WearPrior.fit(wear_arr, fail_arr, q, e, cfg.prior_shrink)
                bank = KalmanBank(prior, n_grid=cfg.n_grid)
                ev = sum(bank.run(o.z, o.z_sd, compute_rul=False).log_evidence for o in obs.values())
                if best is None or ev > best[0]:
                    best = (ev, q, e)
        assert best is not None
        self.prior = WearPrior.fit(wear_arr, fail_arr, best[1], best[2], cfg.prior_shrink)
        self.report = {
            "design": self.design.to_dict(),
            "noise": self.noise.to_dict(),
            "nominal_residual": dict(zip(SENSORS, map(float, nominal_resid), strict=True)),
            "drift_fit_rms_sigma": drift_rms,
            "fault_direction": {"eta_hpc": float(self.direction[0]), "flow_hpc": float(self.direction[1])},
            "eol_health_corr": float(np.corrcoef(eol_health.T)[0, 1]),
            "prior": self.prior.to_dict(),
            "log_evidence": float(best[0]),
        }
        return self

    # ------------------------------------------------------------------ inference
    def observe(self, engine: pd.DataFrame) -> Observation:
        """Gas-path analysis: per-cycle HPC health and damage, with uncertainty."""
        assert self.noise is not None
        y = cal.sensor_matrix(engine)
        base = cal.engine_baseline(self.design, y, self.cfg.baseline_cycles)
        h, cov = cal.estimate_states(self.design, y, self.noise.sigma, base)
        u = self.direction if self.direction is not None else np.array([1.0, 0.0])
        z = h @ u
        z_sd = np.sqrt(np.einsum("i,nij,j->n", u, cov, u))
        return Observation(engine["cycle"].to_numpy(dtype=float), h, z, z_sd, base)

    def track(self, engine: pd.DataFrame, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)) -> FilterResult:
        assert self.prior is not None
        o = self.observe(engine)
        return KalmanBank(self.prior, n_grid=self.cfg.n_grid).run(o.z, o.z_sd, quantiles=quantiles)

    def predict_sensors(self, damage: FloatArray, baseline: FloatArray | None = None) -> FloatArray:
        assert self.direction is not None
        y = sensors(self.design, np.outer(damage, self.direction))
        return y + (0.0 if baseline is None else baseline)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.report | {"config": asdict(self.cfg)}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> DigitalTwin:
        """Rebuild a calibrated twin from :meth:`save` output (no refitting)."""
        rep = json.loads(Path(path).read_text())
        cfg = TwinConfig.from_dict(rep["config"]) if "config" in rep else TwinConfig()
        noise = cal.NoiseModel(
            sigma=np.array([rep["noise"]["sigma"][s] for s in SENSORS]),
            sigma_between=np.array([rep["noise"]["sigma_between"][s] for s in SENSORS]),
        )
        pr = rep["prior"]
        prior = WearPrior(
            np.array(pr["log_ab_mean"]),
            np.array(pr["log_ab_cov"]),
            pr["fail_mean"],
            pr["fail_sd"],
            pr["process_sd"],
            pr["meas_sd_extra"],
        )
        direction = np.array([rep["fault_direction"]["eta_hpc"], rep["fault_direction"]["flow_hpc"]])
        return cls(cfg, EngineDesign.from_dict(rep["design"]), noise, direction, prior, rep)
