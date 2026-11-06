"""Zero-dimensional (lumped-parameter) gas-path model of a two-spool turbofan.

The model is a steady-state Brayton-cycle calculation per flight cycle: each operating
snapshot is solved from component relations (isentropic efficiencies, pressure ratios,
shaft power balances, combustor energy balance) rather than learned from data.

Health is described by four dimensionless *loss* parameters, all zero for a new engine:

* ``eta_hpc``  - fractional loss of high-pressure-compressor isentropic efficiency
* ``flow_hpc`` - fractional loss of HPC flow capacity (fouling, tip clearance)
* ``eta_fan``  - fractional loss of fan efficiency
* ``flow_fan`` - fractional loss of fan flow capacity

Closure (how the engine "answers" a degraded component): the controller holds fan speed,
so fan pressure ratio is fixed; the reduced HPC swallowing capacity moves both compressors
along their speed lines (``kappa_lpc``, ``kappa_hpc``); the HPT is sized by HPC work and
the LPT must still drive the fan, which sets turbine inlet temperature T40 by a scalar
root-find. Everything is vectorised over arrays of health parameters so a particle
filter can evaluate thousands of hypotheses per cycle.

Outputs are reported in C-MAPSS units (degrees Rankine, psia, pps/psi) for the nine
gas-path sensors the model covers.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

RANKINE_TO_K = 5.0 / 9.0

#: C-MAPSS sensor columns produced by the model, in output order.
SENSORS: tuple[str, ...] = ("T24", "T30", "T50", "P30", "Ps30", "phi", "BPR", "W31", "W32")

#: Health-parameter names, in state-vector order.
HEALTH: tuple[str, ...] = ("eta_hpc", "flow_hpc", "eta_fan", "flow_fan")


@dataclass(frozen=True)
class EngineDesign:
    """Design-point constants.

    Gas properties and component efficiencies are textbook values. The ``*_0``,
    ``gain_*``, ``kappa_*``, ``mach_30`` and ``nozzle_ratio`` defaults are the values
    :mod:`turbotwin.calibration` fits on C-MAPSS FD001 (rounded), so the default engine
    is a realistic one; a twin always re-fits them from its own training data.
    """

    # gas properties
    gamma_c: float = 1.40
    cp_c: float = 1004.5  # J/(kg K), cold section
    gamma_h: float = 1.33
    cp_h: float = 1150.0  # J/(kg K), hot section
    lhv: float = 43.0e6  # J/kg, jet fuel
    eta_burner: float = 0.99
    eta_mech: float = 0.99
    dp_burner: float = 0.05  # fractional total-pressure loss
    # ambient (sea-level static, as in FD001/FD003)
    t2_r: float = 518.67
    p2_psia: float = 14.62
    # fan / bypass
    pi_fan: float = 21.61 / 14.62  # P15/P2 in the data
    eta_fan_0: float = 0.90
    bpr_0: float = 8.4184
    # low-pressure compressor (booster)
    pi_lpc_0: float = 1.9481
    eta_lpc: float = 0.88
    # high-pressure compressor
    pi_hpc_0: float = 19.449
    eta_hpc_0: float = 0.90785
    # turbines
    eta_hpt: float = 0.90
    eta_lpt: float = 0.90
    nozzle_ratio: float = 1.3976  # P50 / P2
    # component matching: compressor pressure ratio response to HPC flow-capacity loss
    kappa_lpc: float = 0.29947
    kappa_hpc: float = 0.43414
    # measurement models
    mach_30: float = 0.95  # HPC-exit Mach number at design (static/total pressure)
    gain_ps30: float = 0.15281
    gain_phi: float = 1.0761e6
    w31_0: float = 38.932
    w32_0: float = 23.359

    def to_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EngineDesign:
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in d.items() if k in known})

    def with_(self, **kw: float) -> EngineDesign:
        return replace(self, **kw)


def _as_health(health: ArrayLike) -> FloatArray:
    h = np.asarray(health, dtype=float)
    if h.shape[-1] == 2:  # HPC-only health vector
        h = np.concatenate([h, np.zeros((*h.shape[:-1], 2))], axis=-1)
    if h.shape[-1] != len(HEALTH):
        raise ValueError(f"health must have last dimension 2 or 4, got {h.shape}")
    return h


def _compressor(t_in: FloatArray, pi: FloatArray, eta: FloatArray, gamma: float) -> FloatArray:
    a = (gamma - 1.0) / gamma
    return t_in * (1.0 + (pi**a - 1.0) / eta)


def solve_cycle(design: EngineDesign, health: ArrayLike) -> dict[str, FloatArray]:
    """Solve the cycle for health parameters of shape ``(..., 2)`` or ``(..., 4)``.

    Returns a dict of arrays of shape ``(...)`` with the nine sensors plus internal
    stations (``T40``, ``T45``, ``fuel``) useful for diagnostics and damage modelling.
    """
    h = _as_health(health)
    d_eta_h, d_flow_h, d_eta_f, d_flow_f = (h[..., i] for i in range(4))
    ac = (design.gamma_c - 1.0) / design.gamma_c
    ah = (design.gamma_h - 1.0) / design.gamma_h

    t2 = design.t2_r * RANKINE_TO_K
    p2 = design.p2_psia

    # --- fan bypass (speed held by control => fixed pressure ratio) ---
    eta_f = design.eta_fan_0 * (1.0 - d_eta_f)
    dt_fan = t2 * (design.pi_fan**ac - 1.0) / eta_f

    # --- core compressors, moved along speed lines by the HPC flow-capacity loss ---
    pi_l = design.pi_lpc_0 * (1.0 + design.kappa_lpc * d_flow_h)
    t24 = _compressor(np.full_like(pi_l, t2), pi_l, np.full_like(pi_l, design.eta_lpc), design.gamma_c)
    p24 = p2 * pi_l
    pi_h = design.pi_hpc_0 * (1.0 - design.kappa_hpc * d_flow_h)
    eta_h = design.eta_hpc_0 * (1.0 - d_eta_h)
    t30 = _compressor(t24, pi_h, eta_h, design.gamma_c)
    p30 = p24 * pi_h

    # core mass flow (corrected-flow scaling at HPC inlet), relative to design
    t24_d = _compressor(np.asarray(t2), np.asarray(design.pi_lpc_0), np.asarray(design.eta_lpc), design.gamma_c)
    w_core = (1.0 - d_flow_h) * (p24 / (p2 * design.pi_lpc_0)) * np.sqrt(t24_d / t24)
    w_fan = 1.0 - d_flow_f
    bpr = design.bpr_0 * w_fan / w_core

    # --- turbines: HPT drives HPC, LPT drives fan + booster ---
    dt_hpt = design.cp_c * (t30 - t24) / (design.cp_h * design.eta_mech)
    work_lpt = (bpr * design.cp_c * dt_fan + design.cp_c * (t24 - t2)) / design.eta_mech  # J/kg core
    p40 = p30 * (1.0 - design.dp_burner)
    p50 = p2 * design.nozzle_ratio

    def lpt_residual(t40: FloatArray) -> FloatArray:
        frac = 1.0 - dt_hpt / (design.eta_hpt * t40)
        pi_hpt = np.clip(frac, 1e-6, None) ** (-1.0 / ah)
        t45 = t40 - dt_hpt
        pi_lpt = np.clip(p40 / pi_hpt / p50, 1.0 + 1e-9, None)
        return design.cp_h * design.eta_lpt * t45 * (1.0 - pi_lpt ** (-ah)) - work_lpt

    # residual is monotone increasing in T40 -> vectorised bisection (robust, branch-free)
    lo = dt_hpt / design.eta_hpt * 1.0001 + 1.0
    hi = np.full_like(lo, 3500.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pos = lpt_residual(mid) > 0.0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    t40 = 0.5 * (lo + hi)
    t45 = t40 - dt_hpt
    t50 = t45 - work_lpt / design.cp_h

    # combustor energy balance -> fuel/air ratio and fuel flow
    far = (design.cp_h * t40 - design.cp_c * t30) / (design.eta_burner * design.lhv - design.cp_h * t40)
    fuel = far * w_core

    # HPC-exit static pressure: Mach number scales with corrected flow W sqrt(T)/P
    t30_d, p30_d = _design_t30_p30(design)
    mach = design.mach_30 * (w_core * np.sqrt(t30 / t30_d) / (p30 / p30_d))
    ps30_true = p30 / (1.0 + 0.5 * (design.gamma_c - 1.0) * mach**2) ** (1.0 / ac)
    ps30 = design.gain_ps30 * ps30_true

    out = {
        "T24": t24 / RANKINE_TO_K,
        "T30": t30 / RANKINE_TO_K,
        "T50": t50 / RANKINE_TO_K,
        "P30": p30,
        "Ps30": ps30,
        "phi": design.gain_phi * fuel / ps30,
        "BPR": bpr,
        "W31": design.w31_0 * w_core,
        "W32": design.w32_0 * w_core,
        "T40": t40 / RANKINE_TO_K,
        "T45": t45 / RANKINE_TO_K,
        "fuel": fuel,
    }
    return out


def _design_t30_p30(design: EngineDesign) -> tuple[float, float]:
    t2 = design.t2_r * RANKINE_TO_K
    ac = (design.gamma_c - 1.0) / design.gamma_c
    t24 = t2 * (1.0 + (design.pi_lpc_0**ac - 1.0) / design.eta_lpc)
    t30 = t24 * (1.0 + (design.pi_hpc_0**ac - 1.0) / design.eta_hpc_0)
    return t30, design.p2_psia * design.pi_lpc_0 * design.pi_hpc_0


def sensors(design: EngineDesign, health: ArrayLike) -> FloatArray:
    """Sensor matrix of shape ``(..., len(SENSORS))``."""
    out = solve_cycle(design, health)
    return np.stack([out[s] for s in SENSORS], axis=-1)


def jacobian(design: EngineDesign, health: ArrayLike | None = None, eps: float = 1e-4) -> FloatArray:
    """Central-difference sensitivity d(sensor)/d(health) at ``health`` (default: new engine).

    Shape ``(len(SENSORS), len(HEALTH))``.
    """
    h0 = np.zeros(len(HEALTH)) if health is None else _as_health(health).astype(float)
    cols = []
    for j in range(len(HEALTH)):
        dh = np.zeros(len(HEALTH))
        dh[j] = eps
        cols.append((sensors(design, h0 + dh) - sensors(design, h0 - dh)) / (2 * eps))
    return np.stack(cols, axis=-1)
