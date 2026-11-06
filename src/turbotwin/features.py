"""Leakage-safe feature tables for the data-driven and hybrid RUL models.

Every feature at cycle ``t`` uses only cycles ``<= t`` of the same engine (rolling
windows and a causal filter), so a test engine truncated at ``t`` sees exactly what a
training row at ``t`` saw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from turbotwin.data import INFORMATIVE
from turbotwin.twin import DigitalTwin

TWIN_FEATURES: tuple[str, ...] = (
    "tw_damage", "tw_damage_sd", "tw_log_a", "tw_log_b", "tw_rul_q10", "tw_rul_q50", "tw_rul_q90",
)  # fmt: skip


def sensor_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in INFORMATIVE if c in df.columns]


def ml_features(df: pd.DataFrame, window: int = 30, short: int = 10) -> pd.DataFrame:
    """Rolling level / trend / scatter of each informative sensor, per engine."""
    cols = sensor_columns(df)
    out = []
    for _, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        t = g["cycle"].astype(float)
        f = pd.DataFrame(index=g.index)
        f["unit"] = g["unit"]
        f["cycle"] = g["cycle"]
        var_t = t.rolling(window, min_periods=2).var()
        for c in cols:
            y = g[c].astype(float)
            f[f"{c}_mean"] = y.rolling(short, min_periods=1).mean()
            f[f"{c}_slope"] = (t.rolling(window, min_periods=2).cov(y) / var_t).fillna(0.0)
            f[f"{c}_std"] = y.rolling(window, min_periods=2).std().fillna(0.0)
        out.append(f)
    return pd.concat(out).loc[df.index]


def twin_features(twin: DigitalTwin, df: pd.DataFrame, rul_cap: float) -> pd.DataFrame:
    """Causal digital-twin posterior summaries for every row of ``df``."""
    parts = []
    for _, g in df.groupby("unit", sort=True):
        g = g.sort_values("cycle")
        r = twin.track(g, quantiles=(0.1, 0.5, 0.9))
        q = np.minimum(r.rul_q, rul_cap)
        parts.append(
            pd.DataFrame(
                {
                    "tw_damage": r.d_mean,
                    "tw_damage_sd": r.d_sd,
                    "tw_log_a": r.log_a_mean,
                    "tw_log_b": r.log_b_mean,
                    "tw_rul_q10": q[:, 0],
                    "tw_rul_q50": q[:, 1],
                    "tw_rul_q90": q[:, 2],
                },
                index=g.index,
            )
        )
    return pd.concat(parts).loc[df.index]


def last_rows(df: pd.DataFrame) -> np.ndarray:
    """Index of the final observed cycle of each engine (the C-MAPSS test query point)."""
    return df.groupby("unit")["cycle"].idxmax().sort_index().to_numpy()
