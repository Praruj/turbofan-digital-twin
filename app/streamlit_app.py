"""Interactive demo: the physics simulator and the calibrated twin on real test engines.

    streamlit run app/streamlit_app.py

Needs ``results/fd001/twin.json`` (from the pipeline) and ``data/raw`` (downloaded by it).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from turbotwin.data import load_test
from turbotwin.simulator import FleetSpec, simulate_fleet
from turbotwin.thermo import SENSORS, EngineDesign, sensors, solve_cycle
from turbotwin.twin import DigitalTwin

ROOT = Path(__file__).resolve().parents[1]
TWIN_PATH = ROOT / "results" / "fd001" / "twin.json"
RAW = ROOT / "data" / "raw"

st.set_page_config(page_title="Turbofan digital twin", layout="wide")
st.title("Hybrid digital twin of a turbofan engine")
st.caption("Gas-path physics + Bayesian damage filter, calibrated on NASA C-MAPSS FD001")


@st.cache_resource
def get_twin() -> DigitalTwin | None:
    return DigitalTwin.load(TWIN_PATH) if TWIN_PATH.exists() else None


twin = get_twin()
design = twin.design if twin is not None else EngineDesign()
tab_sim, tab_twin, tab_fleet = st.tabs(["Physics simulator", "Twin on a real test engine", "Simulated fleet"])

with tab_sim:
    st.subheader("What does each kind of damage do to the sensors?")
    c1, c2 = st.columns(2)
    with c1:
        eta_h = st.slider("HPC efficiency loss (%)", 0.0, 4.0, 1.5, 0.1) / 100
        flow_h = st.slider("HPC flow-capacity loss (%)", 0.0, 4.0, 2.0, 0.1) / 100
    with c2:
        eta_f = st.slider("Fan efficiency loss (%)", 0.0, 4.0, 0.0, 0.1) / 100
        flow_f = st.slider("Fan flow-capacity loss (%)", 0.0, 4.0, 0.0, 0.1) / 100
    new = solve_cycle(design, np.zeros(4))
    worn = solve_cycle(design, np.array([eta_h, flow_h, eta_f, flow_f]))
    rows = [
        {"sensor": s, "new engine": float(new[s]), "degraded": float(worn[s]), "change": float(worn[s] - new[s])}
        for s in (*SENSORS, "T40")
    ]
    df = pd.DataFrame(rows)
    if twin is not None and twin.noise is not None:
        sig = dict(zip(SENSORS, twin.noise.sigma, strict=True))
        df["change / sensor noise"] = [r["change"] / sig[r["sensor"]] if r["sensor"] in sig else np.nan for r in rows]
    st.dataframe(df.style.format(precision=3), width="stretch", hide_index=True)
    st.caption("T40 (turbine inlet temperature) is not measured in C-MAPSS; the model infers it.")

with tab_twin:
    if twin is None or not RAW.exists():
        st.warning("Run `python -m turbotwin.pipeline --config configs/fd001.yaml` first.")
    else:
        test = load_test(RAW, "FD001")
        unit = st.selectbox("Test engine", sorted(test["unit"].unique()), index=30)
        g = test[test["unit"] == unit].sort_values("cycle")
        upto = st.slider("Cycles observed so far", 31 if len(g) > 31 else 1, len(g), len(g))
        g = g.iloc[:upto]
        obs = twin.observe(g)
        res = twin.track(g)
        assert twin.prior is not None
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].plot(obs.cycles, obs.z, ".", color="#999", ms=3, label="gas-path estimate")
        ax[0].plot(obs.cycles, res.d_mean, color="#1b7837", label="filtered damage")
        ax[0].fill_between(obs.cycles, res.d_mean - 2 * res.d_sd, res.d_mean + 2 * res.d_sd, color="#1b7837", alpha=0.2)
        ax[0].axhline(twin.prior.fail_mean, ls="--", color="k", lw=0.8, label="failure threshold")
        ax[0].set_xlabel("cycle")
        ax[0].set_ylabel("HPC damage")
        ax[0].legend(fontsize=8)
        ax[1].plot(obs.cycles, g["rul"], "k-", lw=1, label="true RUL")
        ax[1].plot(obs.cycles, res.rul_q[:, 1], color="#1b7837", label="posterior median")
        ax[1].fill_between(obs.cycles, res.rul_q[:, 0], res.rul_q[:, 2], color="#1b7837", alpha=0.2, label="80%")
        ax[1].set_xlabel("cycle")
        ax[1].set_ylabel("RUL")
        ax[1].set_ylim(0, 350)
        ax[1].legend(fontsize=8)
        st.pyplot(fig)
        q = res.rul_q[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted RUL (median)", f"{q[1]:.0f} cycles")
        m2.metric("80% interval", f"{q[0]:.0f} - {q[2]:.0f}")
        m3.metric("True RUL", f"{int(g['rul'].iloc[-1])}")
        st.caption("Physics-only estimate; the ML and hybrid numbers are in the README benchmark.")

with tab_fleet:
    st.subheader("Generate run-to-failure data with controllable wear and noise")
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Engines", 1, 20, 5)
    wear_scale = c2.slider("Wear-rate multiplier", 0.5, 2.0, 1.0, 0.1)
    noise_scale = c3.slider("Sensor-noise multiplier", 0.0, 3.0, 1.0, 0.1)
    base = FleetSpec()
    spec = FleetSpec(
        log_ab_mean=(base.log_ab_mean[0] + np.log(wear_scale), base.log_ab_mean[1] + np.log(wear_scale)),
        log_ab_cov=base.log_ab_cov,
    )
    sigma = twin.noise.sigma if twin is not None and twin.noise is not None else 0.01 * sensors(design, [0, 0])
    sim = simulate_fleet(design, n, spec, noise_scale * sigma, seed=st.number_input("seed", 0, 999, 0))
    s = st.selectbox("Sensor", SENSORS, index=2)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for _, gg in sim.groupby("unit"):
        ax.plot(gg["cycle"], gg[s], lw=0.8)
    ax.set_xlabel("cycle")
    ax.set_ylabel(s)
    st.pyplot(fig)
    st.caption(f"Simulated lifetimes: {sorted(sim.groupby('unit')['cycle'].max().tolist())}")
