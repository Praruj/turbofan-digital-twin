"""Calibrate a twin on a simulated fleet with known damage and check what it recovers."""

import numpy as np
import pandas as pd
import pytest

from turbotwin import calibration as cal
from turbotwin.features import last_rows, ml_features
from turbotwin.metrics import bootstrap_ci, nasa_score, paired_diff_ci, rmse
from turbotwin.simulator import FleetSpec, simulate_fleet
from turbotwin.thermo import EngineDesign
from turbotwin.twin import DigitalTwin, TwinConfig

SIGMA = np.array([0.302, 4.03, 3.97, 0.405, 0.101, 0.303, 0.0201, 0.0999, 0.0613])


@pytest.fixture(scope="module")
def fleet() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = simulate_fleet(EngineDesign(), 30, FleetSpec(), SIGMA, 0.5 * SIGMA, seed=0)
    te = simulate_fleet(EngineDesign(), 15, FleetSpec(), SIGMA, 0.5 * SIGMA, seed=1, truncate=True)
    return tr, te


@pytest.fixture(scope="module")
def twin(fleet: tuple[pd.DataFrame, pd.DataFrame]) -> DigitalTwin:
    cfg = TwinConfig(n_grid=11, process_sd_grid=(3e-5,), meas_extra_grid=(0.0,))
    return DigitalTwin(cfg).fit(fleet[0])


def test_simulated_fleet_is_well_formed(fleet: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tr, te = fleet
    assert tr.groupby("unit")["rul"].min().eq(0).all()  # run to failure
    assert (te.groupby("unit")["rul"].min() >= 0).all()
    assert tr.groupby("unit")["cycle"].max().between(80, 600).all()


def test_noise_estimate_close_to_truth(fleet: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    n = cal.estimate_noise(fleet[0], 30)
    np.testing.assert_allclose(n.sigma, SIGMA, rtol=0.15)


def test_recovers_fault_direction(twin: DigitalTwin) -> None:
    assert twin.direction is not None
    u = np.array(FleetSpec().direction) / np.linalg.norm(FleetSpec().direction)
    assert float(twin.direction @ u) > 0.98


def test_tracks_true_damage(twin: DigitalTwin, fleet: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    assert twin.prior is not None
    errs = []
    for _, g in fleet[1].groupby("unit"):
        r = twin.track(g)
        errs.append(r.d_mean - g["damage"].to_numpy())
    err = np.sqrt(np.mean(np.concatenate(errs) ** 2))
    assert err < 0.2 * twin.prior.fail_mean


def test_physics_rul_beats_naive(twin: DigitalTwin, fleet: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    te = fleet[1]
    y, p = [], []
    for _, g in te.groupby("unit"):
        y.append(g["rul"].iloc[-1])
        p.append(twin.track(g).rul_q[-1, 1])
    y_arr, p_arr = np.array(y, float), np.array(p)
    naive = np.full_like(y_arr, y_arr.mean())  # oracle mean: an optimistic naive baseline
    assert rmse(y_arr, p_arr) < rmse(y_arr, naive)


def test_features_are_causal(fleet: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    tr = fleet[0]
    g = tr[tr["unit"] == 1]
    full = ml_features(g)
    cut = ml_features(g.iloc[:50])
    pd.testing.assert_frame_equal(full.iloc[:50], cut)


def test_last_rows(fleet: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    te = fleet[1]
    idx = last_rows(te)
    assert len(idx) == te["unit"].nunique()
    assert (te.loc[idx].groupby("unit")["cycle"].max() == te.groupby("unit")["cycle"].max()).all()


def test_metrics() -> None:
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == 0.0
    assert nasa_score(y, y + 10) > nasa_score(y, y - 10)  # late is penalised more
    est, lo, hi = bootstrap_ci(y, y + 1, rmse, n_boot=100)
    assert lo <= est <= hi
    d, _dlo, dhi = paired_diff_ci(y, y + 1, y + 3, rmse, n_boot=100)
    assert d == pytest.approx(-2.0) and dhi < 0


def test_mondrian_conformal_bands() -> None:
    from turbotwin.models import conformal_halfwidths

    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 125, 5000)
    resid = np.abs(rng.normal(0, 1 + pred / 10))  # error grows with predicted RUL
    q, per_band = conformal_halfwidths(resid, pred, np.array([10.0, 110.0]), (40.0, 80.0), 0.8)
    assert per_band[0] < per_band[1] < per_band[2]
    assert q[0] == per_band[0] and q[1] == per_band[2]
