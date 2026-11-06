import numpy as np
import pytest

from turbotwin.degradation import KalmanBank, WearPrior, damage_curve, fit_wear
from turbotwin.simulator import wear_ode


def test_ode_matches_closed_form() -> None:
    a, b = 1e-5, 0.02
    d = wear_ode(a, b, 200, 0.0, np.random.default_rng(0))
    t = np.arange(1, 201, dtype=float)
    np.testing.assert_allclose(d, damage_curve(t, a, b), rtol=1e-5)


def test_fit_wear_recovers_parameters() -> None:
    t = np.arange(1, 251, dtype=float)
    a, b = 1.2e-5, 0.018
    d = damage_curve(t, a, b) + np.random.default_rng(1).normal(0, 2e-4, len(t))
    ah, bh = fit_wear(t, d)
    assert ah == pytest.approx(a, rel=0.3)
    assert bh == pytest.approx(b, rel=0.1)


def _prior() -> WearPrior:
    return WearPrior(
        mean=np.log([1e-5, 0.019]),
        cov=np.diag([0.2, 0.06]),
        fail_mean=0.027,
        fail_sd=0.004,
        process_sd=3e-5,
        meas_sd_extra=0.0,
    )


def test_filter_tracks_damage_and_rul_shrinks() -> None:
    a, b = 1e-5, 0.02
    t = np.arange(1, 181, dtype=float)
    true = damage_curve(t, a, b)
    z = true + np.random.default_rng(2).normal(0, 2e-3, len(t))
    res = KalmanBank(_prior(), n_grid=15).run(z, np.full(len(t), 2e-3))
    assert np.sqrt(np.mean((res.d_mean[-50:] - true[-50:]) ** 2)) < 2e-3
    # posterior median RUL decreases as the engine wears
    assert res.rul_q[-1, 1] < res.rul_q[60, 1]
    # quantiles are ordered
    assert np.all(np.diff(res.rul_q, axis=1) >= -1e-9)


def test_filter_is_deterministic() -> None:
    z = np.linspace(0, 0.01, 50)
    r1 = KalmanBank(_prior(), n_grid=9).run(z, np.full(50, 2e-3))
    r2 = KalmanBank(_prior(), n_grid=9).run(z, np.full(50, 2e-3))
    np.testing.assert_array_equal(r1.rul_q, r2.rul_q)
    assert r1.log_evidence == r2.log_evidence


def test_evidence_prefers_true_noise_level() -> None:
    t = np.arange(1, 151, dtype=float)
    z = damage_curve(t, 1e-5, 0.02) + np.random.default_rng(3).normal(0, 2e-3, len(t))
    good = KalmanBank(_prior(), n_grid=9).run(z, np.full(len(t), 2e-3), compute_rul=False).log_evidence
    bad = KalmanBank(_prior(), n_grid=9).run(z, np.full(len(t), 2e-4), compute_rul=False).log_evidence
    assert good > bad
