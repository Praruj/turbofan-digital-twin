import numpy as np
import pytest

from turbotwin.thermo import HEALTH, SENSORS, EngineDesign, jacobian, sensors, solve_cycle


@pytest.fixture
def design() -> EngineDesign:
    return EngineDesign()


def test_new_engine_matches_fd001_levels(design: EngineDesign) -> None:
    # defaults are the FD001-calibrated constants: fleet-mean early-life levels
    y = dict(zip(SENSORS, sensors(design, np.zeros(2)), strict=True))
    assert y["T24"] == pytest.approx(642.4, abs=0.5)
    assert y["T30"] == pytest.approx(1587, abs=3)
    assert y["P30"] == pytest.approx(554, abs=1)


def test_turbine_power_balance_closes(design: EngineDesign) -> None:
    out = solve_cycle(design, np.array([[0.0, 0.0], [0.02, 0.03]]))
    # temperature ordering through the hot section
    assert np.all(out["T40"] > out["T45"])
    assert np.all(out["T45"] > out["T50"])
    assert np.all(out["T30"] > out["T24"])


def test_hpc_degradation_signs_match_cmapss(design: EngineDesign) -> None:
    # signs of FD001 end-of-life drift (see validation): T30, T50 up; P30, phi, W31 down
    j = dict(zip(SENSORS, jacobian(design)[:, :2] @ np.array([0.58, 0.82]), strict=True))
    assert j["T30"] > 0 and j["T50"] > 0 and j["T24"] > 0
    assert j["P30"] < 0 and j["phi"] < 0 and j["W31"] < 0 and j["W32"] < 0
    assert j["BPR"] > 0


def test_efficiency_loss_burns_more_fuel(design: EngineDesign) -> None:
    f0 = solve_cycle(design, np.zeros(2))["fuel"]
    f1 = solve_cycle(design, np.array([0.02, 0.0]))["fuel"]
    assert f1 > f0


def test_vectorised_equals_loop(design: EngineDesign) -> None:
    rng = np.random.default_rng(0)
    h = rng.uniform(0, 0.03, size=(7, 4))
    batch = sensors(design, h)
    loop = np.stack([sensors(design, hi) for hi in h])
    np.testing.assert_allclose(batch, loop, rtol=1e-10)


def test_jacobian_shape(design: EngineDesign) -> None:
    assert jacobian(design).shape == (len(SENSORS), len(HEALTH))


def test_bad_health_shape(design: EngineDesign) -> None:
    with pytest.raises(ValueError):
        sensors(design, np.zeros(3))
