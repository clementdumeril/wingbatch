"""The vectorised systems against the scalar, world-frame, line-by-line transcription of the gz-sim C++."""

import numpy as np
import pytest

from helpers import advanced_plane, random_states, world_wrench
from wingbatch.reference import ScalarAircraft


@pytest.mark.parametrize("wind", [(0.0, 0.0, 0.0), (3.0, -2.0, 0.5)])
@pytest.mark.parametrize("cem_overwrite", [False, True])
def test_wrench_matches_the_line_by_line_reference(wind, cem_overwrite):
    ac = advanced_plane(300, wind=wind, gz_cem_overwrite=cem_overwrite)
    X = random_states(ac, np.random.default_rng(1))
    F, T, info = world_wrench(ac, X)
    assert np.mean(np.abs(info["lift_drag0"]["alpha"]) > ac.lift_drag[0].system.params.alpha_stall) > 0.3
    ref = ScalarAircraft(ac)
    for i in range(ac.n):
        f, t = ref.wrench_world(list(X[i]))
        np.testing.assert_allclose(F[i], f.tolist(), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(T[i], t.tolist(), rtol=1e-11, atol=1e-11)


def test_derivatives_and_rk4_match_the_reference():
    ac = advanced_plane(20, wind=(2.0, 1.0, 0.0))
    rng = np.random.default_rng(2)
    X = random_states(ac, rng, stall=False)
    s = rng.uniform(-0.5, 0.5, (ac.n, ac.n_servos))
    m = rng.uniform(0.0, 1600.0, (ac.n, ac.n_motors))
    ref = ScalarAircraft(ac)
    d, _ = ac.derivatives(X, s, m)
    for i in range(ac.n):
        np.testing.assert_allclose(d[i], ref.derivatives(list(X[i]), list(s[i]), list(m[i])), rtol=1e-10, atol=1e-10)
    ac.reset(X)
    xs = [list(X[i]) for i in range(ac.n)]
    for _ in range(100):  # 1 s
        ac.step(s, m, 0.01)
        xs = [ref.step(x, list(s[i]), list(m[i]), 0.01) for i, x in enumerate(xs)]
    np.testing.assert_allclose(ac.state, np.array(xs), rtol=1e-8, atol=1e-8)


def test_alpha_exactly_zero_drops_the_force_as_gazebo():
    """CL uses alpha/|alpha|: NaN at alpha == 0, so Correct() zeroes the whole lift-drag force for that call."""
    ac = advanced_plane(1)
    X = ac.make_state(velocity=(12.0, 0.0, 0.0), motors=0.0)  # facing +x (east), w = 0 exactly
    F, T, info = world_wrench(ac, X)
    assert info["lift_drag0"]["alpha"][0] == 0.0
    assert np.all(F[0] == 0.0)
    assert np.any(T[0] != 0.0)  # the moments survive
    ref = ScalarAircraft(ac)
    f, t = ref.wrench_world(list(X[0]))
    np.testing.assert_allclose(T[0], t.tolist(), rtol=1e-12, atol=1e-14)
    ac2 = advanced_plane(1, alpha_zero="continuous")
    F2, _, _ = world_wrench(ac2, X)
    assert F2[0, 2] > 0  # lift with sign(0) = +1


def test_no_wrench_below_1e9_in_plane_speed_and_no_nan():
    ac = advanced_plane(3)
    X = ac.make_state(velocity=[(0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 0.0)], motors=0.0)
    X[1, 7:10] = (0.0, 5.0, 0.0)  # pure sideslip: zero in-plane speed
    F, T, _ = world_wrench(ac, X)
    assert np.all(np.isfinite(F)) and np.all(np.isfinite(T))
    assert np.all(F[0] == 0) and np.all(T[0] == 0)
    np.testing.assert_array_equal(F[1], 0.0)
    ac.reset(X)
    for _ in range(10):
        ac.step(0.0, 0.0, 0.01)
    assert np.all(np.isfinite(ac.state))


def test_per_aircraft_and_field_wind_match_the_reference():
    rng = np.random.default_rng(8)
    W = rng.uniform(-4, 4, (40, 3))
    import wingbatch as wb
    for wind in (W, wb.WindField(lambda p, t: 0.02 * p + np.array([1.0, -2.0, 0.3]))):
        ac = advanced_plane(40, wind=wind)
        X = random_states(ac, rng)
        F, T, _ = world_wrench(ac, X)
        for i in range(0, 40, 7):
            f, t = ScalarAircraft(ac, index=i).wrench_world(list(X[i]))
            np.testing.assert_allclose(F[i], f.tolist(), rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(T[i], t.tolist(), rtol=1e-11, atol=1e-11)
