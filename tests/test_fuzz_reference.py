"""The vectorised systems against the line-by-line reference far outside the flight envelope of test_reference.py:
zero and 1e-10 m/s, backwards flight, sideslip near +-90 deg, vertical and inverted attitudes, 20 rad/s rates,
servo angles at and beyond the joint limits, negative and over-limit rotor speeds, wind stronger than the motion.

gz-sim's AdvancedLiftDrag has three input singularities where a round-off difference of the inputs changes the
output by O(1) and the world-frame reference cannot be compared with the link-frame code:

* ``beta = atan2(v_y, v_x)`` with both ~0 (air velocity along the body z axis: flat vertical fall);
* the branch cut of beta at +-pi (backwards flight with ~0 sideslip velocity);
* the branch cut of alpha at +-pi (backwards flight with ~0 vertical velocity) and ``alpha / |alpha|`` at 0.

Those states are excluded from the comparison here and checked separately with inputs that are exact in both
frames (identity attitude, zero rates).
"""

import numpy as np
import pytest

from helpers import advanced_plane, world_wrench
from wingbatch import _vec as V
from wingbatch.reference import ScalarAircraft


def extreme_states(ac, rng):
    n = ac.n
    speed = rng.choice([0.0, 1e-10, 1e-9, 2e-9, 0.5, 5.0, 15.0, 40.0, 100.0], n) * rng.uniform(0.5, 1.5, n)
    speed[rng.random(n) < 0.1] = 0.0
    alpha = rng.choice([np.pi, -np.pi, np.pi - 1e-9, 0.339, -0.339, 1e-300, np.pi / 2, -np.pi / 2], n)
    alpha = np.where(rng.random(n) < 0.5, alpha, rng.uniform(-np.pi, np.pi, n))
    beta = rng.choice([0.0, np.pi / 2, -np.pi / 2, np.pi / 2 - 1e-12, 1.5], n)
    beta = np.where(rng.random(n) < 0.5, beta, rng.uniform(-np.pi / 2, np.pi / 2, n))
    vb = np.stack([speed * np.cos(alpha) * np.cos(beta), speed * np.sin(beta), -speed * np.sin(alpha) * np.cos(beta)], 1)
    q = rng.normal(size=(n, 4))
    special = rng.random(n) < 0.3
    choices = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [np.sqrt(.5), 0, np.sqrt(.5), 0], [np.sqrt(.5), 0, -np.sqrt(.5), 0],
                        [0, 0, 0, 1], [0, 0, 1, 0]], float)  # level, inverted, nose up/down vertical, reversed
    q[special] = choices[rng.integers(0, len(choices), special.sum())]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    vw = np.stack(V.rotate(V.quat_to_matrix(*q.T), tuple(vb.T)), 1)
    omega = rng.choice([0.0, 20.0, -20.0], (n, 3)) * rng.uniform(0.5, 1.0, (n, 3))
    servos = rng.choice([-0.78, 0.78, -1.5, 1.5, 0.0, -0.7800001, 0.7800001], (n, ac.n_servos))
    motors = rng.choice([-500.0, 0.0, 1e-12, 3500.0, 5000.0, -5000.0, 1600.0], (n, ac.n_motors))
    return ac.make_state(position=rng.uniform(-1e3, 1e3, (n, 3)), quaternion=q, velocity=vw, omega=omega,
                         servos=servos, motors=motors)


def regular(ac, X):
    """Rows away from the singularities listed in the module docstring (relative margin 1e-6)."""
    R = V.quat_to_matrix(*X[:, 3:7].T)
    wind = ac._wind(X[:, 0:3], 0.0)
    vb = np.stack(V.rotate_inv(R, tuple((X[:, 7:10] - wind).T)), 1)
    va = vb + np.cross(X[:, 10:13], ac.lift_drag[0].system.cp)
    s = np.linalg.norm(va, axis=1)
    tol = 1e-6 * s
    beta_undefined = np.hypot(va[:, 0], va[:, 1]) <= tol
    backwards = va[:, 0] < 0
    beta_cut = backwards & (np.abs(va[:, 1]) <= tol)
    alpha_cut = backwards & (np.abs(va[:, 2]) <= tol)
    alpha_zero = ~backwards & (np.abs(va[:, 2]) <= tol)
    threshold = np.abs(np.hypot(va[:, 0], va[:, 2]) - 1e-9) < 1e-12
    return ~(beta_undefined | beta_cut | alpha_cut | alpha_zero | threshold)


@pytest.mark.parametrize("kw", [{}, {"gz_cem_overwrite": True}, {"wind": (30.0, -20.0, 5.0)}],
                         ids=["default", "cem_overwrite", "strong_wind"])
def test_wrench_matches_the_reference_over_extreme_states(kw):
    ac = advanced_plane(1500, **kw)
    X = extreme_states(ac, np.random.default_rng(11))
    keep = regular(ac, X)
    assert keep.mean() > 0.6
    F, T, info = world_wrench(ac, X)
    assert np.all(np.isfinite(F)) and np.all(np.isfinite(T))
    ref = ScalarAircraft(ac)
    for i in np.flatnonzero(keep):
        f, t = ref.wrench_world(list(X[i]))
        r = np.array(f.tolist() + t.tolist())
        scale = 1.0 + np.abs(r).max()  # forces reach 1e3 N at 150 m/s and 5000 rad/s
        np.testing.assert_allclose(np.r_[F[i], T[i]], r, rtol=0, atol=1e-11 * scale, err_msg=f"row {i}")


def test_derivatives_match_the_reference_over_extreme_states_and_commands():
    ac = advanced_plane(600)
    rng = np.random.default_rng(12)
    X = extreme_states(ac, rng)
    s = rng.choice([-0.78, 0.78, -2.0, 2.0, 0.0, 0.3], (ac.n, ac.n_servos))
    m = rng.choice([-500.0, 0.0, 3500.0, 9000.0, 1600.0], (ac.n, ac.n_motors))
    d, _ = ac.derivatives(X, s, m)
    assert np.all(np.isfinite(d))
    ref = ScalarAircraft(ac)
    for i in np.flatnonzero(regular(ac, X)):
        r = np.array(ref.derivatives(list(X[i]), list(s[i]), list(m[i])))
        np.testing.assert_allclose(d[i], r, rtol=0, atol=1e-10 * (1.0 + np.abs(r).max()), err_msg=f"row {i}")


def test_branch_cuts_agree_with_the_reference_just_off_the_cut():
    """Identity attitude, zero rates, velocities 1e-12 m/s off each cut: every operation is exact in both codes, so
    both must land on the same side. (Exactly on a cut the side is set by the sign of a zero, which the two codes
    do not propagate alike: not comparable.) Across the beta cut the roll and yaw moments flip sign: that jump is
    gz-sim's, not round-off."""
    ac = advanced_plane(8)
    e = 1e-12
    vel = np.array([[-10.0, e, -1.0], [-10.0, -e, -1.0],       # beta cut
                    [-10.0, 1.0, e], [-10.0, 1.0, -e],         # alpha cut
                    [e, e, -10.0], [-e, e, -10.0], [e, -e, -10.0], [-e, -e, -10.0]])  # flat fall
    X = ac.make_state(velocity=vel, motors=0.0)
    F, T, info = world_wrench(ac, X)
    d = info["lift_drag0"]
    np.testing.assert_allclose(np.abs(d["beta"][:2]), np.pi, atol=1e-12)
    assert d["beta"][0] == -d["beta"][1]
    np.testing.assert_allclose(np.abs(d["alpha"][2:4]), np.pi, atol=1e-12)
    assert d["alpha"][2] == -d["alpha"][3]
    np.testing.assert_allclose(d["beta"][4:], [-np.pi / 4, -3 * np.pi / 4, np.pi / 4, 3 * np.pi / 4], atol=1e-12)
    np.testing.assert_allclose(T[0, [0, 2]], -T[1, [0, 2]], rtol=1e-9)
    assert abs(T[0, 2]) > 1.0
    ref = ScalarAircraft(ac)
    for i in range(ac.n):
        f, t = ref.wrench_world(list(X[i]))
        np.testing.assert_allclose(F[i], f.tolist(), rtol=1e-12, atol=1e-12, err_msg=f"row {i}")
        np.testing.assert_allclose(T[i], t.tolist(), rtol=1e-12, atol=1e-12, err_msg=f"row {i}")


def test_in_plane_speed_threshold_matches_the_reference():
    ac = advanced_plane(4)
    vel = np.array([[1e-9, 0, 0], [1.0000001e-9, 0, 0], [0.9999999e-9, 0, -0.0], [1e-10, 0, -1e-10]])
    vel[:, 2] -= 1e-30  # avoid alpha == 0 exactly
    X = ac.make_state(velocity=vel, motors=0.0)
    F, T, info = world_wrench(ac, X)
    assert np.all(F[[0, 2, 3]] == 0) and np.all(T[[0, 2, 3]] == 0)
    ref = ScalarAircraft(ac)
    for i in range(ac.n):
        f, t = ref.wrench_world(list(X[i]))
        np.testing.assert_allclose(np.r_[F[i], T[i]], f.tolist() + t.tolist(), rtol=1e-12, atol=1e-30)


def test_non_finite_states_give_the_same_wrench_as_the_reference():
    """NaN / inf in the air data: Correct() zeroes the lift-drag force in both; the motor terms stay NaN."""
    ac = advanced_plane(4)
    X = ac.make_state(velocity=(12.0, 0.0, -1.0), motors=500.0)
    X[0, 7] = np.nan
    X[1, 13] = np.inf  # servo_0: inf * CD_ctrl etc.
    X[2, 10] = np.nan  # roll rate
    X[3, ac.idx["motors"]] = np.nan
    F, T, _ = world_wrench(ac, X)
    ref = ScalarAircraft(ac)
    for i in range(ac.n):
        f, t = ref.wrench_world(list(X[i]))
        np.testing.assert_allclose(F[i], f.tolist(), rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=f"row {i}")
        np.testing.assert_allclose(T[i], t.tolist(), rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=f"row {i}")
