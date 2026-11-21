"""Rigid body with an independent matrix reference, DART >= 6.13 FreeJoint step, frames, trim, API edge cases
(regressions from the review)."""
import numpy as np
import pytest

import wingbatch as wb
from wingbatch import _vec as V
from wingbatch.frames import _matrix_to_quat, enu_to_ned, euler_ned_frd, quaternion_from_euler_ned_frd
from wingbatch.trim import trim

BOX = """<?xml version="1.0"?>
<sdf version="1.9"><model name="box">
  <link name="body"><inertial><pose>0.1 -0.05 0.02 0 0 0</pose><mass>2.0</mass>
    <inertia><ixx>0.3</ixx><iyy>0.5</iyy><izz>0.7</izz><ixy>0.01</ixy><ixz>0.02</ixz><iyz>-0.03</iyz></inertia>
  </inertial></link>
  <link name="nograv"><pose>-0.3 0.2 0.1 0 0 0</pose><gravity>0</gravity>
    <inertial><mass>0.5</mass><inertia><ixx>0.01</ixx><iyy>0.02</iyy><izz>0.03</izz></inertia></inertial></link>
  <joint name="fix" type="fixed"><parent>body</parent><child>nograv</child></joint>
</model></sdf>"""


@pytest.fixture
def forced_box(tmp_path):
    f = tmp_path / "box.sdf"
    f.write_text(BOX)
    rng = np.random.default_rng(0)
    n = 4
    Fb, pb, Cb = rng.normal(size=(n, 3)), 0.3 * rng.normal(size=(n, 3)), 0.1 * rng.normal(size=(n, 3))
    To = np.cross(pb, Fb) + Cb

    def make(**kw):
        ac = wb.Aircraft(f, n, warn=False, **kw)
        ac.wrenches = lambda X, t=None: (tuple(Fb.T.copy()), tuple(To.T.copy()), {})
        return ac
    q = rng.normal(size=(n, 4))
    X = make().make_state(position=rng.normal(size=(n, 3)), quaternion=q, velocity=5 * rng.normal(size=(n, 3)),
                          omega=2 * rng.normal(size=(n, 3)))
    return make, X, Fb, pb, Cb


def test_newton_euler_offset_com_partial_gravity_matches_world_frame_reference(forced_box):
    make, X, Fb, pb, Cb = forced_box
    ac = make()
    mp, g = ac.mass, ac.gravity
    dX, _ = ac.derivatives(X, np.zeros((ac.n, 0)), np.zeros((ac.n, 0)))
    for i in range(ac.n):
        R = np.array(V.quat_to_matrix(*X[i, 3:7])).reshape(3, 3)
        w = R @ X[i, 10:13]
        Iw = R @ mp.inertia @ R.T
        tau = R @ (np.cross(pb[i] - mp.com, Fb[i]) + Cb[i]) + np.cross(R @ (mp.gravity_com - mp.com), mp.gravity_mass * g)
        wd = np.linalg.solve(Iw, tau - np.cross(w, Iw @ w))
        rc = R @ mp.com
        a_o = (R @ Fb[i] + mp.gravity_mass * g) / mp.mass - np.cross(wd, rc) - np.cross(w, np.cross(w, rc))
        np.testing.assert_allclose(dX[i, 7:10], a_o, atol=1e-12)
        np.testing.assert_allclose(dX[i, 10:13], R.T @ wd, atol=1e-12)


def test_semi_implicit_step_equals_dart_6_13_freejoint(forced_box):
    """dart/dynamics/FreeJoint.cpp (v6.13): integrateVelocities adds w x v to the spatial acceleration, Euler on
    [w; v_body]; integratePositions Q <- Q [exp(w dt), v dt], then re-expresses V with AdR(Qdiff^-1)."""
    make, X, *_ = forced_box
    ac = make(integrator="semi_implicit_euler")
    dX, _ = make().derivatives(X, np.zeros((ac.n, 0)), np.zeros((ac.n, 0)))
    ac.reset(X)
    dt = 0.004
    Y = ac.step(dt=dt)
    for i in range(ac.n):
        R = np.array(V.quat_to_matrix(*X[i, 3:7])).reshape(3, 3)
        w, vb = X[i, 10:13], R.T @ X[i, 7:10]
        a_spatial = R.T @ dX[i, 7:10] - np.cross(w, vb)
        Vn = np.r_[w, vb] + dt * np.r_[dX[i, 10:13], a_spatial + np.cross(w, vb)]
        th = np.linalg.norm(Vn[:3] * dt); k = Vn[:3] * dt / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rd = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        q = _matrix_to_quat(R @ Rd)
        np.testing.assert_allclose(Y[i, 0:3], X[i, 0:3] + R @ Vn[3:] * dt, atol=1e-12)
        np.testing.assert_allclose(Y[i, 7:10], R @ Vn[3:], atol=1e-12)        # = R_new AdR(Rd^T) v
        np.testing.assert_allclose(Y[i, 10:13], Rd.T @ Vn[:3], atol=1e-12)
        np.testing.assert_allclose(Y[i, 3:7] * np.sign(Y[i, 3] * q[0]), q, atol=1e-12)


@pytest.mark.parametrize("gamma", [-0.1, 0.0, 0.12])
def test_trim_is_steady_with_requested_speed_gamma_and_no_sideslip(gamma):
    ac = wb.Aircraft("advanced_plane", 3, warn=False)
    Va = np.array([15.0, 19.0, 24.0])
    r = trim(ac, Va, gamma=gamma, heading=[0.0, 1.0, -2.0])
    assert r["ok"].all()
    ac.reset(r["state"])
    s, a, b = ac.airspeed()
    np.testing.assert_allclose(s, Va, atol=1e-9); np.testing.assert_allclose(b, 0, atol=1e-9)
    vn = enu_to_ned(r["state"][:, 7:10])
    np.testing.assert_allclose(np.arcsin(-vn[:, 2] / Va), gamma, atol=1e-12)  # climb positive
    for _ in range(1000):
        ac.step(r["servo_cmd"], r["motor_cmd"], 0.004)
    np.testing.assert_allclose(ac.state[:, 7:10], r["state"][:, 7:10], atol=1e-6)
    np.testing.assert_allclose(ac.omega, 0, atol=1e-8)


def test_trim_respects_the_aircraft_wind():
    ac = wb.Aircraft("advanced_plane", 1, warn=False, wind=(0.0, 5.0, 0.0))
    r = trim(ac, 18.0)
    ac.reset(r["state"])
    s, a, b = ac.airspeed()
    assert not r["ok"].any() or (abs(s[0] - 18.0) < 1e-6 and abs(b[0]) < 1e-6)


def test_rollout_accepts_none_like_step():
    ac = wb.Aircraft("advanced_plane", 2, warn=False)
    out = ac.rollout(None, np.full((3, 2, 1), 500.0))
    assert np.isfinite(out).all()


def test_rollout_2d_commands_are_constant_and_need_steps():
    """A 2-D command is (n, k), held constant: never read as a time sequence, even when T would equal n."""
    ac = wb.Aircraft("advanced_plane", 3, warn=False)
    with pytest.raises(ValueError, match="steps"):
        ac.rollout(np.zeros((3, ac.n_servos)), 500.0)
    assert ac.rollout(np.zeros((3, ac.n_servos)), 500.0, steps=4).shape == (5, 3, ac.nx)


def test_rollout_broadcasts_a_length_one_time_axis():
    ac = wb.Aircraft("advanced_plane", 3, warn=False)
    assert ac.rollout(np.zeros((1, 3, ac.n_servos)), np.full((5, 3, 1), 500.0)).shape[0] == 6


def test_failed_reset_keeps_previous_state():
    ac = wb.Aircraft("advanced_plane", 3, warn=False)
    before = ac.state.copy()
    with pytest.raises(ValueError):
        ac.reset(np.zeros((2, ac.nx)))
    np.testing.assert_array_equal(ac.state, before)


@pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2])
def test_euler_at_gimbal_lock_reproduces_the_attitude(pitch):
    q = quaternion_from_euler_ned_frd(0.3, pitch, 0.5)
    q2 = quaternion_from_euler_ned_frd(*[a[0] for a in euler_ned_frd(q)])
    assert 1 - abs(q[0] @ q2[0]) < 1e-12
