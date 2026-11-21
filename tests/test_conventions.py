"""Signs and conventions, checked on the advanced_plane in level flight. Documented in the README."""

import numpy as np
import pytest

import wingbatch as wb
from helpers import advanced_plane
from wingbatch import px4
from wingbatch.frames import euler_ned_frd, quaternion_from_euler_ned_frd


@pytest.fixture(scope="module")
def trimmed():
    ac = advanced_plane(1)
    r = wb.trim(ac, 12.0)
    assert r["ok"][0]
    return ac, r


def torque_change(ac, r, servo, delta):
    X = r["state"].copy()
    X[:, ac.idx["servos"].start + ac.servo_names.index(servo)] += delta
    _, T0, _ = ac.wrenches(r["state"])
    _, T1, _ = ac.wrenches(X)
    return np.array([T1[i][0] - T0[i][0] for i in range(3)])  # body FLU


def test_control_surface_signs(trimmed):
    """Positive joint angles: servo_2 (elevator, direction -1) pitches the nose up; servo_0 (left elevon) rolls
    left, servo_1 (right elevon) rolls right; servo_3 (rudder) yaws the nose right. In body FLU axes: roll right
    (right wing down) is positive about x, nose up is negative about y, nose right is negative about z."""
    ac, r = trimmed
    d = np.radians(2.0)
    t = torque_change(ac, r, "servo_2", d)
    assert t[1] < 0  # nose up = negative about FLU y (left)
    t = torque_change(ac, r, "servo_0", d)
    assert t[0] < 0  # roll left = negative about x (right wing up)
    t = torque_change(ac, r, "servo_1", d)
    assert t[0] > 0  # roll right
    t = torque_change(ac, r, "servo_3", d)
    assert t[2] < 0  # nose right = negative about FLU z (up)
    for flap in ("servo_4", "servo_5"):  # not in AdvancedLiftDrag: no effect
        assert np.all(torque_change(ac, r, flap, d) == 0)


def test_px4_allocation_signs_agree(trimmed):
    """PX4's airframe 4008: CS0 roll torque -0.5, CS1 +0.5, CS2 pitch +1, CS3 yaw +1. A positive roll command
    (right) therefore drives servo_0 negative and servo_1 positive, which rolls right here too."""
    ac, r = trimmed
    t = torque_change(ac, r, "servo_0", -0.05) + torque_change(ac, r, "servo_1", 0.05)
    assert t[0] > 0


def test_thrust_drag_and_stall(trimmed):
    ac, r = trimmed
    X = r["state"].copy()
    F, T, info = ac.wrenches(X)
    assert info["thrust0"][0] > 0
    X0 = X.copy()
    X0[:, ac.idx["motors"]] = 0.0
    F0, _, _ = ac.wrenches(X0)
    v = X[0, 7:10]
    from wingbatch import _vec as V
    R = np.array(V.quat_to_matrix(*X[0, 3:7])).reshape(3, 3)
    assert (R @ np.array([f[0] for f in F0])) @ v < 0  # gliding: the air force opposes the motion
    # lift coefficient past the stall: sigmoid blend down from the linear curve
    sysm = ac.lift_drag[0].system
    alphas = np.radians(np.arange(-5, 40, 0.5) + 0.25)  # avoid alpha == 0 exactly (NaN lift, as gz)
    ones = np.ones_like(alphas)
    vb = (12 * np.cos(alphas), 0 * ones, -12 * np.sin(alphas))
    _, _, inf = sysm.wrench(vb, (0 * ones, 0 * ones, 0 * ones), np.zeros((len(alphas), ac.n_servos)))
    CL = inf["CL"]
    peak = np.degrees(alphas[np.argmax(CL)])
    assert 12 < peak < 20 and CL.max() < 1.3 and CL[-1] < 0.8 * CL.max()
    np.testing.assert_allclose(inf["alpha"], alphas, atol=1e-12)


def test_frames_and_px4_mapping():
    q = quaternion_from_euler_ned_frd(0.3, -0.2, 1.0)
    roll, pitch, yaw = euler_ned_frd(q)
    np.testing.assert_allclose([roll[0], pitch[0], yaw[0]], [0.3, -0.2, 1.0], atol=1e-12)
    ac = advanced_plane(1)
    ac.reset(quaternion=quaternion_from_euler_ned_frd(0.0, 0.0, 0.0)[0])  # heading north
    from wingbatch import _vec as V
    R = np.array(V.quat_to_matrix(*ac.quaternion[0])).reshape(3, 3)
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)  # nose to +y (north) in ENU
    assert px4.motor_speed(0.0) == 10.0 and px4.motor_speed(1.0) == 1600.0
    np.testing.assert_allclose(px4.servo_angle([-1.0, 0.0, 1.0]), np.radians([-45.0, 0.0, 45.0]))
    np.testing.assert_allclose(px4.throttle(px4.motor_speed(0.37)), 0.37)
