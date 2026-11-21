"""Rigid body, integrators, wind, batching."""

import numpy as np
import pytest

import wingbatch as wb
from helpers import advanced_plane, random_states

BOX = """<?xml version="1.0"?>
<sdf version="1.9"><model name="box">
  <link name="body">
    <inertial><pose>{com} 0 0 0</pose><mass>2.0</mass>
      <inertia><ixx>0.3</ixx><iyy>0.5</iyy><izz>0.7</izz><ixy>0.01</ixy><ixz>0.02</ixz><iyz>-0.03</iyz></inertia>
    </inertial>
  </link>
</model></sdf>"""


def box(tmp_path, com="0 0 0", n=1, **kw):
    f = tmp_path / "box.sdf"
    f.write_text(BOX.format(com=com))
    return wb.Aircraft(f, n, warn=False, **kw)


@pytest.mark.parametrize("integrator", ["rk4", "semi_implicit_euler"])
def test_free_fall(tmp_path, integrator):
    ac = box(tmp_path, integrator=integrator)
    ac.reset(position=(0.0, 0.0, 100.0), velocity=(1.0, 2.0, 3.0))
    for _ in range(100):
        ac.step(dt=0.01)
    t = 1.0
    if integrator == "rk4":  # exact for a quadratic
        np.testing.assert_allclose(ac.position[0], [1.0, 2.0, 100 + 3 * t - 0.5 * 9.8 * t * t], atol=1e-10)
    else:  # semi-implicit Euler: z_k = z0 + v0 t - g dt^2 k (k + 1) / 2
        np.testing.assert_allclose(ac.position[0, 2], 100 + 3 * t - 9.8 * 0.01**2 * 100 * 101 / 2, atol=1e-10)
    np.testing.assert_allclose(ac.velocity[0], [1.0, 2.0, 3.0 - 9.8 * t], atol=1e-10)


def test_torque_free_tumbling_conserves_momentum_and_energy(tmp_path):
    """Off-centre centre of mass, products of inertia, no gravity: the centre of mass moves in a straight line,
    the world angular momentum about it and the kinetic energy stay constant."""
    ac = box(tmp_path, com="0.1 -0.05 0.02", world=wb.World(gravity=(0.0, 0.0, 0.0)))
    ac.reset(velocity=(1.0, 0.0, 0.0), omega=(0.5, 2.0, -1.0))
    from wingbatch import _vec as V
    mp = ac.mass

    def invariants():
        q, w = ac.quaternion[0], ac.omega[0]
        R = np.array(V.quat_to_matrix(*q)).reshape(3, 3)
        com = ac.position[0] + R @ mp.com
        vcom = ac.velocity[0] + np.cross(R @ w, R @ mp.com)
        L = R @ (mp.inertia @ w)
        E = 0.5 * w @ mp.inertia @ w
        return com, vcom, L, E

    com0, v0, L0, E0 = invariants()
    for _ in range(500):
        ac.step(dt=0.002)
    com, v, L, E = invariants()
    np.testing.assert_allclose(v, v0, atol=1e-12)
    np.testing.assert_allclose(com, com0 + v0 * 1.0, atol=1e-10)
    np.testing.assert_allclose(L, L0, rtol=1e-7)
    np.testing.assert_allclose(E, E0, rtol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(ac.quaternion[0]), 1.0, atol=1e-14)


def test_uniform_wind_only_carries_the_aircraft():
    """Same air-relative initial state with and without a uniform wind: identical air-relative flights, the windy
    one displaced by W t."""
    W = np.array([4.0, -3.0, 0.5])
    calm = advanced_plane(1)
    r = wb.trim(calm, 13.0)
    calm.reset(r["state"])
    windy = advanced_plane(1, wind=W)
    X = r["state"].copy()
    X[:, 7:10] += W
    windy.reset(X)
    cmd = r["servo_cmd"] + np.array([0.05, -0.05, 0.0, 0.0, 0.1, 0.02])
    for _ in range(300):
        calm.step(cmd, r["motor_cmd"], 0.01)
        windy.step(cmd, r["motor_cmd"], 0.01)
    np.testing.assert_allclose(windy.position, calm.position + W * 3.0, atol=1e-8)
    np.testing.assert_allclose(windy.velocity - W, calm.velocity, atol=1e-9)
    np.testing.assert_allclose(windy.quaternion, calm.quaternion, atol=1e-10)


def test_every_aircraft_of_a_batch_flies_as_if_alone():
    rng = np.random.default_rng(5)
    batch = advanced_plane(64)
    X = random_states(batch, rng, stall=False)
    s = rng.uniform(-0.4, 0.4, (64, batch.n_servos))
    m = rng.uniform(0, 1600, (64, 1))
    batch.reset(X)
    for _ in range(50):
        batch.step(s, m, 0.01)
    for i in (0, 17, 63):
        one = advanced_plane(1)
        one.reset(X[i:i + 1])
        for _ in range(50):
            one.step(s[i:i + 1], m[i:i + 1], 0.01)
        np.testing.assert_allclose(one.state[0], batch.state[i], rtol=1e-12, atol=1e-12)


def test_semi_implicit_euler_converges_to_rk4():
    """The Gazebo-like stepping is first order: halving its step halves its distance to the RK4 flight."""
    def fly(integrator, dt):
        ac = advanced_plane(1, integrator=integrator)
        r = wb.trim(ac, 12.0)
        ac.reset(r["state"])
        cmd = r["servo_cmd"] + np.array([0.1, -0.1, 0.0, 0.0, 0.1, 0.05])
        for _ in range(int(round(2.0 / dt))):
            ac.step(cmd, r["motor_cmd"] * 1.2, dt)
        return ac.position[0]
    ref = fly("rk4", 0.001)
    e1 = np.linalg.norm(fly("semi_implicit_euler", 0.004) - ref)
    e2 = np.linalg.norm(fly("semi_implicit_euler", 0.002) - ref)
    assert e1 < 0.5
    assert 1.6 < e1 / e2 < 2.5


def test_rotor_filter_discrete_matches_gz_and_continuous_lag():
    ac = advanced_plane(1)
    mot = ac.motors[0]
    w = np.array([10.0])
    for _ in range(250):  # 1 s at gz's 4 ms
        w = mot.filter_step(w, np.array([1600.0]), 0.004)
    assert abs(w[0] - 1600.0) < 1e-6
    w1 = mot.filter_step(np.array([100.0]), np.array([1600.0]), 0.004)
    np.testing.assert_allclose(w1, 1600 + (100 - 1600) * np.exp(-0.004 / 0.0125))
    w2 = mot.filter_step(np.array([1600.0]), np.array([100.0]), 0.004)
    np.testing.assert_allclose(w2, 100 + (1600 - 100) * np.exp(-0.004 / 0.025))
    assert mot.command(np.array([5000.0]))[0] == 3500.0  # maxRotVelocity of the SDF
