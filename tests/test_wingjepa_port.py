"""wingbatch against the earlier, independent port (WingJEPA / GeneticWing ``gzplane.py``, a transcription of the
Fixed-Wing C++ model): same forces and moments, same trajectories when the same simplifications are applied."""

import numpy as np
import pytest

import wingbatch as wb
from wingbatch import px4
from fixtures import wingjepa_advanced_plane as gz
from wingbatch import _vec as V
from wingbatch.frames import flu_to_frd, quaternion_from_euler_ned_frd
from wingbatch.sdf import MODELS_DIR

SERVO_ORDER = ["servo_0", "servo_1", "servo_4", "servo_5", "servo_2", "servo_3"]


@pytest.fixture(scope="module")
def exact_axis_model(tmp_path_factory):
    """The bundled SDF with the rotor pitched by exactly pi/2 (the SDF says 1.57, the old port assumes pi/2)."""
    text = (MODELS_DIR / "advanced_plane" / "model.sdf").read_text()
    assert "<pose>0.3 0 0.0 0 1.57 0</pose>" in text
    d = tmp_path_factory.mktemp("advanced_plane")
    (d / "model.sdf").write_text(text.replace("<pose>0.3 0 0.0 0 1.57 0</pose>",
                                              f"<pose>0.3 0 0.0 0 {np.pi / 2!r} 0</pose>"))
    return d / "model.sdf"


def to_wingbatch(ac, Y, gust_b):
    """gzplane state (FRD body, NED, Euler) and body-axes wind -> wingbatch state and world wind."""
    q = quaternion_from_euler_ned_frd(Y[:, gz.PHI], Y[:, gz.TH], Y[:, gz.PSI])
    R = V.quat_to_matrix(*q.T)
    vel = np.stack(V.rotate(R, tuple(flu_to_frd(Y[:, gz.U:gz.W + 1]).T)), 1)
    wind = np.stack(V.rotate(R, tuple(flu_to_frd(gust_b).T)), 1)
    de, da, dr = Y[:, gz.DE], Y[:, gz.DA], Y[:, gz.DR]
    joints = {"servo_0": da, "servo_1": -da, "servo_2": -de, "servo_3": -dr}
    servos = np.stack([joints.get(nm, np.zeros(len(Y))) for nm in ac.servo_names], 1)
    pos = np.stack([Y[:, gz.YE], Y[:, gz.XN], Y[:, gz.H]], 1)
    X = ac.make_state(position=pos, quaternion=q, velocity=vel, omega=flu_to_frd(Y[:, gz.P:gz.R + 1]),
                      servos=servos, motors=Y[:, gz.OM:gz.OM + 1])
    return X, wind


def random_gz_states(n, rng):
    Y = np.zeros((n, gz.N_STATE))
    Y[:, gz.U] = rng.uniform(4.0, 22.0, n)
    Y[:, gz.V] = rng.uniform(-3.0, 3.0, n)
    Y[:, gz.W] = rng.uniform(-4.0, 9.0, n)  # up to well past the stall
    Y[:, gz.P:gz.R + 1] = rng.uniform(-1.5, 1.5, (n, 3))
    Y[:, gz.PHI] = rng.uniform(-1.2, 1.2, n)
    Y[:, gz.TH] = rng.uniform(-0.8, 0.8, n)
    Y[:, gz.PSI] = rng.uniform(-3.0, 3.0, n)
    Y[:, gz.H] = 200.0
    Y[:, gz.DE:gz.DR + 1] = rng.uniform(-0.7, 0.7, (n, 3))
    Y[:, gz.OM] = rng.uniform(10.0, 1600.0, n)
    return Y, rng.uniform(-3.0, 3.0, (n, 3))


def test_forces_and_moments_match_the_earlier_port(exact_axis_model):
    rng = np.random.default_rng(3)
    Y, gust_b = random_gz_states(400, rng)
    old = gz.AdvancedPlane(len(Y))
    Fo = old.forces(Y, gust_b)
    ac = wb.Aircraft(exact_axis_model, len(Y), warn=False)
    X, wind = to_wingbatch(ac, Y, gust_b)
    ac.wind = wb.UniformWind(wind)
    F, T, _ = ac.wrenches(X)
    F_frd, T_frd = flu_to_frd(np.stack(F, 1)), flu_to_frd(np.stack(T, 1))
    np.testing.assert_allclose(F_frd, Fo["F_aero"] + Fo["F_prop"], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(T_frd, Fo["M_aero"] + Fo["M_prop"], rtol=1e-11, atol=1e-11)


def _fly_both(model, dt, seconds=3.0, n=30):
    rng = np.random.default_rng(4)
    Y, _ = random_gz_states(n, rng)
    Y[:, gz.W] = rng.uniform(-1.0, 2.0, n)
    Y[:, gz.DE:gz.DR + 1] = rng.uniform(-0.3, 0.3, (n, 3))
    p = gz.ADVANCED_PLANE
    mp = wb.MassProperties.simple(p["mass"], [p["Ixx"], p["Iyy"], p["Izz"]])
    ac = wb.Aircraft(model, n, mass_properties=mp, rotor_gyro=False, warn=False)  # the old port has no rotor momentum
    X, _ = to_wingbatch(ac, Y, np.zeros((n, 3)))
    ac.reset(X)
    old = gz.AdvancedPlane(n)
    cmd = rng.uniform(-0.3, 0.3, (n, 3))
    thr = rng.uniform(0.2, 0.9, n)
    zero = np.zeros((n, 3))
    servo_cmd = to_wingbatch(ac, np.column_stack([np.zeros((n, gz.DE)), cmd, np.zeros(n)]), zero)[0][:, ac.idx["servos"]]
    motor_cmd = px4.motor_speed(thr)[:, None]
    for _ in range(int(round(seconds / dt))):
        Y = old.step(Y, cmd, thr, zero, zero, dt)
        ac.step(servo_cmd, motor_cmd, dt)
    Xo, _ = to_wingbatch(ac, Y, zero)
    qdot = np.abs(np.sum(ac.quaternion * Xo[:, 3:7], 1))  # same rotation up to sign
    return (np.abs(ac.position - Xo[:, 0:3]).max(), np.abs(ac.omega - Xo[:, 10:13]).max(),
            np.abs(1.0 - qdot).max())


def test_trajectories_match_the_earlier_port_with_its_simplifications(exact_axis_model):
    """Same mass properties (1.005 kg at the origin, the old port's inertia), calm air, RK4, 3 s from random
    (violent) states with random held commands. One integrates Euler angles, the other a quaternion: the gap is
    RK4's truncation error, so it shrinks like dt^4."""
    pos10, w10, q10 = _fly_both(exact_axis_model, 0.01)
    pos5, w5, q5 = _fly_both(exact_axis_model, 0.005)
    assert pos10 < 1e-3 and w10 < 1e-3
    assert pos10 / pos5 > 8 and w10 / w5 > 8  # fourth order (16 in the limit)
    assert pos5 < 5e-5
