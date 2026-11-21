"""JointPositionController + DART joint: literal discrete emulation vs wingbatch's first-order reduction.

gz-math PID::Update (gz-math7): cmd = offset - p*e - i*int - d*(e - e_last)/h, clamped; e = q - q_cmd.
DART GenericJoint (v6.13 detail/GenericJoint.hpp:1931): damping implicit, (I + h c) qd' = I qd + h f.
DART semi-implicit position update: q' = q + h qd'.  Joint limits: hard clamp (approximation of the limit constraint).
Run: cd D:\\Work\\wingbatch && PYTHONPATH="src;tests" python -m pytest <this file> -q
"""
import numpy as np
import pytest

from wingbatch import Aircraft
from wingbatch.systems.joint_position_controller import JointPositionControllers, ServoParams, check_reduction


def gz_servo(p, d, c, I, q_cmd, h, steps, cmd_max=1000.0, lo=-0.78, hi=0.78):
    q = qd = 0.0
    e_last = 0.0  # PID pErrLast starts at 0
    out = []
    for _ in range(steps):
        e = q - q_cmd
        f = float(np.clip(-p * e - d * (e - e_last) / h, -cmd_max, cmd_max))
        e_last = e
        qd = (I * qd + h * f) / (I + h * c)
        q = q + h * qd
        if q > hi or q < lo:
            q, qd = min(max(q, lo), hi), 0.0
        out.append(q)
    return np.array(out)


def test_advanced_plane_servo_matches_dart_discrete():
    ac = Aircraft("advanced_plane", 1, integrator="semi_implicit_euler", warn=False)
    sv = ac.servos
    h, steps = 0.004, 100
    q = np.zeros((1, sv.p.size))
    cmd = np.full((1, sv.p.size), 0.5)
    traj = []
    for _ in range(steps):
        q = sv.clip(q + h * sv.rate(q, cmd))
        traj.append(q[0, 0])
    I = ac.servos.servos[0].inertia
    ref = gz_servo(10.0, 0.0, 1.0, I, 0.5, h, steps)
    # one-step lag of the PID kick aside, the reduction tracks DART to ~I/(h c) relative
    assert np.max(np.abs(np.array(traj) - ref)) < 1e-3


@pytest.mark.parametrize("damping,d_gain", [(1.0, 1.5), (0.0, 1.0)])
def test_check_reduction_rejects_explicit_d_term_unstable_in_gz(damping, d_gain):
    """The PID derivative term is explicit (finite difference of the previous errors), not implicit like DART's
    joint damping: with d_gain > damping + 2 I / h the gz joint diverges / chatters, but check_reduction accepts
    it, so wingbatch refuses any d_gain."""
    I, h = 1e-6, 0.004
    q_gz = gz_servo(10.0, d_gain, damping, I, 0.3, h, 200, lo=-1e16, hi=1e16)
    assert not np.all(np.isfinite(q_gz)) or np.max(np.abs(q_gz)) > 1.0  # gz: diverges or chatters (target 0.3)
    sp = ServoParams("j", p_gain=10.0, i_gain=0.0, d_gain=d_gain, damping=damping, inertia=I)
    with pytest.raises(NotImplementedError):
        check_reduction(sp)
