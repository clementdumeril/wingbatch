import numpy as np

import wingbatch as wb
from wingbatch.frames import quaternion_from_euler_ned_frd


def random_states(ac, rng, n=None, stall=True):
    """Random flight states over a wide envelope: speeds 3-25 m/s, angles of attack up to about +-60 deg (past the
    stall), sideslip, rates, any attitude, servo angles across their range, rotor 0-1600 rad/s."""
    n = ac.n if n is None else n
    speed = rng.uniform(3.0, 25.0, n)
    alpha = rng.uniform(-1.0, 1.0, n) if stall else rng.uniform(-0.2, 0.25, n)
    beta = rng.uniform(-0.4, 0.4, n)
    vb = np.stack([speed * np.cos(alpha) * np.cos(beta), speed * np.sin(beta), -speed * np.sin(alpha) * np.cos(beta)], 1)
    q = quaternion_from_euler_ned_frd(rng.uniform(-np.pi, np.pi, n), rng.uniform(-1.4, 1.4, n),
                                      rng.uniform(-np.pi, np.pi, n))
    from wingbatch import _vec as V
    R = V.quat_to_matrix(*q.T)
    vw = np.stack(V.rotate(R, tuple(vb.T)), 1)
    X = ac.make_state(position=rng.uniform(-100, 100, (n, 3)), quaternion=q, velocity=vw,
                      omega=rng.uniform(-2.0, 2.0, (n, 3)),
                      servos=rng.uniform(-0.78, 0.78, (n, ac.n_servos)),
                      motors=rng.uniform(0.0, 1600.0, (n, ac.n_motors)))
    return X


def world_wrench(ac, X, t=0.0):
    """Aircraft.wrenches rotated to world axes (torque about the body origin)."""
    from wingbatch import _vec as V
    F, T, info = ac.wrenches(X, t)
    R = V.quat_to_matrix(*X[:, 3:7].T)
    return np.stack(V.rotate(R, F), 1), np.stack(V.rotate(R, T), 1), info


def advanced_plane(n=1, **kw):
    kw.setdefault("warn", False)
    return wb.Aircraft("advanced_plane", n, **kw)
