"""Steady straight flight (trim) for a batch, by a vectorised Levenberg-Marquardt.

Unknowns per aircraft: angle of attack, bank angle, three virtual controls (roll, pitch, yaw: each a fixed
combination of servo joint angles, see ``mixing``) and the rotor speed. Equations: zero linear and angular
acceleration with zero sideslip and zero angular velocity, relative to the aircraft's air (a uniform wind is added to
the ground velocity). The bank angle balances the propeller's reaction torque and side force.
"""

from __future__ import annotations

import numpy as np

from . import _vec as V
from .frames import ENU_TO_NED, FLU_TO_FRD, _matrix_to_quat

ADVANCED_PLANE_MIXING = {
    "roll": {"servo_0": 1.0, "servo_1": -1.0},
    "pitch": {"servo_2": 1.0},
    "yaw": {"servo_3": 1.0},
}


def _mixing_matrix(ac, mixing):
    M = np.zeros((ac.n_servos, 3))
    for j, axis in enumerate(("roll", "pitch", "yaw")):
        for name, k in mixing.get(axis, {}).items():
            if name not in ac.servo_names:
                raise ValueError(f"mixing: {name!r} is not a servo joint ({ac.servo_names})")
            M[ac.servo_names.index(name), j] = k
    return M


def steady_state(ac, airspeed, alpha, gamma=0.0, bank=0.0, heading=0.0, altitude=100.0, servos=None, motors=None):
    """State of steady flight: airspeed (m/s) at angle of attack ``alpha``, flight-path angle ``gamma``, bank about
    the velocity vector, heading (aerospace angles, rad, heading from north), zero sideslip and angular velocity."""
    n = ac.n
    alpha, gamma, bank, heading, Va, altitude = (np.broadcast_to(np.asarray(a, float), (n,))
                                                 for a in (alpha, gamma, bank, heading, airspeed, altitude))
    q = _attitude_for(ac, alpha, gamma, bank, heading)
    R = V.quat_to_matrix(q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    pos = np.stack([np.zeros(n), np.zeros(n), altitude], -1)
    vw = np.stack(V.rotate(R, (Va * np.cos(alpha), np.zeros(n), -Va * np.sin(alpha))), -1) + ac._wind(pos, ac.t)
    return ac.make_state(position=pos, quaternion=q, velocity=vw, servos=servos, motors=motors)


def _attitude_for(ac, alpha, gamma, bank, heading):
    """Body attitude with velocity vector at flight-path angle gamma and heading, alpha and bank about it."""
    cg, sg, ch, sh = np.cos(gamma), np.sin(gamma), np.cos(heading), np.sin(heading)
    cb, sb, ca, sa = np.cos(bank), np.sin(bank), np.cos(alpha), np.sin(alpha)
    # wind axes in NED: R_w = Rz(heading) Ry(gamma) Rx(bank); body = R_w Ry(alpha)
    Rz = np.array([[ch, -sh, 0 * ch], [sh, ch, 0 * ch], [0 * ch, 0 * ch, 1 + 0 * ch]])
    Ry_g = np.array([[cg, 0 * cg, sg], [0 * cg, 1 + 0 * cg, 0 * cg], [-sg, 0 * cg, cg]])
    Rx_b = np.array([[1 + 0 * cb, 0 * cb, 0 * cb], [0 * cb, cb, -sb], [0 * cb, sb, cb]])
    Ry_a = np.array([[ca, 0 * ca, sa], [0 * ca, 1 + 0 * ca, 0 * ca], [-sa, 0 * ca, ca]])
    Rn = np.einsum("ijn,jkn,kln,lmn->imn", Rz, Ry_g, Rx_b, Ry_a)  # NED <- FRD
    Re = np.einsum("ij,jkn,kl->iln", ENU_TO_NED.T, Rn, FLU_TO_FRD.T)  # ENU <- FLU
    return _matrix_to_quat(Re)


def trim(ac, airspeed, gamma=0.0, altitude=100.0, heading=0.0, mixing=None, max_rotor: float | None = None,
         iters: int = 60, tol: float = 1e-8):
    """Trim every aircraft of ``ac`` (an :class:`wingbatch.Aircraft`).

    Returns a dict: ``state`` (n, nx) ready for :meth:`Aircraft.reset`, ``servo_cmd`` (n, k), ``motor_cmd``
    (n, m), ``alpha``, ``bank``, ``controls`` (n, 3: roll, pitch, yaw virtual controls, rad), ``ok`` (n,),
    ``residual`` (n,). ``mixing`` defaults to the advanced_plane's (elevons for roll, elevator, rudder).

    ``ok`` means: converged, servos within their joint limits, rotor speed in (0, ``max_rotor``) (default: the
    SDF ``maxRotVelocity``; pass e.g. 1600 for PX4's ESC range) and angle of attack below ``alpha_stall`` (at very
    low speed the solver can otherwise "trim" a propeller hang at 80 deg)."""
    if ac.n_motors != 1:
        raise NotImplementedError("trim handles exactly one motor")
    mixing = ADVANCED_PLANE_MIXING if mixing is None else mixing
    Mx = _mixing_matrix(ac, mixing)
    n = ac.n
    Va, gamma, h, psi = (np.broadcast_to(np.asarray(a, float), (n,)).copy() for a in (airspeed, gamma, altitude, heading))
    wmax = ac.motors[0].params.maxRotVelocity
    x = np.stack([np.full(n, 0.05), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, 0.5 * wmax)], 1)
    scale_x = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1000.0])

    def unpack(x):
        a, phi, cr, cpch, cy, w = x.T
        servos = np.stack([cr, cpch, cy], 1) @ Mx.T
        q = _attitude_for(ac, a, gamma, phi, psi)
        ca, sa = np.cos(a), np.sin(a)
        R = V.quat_to_matrix(q[:, 0], q[:, 1], q[:, 2], q[:, 3])
        vb = (Va * ca, np.zeros(n), -Va * sa)
        pos = np.stack([np.zeros(n), np.zeros(n), h], -1)
        vw = np.stack(V.rotate(R, vb), -1) + ac._wind(pos, ac.t)
        X = ac.make_state(position=pos, quaternion=q, velocity=vw, servos=servos, motors=w[:, None])
        return X, servos

    def residual(x):
        X, _ = unpack(x)
        F, T, _ = ac.wrenches(X, ac.t)
        c = X.T
        acc, wdot = ac._accelerations(c[3], c[4], c[5], c[6], (c[10], c[11], c[12]), F, T)
        R = V.quat_to_matrix(c[3], c[4], c[5], c[6])
        ab = V.rotate_inv(R, acc)
        return np.stack([*ab, *wdot], 1)

    lo = np.array([-0.3, -0.5, -0.5, -0.5, -0.5, -wmax])
    eps = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-3])
    for _ in range(iters):
        r = residual(x)
        J = np.empty((n, 6, 6))
        for j in range(6):
            xk = x.copy()
            xk[:, j] += eps[j]
            J[:, :, j] = (residual(xk) - r) / eps[j]
        Js = J * scale_x[None, None, :]
        JT = np.swapaxes(Js, 1, 2)
        A = JT @ Js
        A = A + 1e-9 * np.eye(6) * np.diagonal(A, axis1=1, axis2=2)[:, None, :] + 1e-14 * np.eye(6)
        dx = np.linalg.solve(A, -(JT @ r[..., None]))[..., 0] * scale_x
        x = x + np.clip(dx, lo * 0.2, -lo * 0.2)
        x[:, 5] = np.clip(x[:, 5], 0.0, wmax)
        if np.all(np.abs(r).max(1) < tol):
            break
    r = residual(x)
    X, servos = unpack(x)
    res = np.abs(r).max(1)
    ok = (res < 1e-6) & np.all(servos >= ac.servos.lower, 1) & np.all(servos <= ac.servos.upper, 1)
    ok &= (x[:, 5] > 0) & (x[:, 5] < (wmax if max_rotor is None else min(max_rotor, wmax)))
    if ac.lift_drag:
        ok &= np.abs(x[:, 0]) < ac.lift_drag[0].system.params.alpha_stall
    return dict(state=X, servo_cmd=servos, motor_cmd=x[:, 5:6], alpha=x[:, 0], bank=x[:, 1], controls=x[:, 2:5],
                ok=ok, residual=res)
