"""Conversions between Gazebo's frames (world ENU, body FLU) and the aerospace ones (NED, FRD)."""

from __future__ import annotations

import numpy as np

from . import _vec as V

ENU_TO_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


def enu_to_ned(v):
    v = np.asarray(v, float)
    return np.stack([v[..., 1], v[..., 0], -v[..., 2]], axis=-1)


ned_to_enu = enu_to_ned


def flu_to_frd(v):
    v = np.asarray(v, float)
    return np.stack([v[..., 0], -v[..., 1], -v[..., 2]], axis=-1)


frd_to_flu = flu_to_frd


def euler_ned_frd(quaternion):
    """Aerospace roll, pitch, yaw (rad) of the FRD body in NED for a gz quaternion (ENU <- FLU), arrays (n,)."""
    q = np.asarray(quaternion, float).reshape(-1, 4)
    R = np.array(V.quat_to_matrix(q[:, 0], q[:, 1], q[:, 2], q[:, 3]))  # (3, 3, n)
    Rn = np.einsum("ij,jkn,kl->iln", ENU_TO_NED, R, FLU_TO_FRD)
    return _euler_zyx(Rn)


def _euler_zyx(R):
    """roll, pitch, yaw of R = Rz(yaw) Ry(pitch) Rx(roll) (entries (3, 3, n)); at pitch = +-90 deg, roll = 0."""
    pitch = np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))
    lock = np.abs(R[2, 0]) > 1 - 1e-12
    roll = np.where(lock, 0.0, np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.where(lock, np.arctan2(-R[0, 1], R[1, 1]), np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def quaternion_from_euler_ned_frd(roll, pitch, yaw):
    """gz quaternion (w, x, y, z), ENU <- FLU, for aerospace roll, pitch, yaw (NED / FRD); arrays (n, 4)."""
    roll, pitch, yaw = (np.atleast_1d(np.asarray(a, float)) for a in (roll, pitch, yaw))
    qn = np.stack(V.euler_to_quat(roll, pitch, yaw), axis=-1)  # NED <- FRD, same ZYX convention
    Rn = np.array(V.quat_to_matrix(qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]))
    Re = np.einsum("ij,jkn,kl->iln", ENU_TO_NED.T, Rn, FLU_TO_FRD.T)
    return _matrix_to_quat(Re)


def _matrix_to_quat(R):
    """Unit quaternion (w >= 0) of a rotation matrix: R (3, 3) -> (4,), or R (3, 3, n) -> (n, 4)."""
    R = np.asarray(R, float)
    single = R.ndim == 2
    if single:
        R = R[:, :, None]
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    q = np.empty((R.shape[2], 4))
    tr = m00 + m11 + m22
    with np.errstate(invalid="ignore", divide="ignore"):
        cases = [
            (tr > 0, lambda s: np.stack([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                                          (R[1, 0] - R[0, 1]) / s], -1), 2.0 * np.sqrt(np.maximum(tr + 1.0, 0))),
            ((m00 >= m11) & (m00 >= m22), lambda s: np.stack([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s,
                                                              (R[0, 2] + R[2, 0]) / s], -1),
             2.0 * np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0))),
            (m11 >= m22, lambda s: np.stack([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s,
                                             (R[1, 2] + R[2, 1]) / s], -1), 2.0 * np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 0))),
            (np.ones_like(tr, bool), lambda s: np.stack([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                                                         (R[1, 2] + R[2, 1]) / s, 0.25 * s], -1),
             2.0 * np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 0))),
        ]
        done = np.zeros(R.shape[2], bool)
        for cond, fn, s in cases:
            sel = cond & ~done
            if np.any(sel):
                q[sel] = fn(s)[sel]
            done |= sel
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)
    return q[0] if single else q
