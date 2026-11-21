"""Batched 3-vector and quaternion helpers.

A batched vector is a tuple ``(x, y, z)`` of arrays of shape ``(n,)``: every formula is written once for the
``n`` aircraft and each operation is a single NumPy call on contiguous memory. Quaternions are ``(w, x, y, z)``
and rotate from the link frame to the world frame, as gz-math's ``Quaterniond``.
"""

from __future__ import annotations

import numpy as np


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(s, a):
    return (s * a[0], s * a[1], s * a[2])


def quat_to_matrix(qw, qx, qy, qz):
    """Rotation matrix entries ``R[i][j]`` (each an array): ``v_world = R @ v_link``. The quaternion need not be
    unit (inside a Runge-Kutta step it is not exactly): the matrix is that of ``q / |q|``."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    s = 2.0 / (qw * qw + xx + yy + zz)
    return ((1 - s * (yy + zz), s * (xy - wz), s * (xz + wy)),
            (s * (xy + wz), 1 - s * (xx + zz), s * (yz - wx)),
            (s * (xz - wy), s * (yz + wx), 1 - s * (xx + yy)))


def rotate(R, v):
    """``R @ v`` (link to world)."""
    return (R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
            R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
            R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2])


def rotate_inv(R, v):
    """``R.T @ v`` (world to link)."""
    return (R[0][0] * v[0] + R[1][0] * v[1] + R[2][0] * v[2],
            R[0][1] * v[0] + R[1][1] * v[1] + R[2][1] * v[2],
            R[0][2] * v[0] + R[1][2] * v[1] + R[2][2] * v[2])


def const_rotate(M, v):
    """``M @ v`` for a constant (3, 3) matrix ``M`` and a batched vector ``v``."""
    return tuple(M[i, 0] * v[0] + M[i, 1] * v[1] + M[i, 2] * v[2] for i in range(3))


def quat_derivative(qw, qx, qy, qz, wx, wy, wz):
    """``dq/dt = q (x) (0, w) / 2`` for a body-frame angular velocity ``w``."""
    return (-0.5 * (qx * wx + qy * wy + qz * wz),
            0.5 * (qw * wx + qy * wz - qz * wy),
            0.5 * (qw * wy + qz * wx - qx * wz),
            0.5 * (qw * wz + qx * wy - qy * wx))


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def quat_exp(wx, wy, wz, dt):
    """Unit quaternion of the rotation ``w * dt`` (body-frame angular velocity held over ``dt``)."""
    angle = np.sqrt(wx * wx + wy * wy + wz * wz) * dt
    half = 0.5 * angle
    small = angle < 1e-12
    k = np.where(small, 0.5 * dt, np.sin(half) / np.where(small, 1.0, angle) * dt)
    return (np.cos(half), k * wx, k * wy, k * wz)


def euler_to_quat(roll, pitch, yaw):
    """SDF / gz-math convention: R = Rz(yaw) Ry(pitch) Rx(roll). Works on scalars and arrays."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy)


def quat_to_euler(qw, qx, qy, qz):
    """Inverse of :func:`euler_to_quat` (roll, pitch, yaw)."""
    from .frames import _euler_zyx

    return _euler_zyx(np.array(quat_to_matrix(qw, qx, qy, qz), dtype=float).reshape(3, 3, -1))
