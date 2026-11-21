# Upstream copyright lines of gz-sim's AdvancedLiftDrag.cc and MulticopterMotorModel.cc:
# Copyright (C) 2019 Open Source Robotics Foundation
# Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
# Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
# Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
# Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
# Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
# Copyright 2016 Geoffrey Hunter <gbmhunter@gmail.com>
# Copyright (C) 2022 Benjamin Perseghetti, Rudis Laboratories
# Copyright (C) 2026 the wingbatch authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Literal transcription of gz-sim (branch gz-sim8, commit f380a94e) AdvancedLiftDrag.cc::Update and
# MulticopterMotorModel.cc::UpdateForcesAndMoments, in the WORLD frame with plain Python floats, statement by
# statement, variable names kept. It shares no code with the vectorised systems.
"""Scalar, world-frame, line-by-line reference of the gz-sim systems, and a scalar simulator built on it.

Two uses: the test oracle for the vectorised code (same numbers to round-off, from an independent reading of the
C++), and the "one aircraft at a time" baseline of the benchmarks. It is slow on purpose: no NumPy.
"""

from __future__ import annotations

import math


# ------------------------------------------------------------------------------------------ gz-math Vector3d
class Vec:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o):
        return Vec(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vec(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s):
        return Vec(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def __truediv__(self, s):
        return Vec(self.x / s, self.y / s, self.z / s)

    def __neg__(self):
        return Vec(-self.x, -self.y, -self.z)

    def Dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def Cross(self, o):
        return Vec(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)

    def Length(self):
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def Normalized(self):
        n = self.Length()
        return Vec(self.x / n, self.y / n, self.z / n)

    def Correct(self):
        """gz-math: non-finite components set to zero."""
        return Vec(*(c if math.isfinite(c) else 0.0 for c in (self.x, self.y, self.z)))

    def tolist(self):
        return [self.x, self.y, self.z]


class Quat:
    """Unit quaternion (w, x, y, z), ``RotateVector`` as gz-math."""

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w, x, y, z):
        self.w, self.x, self.y, self.z = float(w), float(x), float(y), float(z)

    def Normalized(self):
        n = math.sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
        return Quat(self.w / n, self.x / n, self.y / n, self.z / n)

    def RotateVector(self, v):
        w, x, y, z = self.w, self.x, self.y, self.z
        # q v q*, expanded
        tx = 2 * (y * v.z - z * v.y)
        ty = 2 * (z * v.x - x * v.z)
        tz = 2 * (x * v.y - y * v.x)
        return Vec(v.x + w * tx + (y * tz - z * ty), v.y + w * ty + (z * tx - x * tz), v.z + w * tz + (x * ty - y * tx))

    def Inverse(self):
        return Quat(self.w, -self.x, -self.y, -self.z)

    def __mul__(self, b):
        aw, ax, ay, az = self.w, self.x, self.y, self.z
        return Quat(aw * b.w - ax * b.x - ay * b.y - az * b.z, aw * b.x + ax * b.w + ay * b.z - az * b.y,
                    aw * b.y - ax * b.z + ay * b.w + az * b.x, aw * b.z + ax * b.y - ay * b.x + az * b.w)


def _v(a):
    return Vec(*a)


# ------------------------------------------------------------------------------------------ AdvancedLiftDrag
def advanced_lift_drag(P, worldLinVel, worldAngVel, rot, AngVel, joint_positions, wind=None, cem_overwrite=False):
    """AdvancedLiftDragPrivate::Update. ``P``: an :class:`AdvancedLiftDragParams`; ``rot``: link world rotation
    (Quat); ``AngVel``: angular velocity in link axes; ``joint_positions``: {joint name: position (rad)}.
    Returns (force, totalTorque) in world axes, torque about the link origin; (None, None) when gz-sim returns
    without applying a wrench."""
    forward = _v(P.forward).Normalized()
    upward = _v(P.upward).Normalized()
    cp = _v(P.cp)
    cpWorld = rot.RotateVector(cp)
    air_velocity = worldLinVel + worldAngVel.Cross(cpWorld)
    if wind is not None:
        air_velocity = worldLinVel + worldAngVel.Cross(cpWorld) - wind
    body_x_axis = rot.RotateVector(forward)
    body_z_axis = -1 * (rot.RotateVector(upward))
    body_y_axis = body_z_axis.Cross(body_x_axis)
    velInLDPlane = air_velocity - air_velocity.Dot(body_y_axis) * body_y_axis
    speedInLDPlane = velInLDPlane.Length()
    if speedInLDPlane <= 1e-9:
        return None, None
    stability_x_axis = velInLDPlane / speedInLDPlane
    stability_y_axis = body_y_axis
    stability_z_axis = stability_x_axis.Cross(stability_y_axis)
    span = math.sqrt(P.area * P.AR)
    mac = P.mac
    if abs(mac - 0.0) <= 1e-9:
        mac = P.area / span
    rr = AngVel.x
    pr = -1 * AngVel.y
    yr = -1 * AngVel.z
    stabx_proj_bodyx = stability_x_axis.Dot(body_x_axis)
    stabx_proj_bodyz = stability_x_axis.Dot(body_z_axis)
    alpha = math.atan2(stabx_proj_bodyz, stabx_proj_bodyx)
    sinAlpha = math.sin(alpha)
    cosAlpha = math.cos(alpha)
    velSW = air_velocity.Dot(body_y_axis)
    velFW = air_velocity.Dot(body_x_axis)
    beta = math.atan2(velSW, velFW)
    dyn_pres = 0.5 * P.air_density * speedInLDPlane * speedInLDPlane
    half_rho_vel = 0.5 * P.air_density * speedInLDPlane
    M, alphaStall = P.M, P.alpha_stall
    sigma = (1 + math.exp(-1 * M * (alpha - alphaStall)) + math.exp(M * (alpha + alphaStall))) / (
        (1 + math.exp(-1 * M * (alpha - alphaStall))) * (1 + math.exp(M * (alpha + alphaStall))))
    sgn = alpha / abs(alpha) if alpha != 0.0 else math.nan  # 0/0 in C++
    CL = (1 - sigma) * (P.CL0 + P.CLa * alpha) + sigma * (2 * sgn * sinAlpha * sinAlpha * cosAlpha)
    CL = CL + P.CLb * beta
    CL_ctrl_tot = CD_ctrl_tot = CY_ctrl_tot = Cell_ctrl_tot = Cem_ctrl_tot = Cen_ctrl_tot = 0.0
    for cs in P.control_surfaces:
        controlAngle = joint_positions.get(cs.name, 0.0) * 180 / math.pi
        CL_ctrl_tot += controlAngle * cs.CL_ctrl * cs.direction
        CD_ctrl_tot += controlAngle * cs.CD_ctrl * cs.direction
        CY_ctrl_tot += controlAngle * cs.CY_ctrl * cs.direction
        Cell_ctrl_tot += controlAngle * cs.Cell_ctrl * cs.direction
        Cem_ctrl_tot += controlAngle * cs.Cem_ctrl * cs.direction
        Cen_ctrl_tot += controlAngle * cs.Cen_ctrl * cs.direction
    CL = CL + CL_ctrl_tot
    lift = (CL * dyn_pres + (P.CLp * (rr * span / 2) * half_rho_vel) + (P.CLq * (pr * mac / 2) * half_rho_vel)
            + (P.CLr * (yr * span / 2) * half_rho_vel)) * (P.area * (-1 * stability_z_axis))
    CD_fp = 2 / (1 + math.exp(P.CD_fp_k1 + P.CD_fp_k2 * (max(P.AR, 1 / P.AR))))
    CD = (1 - sigma) * (P.CD0 + (CL * CL) / (math.pi * P.AR * P.eff)) + sigma * abs(
        CD_fp * (0.5 - 0.5 * math.cos(2 * alpha)))
    CD = CD + CD_ctrl_tot
    drag = (CD * dyn_pres + (P.CDp * (rr * span / 2) * half_rho_vel) + (P.CDq * (pr * mac / 2) * half_rho_vel)
            + (P.CDr * (yr * span / 2) * half_rho_vel)) * (P.area * (-1 * stability_x_axis))
    CY = P.CYa * alpha + P.CYb * beta + CY_ctrl_tot
    sideforce = (CY * dyn_pres + (P.CYp * (rr * span / 2) * half_rho_vel) + (P.CYq * (pr * mac / 2) * half_rho_vel)
                 + (P.CYr * (yr * span / 2) * half_rho_vel)) * (P.area * stability_y_axis)
    if alpha > alphaStall:
        Cem = P.Cem0 + (P.Cema * alphaStall + P.Cema_stall * (alpha - alphaStall))
    elif alpha < -alphaStall:
        Cem = P.Cem0 + (-P.Cema * alphaStall + P.Cema_stall * (alpha + alphaStall))
    else:
        Cem = P.Cem0 + P.Cema * alpha
    if cem_overwrite:  # gz-sim <= 8.14.0
        Cem = P.Cemb * beta
    else:
        Cem += P.Cemb * beta
    Cem += Cem_ctrl_tot
    pm = ((Cem * dyn_pres) + (P.Cemp * (rr * span / 2) * half_rho_vel) + (P.Cemq * (pr * mac / 2) * half_rho_vel)
          + (P.Cemr * (yr * span / 2) * half_rho_vel)) * (P.area * mac * body_y_axis)
    Cell = P.Cella * alpha + P.Cellb * beta + Cell_ctrl_tot
    rm = ((Cell * dyn_pres) + (P.Cellp * (rr * span / 2) * half_rho_vel) + (P.Cellq * (pr * mac / 2) * half_rho_vel)
          + (P.Cellr * (yr * span / 2) * half_rho_vel)) * (P.area * span * body_x_axis)
    Cen = P.Cena * alpha + P.Cenb * beta + Cen_ctrl_tot
    ym = ((Cen * dyn_pres) + (P.Cenp * (rr * span / 2) * half_rho_vel) + (P.Cenq * (pr * mac / 2) * half_rho_vel)
          + (P.Cenr * (yr * span / 2) * half_rho_vel)) * (P.area * span * body_z_axis)
    moment = pm + rm + ym
    force = lift + drag + sideforce
    torque = moment
    force = force.Correct()
    torque = torque.Correct()
    totalTorque = torque + cpWorld.Cross(force)
    return force, totalTorque


# ------------------------------------------------------------------------------------------ MulticopterMotorModel
def motor_forces(P, joint_velocity, link_rot, link_com_offset, link_world_lin_vel, joint_axis_world, wind,
                 parent_rot):
    """Forces of MulticopterMotorModel (velocity type) in world axes.

    Returns (force on the rotor link, applied at its centre of mass; torque about the rotor link origin from that
    force, as Link::AddWorldForce; torque on the parent link)."""
    realMotorVelocity = joint_velocity * P.rotorVelocitySlowdownSim
    realMotorVelocitySign = (realMotorVelocity > 0) - (realMotorVelocity < 0)
    thrust = P.turningDirection * realMotorVelocitySign * realMotorVelocity * realMotorVelocity * P.motorConstant
    f_thrust = link_rot.RotateVector(Vec(0, 0, thrust))
    relativeWindVelocityWorld = link_world_lin_vel - wind
    jointAxis = joint_axis_world
    bodyVelocityPerpendicular = relativeWindVelocityWorld - (relativeWindVelocityWorld.Dot(jointAxis) * jointAxis)
    airDrag = -abs(realMotorVelocity) * P.rotorDragCoefficient * bodyVelocityPerpendicular
    force = f_thrust + airDrag
    posComWorldCoord = link_rot.RotateVector(link_com_offset)
    torque_from_force = posComWorldCoord.Cross(f_thrust) + posComWorldCoord.Cross(airDrag)
    poseDifferenceRot = parent_rot.Inverse() * link_rot
    dragTorque = Vec(0, 0, -P.turningDirection * thrust * P.momentConstant)
    dragTorqueParentFrame = poseDifferenceRot.RotateVector(dragTorque)
    parentWorldTorque = parent_rot.RotateVector(dragTorqueParentFrame)
    rollingMoment = -abs(realMotorVelocity) * P.rollingMomentCoefficient * bodyVelocityPerpendicular
    parentWorldTorque = parentWorldTorque + rollingMoment
    return force, torque_from_force, parentWorldTorque, thrust


# ------------------------------------------------------------------------------------------ scalar simulator
class ScalarAircraft:
    """One aircraft, plain Python: the reference systems above, the same rigid body and RK4 as
    :class:`wingbatch.Aircraft`. Built from an Aircraft (``ac``) to share the parsed SDF (parameters, link poses,
    mass properties): it checks the physics and the frames, not the parsing. The wind is ``ac``'s, for aircraft
    ``index``, at the current position and at t = 0; links without gravity are not supported."""

    def __init__(self, ac, index: int = 0):
        import numpy as np  # only to read the parsed model

        self.ac = ac
        mp = ac.mass
        self.m = mp.mass
        self.com = Vec(*mp.com)
        self.I = [list(map(float, r)) for r in mp.inertia]
        self.Iinv = [list(map(float, r)) for r in np.linalg.inv(mp.inertia)]
        self.g = Vec(*ac.gravity)
        if mp.gravity_mass != mp.mass:
            raise NotImplementedError("ScalarAircraft: links without gravity are not supported")
        self.index = index
        from .systems.wind import UniformWind

        w = getattr(ac.wind, "velocity", None)
        self._const_wind = Vec(*w) if isinstance(ac.wind, UniformWind) and np.shape(w) == (3,) else None
        self.wind = self._const_wind or Vec()
        self.ld = [(a.system.params, _quat_from_matrix(a.R), Vec(*a.p)) for a in ac.lift_drag]
        self.motors = [(mm.params, _quat_from_matrix(mm.rotor_R), Vec(*mm.origin),
                        Vec(*(mm.rotor_R.T @ (mm.com - mm.origin))), Vec(*mm.axis)) for mm in ac.motors]
        self.spin = [mm.spin_inertia if ac.rotor_gyro else 0.0 for mm in ac.motors]
        self.servo_names = list(ac.servo_names)
        sv = ac.servos
        self.sv = [(float(sv.p[i]), float(sv.c[i]), float(sv.cmd_min[i]), float(sv.cmd_max[i]), float(sv.lower[i]),
                    float(sv.upper[i])) for i in range(len(self.servo_names))]
        self.cem_overwrite = any(a.system.gz_cem_overwrite for a in ac.lift_drag)
        self.k, self.nm = len(self.servo_names), len(self.motors)

    def _mat(self, A, v):
        return Vec(A[0][0] * v.x + A[0][1] * v.y + A[0][2] * v.z, A[1][0] * v.x + A[1][1] * v.y + A[1][2] * v.z,
                   A[2][0] * v.x + A[2][1] * v.y + A[2][2] * v.z)

    def wrench_world(self, x):
        """World-frame force and torque about the body origin, from the reference systems."""
        x = [float(v) for v in x]
        if self._const_wind is None:  # per-aircraft or position-dependent wind
            import numpy as np

            pos = np.zeros((self.ac.n, 3))
            pos[:] = x[0:3]
            self.wind = Vec(*self.ac._wind(pos, 0.0)[self.index])
        q = Quat(*x[3:7]).Normalized()
        v = Vec(*x[7:10])
        wb = Vec(*x[10:13])
        ww = q.RotateVector(wb)
        joints = dict(zip(self.servo_names, x[13:13 + self.k]))
        F, T = Vec(), Vec()
        for P, ql, pl in self.ld:
            rot = q * ql
            pw = q.RotateVector(pl)
            vlink = v + ww.Cross(pw)
            angv = ql.Inverse().RotateVector(wb)
            f, t = advanced_lift_drag(P, vlink, ww, rot, angv, joints, self.wind, self.cem_overwrite)
            if f is None:
                continue
            F = F + f
            T = T + t + pw.Cross(f)
        for i, (P, qr, origin, com_off, axis) in enumerate(self.motors):
            rot = q * qr
            pw = q.RotateVector(origin)
            vr = v + ww.Cross(pw)
            jv = P.turningDirection * x[13 + self.k + i] / P.rotorVelocitySlowdownSim
            f, t_link, t_parent, _ = motor_forces(P, jv, rot, com_off, vr, q.RotateVector(axis), self.wind, q)
            F = F + f
            T = T + t_link + pw.Cross(f) + t_parent
        return F, T

    def derivatives(self, x, servo_cmd, motor_cmd):
        x, servo_cmd, motor_cmd = ([float(v) for v in a] for a in (x, servo_cmd, motor_cmd))
        F, Tw = self.wrench_world(x)
        q = Quat(*x[3:7]).Normalized()
        qi = q.Inverse()
        wb = Vec(*x[10:13])
        Fb, Tb = qi.RotateVector(F), qi.RotateVector(Tw)
        rc = self.com
        tc = Tb - rc.Cross(Fb)
        d_servo, d_motor = [], []
        for i, (p, c, lo_f, hi_f, lo, hi) in enumerate(self.sv):
            qj = x[13 + i]
            rate = min(max(p * (servo_cmd[i] - qj), lo_f), hi_f) / c
            if (qj >= hi and rate > 0) or (qj <= lo and rate < 0):
                rate = 0.0
            d_servo.append(rate)
        h, hdot = Vec(), Vec()
        for i, (P, _q, _o, _c, axis) in enumerate(self.motors):
            w = x[13 + self.k + i]
            ref = min(motor_cmd[i], P.maxRotVelocity)
            tau = P.timeConstantUp if ref > w else P.timeConstantDown
            d_motor.append((ref - w) / tau)
            # rotor joint spun by gz-sim at turningDirection * w / rotorVelocitySlowdownSim
            k = self.spin[i] * P.turningDirection / P.rotorVelocitySlowdownSim
            h = h + axis * (k * w)
            hdot = hdot + axis * (k * d_motor[-1])
        wdot = self._mat(self.Iinv, tc - wb.Cross(self._mat(self.I, wb) + h) - hdot)
        a_com = F / self.m + self.g
        a_o = a_com - q.RotateVector(wdot.Cross(rc) + wb.Cross(wb.Cross(rc)))
        qw, qx, qy, qz = x[3:7]
        dq = [-0.5 * (qx * wb.x + qy * wb.y + qz * wb.z), 0.5 * (qw * wb.x + qy * wb.z - qz * wb.y),
              0.5 * (qw * wb.y + qz * wb.x - qx * wb.z), 0.5 * (qw * wb.z + qx * wb.y - qy * wb.x)]
        d = list(x[7:10]) + dq + a_o.tolist() + wdot.tolist() + d_servo + d_motor
        return d

    def step(self, x, servo_cmd, motor_cmd, dt):
        f = self.derivatives
        k1 = f(x, servo_cmd, motor_cmd)
        k2 = f([a + 0.5 * dt * b for a, b in zip(x, k1)], servo_cmd, motor_cmd)
        k3 = f([a + 0.5 * dt * b for a, b in zip(x, k2)], servo_cmd, motor_cmd)
        k4 = f([a + dt * b for a, b in zip(x, k3)], servo_cmd, motor_cmd)
        xn = [a + dt / 6 * (b + 2 * c + 2 * d + e) for a, b, c, d, e in zip(x, k1, k2, k3, k4)]
        n = math.sqrt(sum(v * v for v in xn[3:7]))
        xn[3:7] = [v / n for v in xn[3:7]]
        for i, (*_x, lo, hi) in enumerate(self.sv):
            xn[13 + i] = min(max(xn[13 + i], lo), hi)
        return xn


def _quat_from_matrix(R):
    from .frames import _matrix_to_quat

    return Quat(*_matrix_to_quat(R))
