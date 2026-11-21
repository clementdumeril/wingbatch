# Upstream copyright lines of gz-sim's MulticopterMotorModel.cc, kept as required by the Apache License:
# Copyright 2015 Fadri Furrer, ASL, ETH Zurich, Switzerland
# Copyright 2015 Michael Burri, ASL, ETH Zurich, Switzerland
# Copyright 2015 Mina Kamel, ASL, ETH Zurich, Switzerland
# Copyright 2015 Janosch Nikolic, ASL, ETH Zurich, Switzerland
# Copyright 2015 Markus Achtelik, ASL, ETH Zurich, Switzerland
# Copyright 2016 Geoffrey Hunter <gbmhunter@gmail.com>
# Copyright (C) 2019 Open Source Robotics Foundation
# Copyright (C) 2022 Benjamin Perseghetti, Rudis Laboratories
# Copyright (C) 2026 the wingbatch authors (NumPy translation, batching)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Derived from gz-sim, src/systems/multicopter_motor_model/MulticopterMotorModel.cc, branch gz-sim8 at commit
# f380a94e5755fc04d8cf509fa0d4541a20e4ceed. Changes: translated to NumPy for a batch of aircraft, the motor
# written in the frame of the rigid body it is attached to, a continuous-time version of the rotor-speed filter
# for the Runge-Kutta integrator (the discrete filter is kept for the Gazebo-like stepping).
"""gz-sim's ``MulticopterMotorModel`` (``motorType`` = ``velocity``), for ``n`` aircraft at once.

Per motor, with ``w`` the filtered rotor speed command (rad/s, what gz-sim stores in the rotor joint velocity
times ``rotorVelocitySlowdownSim`` and ``turningDirection``):

* thrust ``T = sign(w) w^2 motorConstant`` along the rotor link z axis, applied at the rotor link's centre of mass;
* rotor drag ``-|w| rotorDragCoefficient v_perp`` on the rotor link, ``v_perp`` the rotor-link-origin air velocity
  with its component along the joint axis removed;
* on the parent link: the drag torque ``(0, 0, -turningDirection T momentConstant)`` in rotor link axes and the
  rolling moment ``-|w| rollingMomentCoefficient v_perp``;
* the command ``min(cmd, maxRotVelocity)`` (no lower bound in gz-sim), filtered by a first-order lag with
  ``timeConstantUp`` when the command exceeds the current speed and ``timeConstantDown`` otherwise.

The thrust direction is the rotor link's z axis at joint angle 0. In gz-sim the rotor link turns about the joint
axis, so when that axis is not exactly the link z axis (the advanced_plane's pitch is 1.57, not pi/2: 8e-4 rad) the
small off-axis thrust component rotates and averages out there, while it stays fixed here (2e-3 relative effect).

gz-sim computes the thrust from the joint velocity of the previous step and only then updates the filter; the
Gazebo-like stepping of :class:`wingbatch.Aircraft` keeps that order, the RK4 stepping uses the continuous lag
``dw/dt = (cmd - w) / tau``.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .. import _vec as V


@dataclass
class MotorParams:
    """``<plugin>`` parameters, SDF names and gz-sim defaults."""

    jointName: str = ""
    linkName: str = ""
    turningDirection: int = -1  # "cw" -> -1, "ccw" -> +1
    timeConstantUp: float = 1.0 / 80.0
    timeConstantDown: float = 1.0 / 40.0
    maxRotVelocity: float = 838.0
    motorConstant: float = 8.54858e-06
    momentConstant: float = 0.016
    rotorDragCoefficient: float = 1.0e-4
    rollingMomentCoefficient: float = 1.0e-6
    rotorVelocitySlowdownSim: float = 10.0
    motorNumber: int = 0
    commandSubTopic: str = "command/motor_speed"

    @classmethod
    def from_sdf(cls, el: ET.Element) -> "MotorParams":
        p = cls()
        p.jointName = (el.findtext("jointName") or "").strip()
        p.linkName = (el.findtext("linkName") or "").strip()
        if not p.jointName or not p.linkName:
            raise ValueError("MulticopterMotorModel needs <jointName> and <linkName>")
        direction = (el.findtext("turningDirection") or "cw").strip()
        if direction not in ("cw", "ccw"):
            raise ValueError("Please only use 'cw' or 'ccw' as turningDirection.")
        p.turningDirection = -1 if direction == "cw" else 1
        motor_type = (el.findtext("motorType") or "velocity").strip()
        if motor_type != "velocity":
            raise NotImplementedError(f"motorType {motor_type!r}: only 'velocity' does anything in gz-sim")
        for name in ("timeConstantUp", "timeConstantDown", "maxRotVelocity", "motorConstant", "momentConstant",
                     "rotorDragCoefficient", "rollingMomentCoefficient", "rotorVelocitySlowdownSim"):
            if el.find(name) is not None:
                setattr(p, name, float(el.findtext(name)))
        for tag in ("actuator_number", "motorNumber"):  # gz-sim prefers actuator_number
            if el.find(tag) is not None:
                p.motorNumber = int(el.findtext(tag))
                break
        if el.find("commandSubTopic") is not None:
            p.commandSubTopic = el.findtext("commandSubTopic").strip()
        return p


class MulticopterMotorModel:
    """One motor, attached to a rigid body.

    Geometry, in the body (canonical link) frame: ``rotor_origin`` (rotor link origin), ``rotor_com`` (its centre
    of mass, where gz-sim's ``AddWorldForce`` applies the thrust and the drag), ``rotor_R`` (rotor link axes) and
    ``axis`` (the joint axis); ``spin_inertia``: rotor link inertia about the joint axis (kg m^2)."""

    def __init__(self, params: MotorParams, rotor_R, rotor_origin, rotor_com, axis, spin_inertia: float = 0.0):
        self.params = params
        self.rotor_R = np.asarray(rotor_R, float)
        self.thrust_dir = self.rotor_R[:, 2].copy()
        self.drag_torque_dir = self.rotor_R[:, 2] * (-params.turningDirection * params.momentConstant)
        self.origin = np.asarray(rotor_origin, float)
        self.com = np.asarray(rotor_com, float)
        a = np.asarray(axis, float)
        self.axis = a / np.linalg.norm(a)
        # rotor joint angular momentum per rad/s of rotor speed: gz-sim sets the joint velocity to
        # turningDirection * w / rotorVelocitySlowdownSim about the joint axis
        self.spin_inertia = float(spin_inertia)
        self.momentum_dir = self.axis * (self.spin_inertia * params.turningDirection / params.rotorVelocitySlowdownSim)

    def command(self, cmd):
        """The reference speed gz-sim keeps from an actuator command."""
        return np.minimum(cmd, self.params.maxRotVelocity)

    def wrench(self, w, v_air_origin, omega):
        """Force and torque in body axes, the torque about the body origin.

        ``w`` (n,): rotor speed; ``v_air_origin``: body-origin velocity minus wind, body axes; ``omega``: body
        angular velocity (body axes)."""
        p = self.params
        T = np.sign(w) * w * w * p.motorConstant
        aw = np.abs(w)
        o, c, a = self.origin, self.com, self.axis
        vr = V.add(v_air_origin, V.cross(omega, (o[0], o[1], o[2])))  # rotor link origin
        va = vr[0] * a[0] + vr[1] * a[1] + vr[2] * a[2]
        vp = (vr[0] - va * a[0], vr[1] - va * a[1], vr[2] - va * a[2])
        kd, kr = -aw * p.rotorDragCoefficient, -aw * p.rollingMomentCoefficient
        F = tuple(T * self.thrust_dir[i] + kd * vp[i] for i in range(3))
        tq = tuple(T * self.drag_torque_dir[i] + kr * vp[i] for i in range(3))
        tq = V.add(tq, V.cross((c[0], c[1], c[2]), F))
        return F, tq, T

    def speed_derivative(self, w, cmd):
        p = self.params
        ref = self.command(cmd)
        tau = np.where(ref > w, p.timeConstantUp, p.timeConstantDown)
        return (ref - w) / tau

    def filter_step(self, w, cmd, dt):
        """gz-sim's discrete ``FirstOrderFilter::UpdateFilter``."""
        p = self.params
        ref = self.command(cmd)
        alpha = np.where(ref > w, math.exp(-dt / p.timeConstantUp), math.exp(-dt / p.timeConstantDown))
        return alpha * w + (1 - alpha) * ref
