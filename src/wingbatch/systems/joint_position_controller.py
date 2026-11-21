# Upstream copyright lines of gz-sim's JointPositionController.cc:
# Copyright (C) 2020 Open Source Robotics Foundation
# Copyright (C) 2023 Benjamin Perseghetti, Rudis Laboratories
# Copyright (C) 2026 the wingbatch authors (reduced-order NumPy model, batching)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Parameters and defaults from gz-sim, src/systems/joint_position_controller/JointPositionController.cc, branch
# gz-sim8 at commit f380a94e5755fc04d8cf509fa0d4541a20e4ceed. Changes: the PID force on the joint and the joint
# damping are reduced to a first-order lag (see the module docstring), evaluated for a batch of aircraft.
"""gz-sim's ``JointPositionController`` driving a control-surface joint, reduced to a first-order lag.

In gz-sim the controller applies the joint force ``f = clip(-p_gain (q - q_cmd) - d_gain dq/dt, cmd_min,
cmd_max)`` (``i_gain`` = 0) and the physics integrates ``I q'' = f - damping q'`` with the joint ``<damping>``.
The control-surface links of fixed-wing models are nearly massless (1e-8 kg, 1e-6 kg m^2 in the advanced_plane),
so the fast pole ``-damping / I`` (about -1e6 1/s there; DART treats the joint damping implicitly, so it stays
stable at any step) is far above any time step and the joint follows the slow pole exactly enough:

    dq/dt = clip(p_gain (q_cmd - q), cmd_min, cmd_max) / damping,   q in [lower, upper]

i.e. a first-order lag of time constant ``damping / p_gain`` (0.1 s for the advanced_plane). The joint limits of
the SDF bound ``q``. ``initial_position`` only sets the first target in gz-sim; the joint itself starts at 0.
:func:`check_reduction` refuses settings where the reduction is not valid (integral or derivative gains, velocity
mode, offsets, or a joint inertia that makes the fast pole slow).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


@dataclass
class ServoParams:
    joint_name: str
    p_gain: float = 1.0
    i_gain: float = 0.1
    d_gain: float = 0.01
    cmd_max: float = 1000.0
    cmd_min: float = -1000.0
    initial_position: float = 0.0
    use_velocity_commands: bool = False
    sub_topic: str = ""
    unsupported: tuple = ()
    # from the joint
    damping: float = 0.0
    lower: float = -1e16
    upper: float = 1e16
    inertia: float = 0.0  # child link inertia about the joint axis (for the validity check)

    @classmethod
    def from_sdf(cls, el: ET.Element) -> "ServoParams":
        names = [e.text.strip() for e in el.findall("joint_name") if e.text]
        if len(names) != 1:
            raise NotImplementedError("JointPositionController with several <joint_name> is not supported")
        p = cls(names[0])
        for tag in ("p_gain", "i_gain", "d_gain", "cmd_max", "cmd_min", "initial_position"):
            if el.find(tag) is not None:
                setattr(p, tag, float(el.findtext(tag)))
        if el.find("use_velocity_commands") is not None:
            p.use_velocity_commands = el.findtext("use_velocity_commands").strip().lower() in ("1", "true")
        p.sub_topic = (el.findtext("topic") or el.findtext("sub_topic") or "").strip()  # <topic> wins in gz-sim
        p.unsupported = tuple(t for t in ("cmd_offset", "use_actuator_msg", "joint_index") if el.find(t) is not None
                              and (el.findtext(t) or "").strip() not in ("0", "false", ""))
        return p

    @property
    def time_constant(self) -> float:
        return self.damping / self.p_gain


def check_reduction(p: ServoParams, dt_min: float = 1e-3) -> None:
    if p.use_velocity_commands:
        raise NotImplementedError(f"{p.joint_name}: use_velocity_commands is not supported")
    if p.i_gain != 0.0:
        raise NotImplementedError(f"{p.joint_name}: i_gain != 0 is not supported (first-order reduction)")
    if p.d_gain != 0.0:
        # gz-math's PID differentiates the error explicitly, DART's joint damping is implicit: a d_gain does not
        # act like damping in Gazebo (it can even make the joint unstable), so it is not reduced here
        raise NotImplementedError(f"{p.joint_name}: d_gain != 0 is not supported (first-order reduction)")
    for tag in p.unsupported:
        raise NotImplementedError(f"{p.joint_name}: <{tag}> is not supported")
    if p.p_gain <= 0 or p.damping <= 0:
        raise NotImplementedError(f"{p.joint_name}: needs p_gain > 0 and a joint <damping> > 0")
    fast = p.inertia / p.damping  # time constant of the neglected pole
    if fast > 0.01 * min(p.time_constant, dt_min):
        raise NotImplementedError(f"{p.joint_name}: joint inertia {p.inertia:g} makes the neglected pole "
                                  f"({fast:g} s) too slow for the first-order reduction")


class JointPositionControllers:
    """All servo joints of an aircraft, as arrays (one column per joint)."""

    def __init__(self, servos: list[ServoParams]):
        self.servos = servos
        self.names = [s.joint_name for s in servos]
        k = len(servos)
        self.p = np.array([s.p_gain for s in servos], float).reshape(k)
        self.c = np.array([s.damping for s in servos], float).reshape(k)
        self.cmd_min = np.array([s.cmd_min for s in servos], float).reshape(k)
        self.cmd_max = np.array([s.cmd_max for s in servos], float).reshape(k)
        self.lower = np.array([s.lower for s in servos], float).reshape(k)
        self.upper = np.array([s.upper for s in servos], float).reshape(k)
        self.initial = np.array([s.initial_position for s in servos], float).reshape(k)

    def rate(self, q, q_cmd):
        """dq/dt (n, k), zero against a limit."""
        f = np.clip(self.p * (q_cmd - q), self.cmd_min, self.cmd_max)
        dq = f / self.c
        return np.where(((q >= self.upper) & (dq > 0)) | ((q <= self.lower) & (dq < 0)), 0.0, dq)

    def clip(self, q):
        return np.clip(q, self.lower, self.upper)
