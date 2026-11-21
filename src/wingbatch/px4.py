"""PX4's gz bridge mapping: normalised actuator outputs to the gz commands :class:`wingbatch.Aircraft` takes.

From PX4-Autopilot ``src/modules/simulation/gz_bridge`` (commit 129446a7, BSD-3-Clause):

* ESC ``i``: PX4's output module maps the motor output ``u`` in [0, 1] linearly to [``SIM_GZ_EC_MINi``,
  ``SIM_GZ_EC_MAXi``] and ``GZMixingInterfaceESC`` publishes that number as the rotor speed command (rad/s).
  The advanced_plane airframe (``4008_gz_advanced_plane``) sets MIN1 = 10, MAX1 = 1600. When disarmed PX4 sends
  ``SIM_GZ_EC_DIS`` (0 by default).
* servo ``i``: the output ``u`` in [-1, 1] is mapped to [``SIM_GZ_SV_MINi``, ``SIM_GZ_SV_MAXi``] = [0, 1000] by
  default, then ``GZMixingInterfaceServo`` publishes ``MINAi + (MAXAi - MINAi) (out - MIN) / (MAX - MIN)`` with
  ``SIM_GZ_SV_MAXA`` = 45 deg and ``SIM_GZ_SV_MINA`` = -45 deg by default, i.e. ``45 deg * u``. When disarmed PX4
  sends ``SIM_GZ_SV_DIS`` = 500, i.e. 0 rad. PX4's servo output ``i`` is published on the gz topic ``servo_i``,
  which reaches the JointPositionController whose ``<sub_topic>`` (or ``<topic>``) is ``servo_i``; in the
  advanced_plane that is the joint of the same name (see :attr:`wingbatch.Aircraft.servo_names` for the column order).

PX4's control allocation (which normalised output each control surface gets for a roll, pitch or yaw torque) is
not reproduced here.
"""

from __future__ import annotations

import numpy as np

ADVANCED_PLANE_ESC = dict(min=10.0, max=1600.0)  # 4008_gz_advanced_plane: SIM_GZ_EC_MIN1, SIM_GZ_EC_MAX1


def motor_speed(u, min_speed: float = ADVANCED_PLANE_ESC["min"], max_speed: float = ADVANCED_PLANE_ESC["max"]):
    """Rotor speed command (rad/s) for the normalised motor output ``u`` in [0, 1] (clipped)."""
    return min_speed + (max_speed - min_speed) * np.clip(u, 0.0, 1.0)


def throttle(speed, min_speed: float = ADVANCED_PLANE_ESC["min"], max_speed: float = ADVANCED_PLANE_ESC["max"]):
    """Inverse of :func:`motor_speed`."""
    return np.clip((np.asarray(speed, float) - min_speed) / (max_speed - min_speed), 0.0, 1.0)


def servo_angle(u, min_angle_deg: float = -45.0, max_angle_deg: float = 45.0):
    """Joint position command (rad) for the normalised servo output ``u`` in [-1, 1] (clipped)."""
    s = (np.clip(u, -1.0, 1.0) + 1.0) / 2.0
    return np.radians(min_angle_deg + (max_angle_deg - min_angle_deg) * s)
