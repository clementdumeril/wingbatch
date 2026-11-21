"""The air the aircraft fly in.

gz-sim gives AdvancedLiftDrag and MulticopterMotorModel one uniform wind: the ``WorldLinearVelocity`` of the
world's wind entity (``<wind><linear_velocity>``, zero in PX4's default world). :class:`UniformWind` is that, one
vector per world or per aircraft.

:class:`WindField` is an extension that gz-sim does not have: any function of position and time (thermals,
shear, turbulence), evaluated at the body origin of each aircraft at every integrator stage and used by both
systems as if it were the uniform wind at that instant. Its results are not Gazebo results.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class UniformWind:
    def __init__(self, velocity=(0.0, 0.0, 0.0)):
        self.velocity = np.asarray(velocity, float)

    def __call__(self, position_world, t):
        return np.broadcast_to(self.velocity, position_world.shape)


class WindField:
    """``fn(position_world (n, 3), t) -> wind velocity (n, 3)``, world (ENU) axes, m/s."""

    def __init__(self, fn: Callable[[np.ndarray, float], np.ndarray]):
        self.fn = fn

    def __call__(self, position_world, t):
        out = np.asarray(self.fn(position_world, t), float)
        if out.shape != position_world.shape:
            raise ValueError(f"WindField function returned shape {out.shape}, expected {position_world.shape}")
        return out
