"""The rigid body: mass properties of a model and the Newton-Euler equations, batched.

All links of the model move as one rigid body: fixed joints are rigid in gz-sim too; revolute joints (control
surfaces, rotor) are taken at their zero position for the mass properties. This neglects the motion of the
control-surface links (1e-8 kg each in the advanced_plane: reaction torques of order 1e-4 N m) and the change of the
rotor's transverse inertia as it turns (0.1 % of Iyy, Izz). The rotor's spin angular momentum is NOT neglected: gz-sim
spins the rotor joint at speed / ``rotorVelocitySlowdownSim`` (160 rad/s at PX4's full throttle, 0.027 N m s with
the advanced_plane's 1.676e-4 kg m^2), and :class:`wingbatch.Aircraft` adds its gyroscopic moment and the reaction to
its spin-up (``rotor_gyro=True``).

Frames are Gazebo's: world ENU (z up), body = the canonical link frame (FLU for the advanced_plane). The state
holds the pose and the world linear velocity of the canonical link origin, as gz-sim's ``WorldPose`` and
``WorldLinearVelocity``, and the angular velocity in link axes, as gz-sim's ``AngularVelocity``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sdf import Model


@dataclass(frozen=True)
class MassProperties:
    mass: float
    com: np.ndarray  # (3,) centre of mass in the body frame (m)
    inertia: np.ndarray  # (3, 3) about the centre of mass, body axes (kg m^2)
    gravity_mass: float  # mass of the links that feel gravity (<gravity>1</gravity>, the default)
    gravity_com: np.ndarray  # their centre of mass

    @classmethod
    def from_model(cls, model: Model) -> "MassProperties":
        """Composite of all links (parallel-axis theorem), in the canonical link frame."""
        items = []
        for name, link in model.links.items():
            X = model.link_pose(name) @ link.inertial.pose  # inertial frame in the body frame
            items.append((link.inertial.mass, X.p, X.R @ link.inertial.inertia @ X.R.T, link.gravity))
        return cls.from_parts(items)

    @classmethod
    def from_parts(cls, items) -> "MassProperties":
        """``items``: (mass, centre of mass (3,), inertia about it (3, 3) in body axes, feels gravity)."""
        m = sum(i[0] for i in items)
        com = sum(i[0] * np.asarray(i[1], float) for i in items) / m
        inertia = np.zeros((3, 3))
        for mi, ci, Ii, _ in items:
            d = np.asarray(ci, float) - com
            inertia += np.asarray(Ii, float) + mi * (d @ d * np.eye(3) - np.outer(d, d))
        mg = sum(i[0] for i in items if i[3])
        cg = sum(i[0] * np.asarray(i[1], float) for i in items if i[3]) / mg if mg > 0 else com
        return cls(float(m), com, inertia, float(mg), np.asarray(cg, float))

    @classmethod
    def simple(cls, mass: float, inertia, com=(0.0, 0.0, 0.0)) -> "MassProperties":
        I = np.asarray(inertia, float)
        if I.shape == (3,):
            I = np.diag(I)
        return cls(float(mass), np.asarray(com, float), I, float(mass), np.asarray(com, float))
