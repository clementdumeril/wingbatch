# Copyright (C) 2019 Open Source Robotics Foundation (gz-sim AdvancedLiftDrag system;
#   authors of the system: Karthik Srivatsan, Frederik Markus)
# Copyright (C) 2026 the wingbatch authors (NumPy translation, batching)
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Derived from gz-sim, src/systems/advanced_lift_drag/AdvancedLiftDrag.cc, branch gz-sim8 at commit
# f380a94e5755fc04d8cf509fa0d4541a20e4ceed (identical to gz-sim 8.15.0+, 9.6.0+, 10.x for this file).
# Changes: translated from C++ to NumPy, evaluated for a batch of aircraft at once, written in the link frame
# instead of the world frame (the same dot and cross products, rotated), optional reproduction of the pitching
# moment of gz-sim <= 8.14.0.
"""gz-sim's ``AdvancedLiftDrag`` system, for ``n`` aircraft at once.

The parameters keep the SDF names (``CL0``, ``Cema``, ``Cellp``, ``alpha_stall``, ``cp``...), with gz-sim's
defaults for the ones the SDF leaves out. ``M``, ``CD_fp_k1`` and ``CD_fp_k2`` are fixed in gz-sim (15, -0.224,
-0.115) and not read from the SDF; ``CLa_stall``, ``CDa_stall``, ``a0`` and ``index`` are not read by gz-sim either
and are ignored here. ``radial_symmetry`` is read by gz-sim but never used.

What :meth:`AdvancedLiftDrag.wrench` reproduces, line by line:

* air velocity at ``cp``: ``v_link_origin + omega x cp - wind`` (uniform world wind);
* body axes from ``forward`` and ``upward`` (``z = -upward``, ``y = z x x``, not re-normalised, as in gz-sim);
* the in-plane speed (spanwise component removed), ``alpha``, ``beta`` (from the full air velocity);
* rates taken from the raw link-frame angular velocity: ``p = w_x``, ``q = -w_y``, ``r = -w_z``;
* the sigmoid stall blend, ``CL``/``CD``/``CY``/``Cem``/``Cell``/``Cen`` with the control-surface terms
  (joint angle in degrees times ``direction``), lift along ``-stability_z``, drag along ``-stability_x``;
* ``Vector3::Correct()`` (non-finite components set to zero), the ``cp x F`` transport, and the early return
  (no wrench) when the in-plane speed is <= 1e-9.

Two gz-sim behaviours are kept on purpose because they are what Gazebo computes:

* ``CL`` uses ``alpha / |alpha|``, which is NaN at exactly ``alpha == 0``; the lift, the drag and the side force
  are then NaN and ``Correct()`` sets the whole force to zero for that evaluation (the moments are kept). Pass
  ``alpha_zero="continuous"`` to use ``sign = +1`` there instead. Whether a state lands exactly on
  ``alpha == 0`` (or on ``alpha = +pi`` rather than ``-pi``, where the ``Cem`` branch jumps) depends on round-off,
  and this port computes in the link frame where gz-sim computes in the world frame: at such exact boundaries
  the two can pick different sides (always the same side for an attitude aligned with the world axes).
* gz-sim up to 8.14.0 overwrote the static pitching moment (``Cem = Cemb * beta`` instead of ``+=``, fixed by
  gazebosim/gz-sim#3817 / #3820, first released in 8.15.0). ``gz_cem_overwrite=True`` reproduces the old
  behaviour, for comparisons with flights logged on those versions.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from .. import _vec as V

COEFFICIENTS = (
    "CL0", "CD0", "Cem0", "CLa", "CYa", "Cella", "Cema", "Cena", "CLb", "CYb", "Cellb", "Cemb", "Cenb",
    "Cema_stall",
    "CDp", "CYp", "CLp", "Cellp", "Cemp", "Cenp",
    "CDq", "CYq", "CLq", "Cellq", "Cemq", "Cenq",
    "CDr", "CYr", "CLr", "Cellr", "Cemr", "Cenr",
)
CONTROL_COEFFICIENTS = ("CD_ctrl", "CY_ctrl", "CL_ctrl", "Cell_ctrl", "Cem_ctrl", "Cen_ctrl")
IGNORED_BY_GZ = ("a0", "CLa_stall", "CDa_stall", "radial_symmetry")


@dataclass
class ControlSurface:
    name: str  # joint name
    direction: float
    CD_ctrl: float
    CY_ctrl: float
    CL_ctrl: float
    Cell_ctrl: float
    Cem_ctrl: float
    Cen_ctrl: float


@dataclass
class AdvancedLiftDragParams:
    """The ``<plugin>`` parameters, SDF names and gz-sim defaults."""

    link_name: str = ""
    CL0: float = 0.0
    CD0: float = 0.0
    Cem0: float = 0.0
    CLa: float = 0.0
    CYa: float = 0.0
    Cella: float = 0.0
    Cema: float = 0.0
    Cena: float = 0.0
    CLb: float = 0.0
    CYb: float = 0.0
    Cellb: float = 0.0
    Cemb: float = 0.0
    Cenb: float = 0.0
    alpha_stall: float = math.pi / 2
    Cema_stall: float = 0.0
    cp: tuple[float, float, float] = (0.0, 0.0, 0.0)
    CDp: float = 0.0
    CYp: float = 0.0
    CLp: float = 0.0
    Cellp: float = 0.0
    Cemp: float = 0.0
    Cenp: float = 0.0
    CDq: float = 0.0
    CYq: float = 0.0
    CLq: float = 0.0
    Cellq: float = 0.0
    Cemq: float = 0.0
    Cenq: float = 0.0
    CDr: float = 0.0
    CYr: float = 0.0
    CLr: float = 0.0
    Cellr: float = 0.0
    Cemr: float = 0.0
    Cenr: float = 0.0
    AR: float = 0.0
    mac: float = 0.0
    eff: float = 0.0
    air_density: float = 1.2041
    area: float = 1.0
    forward: tuple[float, float, float] = (1.0, 0.0, 0.0)
    upward: tuple[float, float, float] = (0.0, 0.0, 1.0)
    control_surfaces: list[ControlSurface] = field(default_factory=list)  # the first num_ctrl_surfaces
    all_surface_names: list[str] = field(default_factory=list)  # every <control_surface> (gz checks them all)
    # fixed in gz-sim (not read from the SDF)
    M: float = 15.0
    CD_fp_k1: float = -0.224
    CD_fp_k2: float = -0.115

    @classmethod
    def from_sdf(cls, el: ET.Element) -> "AdvancedLiftDragParams":
        p = cls()
        for name in COEFFICIENTS + ("AR", "mac", "eff", "area"):
            if el.find(name) is not None:
                setattr(p, name, float(el.findtext(name)))
        if el.find("alpha_stall") is not None:
            p.alpha_stall = float(el.findtext("alpha_stall"))
        if el.find("air_density") is not None:
            p.air_density = float(el.findtext("air_density"))
        for name in ("cp", "forward", "upward"):
            if el.find(name) is not None:
                setattr(p, name, tuple(float(s) for s in el.findtext(name).split()))
        p.link_name = (el.findtext("link_name") or "").strip()
        if not p.link_name:
            raise ValueError("AdvancedLiftDrag requires the 'link_name' parameter")
        surfaces = []
        for cs in el.findall("control_surface"):
            def num(tag, cs=cs):
                text = cs.findtext(tag)
                if text is None:
                    raise ValueError(f"AdvancedLiftDrag control surface without <{tag}>")
                return float(text)
            surfaces.append(ControlSurface(cs.findtext("name").strip(), num("direction"),
                                           *(num(t) for t in CONTROL_COEFFICIENTS)))
        n_ctrl = int(el.findtext("num_ctrl_surfaces") or 0)
        if n_ctrl > len(surfaces):
            raise ValueError(f"num_ctrl_surfaces = {n_ctrl} but only {len(surfaces)} <control_surface> given "
                             "(out-of-bounds access in gz-sim)")
        p.control_surfaces = surfaces[:n_ctrl]  # gz-sim loops over the first num_ctrl_surfaces only
        p.all_surface_names = [cs.name for cs in surfaces]
        return p


class AdvancedLiftDrag:
    """Batched ``AdvancedLiftDrag`` for one plugin instance.

    ``joint_index[i]`` is the column of the joint-position array that holds control surface ``i`` (or -1 for a
    joint that is not actuated, whose position stays 0)."""

    def __init__(self, params: AdvancedLiftDragParams, joint_index=None, *, alpha_zero: str = "gazebo",
                 gz_cem_overwrite: bool = False):
        if alpha_zero not in ("gazebo", "continuous"):
            raise ValueError("alpha_zero must be 'gazebo' or 'continuous'")
        self.params = p = params
        self.alpha_zero = alpha_zero
        self.gz_cem_overwrite = gz_cem_overwrite
        fwd = np.asarray(p.forward, float)
        if np.linalg.norm(fwd) < 1e-9:
            raise ValueError("Forward vector length is zero. This is not valid.")
        self.bx = fwd / np.linalg.norm(fwd)
        up = np.asarray(p.upward, float)
        if np.linalg.norm(up) < 1e-9:
            raise ValueError("AdvancedLiftDrag: zero <upward> vector")
        self.bz = -up / np.linalg.norm(up)
        if not (p.AR > 0 and p.eff > 0 and p.area > 0):
            raise ValueError(f"AdvancedLiftDrag needs <AR>, <eff> and <area> > 0 (got {p.AR}, {p.eff}, {p.area}); "
                             "gz-sim would compute a non-finite force and drop it")
        self.by = np.cross(self.bz, self.bx)  # not normalised, as gz-sim
        self.cp = np.asarray(p.cp, float)
        self.span = math.sqrt(p.area * p.AR)
        self.mac = p.mac if abs(p.mac) > 1e-9 else p.area / self.span
        self.CD_fp = 2.0 / (1.0 + math.exp(p.CD_fp_k1 + p.CD_fp_k2 * max(p.AR, 1.0 / p.AR)))
        self.induced = 1.0 / (math.pi * p.AR * p.eff)
        self.joint_index = np.full(len(p.control_surfaces), -1, int) if joint_index is None else \
            np.asarray(joint_index, int)
        # per-degree control derivatives times direction, as a (6, n_surfaces) matrix
        ctrl = np.array([[getattr(cs, c) * cs.direction for cs in p.control_surfaces]
                         for c in CONTROL_COEFFICIENTS], float).reshape(6, -1)
        self._ctrl = ctrl * (180.0 / math.pi)  # per radian of joint angle

    # ------------------------------------------------------------------------------------------
    def wrench(self, v_air, w_link, joints):
        """Force and torque in the plugin link frame, the torque about the link origin.

        ``v_air`` (3-tuple of (n,)): link-origin velocity minus wind, in link axes; ``w_link``: angular velocity
        in link axes; ``joints`` (n, n_joints): joint positions (rad). Returns ``(force, torque, info)``."""
        p = self.params
        bx, by, bz, cp = self.bx, self.by, self.bz, self.cp
        # air velocity at cp: v + w x cp (link axes)
        wxcp = V.cross(w_link, (cp[0], cp[1], cp[2]))
        va = V.add(v_air, wxcp)
        va_y = va[0] * by[0] + va[1] * by[1] + va[2] * by[2]
        vin = (va[0] - va_y * by[0], va[1] - va_y * by[1], va[2] - va_y * by[2])
        speed = np.sqrt(vin[0] ** 2 + vin[1] ** 2 + vin[2] ** 2)
        live = speed > 1e-9
        inv = 1.0 / np.where(live, speed, 1.0)
        sx = (vin[0] * inv, vin[1] * inv, vin[2] * inv)  # stability x
        sz = V.cross(sx, (by[0], by[1], by[2]))  # stability z
        rr, pr, yr = w_link[0], -w_link[1], -w_link[2]
        alpha = np.arctan2(sx[0] * bz[0] + sx[1] * bz[1] + sx[2] * bz[2], sx[0] * bx[0] + sx[1] * bx[1] + sx[2] * bx[2])
        sa, ca = np.sin(alpha), np.cos(alpha)
        beta = np.arctan2(va_y, va[0] * bx[0] + va[1] * bx[1] + va[2] * bx[2])
        dyn = 0.5 * p.air_density * speed * speed
        hrv = 0.5 * p.air_density * speed
        M, a_s = p.M, p.alpha_stall
        e_up = np.exp(-M * (alpha - a_s))
        e_dn = np.exp(M * (alpha + a_s))
        sigma = (1 + e_up + e_dn) / ((1 + e_up) * (1 + e_dn))
        if self.alpha_zero == "gazebo":
            with np.errstate(invalid="ignore", divide="ignore"):
                sgn = alpha / np.abs(alpha)
        else:
            sgn = np.where(alpha >= 0, 1.0, -1.0)
        CL = (1 - sigma) * (p.CL0 + p.CLa * alpha) + sigma * (2 * sgn * sa * sa * ca)
        CL = CL + p.CLb * beta
        if self._ctrl.shape[1] and joints.shape[1] and np.any(self.joint_index >= 0):
            q = np.where(self.joint_index[None, :] >= 0, joints[:, np.maximum(self.joint_index, 0)], 0.0)
            dC = q @ self._ctrl.T  # (n, 6): CD, CY, CL, Cell, Cem, Cen
            dCD, dCY, dCL, dCell, dCem, dCen = dC.T
        else:
            dCD = dCY = dCL = dCell = dCem = dCen = 0.0
        CL = CL + dCL
        ph, qh, rh = rr * self.span / 2 * hrv, pr * self.mac / 2 * hrv, yr * self.span / 2 * hrv

        def rate(a, b, c):
            return a * ph + b * qh + c * rh

        A = p.area
        L = (CL * dyn + rate(p.CLp, p.CLq, p.CLr)) * A
        CD = (1 - sigma) * (p.CD0 + CL * CL * self.induced) + sigma * np.abs(self.CD_fp * (0.5 - 0.5 * np.cos(2 * alpha)))
        CD = CD + dCD
        D = (CD * dyn + rate(p.CDp, p.CDq, p.CDr)) * A
        CY = p.CYa * alpha + p.CYb * beta + dCY
        Y = (CY * dyn + rate(p.CYp, p.CYq, p.CYr)) * A
        Cem = np.where(alpha > a_s, p.Cem0 + (p.Cema * a_s + p.Cema_stall * (alpha - a_s)),
                       np.where(alpha < -a_s, p.Cem0 + (-p.Cema * a_s + p.Cema_stall * (alpha + a_s)),
                                p.Cem0 + p.Cema * alpha))
        if self.gz_cem_overwrite:
            Cem = p.Cemb * beta + 0.0 * alpha
        else:
            Cem = Cem + p.Cemb * beta
        Cem = Cem + dCem
        Pm = ((Cem * dyn) + rate(p.Cemp, p.Cemq, p.Cemr)) * (A * self.mac)
        Cell = p.Cella * alpha + p.Cellb * beta + dCell
        Rm = ((Cell * dyn) + rate(p.Cellp, p.Cellq, p.Cellr)) * (A * self.span)
        Cen = p.Cena * alpha + p.Cenb * beta + dCen
        Ym = ((Cen * dyn) + rate(p.Cenp, p.Cenq, p.Cenr)) * (A * self.span)

        F = tuple(-L * sz[i] - D * sx[i] + Y * by[i] for i in range(3))
        T = tuple(Pm * by[i] + Rm * bx[i] + Ym * bz[i] for i in range(3))
        # Vector3::Correct(), then nothing at all below 1e-9 m/s (gz-sim returns before applying a wrench)
        F = tuple(np.where(live & np.isfinite(f), f, 0.0) for f in F)
        T = tuple(np.where(live & np.isfinite(t), t, 0.0) for t in T)
        tot = V.add(T, V.cross((cp[0], cp[1], cp[2]), F))
        info = dict(alpha=alpha, beta=beta, speed=speed, qbar=dyn, sigma=sigma, CL=CL, CD=CD, lift=L, drag=D,
                    side=Y)
        return F, tot, info
