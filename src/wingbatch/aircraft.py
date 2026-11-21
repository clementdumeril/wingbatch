"""A batch of ``n`` aircraft read from one SDF model, flown with the gz-sim systems it declares.

State, one row per aircraft (columns, see :attr:`Aircraft.idx`)::

    x y z            world position of the canonical link origin (ENU, m)       = gz WorldPose
    qw qx qy qz      orientation, link -> world                                  = gz WorldPose
    vx vy vz         world velocity of the canonical link origin (m/s)           = gz WorldLinearVelocity
    p q r            angular velocity in link axes (rad/s)                       = gz AngularVelocity
    servo joints     positions of the joints driven by JointPositionController (rad)
    rotors           filtered speed of each MulticopterMotorModel (rad/s)

Inputs per step: the servo joint position commands (rad, as published on gz's ``servo_N`` topics) and the motor
speed commands (rad/s, as on ``command/motor_speed``). :mod:`wingbatch.px4` maps PX4's normalised actuator
outputs to these.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np

from . import _vec as V
from .physics import MassProperties
from .sdf import Model, World, load_model
from .systems.advanced_lift_drag import AdvancedLiftDrag, AdvancedLiftDragParams
from .systems.joint_position_controller import JointPositionControllers, ServoParams, check_reduction
from .systems.multicopter_motor_model import MotorParams, MulticopterMotorModel
from .systems.wind import UniformWind, WindField

INTEGRATORS = ("rk4", "semi_implicit_euler")


@dataclass
class _Attached:
    """A system on a link that may differ from the canonical one: the link pose in the body frame."""

    system: object
    R: np.ndarray
    p: np.ndarray
    identity: bool


class Aircraft:
    """``n`` copies of the model in one array.

    Parameters
    ----------
    model: a :class:`wingbatch.sdf.Model`, a path to a ``model.sdf`` or model directory, or the name of a bundled
        model (``"advanced_plane"``).
    n: number of aircraft.
    world: gravity and uniform wind (:class:`wingbatch.World`, PX4's default world if omitted).
    wind: overrides ``world.wind``: a 3-vector, an (n, 3) array, or a callable ``wind(position (n, 3), t)``.
    integrator: ``"rk4"`` (default) or ``"semi_implicit_euler"`` (Gazebo-like order of operations, not
        bit-identical to DART).
    alpha_zero, gz_cem_overwrite: see :mod:`wingbatch.systems.advanced_lift_drag`.
    mass_properties: overrides the composite of the SDF links.
    chunk: step large batches in blocks of this many aircraft (cache-friendly); ``None`` for one block. Only with a
        :class:`UniformWind` or :class:`WindField` (a raw callable always sees the whole batch).
    rotor_gyro: include the angular momentum of the rotor joints that gz-sim spins (at speed /
        ``rotorVelocitySlowdownSim``): their gyroscopic moment and the reaction to their spin-up (default True).
    """

    def __init__(self, model="advanced_plane", n: int = 1, *, world: World | None = None, wind=None,
                 integrator: str = "rk4", alpha_zero: str = "gazebo", gz_cem_overwrite: bool = False,
                 mass_properties: MassProperties | None = None, rotor_gyro: bool = True, chunk: int | None = 1024,
                 resource_paths=(), warn: bool = True):
        if integrator not in INTEGRATORS:
            raise ValueError(f"integrator must be one of {INTEGRATORS}")
        if not isinstance(model, Model):
            model = load_model(model, resource_paths)
        self.model = model
        self.n = int(n)
        self.chunk = chunk
        self._rows = None
        self.integrator = integrator
        self.world = world or World()
        self.gravity = np.asarray(self.world.gravity, float)
        if wind is None:
            wind = self.world.wind
        self.wind = wind if callable(wind) else UniformWind(np.asarray(wind, float))
        self.mass = mass_properties or MassProperties.from_model(model)
        I = self.mass.inertia
        self._I, self._Iinv = I, np.linalg.inv(I)

        self.warnings = list(model.warnings)
        # servos (JointPositionController), in SDF order
        servos = []
        for pl in model.plugins_of("JointPositionController"):
            sp = ServoParams.from_sdf(pl.element)
            sp.joint_name = pl.scoped(sp.joint_name)
            j = model.joints.get(sp.joint_name)
            if j is None:
                raise ValueError(f"JointPositionController: no joint {sp.joint_name!r}")
            if j.axis is None:
                raise ValueError(f"joint {j.name!r} has no <axis>")
            sp.damping, sp.lower, sp.upper = j.damping, j.lower, j.upper
            child = model.links[j.child].inertial
            Xc = model.link_pose(j.child) @ child.pose
            arm = Xc.p - (model.links[model.canonical_link].pose.inverse() @ j.pose).p
            axis = model.links[model.canonical_link].pose.R.T @ j.axis
            Rc = Xc.R
            sp.inertia = float(axis @ (Rc @ child.inertia @ Rc.T) @ axis
                               + child.mass * (arm @ arm - (arm @ axis) ** 2))
            check_reduction(sp)
            servos.append(sp)
        self.servos = JointPositionControllers(servos)
        self.servo_names = self.servos.names

        # motors
        self.motors: list[MulticopterMotorModel] = []
        for pl in model.plugins_of("MulticopterMotorModel"):
            mp = MotorParams.from_sdf(pl.element)
            mp.jointName, mp.linkName = pl.scoped(mp.jointName), pl.scoped(mp.linkName)
            j = model.joints.get(mp.jointName)
            if j is None or mp.linkName not in model.links:
                raise ValueError(f"MulticopterMotorModel: joint {mp.jointName!r} or link {mp.linkName!r} missing")
            if j.parent not in model.links:
                raise ValueError(f"MulticopterMotorModel: parent of {mp.jointName!r} must be a link")
            if j.axis is None:
                raise ValueError(f"joint {j.name!r} has no <axis>")
            X = model.link_pose(mp.linkName)
            inertial = model.links[mp.linkName].inertial
            Xi = X @ inertial.pose
            axis = model.links[model.canonical_link].pose.R.T @ j.axis
            axis = axis / np.linalg.norm(axis)
            spin_inertia = float(axis @ (Xi.R @ inertial.inertia @ Xi.R.T) @ axis)
            self.motors.append(MulticopterMotorModel(mp, X.R, X.p, Xi.p, axis, spin_inertia))
        self.motor_names = [m.params.jointName for m in self.motors]
        numbers = sorted(m.params.motorNumber for m in self.motors)
        if numbers != list(range(len(self.motors))):
            raise NotImplementedError(f"motor numbers {numbers}: wingbatch needs actuator numbers 0..m-1")
        self.motors.sort(key=lambda m: m.params.motorNumber)  # column i of motor_cmd = actuator number i
        self.motor_names = [m.params.jointName for m in self.motors]
        self.rotor_gyro = bool(rotor_gyro)

        # lift-drag
        self.lift_drag: list[_Attached] = []
        for pl in model.plugins_of("AdvancedLiftDrag"):
            ap = AdvancedLiftDragParams.from_sdf(pl.element)
            ap.link_name = pl.scoped(ap.link_name)
            if ap.link_name not in model.links:
                raise ValueError(f"AdvancedLiftDrag: link {ap.link_name!r} not found")
            for name in ap.all_surface_names:  # gz-sim disables the plugin if any is missing
                if pl.scoped(name) not in model.joints:
                    raise ValueError(f"AdvancedLiftDrag: joint {pl.scoped(name)!r} not found")
            index = []
            for cs in ap.control_surfaces:
                cs.name = pl.scoped(cs.name)
                if cs.name in self.servo_names:
                    index.append(self.servo_names.index(cs.name))
                else:
                    index.append(-1)
                    self.warnings.append(f"control surface {cs.name!r} has no JointPositionController: held at 0")
            sysm = AdvancedLiftDrag(ap, index, alpha_zero=alpha_zero, gz_cem_overwrite=gz_cem_overwrite)
            X = model.link_pose(ap.link_name)
            ident = np.allclose(X.R, np.eye(3), atol=0, rtol=0) and not np.any(X.p)
            self.lift_drag.append(_Attached(sysm, X.R, X.p, ident))
        if warn:
            for msg in self.warnings:
                warnings.warn(f"{model.name}: {msg}", stacklevel=2)

        k, m = len(self.servo_names), len(self.motors)
        self.n_servos, self.n_motors = k, m
        self.idx = dict(pos=slice(0, 3), quat=slice(3, 7), vel=slice(7, 10), omega=slice(10, 13),
                        servos=slice(13, 13 + k), motors=slice(13 + k, 13 + k + m))
        self.nx = 13 + k + m
        self.t = 0.0
        self.state = self.make_state()
        self.last_info: dict | None = None

    # ------------------------------------------------------------------------------------------ state
    def make_state(self, position=(0.0, 0.0, 100.0), quaternion=(1.0, 0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0),
                   omega=(0.0, 0.0, 0.0), servos=None, motors=None) -> np.ndarray:
        """A state array (n, nx); every argument broadcasts to its (n, width) block."""
        X = np.zeros((self.n, self.nx), order="F")
        X[:, self.idx["pos"]] = position
        q = np.broadcast_to(np.asarray(quaternion, float), (self.n, 4))
        X[:, self.idx["quat"]] = q / np.linalg.norm(q, axis=1, keepdims=True)
        X[:, self.idx["vel"]] = velocity
        X[:, self.idx["omega"]] = omega
        X[:, self.idx["servos"]] = 0.0 if servos is None else servos  # gz joints start at 0
        X[:, self.idx["motors"]] = 0.0 if motors is None else motors
        return X

    def reset(self, state=None, t: float = 0.0, **kwargs) -> np.ndarray:
        """Set the state (an (n, nx) array, or the arguments of :meth:`make_state`) and the time."""
        X = np.array(state if state is not None else self.make_state(**kwargs), dtype=float, order="F", copy=True)
        if X.shape != (self.n, self.nx):
            raise ValueError(f"state must have shape {(self.n, self.nx)}, got {X.shape}")
        self.state, self.t = X, float(t)
        return self.state

    def _wind(self, pos, t):
        if self._rows is not None and isinstance(self.wind, UniformWind) and self.wind.velocity.ndim == 2:
            return self.wind.velocity[self._rows]  # per-aircraft uniform wind, for the block being stepped
        w = np.asarray(self.wind(pos, t), float)
        if w.shape != pos.shape:
            w = np.broadcast_to(w, pos.shape)  # raises on a wrong shape (a transposed (3, n) array included)
        return w

    # ------------------------------------------------------------------------------------------ wrenches
    def wrenches(self, X, t=None):
        """Sum of the systems' forces and torques on the body (body axes, torque about the body origin).

        Returns ``(F, tau, info)`` with F, tau 3-tuples of (n,) arrays, and per-system air data in ``info``."""
        t = self.t if t is None else t
        c = X.T  # (nx, n): contiguous rows with a Fortran-ordered X
        qw, qx, qy, qz = c[3], c[4], c[5], c[6]
        R = V.quat_to_matrix(qw, qx, qy, qz)
        wind = self._wind(X[:, 0:3], t)
        vrel = V.rotate_inv(R, (c[7] - wind[:, 0], c[8] - wind[:, 1], c[9] - wind[:, 2]))
        w = (c[10], c[11], c[12])
        joints = X[:, self.idx["servos"]]
        nb = X.shape[0]
        F = (np.zeros(nb), np.zeros(nb), np.zeros(nb))
        T = (np.zeros(nb), np.zeros(nb), np.zeros(nb))
        info = {}
        for i, att in enumerate(self.lift_drag):
            if att.identity:
                f, tq, inf = att.system.wrench(vrel, w, joints)
            else:
                p = att.p
                vl = V.add(vrel, V.cross(w, (p[0], p[1], p[2])))
                f, tq, inf = att.system.wrench(V.const_rotate(att.R.T, vl), V.const_rotate(att.R.T, w), joints)
                f = V.const_rotate(att.R, f)
                tq = V.add(V.const_rotate(att.R, tq), V.cross((p[0], p[1], p[2]), f))
            F, T = V.add(F, f), V.add(T, tq)
            info[f"lift_drag{i}"] = inf
        wm = X[:, self.idx["motors"]]
        for i, mot in enumerate(self.motors):
            f, tq, thrust = mot.wrench(wm[:, i], vrel, w)
            F, T = V.add(F, f), V.add(T, tq)
            info[f"thrust{i}"] = thrust
        return F, T, info

    # ------------------------------------------------------------------------------------------ dynamics
    def _rotor_momentum(self, W):
        """Angular momentum of the spinning rotor joints relative to the body (body axes), from rotor speeds W
        (n, m): gz-sim spins each rotor joint at ``turningDirection * w / rotorVelocitySlowdownSim``."""
        h = (0.0, 0.0, 0.0)
        for i, mot in enumerate(self.motors):
            hd = mot.momentum_dir
            h = (h[0] + hd[0] * W[:, i], h[1] + hd[1] * W[:, i], h[2] + hd[2] * W[:, i])
        return h

    def derivatives(self, X, servo_cmd, motor_cmd, t=None):
        """dX/dt for the continuous model (RK4); also returns the wrench info."""
        if np.ndim(servo_cmd) < 2 or np.shape(servo_cmd)[0] != X.shape[0]:
            servo_cmd, motor_cmd = self._commands(servo_cmd, motor_cmd)
        F, Tq, info = self.wrenches(X, t)
        c = X.T
        dX = np.empty((X.shape[0], self.nx), order="F")
        d = dX.T
        qw, qx, qy, qz = c[3], c[4], c[5], c[6]
        w = (c[10], c[11], c[12])
        s, mo = self.idx["servos"], self.idx["motors"]
        if self.n_servos:
            dX[:, s] = self.servos.rate(X[:, s], servo_cmd)
        for i, mot in enumerate(self.motors):
            dX[:, mo.start + i] = mot.speed_derivative(X[:, mo.start + i], motor_cmd[:, i])
        h = hdot = None
        if self.rotor_gyro and self.n_motors:
            h, hdot = self._rotor_momentum(X[:, mo]), self._rotor_momentum(dX[:, mo])
        acc, wdot = self._accelerations(qw, qx, qy, qz, w, F, Tq, h, hdot)
        d[0], d[1], d[2] = c[7], c[8], c[9]
        d[3], d[4], d[5], d[6] = V.quat_derivative(qw, qx, qy, qz, *w)
        d[7], d[8], d[9] = acc
        d[10], d[11], d[12] = wdot
        return dX, info

    def _accelerations(self, qw, qx, qy, qz, w, F, Tq, h=None, hdot=None):
        """World acceleration of the body origin and body angular acceleration, from the external force F and
        torque Tq about the origin (body axes), with the rotor angular momentum h and its rate hdot (body axes):
        ``I wdot = tau_c - w x (I w + h) - hdot``."""
        mp = self.mass
        R = V.quat_to_matrix(qw, qx, qy, qz)
        rc = (mp.com[0], mp.com[1], mp.com[2])
        # torque about the centre of mass: external torque about the origin minus r_c x F, plus gravity's
        # moment when some links do not feel gravity
        tc = V.sub(Tq, V.cross(rc, F))
        if mp.gravity_mass != mp.mass:
            gb = V.rotate_inv(R, tuple(np.full_like(qw, gi * mp.gravity_mass) for gi in self.gravity))
            d = mp.gravity_com - mp.com
            tc = V.add(tc, V.cross((d[0], d[1], d[2]), gb))
        L = V.const_rotate(self._I, w)
        if h is not None:
            L = V.add(L, h)
            tc = V.sub(tc, hdot)
        wdot = V.const_rotate(self._Iinv, V.sub(tc, V.cross(w, L)))
        g = self.gravity * (mp.gravity_mass / mp.mass)
        Fw = V.rotate(R, F)
        a_com = (Fw[0] / mp.mass + g[0], Fw[1] / mp.mass + g[1], Fw[2] / mp.mass + g[2])
        # origin acceleration: a_c - wdot x r_c - w x (w x r_c), rotated to world
        rel = V.add(V.cross(wdot, rc), V.cross(w, V.cross(w, rc)))
        relw = V.rotate(R, rel)
        return V.sub(a_com, relw), wdot

    # ------------------------------------------------------------------------------------------ stepping
    def _commands(self, servo_cmd, motor_cmd):
        """Broadcast to (n, n_servos) and (n, n_motors); a 1-D array is one value per servo / motor."""
        s = np.broadcast_to(np.asarray(0.0 if servo_cmd is None else servo_cmd, float), (self.n, self.n_servos))
        m = np.broadcast_to(np.asarray(0.0 if motor_cmd is None else motor_cmd, float), (self.n, self.n_motors))
        return s, m

    def step(self, servo_cmd=None, motor_cmd=None, dt: float = 0.01) -> np.ndarray:
        """Advance every aircraft by ``dt`` seconds with the commands held; returns the new state.

        ``servo_cmd``: joint position commands (rad), broadcastable to (n, n_servos), columns in
        :attr:`servo_names` order; ``motor_cmd``: rotor speed commands (rad/s), broadcastable to (n, n_motors),
        column i = actuator number i. ``None`` means zero."""
        s, m = self._commands(servo_cmd, motor_cmd)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(m))):
            raise ValueError("non-finite servo or motor command")
        X, t = self.state, self.t
        chunk = self.chunk
        if chunk and self.n > chunk and isinstance(self.wind, (UniformWind, WindField)):
            # blocks of ~1000 aircraft keep every temporary array in the CPU cache (about 2x faster at 1e5)
            Xn = np.empty((self.n, self.nx), order="F")
            infos = []
            try:
                for a in range(0, self.n, chunk):
                    self._rows = slice(a, min(a + chunk, self.n))
                    Xn[self._rows], info = self._advance(X[self._rows], s[self._rows], m[self._rows], t, dt)
                    infos.append(info)
            finally:
                self._rows = None
            info = _concat_info(infos)
        else:
            Xn, info = self._advance(X, s, m, t, dt)
        self.state, self.t, self.last_info = Xn, t + dt, info
        return self.state

    def _advance(self, X, s, m, t, dt):
        if self.integrator == "rk4":
            k1, info = self.derivatives(X, s, m, t)
            k2, _ = self.derivatives(X + 0.5 * dt * k1, s, m, t + 0.5 * dt)
            k3, _ = self.derivatives(X + 0.5 * dt * k2, s, m, t + 0.5 * dt)
            k4, _ = self.derivatives(X + dt * k3, s, m, t + dt)
            Xn = X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            Xn, info = self._semi_implicit_euler(X, s, m, t, dt)
        q = Xn[:, 3:7]
        Xn[:, 3:7] = q / np.sqrt(np.einsum("ij,ij->i", q, q))[:, None]
        if self.n_servos:
            sl = self.idx["servos"]
            Xn[:, sl] = self.servos.clip(Xn[:, sl])
        return np.asfortranarray(Xn), info

    def _semi_implicit_euler(self, X, s, m, t, dt):
        """Gazebo's order: the systems compute their wrenches from the current state (the thrust from the rotor
        speed of the previous step), the rotor filter is updated, then the velocities are integrated and the
        pose follows with the new velocities (the free-body update of DART >= 6.13's FreeJoint: Euler on the
        body angular velocity and world linear velocity, then ``q <- q exp(w dt)``)."""
        F, Tq, info = self.wrenches(X, t)
        c = X.T
        qw, qx, qy, qz = c[3], c[4], c[5], c[6]
        w = (c[10], c[11], c[12])
        Xn = np.empty_like(X, order="F")
        sl, mo = self.idx["servos"], self.idx["motors"]
        if self.n_servos:
            Xn[:, sl] = X[:, sl] + dt * self.servos.rate(X[:, sl], s)
        for i, mot in enumerate(self.motors):
            Xn[:, mo.start + i] = mot.filter_step(X[:, mo.start + i], m[:, i], dt)
        h = hdot = None
        if self.rotor_gyro and self.n_motors:
            # the rotor joint reaches its new speed within the step: the body takes the opposite impulse
            h = self._rotor_momentum(X[:, mo])
            hdot = self._rotor_momentum((Xn[:, mo] - X[:, mo]) / dt)
        acc, wdot = self._accelerations(qw, qx, qy, qz, w, F, Tq, h, hdot)
        dn = Xn.T
        for i in range(3):
            dn[7 + i] = c[7 + i] + dt * acc[i]
            dn[10 + i] = c[10 + i] + dt * wdot[i]
            dn[i] = c[i] + dt * dn[7 + i]
        dq = V.quat_exp(dn[10], dn[11], dn[12], dt)
        dn[3], dn[4], dn[5], dn[6] = V.quat_multiply((qw, qx, qy, qz), dq)
        return Xn, info

    def rollout(self, servo_cmd=None, motor_cmd=None, dt: float = 0.01, steps: int | None = None) -> np.ndarray:
        """Fly a command sequence; returns the states (T + 1, n, nx), the first one the current state.

        Commands are 3-D sequences (T, n, k) / (T, n, m), or anything broadcastable to (n, k) / (n, m), which is
        held constant (then give ``steps``)."""
        seqs = []
        for cmd, width in ((servo_cmd, self.n_servos), (motor_cmd, self.n_motors)):
            a = np.asarray(0.0 if cmd is None else cmd, float)
            seqs.append(a if a.ndim == 3 else np.broadcast_to(a, (self.n, width))[None])
        T = max(len(seqs[0]), len(seqs[1])) if steps is None else int(steps)
        if steps is None and len(seqs[0]) == 1 and len(seqs[1]) == 1:
            raise ValueError("constant commands: say how many steps")
        for a in seqs:
            if len(a) not in (1, T):
                raise ValueError(f"command sequence of length {len(a)} for {T} steps")
        out = np.empty((T + 1, self.n, self.nx))
        out[0] = self.state
        for k in range(T):
            s = seqs[0][k if len(seqs[0]) > 1 else 0]
            mcmd = seqs[1][k if len(seqs[1]) > 1 else 0]
            out[k + 1] = self.step(s, mcmd, dt)
        return out

    # ------------------------------------------------------------------------------------------ views
    @property
    def position(self):
        return self.state[:, 0:3]

    @property
    def quaternion(self):
        return self.state[:, 3:7]

    @property
    def velocity(self):
        return self.state[:, 7:10]

    @property
    def omega(self):
        return self.state[:, 10:13]

    def airspeed(self):
        """``(speed, alpha, beta)`` of the first lift-drag system at the current state: in-plane air speed at
        ``cp`` (m/s), angle of attack and sideslip (rad)."""
        F, T, info = self.wrenches(self.state)
        d = info.get("lift_drag0", {})
        return d.get("speed"), d.get("alpha"), d.get("beta")

    def __repr__(self) -> str:
        return (f"Aircraft({self.model.name!r}, n={self.n}, servos={self.servo_names}, motors={self.motor_names}, "
                f"mass={self.mass.mass:.4g} kg, integrator={self.integrator!r})")


def _concat_info(infos):
    out = {}
    for k, v in infos[0].items():
        out[k] = _concat_info([i[k] for i in infos]) if isinstance(v, dict) else np.concatenate([i[k] for i in infos])
    return out


def bundled_models() -> list[str]:
    from .sdf import MODELS_DIR
    return sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir() and (p / "model.sdf").exists()
                  and os.path.basename(p) != "LW20")
