# Frozen copy, for regression tests only, of GeneticWing's src/geneticwing/gzplane.py (19 Nov 2026), itself
# vendored from WingJEPA (github.com/clementdumeril/WingJEPA, wingjepa/sim/advanced_plane.py, commit e0357c6), a
# NumPy port of the Fixed-Wing C++ model (cpp/include/fixed_wing/dynamics.h). Same author as wingbatch. It is the
# independent earlier port wingbatch is checked against: body FRD axes, Euler angles, conventions de > 0 nose down,
# da > 0 rolls left, dr > 0 yaws left. Not part of the package.
"""PX4's ``advanced_plane`` as Gazebo (gz-sim) flies it, vectorized over a batch of flights.

Vendored from WingJEPA (github.com/clementdumeril/WingJEPA, ``wingjepa/sim/advanced_plane.py`` and
``configs/advanced_plane.yaml``, commit e0357c6), itself a port of the Fixed-Wing C++ model checked against it to
1e-9: the AdvancedLiftDrag system with the coefficients of ``models/advanced_plane/model.sdf`` (air velocity and
forces at the centre of pressure, post-stall blend, control derivatives per joint degree), the
MulticopterMotorModel propeller, first-order servos, a 6-DoF rigid body. Changes: the per-flight randomization
and the dataset channels were left out, ``derivatives`` also returns the inertial acceleration and accepts
precomputed forces, and ``throttle_for_thrust`` inverts the motor's steady state. The physics is the same as :func:`geneticwing.sixdof.advanced_plane` (tests/test_gzplane.py checks the
aerodynamic forces and moments against it); this module adds the propeller and Gazebo's servos, and runs ``n``
aircraft at once, which is what makes 6-DoF searches affordable.

State Y (n, 16): u v w | p q r | phi theta psi | north east h | de da dr (positions) | omega (rotor, rad/s).
Body FRD, NED attitude, ``h`` up. u, v, w are relative to the mean wind passed as ``wind_ned`` (which only
displaces the aircraft); ``gust_b`` (body axes) enters the aerodynamics. :class:`geneticwing.gzplant.GazeboPlant`
passes the whole wind as ``gust_b``, so u, v, w are ground-relative there.

Control surfaces, WingJEPA / UMN conventions: de > 0 nose down, da > 0 rolls LEFT, dr > 0 yaws LEFT (the opposite of
:mod:`geneticwing.sixdof` for da and dr). Inputs: de, da, dr (rad) and throttle (0-1).
"""

from __future__ import annotations

import numpy as np

(U, V, W, P, Q, R, PHI, TH, PSI, XN, YE, H, DE, DA, DR, OM) = range(16)
N_STATE = 16
D2R, R2D = np.pi / 180, 180 / np.pi

# Nominal values: Fixed-Wing/cpp/include/fixed_wing/dynamics.h (namespace advanced), from the SDF.
ADVANCED_PLANE = dict(
    mass=1.005,  # kg: base link 1 kg + propeller 5 g
    Ixx=0.197563975,  # kg m2: SDF + propeller about its axis
    Iyy=0.146509604,  # SDF + propeller (parallel axis, 0.3 m)
    Izz=0.148317604,
    Ixz=0.0,
    g=9.8,  # Gazebo's default world
    rho=1.2041,  # Gazebo's constant air density
    density_model="constant",  # or "isa"
    S=0.34, c=0.22, AR=6.5,
    cp=-0.12,  # m, centre of pressure on body x (forward +): 12 cm behind the centre of mass
    CL0=0.15188, CLa=5.015, CD0=0.029, eff=0.97, Cm0=0.075, Cma=-0.463966,
    CYb=-0.258244, Clb=-0.039250, Cnb=0.100826,
    CDp=0.0, CYp=0.065861, CLp=0.0, Clp=-0.487407, Cmp=0.0, Cnp=-0.040416,
    CDq=0.055166, CYq=0.0, CLq=7.971792, Clq=0.0, Cmq=-12.140140, Cnq=0.0,
    CDr=0.0, CYr=0.230299, CLr=0.0, Clr=0.078165, Cmr=0.0, Cnr=-0.089947,
    alpha_stall=0.3391428111, M=15.0, Cma_stall=0.0, CD_fp_k1=-0.224, CD_fp_k2=-0.115,
    # per DEGREE of joint angle, times the joint direction: [servo_0 left elevon, servo_1 right elevon,
    # servo_2 elevator, servo_3 rudder]
    CD_ctrl=[-0.000059, -0.000059, -0.000274, 0.0],
    CY_ctrl=[0.000171, -0.000171, 0.0, -0.003913],
    CL_ctrl=[-0.011940, -0.011940, -0.010696, 0.0],
    Cl_ctrl=[-0.003331, 0.003331, 0.0, -0.000257],
    Cm_ctrl=[0.001498, 0.001498, 0.025798, 0.0],
    Cn_ctrl=[-0.000057, 0.000057, 0.0, 0.001613],
    # propeller (MulticopterMotorModel)
    rotor_x=0.3, omega_min=10.0, omega_max=1600.0, motor_constant=8.54858e-06, moment_constant=0.01,
    rotor_drag=8.06428e-05, rolling_moment=1.0e-06, motor_tau_up=0.0125, motor_tau_down=0.025,
    # surfaces: Gazebo's joint controller against the joint damping, first order, no rate limit, +-45 deg
    servo_tau=0.1,
    act_max=[np.pi / 4] * 3,
    act_min=[-np.pi / 4] * 3,
)

# joint angle (rad) of [servo_0, servo_1, servo_2, servo_3] = JOINTS @ [de, da, dr]
JOINTS = np.array([[0.0, 1.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [-1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0]])
RATE = ("CDp", "CYp", "CLp", "Clp", "Cmp", "Cnp", "CDq", "CYq", "CLq", "Clq", "Cmq", "Cnq",
        "CDr", "CYr", "CLr", "Clr", "Cmr", "Cnr")


def isa_density(h):
    """1976 standard atmosphere, troposphere."""
    T = 288.15 - 0.0065 * h
    return 1.225 * (T / 288.15) ** 4.2559


def dcm_body_to_ned(phi, th, psi):
    """Matrices (n, 3, 3): v_NED = C @ v_body."""
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
    C = np.empty(phi.shape + (3, 3))
    C[:, 0, 0], C[:, 0, 1], C[:, 0, 2] = ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp
    C[:, 1, 0], C[:, 1, 1], C[:, 1, 2] = ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp
    C[:, 2, 0], C[:, 2, 1], C[:, 2, 2] = -st, sf * ct, cf * ct
    return C


def control_matrix(p):
    """(6, 3): increments of [CD, CY, CL, Cl, Cm, Cn] per rad of [de, da, dr]."""
    ctrl = np.array([p["CD_ctrl"], p["CY_ctrl"], p["CL_ctrl"], p["Cl_ctrl"], p["Cm_ctrl"], p["Cn_ctrl"]], float)
    return ctrl @ JOINTS * R2D


def stall_blend(alpha, M, alpha_stall):
    """AdvancedLiftDrag sigma: 0 in attached flow, 1 past the stall (about 0.5 at +-alpha_stall)."""
    with np.errstate(over="ignore", invalid="ignore"):
        up = np.exp(-M * (alpha - alpha_stall))
        down = np.exp(M * (alpha + alpha_stall))
        s = (1 + up + down) / ((1 + up) * (1 + down))
    return np.where(np.isfinite(s), s, 1.0)  # far past the stall: inf / inf, the limit is 1


class AdvancedPlane:
    """Batch of ``n`` advanced_planes; ``p`` the parameters (default :data:`ADVANCED_PLANE`)."""

    def __init__(self, n: int, p: dict | None = None):
        p = dict(ADVANCED_PLANE if p is None else p)
        self.p, self.n = p, n
        self.mass = np.full(n, float(p["mass"]))
        I = np.zeros((n, 3, 3))
        I[:, 0, 0], I[:, 1, 1], I[:, 2, 2] = p["Ixx"], p["Iyy"], p["Izz"]
        I[:, 0, 2] = I[:, 2, 0] = -p["Ixz"]
        self.I, self.Iinv = I, np.linalg.inv(I)
        self.S, self.c, self.AR = float(p["S"]), float(p["c"]), float(p["AR"])
        self.b = float(np.sqrt(self.S * self.AR))
        self.motor_k = float(p["motor_constant"])
        self.cp = float(p["cp"])
        self.servo_tau = np.full(n, float(p["servo_tau"]))
        k = {name: float(p[name]) for name in ("CL0", "CLa", "CD0", "Cm0", "Cma", "CYb", "Clb", "Cnb") + RATE}
        k["induced"] = 1 / (np.pi * self.AR * p["eff"])
        self.k = k
        self.G = control_matrix(p)  # (6, 3)
        self.CD_fp = 2 / (1 + np.exp(p["CD_fp_k1"] + p["CD_fp_k2"] * max(self.AR, 1 / self.AR)))
        self.act_min, self.act_max = np.asarray(p["act_min"], float), np.asarray(p["act_max"], float)

    def rho(self, h):
        if self.p.get("density_model", "constant") == "isa":
            return isa_density(h)
        return np.full(np.shape(h), float(self.p["rho"]))

    # --- forces and moments --------------------------------------------------------------------
    def forces(self, Y, gust_b):
        """Forces and moments in body FRD axes about the centre of mass, and the AdvancedLiftDrag air data.

        ``F_aero, M_aero`` (n, 3): the lift-drag system's force and moment (the centre of pressure's lever arm
        included); ``T`` thrust, ``F_prop, M_prop`` the propeller's; ``V_ld, alpha_cp, beta_cp``: speed in the
        lift-drag plane and the angles at the centre of pressure, as AdvancedLiftDrag uses them; ``qbar``."""
        k = self.k
        vr = Y[:, U:W + 1] - gust_b
        pr, qr, rr = Y[:, P], Y[:, Q], Y[:, R]
        rho = self.rho(Y[:, H])
        cp = self.cp
        u, v, w = vr[:, 0], vr[:, 1] + rr * cp, vr[:, 2] - qr * cp  # v + omega x (cp, 0, 0)
        V = np.sqrt(u**2 + w**2)  # the sideslip component does not count
        live = V > 1e-9
        alpha, beta = np.arctan2(w, u), np.arctan2(v, u)
        sa, ca = np.sin(alpha), np.cos(alpha)
        dyn, hrv = 0.5 * rho * V**2, 0.5 * rho * V
        sig = stall_blend(alpha, self.p["M"], self.p["alpha_stall"])
        dC = Y[:, DE:DR + 1] @ self.G.T  # [CD, CY, CL, Cl, Cm, Cn] increments
        b, c, S = self.b, self.c, self.S
        ph, qh, rh = pr * b / 2 * hrv, qr * c / 2 * hrv, rr * b / 2 * hrv

        def rate(C):
            return k[C + "p"] * ph + k[C + "q"] * qh + k[C + "r"] * rh

        sign = np.where(alpha >= 0, 1.0, -1.0)
        CL = (1 - sig) * (k["CL0"] + k["CLa"] * alpha) + sig * (2 * sign * sa**2 * ca) + dC[:, 2]
        lift = (CL * dyn + rate("CL")) * S
        CD = ((1 - sig) * (k["CD0"] + CL**2 * k["induced"])
              + sig * np.abs(self.CD_fp * (0.5 - 0.5 * np.cos(2 * alpha))) + dC[:, 0])
        drag = (CD * dyn + rate("CD")) * S
        CY = k["CYb"] * beta + dC[:, 1]
        side = (CY * dyn + rate("CY")) * S
        a_s, Cma_s = self.p["alpha_stall"], self.p["Cma_stall"]
        Cm = np.where(alpha > a_s, k["Cm0"] + k["Cma"] * a_s + Cma_s * (alpha - a_s),
                      np.where(alpha < -a_s, k["Cm0"] - k["Cma"] * a_s + Cma_s * (alpha + a_s),
                               k["Cm0"] + k["Cma"] * alpha)) + dC[:, 4]
        Cl = k["Clb"] * beta + dC[:, 3]
        Cn = k["Cnb"] * beta + dC[:, 5]
        F_aero = np.stack([lift * sa - drag * ca, side, -lift * ca - drag * sa], 1)
        M_aero = np.stack([(Cl * dyn + rate("Cl")) * S * b,
                           (Cm * dyn + rate("Cm")) * S * c,
                           (Cn * dyn + rate("Cn")) * S * b], 1)
        M_aero[:, 1] += -cp * F_aero[:, 2]  # force applied at the centre of pressure: + r x F
        M_aero[:, 2] += cp * F_aero[:, 1]
        F_aero = np.where(live[:, None], F_aero, 0.0)
        M_aero = np.where(live[:, None], M_aero, 0.0)

        # propeller (MulticopterMotorModel), rotor at (rotor_x, 0, 0)
        pp = self.p
        omega = Y[:, OM]
        T = self.motor_k * omega**2
        rx = pp["rotor_x"]
        ay, az = vr[:, 1] + rr * rx, vr[:, 2] - qr * rx
        Fry, Frz = -omega * pp["rotor_drag"] * ay, -omega * pp["rotor_drag"] * az
        F_prop = np.stack([T, Fry, Frz], 1)
        M_prop = np.stack([pp["moment_constant"] * T,
                           -omega * pp["rolling_moment"] * ay - rx * Frz,
                           -omega * pp["rolling_moment"] * az + rx * Fry], 1)
        return dict(F_aero=F_aero, M_aero=M_aero, T=T, F_prop=F_prop, M_prop=M_prop, V_ld=V, alpha_cp=alpha,
                    beta_cp=beta, qbar=dyn, sigma=sig, lift=lift, drag=drag, side=side)

    # --- dynamics -------------------------------------------------------------------------------
    def omega_command(self, thr):
        p = self.p
        return p["omega_min"] + (p["omega_max"] - p["omega_min"]) * np.clip(thr, 0.0, 1.0)

    def throttle_for_thrust(self, thrust):
        """The throttle whose steady rotor speed gives ``thrust`` (N)."""
        p = self.p
        omega = np.sqrt(np.maximum(thrust, 0.0) / self.motor_k)
        return np.clip((omega - p["omega_min"]) / (p["omega_max"] - p["omega_min"]), 0.0, 1.0)

    def derivatives(self, Y, cmd, thr, gust_b, wind_ned, actuator=True, forces=None):
        """cmd (n, 3): de, da, dr; thr (n,): throttle; wind_ned (n, 3): mean wind, which only displaces the aircraft.
        Returns dY and the inertial acceleration (NED) of the centre of mass."""
        p = self.p
        F = self.forces(Y, gust_b) if forces is None else forces
        g, m = p["g"], self.mass
        u, v, w = Y[:, U], Y[:, V], Y[:, W]
        pr, qr, rr = Y[:, P], Y[:, Q], Y[:, R]
        phi, th, psi = Y[:, PHI], Y[:, TH], Y[:, PSI]
        sf, cf, st, ct = np.sin(phi), np.cos(phi), np.sin(th), np.cos(th)
        Fb = F["F_aero"] + F["F_prop"]
        dY = np.empty_like(Y)
        dY[:, U] = rr * v - qr * w - g * st + Fb[:, 0] / m
        dY[:, V] = pr * w - rr * u + g * ct * sf + Fb[:, 1] / m
        dY[:, W] = qr * u - pr * v + g * ct * cf + Fb[:, 2] / m
        om = Y[:, P:R + 1]
        M = F["M_aero"] + F["M_prop"]
        Iw = np.einsum("nij,nj->ni", self.I, om)
        dY[:, P:R + 1] = np.einsum("nij,nj->ni", self.Iinv, M - np.cross(om, Iw))
        dY[:, PHI] = pr + (qr * sf + rr * cf) * np.tan(th)
        dY[:, TH] = qr * cf - rr * sf
        dY[:, PSI] = (qr * sf + rr * cf) / ct
        C = dcm_body_to_ned(phi, th, psi)
        Ve = np.einsum("nij,nj->ni", C, Y[:, U:W + 1]) + wind_ned
        dY[:, XN], dY[:, YE], dY[:, H] = Ve[:, 0], Ve[:, 1], -Ve[:, 2]
        if actuator:
            dY[:, DE:DR + 1] = (cmd - Y[:, DE:DR + 1]) / self.servo_tau[:, None]
        else:
            dY[:, DE:DR + 1] = 0.0
        om_cmd = self.omega_command(thr)
        tau = np.where(om_cmd < Y[:, OM], p["motor_tau_down"], p["motor_tau_up"])
        dY[:, OM] = (om_cmd - Y[:, OM]) / tau
        acc_ned = np.einsum("nij,nj->ni", C, Fb / m[:, None]) + np.array([0.0, 0.0, g])
        return dY, acc_ned

    def post_step(self, Y):
        """Surfaces and rotor speed clipped after a step, as the C++ post_integration."""
        Y[:, DE:DR + 1] = np.clip(Y[:, DE:DR + 1], self.act_min, self.act_max)
        Y[:, OM] = np.clip(Y[:, OM], self.p["omega_min"], self.p["omega_max"])
        return Y

    def step(self, Y, cmd, thr, gust_b, wind_ned, dt, actuator=True):
        """One RK4 step, inputs and wind held over it (WingJEPA's step)."""
        cmd = np.clip(cmd, self.act_min, self.act_max)
        thr = np.clip(thr, 0.0, 1.0)

        def f(y):
            return self.derivatives(y, cmd, thr, gust_b, wind_ned, actuator)[0]

        k1 = f(Y)
        k2 = f(Y + 0.5 * dt * k1)
        k3 = f(Y + 0.5 * dt * k2)
        k4 = f(Y + dt * k3)
        return self.post_step(Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    # --- trim -----------------------------------------------------------------------------------
    def trim(self, airspeed, gamma=0.0, h=100.0, psi=0.0, iters=60):
        """Steady straight flight, calm air, zero sideslip. Unknowns: alpha, de, throttle, omega, phi, da, dr.
        Vectorized Levenberg-Marquardt. Returns Y (n, 16), [de, da, dr] (n, 3), throttle (n,), ok (n,)."""
        n = self.n
        Vt, h, psi = (np.broadcast_to(x, (n,)).astype(float) for x in (airspeed, h, psi))
        z = np.zeros(n)
        x = np.stack([np.full(n, 0.05), z, np.full(n, 0.25), np.full(n, 400.0), z, z, z], 1)
        scale = np.array([1, 1, 10, 0.01, 1, 10, 10])

        def residual(x):
            a, de, thr, om, phi, da, dr = x.T
            Y = np.zeros((n, N_STATE))
            Y[:, U], Y[:, W] = Vt * np.cos(a), Vt * np.sin(a)
            Y[:, PHI], Y[:, TH], Y[:, PSI], Y[:, H] = phi, a + gamma, psi, h
            Y[:, DE], Y[:, DA], Y[:, DR], Y[:, OM] = de, da, dr, om
            d, _ = self.derivatives(Y, Y[:, DE:DR + 1], thr, np.zeros((n, 3)), np.zeros((n, 3)), actuator=False)
            return d[:, [U, W, Q, OM, V, P, R]] * scale, Y

        eps = np.array([1e-5, 1e-5, 1e-5, 1e-2, 1e-5, 1e-5, 1e-5])
        lo = np.array([-0.05, -0.05, -0.1, -100, -0.05, -0.05, -0.05])
        for _ in range(iters):
            r, _ = residual(x)
            Jm = np.empty((n, 7, 7))
            for j in range(7):
                step = np.full(n, eps[j])
                if j == 2:  # throttle at the upper stop: backward difference
                    step = np.where(x[:, 2] + eps[2] > 1.0, -eps[2], eps[2])
                xk = x.copy()
                xk[:, j] += step
                Jm[:, :, j] = (residual(xk)[0] - r) / step[:, None]
            JT = np.swapaxes(Jm, 1, 2)
            JTJ = JT @ Jm
            A = JTJ + 1e-6 * np.eye(7) * np.diagonal(JTJ, axis1=1, axis2=2)[:, None, :] + 1e-12 * np.eye(7)
            dx = np.linalg.solve(A, -(JT @ r[..., None]))[..., 0]
            x = x + np.clip(dx, lo, -lo)
            x[:, 2] = np.clip(x[:, 2], 0, 1)
        r, Y = residual(x)
        ok = np.abs(r).max(1) < 1e-6
        ok &= np.all(np.abs(x[:, [1, 5, 6]]) <= self.act_max, axis=1)
        return Y, x[:, [1, 5, 6]], x[:, 2], ok
