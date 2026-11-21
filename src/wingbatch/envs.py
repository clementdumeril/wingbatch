"""A ``gymnasium.vector.VectorEnv`` over :class:`wingbatch.Aircraft`: all sub-environments in one NumPy batch.

:class:`LevelFlightVectorEnv` is a minimal example task (hold an altitude and an airspeed, wings level), meant
as a template: change :meth:`LevelFlightVectorEnv.reward` and :meth:`LevelFlightVectorEnv.observe` for your
own task. Actions are PX4-style normalised actuator outputs: one value in [-1, 1] per control surface of the
model's AdvancedLiftDrag (mapped to +-45 deg like PX4's gz bridge, then bounded by the joint limits) and one in
[-1, 1] per motor (mapped to throttle [0, 1], then to rotor speed with the advanced_plane's ESC range).

Requires ``gymnasium >= 1.0`` (``pip install wingbatch[gym]``). Autoreset follows gymnasium's default
``NEXT_STEP`` mode: the step after an episode ends resets that sub-environment and ignores its action.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium.vector import AutoresetMode, VectorEnv
    from gymnasium.vector.utils import batch_space
except ImportError as e:  # pragma: no cover
    raise ImportError("wingbatch.envs needs gymnasium >= 1.0: pip install 'wingbatch[gym]'") from e

from . import px4
from .aircraft import Aircraft
from .frames import euler_ned_frd
from .trim import trim


class LevelFlightVectorEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP, "render_modes": []}

    def __init__(self, num_envs: int = 64, model="advanced_plane", airspeed: float = 12.0, altitude: float = 100.0,
                 control_dt: float = 0.02, physics_dt: float = 0.01, max_episode_steps: int = 1500, **aircraft_kw):
        aircraft_kw.setdefault("warn", False)
        self.num_envs = int(num_envs)
        self.ac = Aircraft(model, self.num_envs, **aircraft_kw)
        self.airspeed, self.altitude = float(airspeed), float(altitude)
        self.control_dt, self.physics_dt = control_dt, physics_dt
        self.substeps = max(1, int(round(control_dt / physics_dt)))
        self.max_episode_steps = max_episode_steps
        surfaces = [cs.name for att in self.ac.lift_drag for cs in att.system.params.control_surfaces]
        self.surface_cols = [self.ac.servo_names.index(s) for s in surfaces if s in self.ac.servo_names]
        n_act = len(self.surface_cols) + self.ac.n_motors
        self.single_action_space = gym.spaces.Box(-1.0, 1.0, (n_act,), np.float32)
        self.single_observation_space = gym.spaces.Box(-np.inf, np.inf, (12 + len(self.surface_cols) + self.ac.n_motors,),
                                                       np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        trimmed = trim(self.ac, self.airspeed, altitude=self.altitude)
        if not np.all(trimmed["ok"]):
            raise ValueError(f"cannot trim at {airspeed} m/s")
        self._trim = trimmed
        self._steps = np.zeros(self.num_envs, int)
        self._autoreset = np.zeros(self.num_envs, bool)
        self._rng = np.random.default_rng()
        self._servo_cmd = trimmed["servo_cmd"].copy()

    # ---------------------------------------------------------------------------------------- task
    def _initial(self, idx):
        X = self._trim["state"][idx].copy()
        k = len(idx)
        rng = self._rng
        X[:, 7:10] *= rng.uniform(0.9, 1.1, (k, 1))
        X[:, 10:13] = rng.uniform(-0.2, 0.2, (k, 3))
        X[:, 2] += rng.uniform(-10.0, 10.0, k)
        return X

    def observe(self):
        ac = self.ac
        X = ac.state
        speed, alpha, beta = self._air
        roll, pitch, _ = euler_ned_frd(X[:, 3:7])
        obs = np.column_stack([(X[:, 2] - self.altitude) / 20.0, X[:, 9] / 5.0, (speed - self.airspeed) / 5.0,
                               alpha, beta, roll, pitch, X[:, 10], X[:, 11], X[:, 12],
                               np.sin(roll), np.cos(roll),
                               X[:, ac.idx["servos"]][:, self.surface_cols] / 0.78,
                               X[:, ac.idx["motors"]] / 1600.0])
        return obs.astype(np.float32)

    def reward(self):
        X = self.ac.state
        speed = self._air[0]
        roll, _, _ = euler_ned_frd(X[:, 3:7])
        dh = (X[:, 2] - self.altitude) / 20.0
        dv = (speed - self.airspeed) / 3.0
        return 1.0 - 0.5 * np.minimum(dh * dh + dv * dv + roll * roll, 2.0)

    def failed(self):
        X = self.ac.state
        roll, pitch, _ = euler_ned_frd(X[:, 3:7])
        return (np.abs(X[:, 2] - self.altitude) > 60.0) | (np.abs(roll) > 1.4) | (np.abs(pitch) > 1.2)

    # ---------------------------------------------------------------------------------------- API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        idx = np.arange(self.num_envs)
        self.ac.reset(self._initial(idx))
        self._steps[:] = 0
        self._autoreset[:] = False
        self._air = self.ac.airspeed()
        return self.observe(), {}

    def actions_to_commands(self, actions):
        a = np.clip(np.asarray(actions, float).reshape(self.num_envs, -1), -1.0, 1.0)
        k = len(self.surface_cols)
        servo = self._servo_cmd.copy()
        servo[:, self.surface_cols] = px4.servo_angle(a[:, :k])
        motor = px4.motor_speed((a[:, k:] + 1.0) / 2.0)
        return servo, motor

    def step(self, actions):
        servo, motor = self.actions_to_commands(actions)
        reset_now = self._autoreset.copy()
        for _ in range(self.substeps):
            self.ac.step(servo, motor, self.physics_dt)
        if np.any(reset_now):  # NEXT_STEP autoreset: these envs restart instead of stepping
            idx = np.flatnonzero(reset_now)
            X = self.ac.state.copy()
            X[idx] = self._initial(idx)
            self.ac.reset(X, t=self.ac.t)
            self._steps[idx] = 0
        self._air = self.ac.airspeed()
        self._steps += 1
        self._steps[reset_now] = 0
        terminated = self.failed() & ~reset_now
        truncated = (self._steps >= self.max_episode_steps) & ~terminated & ~reset_now
        reward = np.where(reset_now, 0.0, self.reward())
        self._autoreset = terminated | truncated
        return self.observe(), reward, terminated, truncated, {}
