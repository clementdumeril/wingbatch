# wingbatch

**gz-sim's fixed-wing physics, read from an SDF model and run for N aircraft at once in NumPy.**

wingbatch re-implements, line by line, the three gz-sim systems that fly the PX4 `advanced_plane`
(`AdvancedLiftDrag`, `MulticopterMotorModel`, `JointPositionController`) and a rigid body, and evaluates them on
arrays of N aircraft. It reads the real `model.sdf` (any model that uses `AdvancedLiftDrag`), keeps the SDF parameter
names, and ships the `advanced_plane` model. There is no rendering, no sensors, no ground and no PX4: only the flight
dynamics, which is what makes thousands of flights per second possible on one CPU core, for reinforcement learning,
genetic or gradient-free optimisation, Monte Carlo studies and system identification.

> wingbatch is **not affiliated with, nor endorsed by**, the PX4 project, the Dronecode Foundation, Open Robotics or
> the Gazebo project. "PX4" and "Gazebo" only describe what it is compatible with.

**Status: v0.1, research code.** The equations are checked against two independent transcriptions to 1e-11, but
**never yet against a flight computed by gz-sim itself** (see [Validation](#validation) and
[docs/validation_plan.md](docs/validation_plan.md)).

## Install

```bash
pip install git+https://github.com/clementdumeril/wingbatch          # numpy only
pip install "wingbatch[gym] @ git+https://github.com/clementdumeril/wingbatch"   # + gymnasium VectorEnv
```

## Quick start

```python
import numpy as np
import wingbatch as wb

ac = wb.Aircraft("advanced_plane", n=1000)          # or a path to any model.sdf using AdvancedLiftDrag
r = wb.trim(ac, airspeed=np.linspace(10, 20, 1000))  # steady level flight, one speed per aircraft
ac.reset(r["state"])

servo = r["servo_cmd"].copy()                        # (1000, 6) joint position commands, rad (gz topics servo_N)
servo[:, ac.servo_names.index("servo_2")] += 0.05     # a little more elevator for everyone
for _ in range(500):                                  # 5 s
    ac.step(servo, r["motor_cmd"], dt=0.01)           # motor_cmd: rotor speed, rad/s (gz command/motor_speed)

print(ac.position[:3])                                # world ENU, m
```

`wingbatch.px4` maps PX4's normalised actuator outputs to those commands (ESC range 10-1600 rad/s of the
`4008_gz_advanced_plane` airframe, servos at 45 deg per unit). `wingbatch.envs.LevelFlightVectorEnv` is a
`gymnasium.vector.VectorEnv` example (hold altitude and airspeed) to start an RL task from.

## What is simulated, and how faithfully

| gz-sim / SDF | wingbatch | Fidelity |
|---|---|---|
| `AdvancedLiftDrag` | `systems/advanced_lift_drag.py` | every statement of `Update()`, including the air velocity at `cp` (`v + w x cp - wind`), the sigmoid stall blend, the control-surface terms per joint degree, `Vector3d::Correct()`, the `cp x F` transport and the early return below 1e-9 m/s |
| `MulticopterMotorModel` (velocity) | `systems/multicopter_motor_model.py` | thrust, rotor drag, rolling moment, drag torque, `maxRotVelocity`, up/down time constants; geometry from the SDF (the rotor is pitched 1.57 rad, not pi/2, and that is kept) |
| `JointPositionController` + joint `<damping>` + `<limit>` | `systems/joint_position_controller.py` | reduced to its first-order lag `tau = damping / p_gain` (0.1 s), the limit of massless surfaces (checked against a discrete emulation of gz-math's PID and DART's implicit damping to 1e-3); integral or derivative gains are refused, because gz-sim's explicit derivative term does not reduce that way |
| the rotor joint gz-sim spins | `Aircraft(rotor_gyro=True)` | its angular momentum (rotor inertia about the joint axis, at speed / `rotorVelocitySlowdownSim`): gyroscopic moment and spin-up reaction, which DART produces through the joint (0.027 N m s at PX4's full throttle) |
| links, `<inertial>`, `<include>`, nested models | `sdf.py`, `physics.py` | composite rigid body of all links (the advanced_plane's lidar adds 21 g and moves the centre of mass 2.5 mm forward); SDFormat 1.4 to 1.11 frame semantics, scoped names |
| world gravity, uniform `<wind>` | `World`, `UniformWind` | as gz-sim; `WindField` (any function of position and time) is an extension gz-sim does not have |
| physics engine (DART) | `Aircraft.step` | RK4 (default), or `semi_implicit_euler`: gz-sim's order of operations and discrete rotor filter, and the free-body update of DART >= 6.13's `FreeJoint` (equal to round-off on a single body) |

Deliberate differences: the control-surface links do not move relative to the body (1e-8 kg each: reaction torques
of order 1e-4 N m), the rotor's transverse inertia does not turn with it (0.1 % of Iyy, Izz) and its thrust axis is
the link z axis at joint angle 0 (the 8e-4 rad off-axis component that averages out in gz-sim stays fixed here:
2e-3 relative); no contact (no ground, no take-off or landing); no sensors; constant air density, as gz-sim.

Two gz-sim behaviours are reproduced on purpose, with a switch:

- **`alpha == 0` exactly.** gz-sim computes `alpha / abs(alpha)`, which is NaN at zero; `Correct()` then zeroes the
  whole force for that update (moments are kept). Starting a flight with `w = 0` exactly hits it. Use
  `Aircraft(..., alpha_zero="continuous")` to take the sign as +1 instead. gz-sim has two other discontinuities at
  exact values (the branch cuts of alpha and beta at +-pi, in backwards flight); at such exact boundaries
  wingbatch, which computes in the link frame, and gz-sim, which computes in the world frame, can fall on
  different sides by round-off.
- **The gz-sim version matters.** Up to **gz-sim 8.14.0**, `AdvancedLiftDrag` overwrote the static pitching moment
  (`Cem = Cemb * beta` instead of `+=`), so `Cem0`, `Cema` and `Cema_stall` had no effect. This was fixed in
  gazebosim/gz-sim#3817 (8.15.0, 9.6.0, 10.x, August 2026). wingbatch computes the fixed version;
  `gz_cem_overwrite=True` reproduces 8.14.0 exactly (its `AdvancedLiftDrag.cc` differs from the ported one by that
  line only). Versions before November 2023 (gazebosim/gz-sim#2226) also ignored the wind.

## Conventions

| | |
|---|---|
| frames | Gazebo's: world ENU (z up), body = the canonical link frame (FLU for the advanced_plane). `wingbatch.frames` converts to NED / FRD and aerospace Euler angles |
| state (n, 13 + servos + motors) | position and world velocity of the canonical link origin (= gz `WorldPose`, `WorldLinearVelocity`), quaternion `w, x, y, z` link -> world, angular velocity in link axes (= gz `AngularVelocity`), servo joint positions (rad), filtered rotor speeds (rad/s) |
| servo order | the order of the `JointPositionController` plugins in the SDF: `ac.servo_names` (`servo_0, servo_1, servo_4, servo_5, servo_2, servo_3` for the advanced_plane; the flaps 4 and 5 have no aerodynamic effect in its SDF) |
| signs (advanced_plane) | positive joint angle: `servo_0` (left elevon) rolls left, `servo_1` (right elevon) rolls right, `servo_2` (elevator) pitches up, `servo_3` (rudder) yaws right. PX4's allocation (`CA_SV_CS0_TRQ_R = -0.5`, ...) agrees (tested) |

## Validation

What is checked, by the test suite (`pytest`, 89 tests, about a minute), on every push. The code was also reviewed
line by line against the gz-sim C++ by five separate reviewers (AdvancedLiftDrag; motor, servo and wrench application;
SDF reader, mass properties and licences; rigid body and integrators; fuzzing), whose findings are fixed and whose
tests are in the suite.

| Check | Tolerance |
|---|---|
| Forces and moments against `reference.py`, a scalar transcription of the C++ in the **world frame**, statement by statement, sharing no code with the vectorised version (it shares the parsed SDF): 1200 random states from 3 to 25 m/s, up to 57 deg angle of attack, any attitude, with and without wind, both gz-sim versions | 1e-11 |
| The same over 5100 extreme states: zero and 1e-10 m/s, backwards flight, sideslip near 90 deg, vertical and inverted attitudes, 20 rad/s, servos beyond their limits, negative and over-limit rotor speeds, wind stronger than the motion (states on gz-sim's exact branch cuts are excluded there and tested separately, 1e-12 off the cut) | 1e-11 |
| AdvancedLiftDrag against a second, independently written world-frame transcription, at edge states (alpha near 0, stall, +-pi/2, +-pi; beta near +-90 deg), random attitudes | 1e-10 |
| MulticopterMotorModel against an independent emulation of `UpdateForcesAndMoments` and `Link::AddWorldForce` that builds every pose from the raw SDF text, on the advanced_plane and on a perturbed copy (rotated, offset canonical link, rotor centre of mass offset, joint pose, axis in the joint frame) | 1e-9 |
| A generic SDF unlike the advanced_plane: plugin on a rotated non-canonical link, non-orthogonal `forward`/`upward`, every coefficient non-zero, a tilted counter-clockwise rotor with an offset centre of mass | 1e-11 |
| Derivatives, and 1 s of RK4 flight, against the scalar reference (rotor momentum included) | 1e-10, 1e-8 |
| Forces and moments against the earlier **independent port** (WingJEPA, NumPy transcription of the Fixed-Wing C++ model, itself checked against the C++ to 1e-11), 400 random states incl. stall and wind | 1e-11 |
| Trajectories against that port with its simplifications (3 s, violent random states, no rotor momentum): the gap is RK4's truncation, it shrinks like dt^4 | < 5e-5 m at 5 ms |
| One `semi_implicit_euler` step against DART 6.13's `FreeJoint` update; servos against a discrete gz-math PID + DART joint emulation | 1e-12; 1e-3 |
| Rigid body against an independent matrix reference (offset centre of mass, gravity-free link); free fall; torque-free tumbling (momentum, energy); uniform wind only carries the aircraft; each aircraft of a batch flies as if alone; trimmed flight stays trimmed for 12 s; Gazebo-like stepping converges to RK4 at first order; rotor filter equals gz-sim's | 1e-7 to 1e-12 |
| Control-surface signs, stall curve, frames and Euler angles (gimbal lock included), PX4 mapping, SDF parsing (every coefficient, includes, nested scopes, frames, SDF 1.4 axes, pose formats, malformed input refused), API edge cases (n = 0, broadcasting, non-finite commands refused) | exact |

What is **not** validated yet:

- **no comparison with a flight computed by gz-sim** (`gz sim`); the plan, with frozen versions and reference traces,
  is in [docs/validation_plan.md](docs/validation_plan.md);
- **no closed-loop comparison with PX4 SITL**; wingbatch contains no autopilot;
- the rigid composite body, the rotor momentum model and the servo reduction are checked against emulations of
  DART's equations, not against DART itself.

## Performance

Aircraft-steps per second on **one CPU core**, the bundled advanced_plane trimmed at 12 m/s with small
perturbations, `python benchmarks/throughput.py` (best of 5 trials; medians in
[benchmarks/results](benchmarks/results)). Machine: Intel Core i7-6700HQ (2015 mobile CPU, 2.6 GHz), Windows 10,
Python 3.11.9, NumPy 2.2.6, BLAS pinned to one thread. The machine was in use during the run: medians are 5 to 40 %
lower, and a dedicated machine will do better.

| N aircraft | RK4, steps/s | vs. scalar loop | semi-implicit Euler, steps/s |
|---:|---:|---:|---:|
| 1 | 210 | 0.2x | 839 |
| 10 | 1,965 | 1.7x | 7,613 |
| 100 | 17,208 | 15x | 42,889 |
| 1,000 | 98,273 | 84x | 384,401 |
| 10,000 | 74,506 | 63x | 290,114 |
| 100,000 | 83,818 | 71x | 318,406 |

The baseline is `wingbatch.reference.ScalarAircraft`, the same physics one aircraft at a time in plain Python:
1,175 RK4 steps/s. In flight time: RK4 at 10 ms flies **about 750 to 1,000 aircraft-seconds per second** for
N >= 1,000; the Gazebo-like stepping at 4 ms (gz-sim's step), about 1,200 to 1,500. For a single aircraft wingbatch
is **slower** than the scalar loop (NumPy's per-call overhead): it is made for batches. Large batches are stepped in
blocks of 1,024 aircraft to stay in the CPU cache (`Aircraft(chunk=...)`). Not measured: the speed of gz-sim itself
on the same model.

## Remarks on the advanced_plane model

wingbatch flies the SDF exactly as written. A few properties of the model are worth knowing when using it far from
PX4's 15 m/s cruise; they are observations, not corrections:

- the moments are transported from `cp` (0.12 m behind the centre of mass) to the centre of mass on top of `Cema`;
  the resulting static margin is large (about 64 % of the chord), so the trim needs a lot of elevator at low speed
  (about 9 deg at 10 m/s);
- the control derivatives act linearly, drag included (`CD_ctrl`), as AVL derivatives taken at one point;
- the inertia tensor is the one of PX4's generic `plane` model (Izz < Ixx);
- the drag polar (`CD0` = 0.029, `eff` = 0.97) gives a lift-to-drag ratio of about 14.

## Related work

No earlier port of `AdvancedLiftDrag` with the advanced_plane to batched array code was found (GitHub, PyPI, arXiv,
PX4 and Gazebo forums, November 2026). Closest projects:

- [IMARLSim](https://github.com/liu-vacancyyy/IMARLSim): PyTorch, batched, ports Gazebo Classic's older `LiftDrag`
  and the motor model for the `standard_vtol`;
- [NeuralPlane](https://github.com/xuecy22/NeuralPlane) (NeurIPS 2024) and FALCON-S (arXiv 2609.06046): GPU-batched
  fixed-wing simulators with their own aircraft models;
- [JSBGym](https://github.com/sryu1/jsbgym), [PyFlyt](https://github.com/jjshoots/PyFlyt): established single-aircraft
  RL environments;
- [recovery_quadtailsitter](https://github.com/rknishi/recovery_quadtailsitter) (CasADi transcription of the
  AdvancedLiftDrag equations inside an MPC) and [AP-PnC](https://github.com/SYSU-HILAB/AP-PnC) (a C++ cut-down
  AdvancedLiftDrag).

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). The systems are derived from gz-sim (Apache-2.0, Open Source Robotics
Foundation; AdvancedLiftDrag by Karthik Srivatsan and Frederik Markus), branch gz-sim8 at commit `f380a94e`; each
derived file states its changes. The bundled models are unmodified copies from
[PX4-gazebo-models](https://github.com/PX4/PX4-gazebo-models) (BSD-3-Clause).
