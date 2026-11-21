# Validating wingbatch against gz-sim itself

Status (November 2026): **not done yet.** wingbatch has been checked against two independent re-implementations of
the same equations (see the README), never against a flight computed by gz-sim. This is the plan to close that gap.
Installing Gazebo is a deliberate step and has not been taken.

## 1. Freeze the versions

| Item | Version to pin | Why |
|---|---|---|
| gz-sim | 8.15.0 or later (Harmonic), and once 8.14.0 | the AdvancedLiftDrag pitching moment changed in 8.15.0 (gazebosim/gz-sim#3817): compare each version with `gz_cem_overwrite` off / on |
| gz-physics | the one shipped with that gz-sim, dartsim engine | the integrator under test |
| PX4-gazebo-models | `e5997f455e33ec071c7c677fc59a65781de496a4` | the bundled `model.sdf` |
| world | PX4's `default.sdf` (4 ms step, gravity 9.8, no wind) plus copies with a uniform `<wind>` | |

Record `gz sim --versions`, the OS and the CPU with every trace.

## 2. Reference traces, open loop (no PX4)

Run `gz sim -s -r --iterations N` headless with a small gz-sim system plugin (or the Python bindings' `TestFixture`)
that, at every step:

1. publishes scripted commands on `/model/advanced_plane/servo_N` and `command/motor_speed` (steps, doublets and
   sweeps on each surface, throttle steps), starting from a pose and velocity set with `/world/.../set_pose` and a
   `LinearVelocityCmd` / `AngularVelocityCmd` on the first step;
2. logs the base_link `WorldPose`, `WorldLinearVelocity`, `AngularVelocity`, every joint position and velocity,
   and the `ExternalWorldWrenchCmd` of base_link and rotor_puller (the summed system wrenches) to CSV at 250 Hz.

Cases: straight flight at 10, 12, 15, 20 m/s; elevator doublets; aileron and rudder doublets; a stall entry; full
throttle climb; each with zero wind and with a uniform 5 m/s cross wind. About 30 s each.

## 3. Comparisons, from the simplest to the hardest

1. **Wrench given the state** (no integration involved): feed each logged state into `Aircraft.wrenches` and compare
   with the logged `ExternalWorldWrenchCmd`. Expected to agree to float round-off; any difference is a porting error.
2. **One step**: from each logged state, one `semi_implicit_euler` step of 4 ms against the next logged state. Measures
   the DART integrator difference (joint constraints, the massless control-surface links, the spinning rotor).
3. **Free flights**: whole open-loop trajectories with RK4 at 10 ms and with the Gazebo-like stepping at 4 ms. Report
   the divergence time and the error growth, and say which effect explains it.

Scripts and traces go in `validation/` with the exact command lines, and the README gets a table: case, tolerance
reached, gz-sim version.

## 4. Closed loop with PX4 (later)

PX4 SITL flies the advanced_plane in gz-sim; log the ULog and the actuator outputs; replay the actuator outputs through
`wingbatch.px4` into wingbatch from the same initial state. This checks the PX4 actuator mapping and gives the
divergence time of a real mission. PX4's controllers themselves are not part of wingbatch.
