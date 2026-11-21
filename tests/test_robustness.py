"""API robustness: model reuse, wind shapes, batch sizes, command broadcasting, determinism, trim flags, long
flights, saturations, non-finite commands, gravity-free links."""

import warnings

import numpy as np
import pytest

import wingbatch as wb
from helpers import advanced_plane, random_states
from wingbatch.sdf import load_model

GENERIC_NO_SERVO = """<?xml version="1.0"?>
<sdf version="1.9"><model name="wing_only">
  <link name="hull"><inertial><mass>1.0</mass><inertia><ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz></inertia>
  </inertial></link>
  <link name="flap"><pose>-0.2 0 0 0 0 0</pose><inertial><mass>1e-8</mass>
    <inertia><ixx>1e-6</ixx><iyy>1e-6</iyy><izz>1e-6</izz></inertia></inertial></link>
  <joint name="flap_joint" type="revolute"><parent>hull</parent><child>flap</child><axis><xyz>0 1 0</xyz></axis></joint>
  <plugin filename="gz-sim-advanced-lift-drag-system" name="gz::sim::systems::AdvancedLiftDrag">
    <link_name>hull</link_name><CL0>0.1</CL0><CLa>4.5</CLa><CD0>0.03</CD0><AR>8</AR><eff>0.9</eff><area>0.5</area>
    <num_ctrl_surfaces>1</num_ctrl_surfaces>
    <control_surface><name>flap_joint</name><direction>1</direction><CD_ctrl>0</CD_ctrl><CY_ctrl>0</CY_ctrl>
      <CL_ctrl>0.01</CL_ctrl><Cell_ctrl>0</Cell_ctrl><Cem_ctrl>0</Cem_ctrl><Cen_ctrl>0</Cen_ctrl></control_surface>
  </plugin>
</model></sdf>"""


# ------------------------------------------------------------------------------------------ model reuse
def test_building_twice_from_one_model_does_not_grow_its_warnings(tmp_path):
    """Regression: Aircraft.__init__ appends 'control surface ... has no JointPositionController' to model.warnings, so
    every Aircraft built from the same Model warns once more than the previous one."""
    f = tmp_path / "w.sdf"
    f.write_text(GENERIC_NO_SERVO)
    model = load_model(f)
    before = list(model.warnings)
    counts = []
    for _ in range(3):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            wb.Aircraft(model, 1)
        counts.append(len(w))
    assert counts == [counts[0]] * 3 and counts[0] >= 1
    assert model.warnings == before  # the parsed model is not mutated


def test_two_aircraft_from_one_model_fly_identically():
    model = load_model("advanced_plane")
    a, b = wb.Aircraft(model, 2, warn=False), wb.Aircraft(model, 2, warn=False)
    X = random_states(a, np.random.default_rng(0))
    a.reset(X), b.reset(X)
    for _ in range(20):
        a.step(0.1, 900.0), b.step(0.1, 900.0)
    np.testing.assert_array_equal(a.state, b.state)


# ------------------------------------------------------------------------------------------ wind
def test_wind_field_of_the_wrong_shape_is_refused():
    wf = wb.WindField(lambda p, t: np.zeros((3, len(p))))  # one row per component: a common slip
    with pytest.raises(ValueError):
        wf(np.zeros((4, 3)), 0.0)


def test_wind_field_of_the_right_shape_and_time_is_used_per_aircraft():
    W = np.array([[1.0, 0.0, 0.0], [0.0, -2.0, 0.5]])
    calls = []

    def fn(p, t):
        calls.append(t)
        return W * (1.0 + t)
    ac = advanced_plane(2, wind=wb.WindField(fn))
    ac2 = advanced_plane(2, wind=W)
    X = ac.make_state(velocity=(12.0, 0.0, -1.0))
    np.testing.assert_array_equal(ac.wrenches(X, 0.0)[0], ac2.wrenches(X, 0.0)[0])
    ac.reset(X)
    ac.step(0.0, 0.0, 0.01)
    assert calls[-4:] == [0.0, 0.005, 0.005, 0.01]  # RK4 stage times


def test_wrong_wind_shapes_raise():
    with pytest.raises(ValueError):
        advanced_plane(3, wind=np.ones((2, 3))).step(0.0, 0.0)
    with pytest.raises(ValueError):
        wb.WindField(lambda p, t: np.ones(3))(np.zeros((3, 3)), 0.0)


# ------------------------------------------------------------------------------------------ batch sizes
def test_empty_batch_steps_and_rolls_out():
    ac = advanced_plane(0)
    assert ac.step(0.0, 0.0).shape == (0, ac.nx)
    assert ac.rollout(np.zeros((3, 0, ac.n_servos)), 0.0).shape == (4, 0, ac.nx)


def test_empty_batch_trims():
    """Regression: trim / steady_state stack an empty list of quaternions for n = 0."""
    ac = advanced_plane(0)
    assert wb.steady_state(ac, 12.0, 0.05).shape == (0, ac.nx)
    assert wb.trim(ac, 12.0)["ok"].shape == (0,)


@pytest.mark.parametrize("n", [1, 3])
def test_command_shapes_broadcast_alike(n):
    ac = advanced_plane(n)
    r = wb.trim(advanced_plane(1), 12.0)
    X = np.repeat(r["state"], n, 0)
    s0, m0 = r["servo_cmd"][0], r["motor_cmd"][0, 0]
    out = []
    for s, m in [(s0, m0), (s0, np.array([m0])), (np.tile(s0, (n, 1)), np.full((n, 1), m0))]:
        ac.reset(X)
        out.append(ac.step(s, m, 0.01).copy())
    for o in out[1:]:
        np.testing.assert_array_equal(o, out[0])
    ac.reset(X)
    ac.step(float(s0[0]), m0)  # scalar servo command: every servo
    np.testing.assert_array_equal(ac.rollout(np.tile(s0, (5, n, 1)), m0)[1:].shape, (5, n, ac.nx))


# ------------------------------------------------------------------------------------------ determinism
def test_bit_for_bit_deterministic():
    def run():
        ac = advanced_plane(257)
        rng = np.random.default_rng(0)
        ac.reset(random_states(ac, rng))
        for _ in range(30):
            ac.step(rng.uniform(-0.9, 0.9, (257, ac.n_servos)), rng.uniform(-100, 4000, (257, 1)), 0.01)
        return ac.state.copy()
    np.testing.assert_array_equal(run(), run())


# ------------------------------------------------------------------------------------------ trim
def test_trim_fails_below_the_stall_speed_and_at_rest():
    ac = advanced_plane(5)
    r = wb.trim(ac, [0.0, 3.0, 5.0, 6.0, 7.0])
    assert not r["ok"].any()


def test_trim_ok_means_wing_borne_flight():
    """Regression: at 1.25-2.75 m/s trim converges to a prop-hang (alpha 76-87 deg, the rotor carrying the weight) and
    flags it ok: ok does not check the angle of attack against alpha_stall."""
    ac = advanced_plane(4)
    r = wb.trim(ac, [1.5, 2.5, 12.0, 25.0])
    a_s = ac.lift_drag[0].system.params.alpha_stall
    assert np.all(~r["ok"] | (np.abs(r["alpha"]) < a_s)), np.degrees(r["alpha"])
    assert r["ok"][2:].all()


def test_trim_is_independent_of_the_batch():
    r1 = wb.trim(advanced_plane(1), 12.0)
    rn = wb.trim(advanced_plane(3), [20.0, 12.0, 9.0])
    np.testing.assert_allclose(rn["state"][1], r1["state"][0], rtol=1e-9, atol=1e-9)


def test_trimmed_state_is_an_equilibrium_of_the_integrator():
    """One RK4 step and 12 s of flight at dt = 0.02 from the trim: straight, level, unit quaternion. (The
    advanced_plane has an unstable spiral mode, Re = 0.05-0.2 1/s, so longer open-loop flights drift physically
    from round-off seeds; keep the horizon short.)"""
    ac = advanced_plane(2)
    r = wb.trim(ac, [18.0, 25.0])
    assert r["ok"].all()
    ac.reset(r["state"])
    X0 = ac.state.copy()
    d, _ = ac.derivatives(X0, r["servo_cmd"], r["motor_cmd"])
    np.testing.assert_allclose(d[:, 7:13], 0.0, atol=1e-10)
    for _ in range(600):
        ac.step(r["servo_cmd"], r["motor_cmd"], 0.02)
    np.testing.assert_allclose(np.linalg.norm(ac.quaternion, axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(ac.velocity, X0[:, 7:10], atol=1e-6)
    np.testing.assert_allclose(ac.position, X0[:, 0:3] + 12.0 * X0[:, 7:10], atol=1e-4)
    np.testing.assert_allclose(ac.quaternion, X0[:, 3:7], atol=1e-7)


# ------------------------------------------------------------------------------------------ saturations
@pytest.mark.parametrize("integrator", ["rk4", "semi_implicit_euler"])
def test_servo_stops_exactly_at_the_joint_limit(integrator):
    ac = advanced_plane(2, integrator=integrator)
    ac.reset(servos=np.array([[1.5] * 6, [-1.5] * 6]))  # start beyond the limits
    for _ in range(200):
        ac.step(np.array([[2.0] * 6, [-2.0] * 6]), 0.0, 0.01)
    np.testing.assert_array_equal(ac.state[:, ac.idx["servos"]], np.array([[0.78] * 6, [-0.78] * 6]))


@pytest.mark.parametrize("integrator", ["rk4", "semi_implicit_euler"])
def test_rotor_command_is_capped_above_but_not_below(integrator):
    """gz-sim: min(cmd, maxRotVelocity), no lower bound (negative speed: negative thrust)."""
    ac = advanced_plane(2, integrator=integrator)
    ac.reset(motors=0.0)
    for _ in range(100):
        ac.step(0.0, np.array([[9000.0], [-500.0]]), 0.01)
    np.testing.assert_allclose(ac.state[:, ac.idx["motors"]][:, 0], [3500.0, -500.0], rtol=1e-6)
    _, _, info = ac.wrenches(ac.state)
    assert info["thrust0"][0] > 0 > info["thrust0"][1]


# ------------------------------------------------------------------------------------------ non-finite
def test_non_finite_command_is_not_silently_absorbed():
    """Regression: a NaN servo command makes the joint NaN, AdvancedLiftDrag's Correct() then zeroes the whole
    lift-drag wrench and the aircraft falls with a finite state: a NaN controller output looks like a stall. Step
    should refuse non-finite commands (or the NaN should reach the state)."""
    ac = advanced_plane(1)
    r = wb.trim(ac, 12.0)
    ac.reset(r["state"])
    cmd = r["servo_cmd"].copy()
    cmd[0, 0] = np.nan
    try:
        for _ in range(10):
            ac.step(cmd, r["motor_cmd"], 0.01)
    except (ValueError, FloatingPointError):
        return
    assert not np.isfinite(ac.state[0, :13]).all()


# ------------------------------------------------------------------------------------------ gravity flag
def test_link_without_gravity(tmp_path):
    """<gravity>0</gravity> on an offset link: the body falls at g m_g / m and pitches/rolls about the centre of
    mass under the moment of the weight (the branch of Aircraft._accelerations the reference does not model)."""
    f = tmp_path / "g.sdf"
    f.write_text("""<sdf version="1.9"><model name="g">
      <link name="a"><inertial><mass>2.0</mass><inertia><ixx>0.3</ixx><iyy>0.4</iyy><izz>0.5</izz></inertia>
      </inertial></link>
      <link name="b"><pose>0.5 0.2 -0.1 0 0 0</pose><gravity>0</gravity><inertial><mass>1.0</mass>
        <inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial></link>
      <joint name="fix" type="fixed"><parent>a</parent><child>b</child></joint></model></sdf>""")
    ac = wb.Aircraft(f, 1, warn=False)
    mp = ac.mass
    assert mp.gravity_mass == 2.0
    np.testing.assert_allclose(mp.gravity_com, 0.0)
    d, _ = ac.derivatives(ac.make_state(), np.zeros((1, 0)), np.zeros((1, 0)))
    Fg = np.array([0.0, 0.0, -9.8 * 2.0])
    wdot = np.linalg.solve(mp.inertia, np.cross(mp.gravity_com - mp.com, Fg))
    a_origin = Fg / 3.0 - np.cross(wdot, mp.com)
    np.testing.assert_allclose(d[0, 10:13], wdot, rtol=1e-12)
    np.testing.assert_allclose(d[0, 7:10], a_origin, rtol=1e-12)
