"""SDF parsing, and a generic AdvancedLiftDrag model unlike the advanced_plane."""

import warnings

import numpy as np
import pytest

import wingbatch as wb
from helpers import advanced_plane, random_states, world_wrench
from wingbatch.reference import ScalarAircraft
from wingbatch.sdf import load_model


def test_advanced_plane_parameters_are_read_from_the_sdf():
    ac = advanced_plane()
    p = ac.lift_drag[0].system.params
    assert (p.CL0, p.CLa, p.CD0, p.eff, p.AR) == (0.15188, 5.015, 0.029, 0.97, 6.5)
    assert (p.Cem0, p.Cema, p.Cemq, p.Cellp, p.Cenr) == (0.075, -0.463966, -12.140140, -0.487407, -0.089947)
    assert p.alpha_stall == 0.3391428111 and p.Cema_stall == 0.0
    assert p.cp == (-0.12, 0.0, 0.0) and p.area == 0.34 and p.mac == 0.22 and p.air_density == 1.2041
    assert [cs.name for cs in p.control_surfaces] == ["servo_0", "servo_1", "servo_2", "servo_3"]
    assert [cs.direction for cs in p.control_surfaces] == [1, 1, -1, 1]
    assert p.control_surfaces[2].Cem_ctrl == -0.025798
    m = ac.motors[0].params
    assert (m.turningDirection, m.maxRotVelocity, m.motorConstant, m.momentConstant) == (-1, 3500, 8.54858e-06, 0.01)
    assert (m.timeConstantUp, m.timeConstantDown, m.rotorDragCoefficient) == (0.0125, 0.025, 8.06428e-05)
    np.testing.assert_allclose(ac.motors[0].thrust_dir, [np.sin(1.57), 0.0, np.cos(1.57)], atol=1e-15)
    np.testing.assert_allclose(ac.motors[0].com, [0.3, 0.0, 0.0])
    np.testing.assert_allclose(ac.motors[0].axis, [1.0, 0.0, 0.0], atol=1e-15)  # use_parent_model_frame
    assert ac.servo_names == ["servo_0", "servo_1", "servo_4", "servo_5", "servo_2", "servo_3"]
    np.testing.assert_allclose(ac.servos.lower, -0.78)
    np.testing.assert_allclose([s.time_constant for s in ac.servos.servos], 0.1)


def test_mass_properties_include_the_lidar_and_the_rotor():
    ac = advanced_plane()
    assert ac.mass.mass == pytest.approx(1.0 + 0.005 + 0.02 + 0.001 + 6e-8)
    np.testing.assert_allclose(ac.mass.com, [(0.005 * 0.3 + 0.02 * 0.05 + 0.001 * 0.05 - 2e-8 * 0.5) / 1.02600006, 0.0,
                                             -0.02 * 0.1 / 1.02600006], rtol=1e-6, atol=1e-9)
    names = set(ac.model.links)
    assert {"base_link", "rotor_puller", "lw20_link", "lidar_sensor_link", "elevator"} <= names


def test_unresolved_include_and_unknown_plugins_are_reported(tmp_path):
    f = tmp_path / "m.sdf"
    f.write_text("""<sdf version="1.9"><model name="m"><link name="a"><inertial><mass>1</mass></inertial></link>
      <include merge="true"><uri>model://does_not_exist</uri></include>
      <plugin filename="libfoo.so" name="foo::Bar"/></model></sdf>""")
    m = load_model(f)
    assert any("does_not_exist" in w for w in m.warnings)
    assert any("foo::Bar" in w for w in m.warnings)
    with pytest.warns(UserWarning):
        wb.Aircraft(f, 1)


def test_pose_formats(tmp_path):
    f = tmp_path / "m.sdf"
    f.write_text("""<sdf version="1.9"><model name="m">
      <link name="a"><inertial><mass>1</mass></inertial></link>
      <link name="b"><pose relative_to="a" degrees="true">1 0 0 0 0 90</pose><inertial><mass>1</mass></inertial></link>
      <link name="c"><pose relative_to="b">1 0 0 0 0 0</pose><inertial><mass>1</mass></inertial></link>
      <link name="d"><pose rotation_format="quat_xyzw">0 0 1 0 0 0.7071067811865476 0.7071067811865476</pose>
        <inertial><mass>1</mass></inertial></link>
    </model></sdf>""")
    m = load_model(f)
    np.testing.assert_allclose(m.link_pose("c").p, [1.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(m.link_pose("d").R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)


GENERIC = """<?xml version="1.0"?>
<sdf version="1.9"><model name="odd_wing">
  <link name="hull"><inertial><pose>0.02 0 -0.01 0 0 0</pose><mass>3.0</mass>
    <inertia><ixx>0.4</ixx><iyy>0.3</iyy><izz>0.6</izz><ixz>0.01</ixz></inertia></inertial></link>
  <link name="wing"><pose>0.1 0.05 0.02 0.1 0.2 0.3</pose><inertial><mass>0.5</mass>
    <inertia><ixx>0.05</ixx><iyy>0.01</iyy><izz>0.06</izz></inertia></inertial></link>
  <joint name="wing_fix" type="fixed"><parent>hull</parent><child>wing</child></joint>
  <link name="flap"><pose relative_to="wing">-0.2 0 0 0 0 0</pose><inertial><mass>1e-8</mass>
    <inertia><ixx>1e-6</ixx><iyy>1e-6</iyy><izz>1e-6</izz></inertia></inertial></link>
  <joint name="flap_joint" type="revolute"><parent>wing</parent><child>flap</child>
    <axis><xyz>0 1 0</xyz><limit><lower>-0.5</lower><upper>0.5</upper></limit><dynamics><damping>2</damping></dynamics></axis></joint>
  <link name="prop"><pose>0.4 0 0 0 1.5 0.1</pose><inertial><pose>0.01 0 0 0 0 0</pose><mass>0.01</mass>
    <inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>2e-5</izz></inertia></inertial></link>
  <joint name="prop_joint" type="revolute"><parent>hull</parent><child>prop</child><axis><xyz>0 0 1</xyz></axis></joint>
  <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
    <joint_name>flap_joint</joint_name><p_gain>8</p_gain><i_gain>0</i_gain><d_gain>0</d_gain></plugin>
  <plugin filename="gz-sim-advanced-lift-drag-system" name="gz::sim::systems::AdvancedLiftDrag">
    <link_name>wing</link_name>
    <CL0>0.1</CL0><CLa>4.5</CLa><CD0>0.03</CD0><Cem0>0.02</Cem0><Cema>-0.6</Cema><AR>8</AR><eff>0.9</eff>
    <CYa>0.05</CYa><Cella>0.01</Cella><Cena>-0.02</Cena><CLb>0.03</CLb><CYb>-0.3</CYb><Cellb>-0.05</Cellb>
    <Cemb>0.04</Cemb><Cenb>0.08</Cenb><alpha_stall>0.3</alpha_stall><Cema_stall>-0.2</Cema_stall>
    <CDp>0.01</CDp><CYp>0.05</CYp><CLp>0.02</CLp><Cellp>-0.4</Cellp><Cemp>0.01</Cemp><Cenp>-0.03</Cenp>
    <CDq>0.04</CDq><CYq>0.01</CYq><CLq>6</CLq><Cellq>0.01</Cellq><Cemq>-10</Cemq><Cenq>0.02</Cenq>
    <CDr>0.01</CDr><CYr>0.2</CYr><CLr>0.03</CLr><Cellr>0.07</Cellr><Cemr>0.02</Cemr><Cenr>-0.08</Cenr>
    <cp>-0.05 0.01 0.02</cp><area>0.5</area><air_density>1.1</air_density>
    <forward>1 0.1 0</forward><upward>0 0.2 1</upward>
    <num_ctrl_surfaces>1</num_ctrl_surfaces>
    <control_surface><name>flap_joint</name><direction>-1</direction><CD_ctrl>0.001</CD_ctrl>
      <CY_ctrl>0.0002</CY_ctrl><CL_ctrl>0.02</CL_ctrl><Cell_ctrl>0.001</Cell_ctrl><Cem_ctrl>-0.01</Cem_ctrl>
      <Cen_ctrl>0.0005</Cen_ctrl></control_surface>
  </plugin>
  <plugin filename="gz-sim-multicopter-motor-model-system" name="gz::sim::systems::MulticopterMotorModel">
    <jointName>prop_joint</jointName><linkName>prop</linkName><turningDirection>ccw</turningDirection>
    <maxRotVelocity>1000</maxRotVelocity><motorConstant>1e-5</motorConstant><momentConstant>0.02</momentConstant>
    <rotorDragCoefficient>1e-4</rotorDragCoefficient><rollingMomentCoefficient>1e-5</rollingMomentCoefficient>
  </plugin>
</model></sdf>"""


def test_generic_model_matches_the_reference(tmp_path):
    """Plugin on a rotated non-canonical link, non-orthogonal forward/upward, no mac (area / span), every
    coefficient non-zero, a ccw rotor on a tilted link with an offset centre of mass."""
    f = tmp_path / "odd.sdf"
    f.write_text(GENERIC)
    ac = wb.Aircraft(f, 200, wind=(1.0, 2.0, -0.5), warn=False)
    assert ac.lift_drag[0].system.mac == pytest.approx(0.5 / np.sqrt(0.5 * 8))
    X = random_states(ac, np.random.default_rng(6))
    X[:, ac.idx["servos"]] = np.random.default_rng(7).uniform(-0.5, 0.5, (200, 1))
    F, T, _ = world_wrench(ac, X)
    ref = ScalarAircraft(ac)
    for i in range(ac.n):
        fr, tr = ref.wrench_world(list(X[i]))
        np.testing.assert_allclose(F[i], fr.tolist(), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(T[i], tr.tolist(), rtol=1e-11, atol=1e-11)
    s = np.full((200, 1), 0.2)
    m = np.full((200, 1), 600.0)
    d, _ = ac.derivatives(X, s, m)
    for i in range(0, 200, 20):
        np.testing.assert_allclose(d[i], ref.derivatives(list(X[i]), [0.2], [600.0]), rtol=1e-10, atol=1e-10)


def test_bad_configurations_are_refused(tmp_path):
    f = tmp_path / "bad.sdf"
    f.write_text(GENERIC.replace("<i_gain>0</i_gain>", "<i_gain>0.5</i_gain>"))
    with pytest.raises(NotImplementedError, match="i_gain"):
        wb.Aircraft(f, 1, warn=False)
    f.write_text(GENERIC.replace("<num_ctrl_surfaces>1</num_ctrl_surfaces>", "<num_ctrl_surfaces>2</num_ctrl_surfaces>"))
    with pytest.raises(ValueError, match="num_ctrl_surfaces"):
        wb.Aircraft(f, 1, warn=False)
