"""SDFormat cases for wingbatch.sdf / MassProperties: the advanced_plane's mass properties by hand, Euler
convention, frame defaults, nested scopes, frames of merged includes, plugins of non-merged includes, SDF 1.4
axes, refused malformed input."""
import math
import warnings

import numpy as np
import pytest

import wingbatch as wb
from wingbatch.sdf import load_model

L = lambda n, extra="": f'<link name="{n}">{extra}<inertial><mass>1</mass></inertial></link>'


def write(tmp_path, body, name="m.sdf", version="1.9"):
    f = tmp_path / name
    f.write_text(f'<sdf version="{version}">{body}</sdf>')
    return f


# ---------------------------------------------------------------- basics
def test_mass_properties_by_hand():
    m = load_model("advanced_plane")
    mp = wb.MassProperties.from_model(m)
    Ry = lambda t: np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    parts = [(1.0, [0, 0, 0], np.diag([0.197563, 0.1458929, 0.1477])),
             (0.005, [0.3, 0, 0], Ry(1.57) @ np.diag([9.75e-07, 0.000166704, 0.000167604]) @ Ry(1.57).T),
             (0.02, [0.05, 0, -0.1], Ry(1.57) @ np.diag([4e-6, 4e-6, 2e-6]) @ Ry(1.57).T),
             (0.001, [0.05, 0, 0], np.eye(3) * 1e-5)]
    for c in ([0, .3, 0], [0, -.3, 0], [0, .15, 0], [0, -.15, 0], [-.5, 0, 0], [-.5, 0, .05]):
        parts.append((1e-8, c, np.eye(3) * 1e-6))
    M = sum(p[0] for p in parts)
    c = sum(p[0] * np.array(p[1], float) for p in parts) / M
    I = sum(p[2] + p[0] * ((np.array(p[1]) - c) @ (np.array(p[1]) - c) * np.eye(3)
                           - np.outer(np.array(p[1]) - c, np.array(p[1]) - c)) for p in parts)
    assert mp.mass == pytest.approx(M, rel=1e-15)
    np.testing.assert_allclose(mp.com, c, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(mp.inertia, I, rtol=1e-12, atol=1e-15)
    print("\nmass", M, "com", c, "\nI", I)


def test_euler_is_extrinsic_xyz(tmp_path):
    f = write(tmp_path, f'<model name="m">{L("a")}{L("b", "<pose>0 0 0 0.3 -0.7 1.1</pose>")}</model>')
    R = load_model(f).link_pose("b").R
    cx, sx, cy, sy, cz, sz = math.cos(.3), math.sin(.3), math.cos(-.7), math.sin(-.7), math.cos(1.1), math.sin(1.1)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    np.testing.assert_allclose(R, Rz @ Ry @ Rx, atol=1e-14)


def test_joint_pose_default_child_and_axis_frames(tmp_path):
    f = write(tmp_path, f'''<model name="m">{L("a")}{L("b", "<pose>1 0 0 0 0 1.5707963267948966</pose>")}
      <joint name="j" type="revolute"><parent>a</parent><child>b</child><pose>0 1 0 0 0 0</pose>
        <axis><xyz>1 0 0</xyz></axis></joint>
      <joint name="k" type="revolute"><parent>a</parent><child>b</child>
        <axis><xyz expressed_in="__model__">1 0 0</xyz></axis></joint></model>''')
    m = load_model(f)
    np.testing.assert_allclose(m.joints["j"].pose.p, [0, 0, 0], atol=1e-12)  # (1,0,0) + Rz90 (0,1,0)
    np.testing.assert_allclose(m.joints["j"].axis, [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(m.joints["k"].axis, [1, 0, 0], atol=1e-12)


# ---------------------------------------------------------------- regressions from the review
def test_doubly_nested_model_names_keep_the_outer_scope(tmp_path):
    f = write(tmp_path, f'''<model name="m">{L("base")}
      <model name="A"><pose>1 0 0 0 0 0</pose>{L("la")}
        <model name="B"><pose>0 1 0 0 0 0</pose>{L("lb")}</model>
        <joint name="ja" type="fixed"><parent>la</parent><child>B::lb</child></joint></model></model>''')
    m = load_model(f)
    assert "A::B::lb" in m.links, sorted(m.links)
    assert m.joints["A::ja"].child in m.links


def test_nested_and_merged_frames_are_kept(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "model.sdf").write_text(
        f'<sdf version="1.9"><model name="sub">{L("s")}<frame name="tip"><pose>0 0 1 0 0 0</pose></frame>'
        '</model></sdf>')
    f = write(tmp_path, f'''<model name="m">{L("base")}
      <include merge="true"><uri>model://sub</uri><pose>2 0 0 0 0 0</pose></include>
      {L("x", '<pose relative_to="tip">0 0 0 0 0 0</pose>')}</model>''')
    m = load_model(f, resource_paths=[tmp_path])
    np.testing.assert_allclose(m.link_pose("x").p, [2, 0, 1])


def test_relative_to_nested_model_frame(tmp_path):
    f = write(tmp_path, f'''<model name="m">{L("base")}
      <model name="N"><pose>0 0 3 0 0 0</pose>{L("n")}</model>
      {L("x", '<pose relative_to="N">1 0 0 0 0 0</pose>')}</model>''')
    m = load_model(f)
    np.testing.assert_allclose(m.link_pose("x").p, [1, 0, 3])


def test_plugins_of_a_non_merged_include_keep_their_scope(tmp_path):
    from wingbatch.sdf import MODELS_DIR
    f = write(tmp_path, '''<model name="wrapper">
      <include><uri>model://advanced_plane</uri><name>plane</name></include></model>''')
    m = load_model(f, resource_paths=[MODELS_DIR])
    assert "plane::servo_0" in m.joints
    wb.Aircraft(m, 1, warn=False)


def test_sdf_1_4_axis_is_in_the_model_frame(tmp_path):
    # libsdformat's 1.4->1.5 converter adds <use_parent_model_frame>true</use_parent_model_frame>
    f = write(tmp_path, f'''<model name="m">{L("a")}{L("b", "<pose>0 0 0 0 0 1.5707963267948966</pose>")}
      <joint name="j" type="revolute"><parent>a</parent><child>b</child><axis><xyz>1 0 0</xyz></axis></joint>
      </model>''', version="1.4")
    np.testing.assert_allclose(load_model(f).joints["j"].axis, [1, 0, 0], atol=1e-12)


def test_zero_axis_is_refused(tmp_path):
    f = write(tmp_path, f'''<model name="m">{L("a")}{L("b")}
      <joint name="j" type="revolute"><parent>a</parent><child>b</child><axis><xyz>0 0 0</xyz></axis></joint>
      </model>''')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            load_model(f)


def test_duplicate_link_from_merged_include_is_refused(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "model.sdf").write_text(
        '<sdf version="1.9"><model name="sub"><link name="base"><inertial><mass>5</mass></inertial></link>'
        '</model></sdf>')
    f = write(tmp_path, f'<model name="m">{L("base")}<include merge="true"><uri>model://sub</uri></include></model>')
    with pytest.raises(ValueError):
        load_model(f, resource_paths=[tmp_path])
