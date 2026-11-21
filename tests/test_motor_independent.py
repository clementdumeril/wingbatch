"""Independent, literal gz-sim MulticopterMotorModel::UpdateForcesAndMoments + Link::AddWorldForce emulation in the
world frame, computing every link / joint pose from the raw SDF text (own rpy parser, own frame composition), and
comparing with wingbatch's MulticopterMotorModel.wrench. Run from D:\\Work\\wingbatch with PYTHONPATH=src;tests.

Variants: the bundled advanced_plane, and a perturbed copy (canonical link not at the model origin and rotated,
rotor CoM offset, rotor joint with a non-trivial pose and an axis in the joint frame, parent link = canonical).
"""
import math
import re
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import wingbatch
from wingbatch import Aircraft
from wingbatch.sdf import MODELS_DIR


def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def pose(text):
    v = [float(s) for s in (text or "0 0 0 0 0 0").split()]
    T = np.eye(4)
    T[:3, :3] = rpy(*v[3:])
    T[:3, 3] = v[:3]
    return T


def axis_angle(a, phi):
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(phi) * K + (1 - math.cos(phi)) * K @ K
    return T


def gz_motor(sdf_text, Xw_model_canon_world, v_canon_world, w_world, wind, w_filtered, phi):
    """Literal C++ port. Returns world force, torque about the canonical link origin, from the rotor plugin."""
    root = ET.fromstring(sdf_text)
    m = root.find("model")
    links = {l.get("name"): l for l in m.findall("link")}
    joints = {j.get("name"): j for j in m.findall("joint")}
    pl = [p for p in m.findall("plugin") if "multicopter" in p.get("filename", "")][0]
    P = {c.tag: c.text.strip() for c in pl}
    canon = m.find("link").get("name")
    X_M = {n: pose(l.findtext("pose")) for n, l in links.items()}  # link in model
    j = joints[P["jointName"]]
    child, parent = j.findtext("child").strip(), j.findtext("parent").strip()
    X_child_joint = pose(j.findtext("pose"))  # SDF 1.5: joint pose relative to child
    xyz = np.array([float(s) for s in j.find("axis").findtext("xyz").split()])
    upmf = (j.find("axis").findtext("use_parent_model_frame") or "0").strip() in ("1", "true")
    X_M_joint = X_M[child] @ X_child_joint
    axis_jf = X_M_joint[:3, :3].T @ xyz if upmf else xyz  # SdfEntityCreator ResolveXyz -> joint frame
    # world poses: the canonical link world pose is the state
    Xw_canon = Xw_model_canon_world
    Xw_model = Xw_canon @ np.linalg.inv(X_M[canon])
    Xw_parent = Xw_model @ X_M[parent]
    # child = parent * X_parent_joint * spin(phi about joint axis, joint frame) * X_joint_child
    X_parent_joint = np.linalg.inv(X_M[parent]) @ X_M_joint
    Xw_child = Xw_parent @ X_parent_joint @ axis_angle(axis_jf, phi) @ np.linalg.inv(X_child_joint)
    # link origin world velocity (rigid; the spin moves the origin only if it is off the axis: neglect)
    r = Xw_child[:3, 3] - Xw_canon[:3, 3]
    v_link = v_canon_world + np.cross(w_world, r)
    td = -1 if P.get("turningDirection", "cw") == "cw" else 1
    slow = float(P.get("rotorVelocitySlowdownSim", 10))
    kT, kM = float(P["motorConstant"]), float(P["momentConstant"])
    kD, kR = float(P["rotorDragCoefficient"]), float(P["rollingMomentCoefficient"])
    jointvel = td * w_filtered / slow
    real = jointvel * slow
    sgn = (real > 0) - (real < 0)
    thrust = td * sgn * real * real * kT
    Rw = Xw_child[:3, :3]
    f1 = Rw @ np.array([0, 0, thrust])
    jointWorld = Xw_child @ X_child_joint
    jointAxis = jointWorld[:3, :3] @ axis_jf
    rel = v_link - wind
    vperp = rel - rel.dot(jointAxis) * jointAxis
    drag = -abs(real) * kD * vperp
    ipose = pose(links[child].find("inertial").findtext("pose"))
    comw = Rw @ ipose[:3, 3]
    # AddWorldForce x2: wrench at the link origin
    F = f1 + drag
    T_origin = np.cross(comw, f1) + np.cross(comw, drag)
    poseDiff = np.linalg.inv(Xw_parent) @ Xw_child
    dragTorque = np.array([0, 0, -td * thrust * kM])
    Tparent = Xw_parent[:3, :3] @ (poseDiff[:3, :3] @ dragTorque) - abs(real) * kR * vperp
    # move to the canonical link origin (free torques + r x F for the force at the rotor origin)
    T = T_origin + Tparent + np.cross(r, F)
    return F, T, thrust


def wb_motor(ac, Xw_canon, v_world, w_world, wind, wfilt):
    R = Xw_canon[:3, :3]
    vb = R.T @ (v_world - wind)
    wb = R.T @ w_world
    mot = ac.motors[0]
    F, T, thr = mot.wrench(np.array([wfilt]), tuple(np.array([c]) for c in vb), tuple(np.array([c]) for c in wb))
    return R @ np.array([f[0] for f in F]), R @ np.array([t[0] for t in T]), thr[0]


def perturbed_sdf(text):
    text = text.replace("<link name='base_link'>\n      <pose>0 0 0 0 0 0</pose>",
                        "<link name='base_link'>\n      <pose>0.1 -0.05 0.02 0.1 -0.2 0.3</pose>")
    assert "0.1 -0.05 0.02 0.1 -0.2 0.3" in text
    # rotor: CoM offset, joint pose offset along the spin axis, axis in the joint frame (link z)
    text = re.sub(r"(<link name='rotor_puller'>\s*<pose>[^<]*</pose>\s*<inertial>\s*<pose>)0 0 0 0 0 0",
                  r"\g<1>0.01 0.02 0.03 0 0 0", text)
    text = text.replace("<child>rotor_puller</child>", "<child>rotor_puller</child>\n      <pose>0 0 0.02 0 0 0</pose>")
    text = text.replace("""<xyz>1 0 0</xyz>
        <limit>
          <lower>-1e+16</lower>""", """<xyz>0 0 1</xyz>
        <limit>
          <lower>-1e+16</lower>""")
    text = text.replace("<use_parent_model_frame>1</use_parent_model_frame>", "<use_parent_model_frame>0</use_parent_model_frame>")
    return text


def run(label, text, tmp):
    d = Path(tmp) / label
    d.mkdir()
    (d / "model.sdf").write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ac = Aircraft(str(d / "model.sdf"), 1, resource_paths=(str(MODELS_DIR),), warn=False)
    rng = np.random.default_rng(1)
    worst0 = worst_phi = 0.0
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w_, x_, y_, z_ = q
        R = np.array([[1 - 2 * (y_ * y_ + z_ * z_), 2 * (x_ * y_ - w_ * z_), 2 * (x_ * z_ + w_ * y_)],
                      [2 * (x_ * y_ + w_ * z_), 1 - 2 * (x_ * x_ + z_ * z_), 2 * (y_ * z_ - w_ * x_)],
                      [2 * (x_ * z_ - w_ * y_), 2 * (y_ * z_ + w_ * x_), 1 - 2 * (x_ * x_ + y_ * y_)]])
        X = np.eye(4)
        X[:3, :3] = R
        X[:3, 3] = rng.normal(size=3)
        v, om, wind = rng.normal(size=3) * 10, rng.normal(size=3), rng.normal(size=3) * 3
        wf = rng.uniform(-200, 1600)
        Fg, Tg, thg = gz_motor(text, X, v, om, wind, wf, 0.0)
        Fw, Tw, thw = wb_motor(ac, X, v, om, wind, wf)
        e = max(np.abs(Fg - Fw).max(), np.abs(Tg - Tw).max(), abs(thg - thw)) / max(1.0, np.abs(Fg).max())
        worst0 = max(worst0, e)
        Fg2, Tg2, _ = gz_motor(text, X, v, om, wind, wf, rng.uniform(0, 2 * math.pi))
        worst_phi = max(worst_phi, max(np.abs(Fg2 - Fw).max(), np.abs(Tg2 - Tw).max()) / max(1.0, np.abs(Fg2).max()))
    print(f"{label}: worst rel diff, rotor angle 0: {worst0:.3e}; random rotor angle: {worst_phi:.3e}")
    return worst0


@pytest.mark.parametrize("variant", ["advanced_plane", "perturbed"])
def test_motor_matches_independent_world_frame_emulation(variant, tmp_path):
    base = (MODELS_DIR / "advanced_plane" / "model.sdf").read_text()
    text = base if variant == "advanced_plane" else perturbed_sdf(base)
    assert run(variant, text, tmp_path) < 1e-9
