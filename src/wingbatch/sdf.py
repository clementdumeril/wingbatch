"""Minimal SDFormat reader: the parts of a ``model.sdf`` that the fixed-wing physics needs.

Read: links (pose, ``<inertial>``, ``<gravity>``), joints (type, parent, child, pose, axis, limits, damping),
``<frame>`` elements, ``<include>`` of other models (resolved through ``model://`` search paths), and the
``<plugin>`` elements, kept as raw XML for the systems to parse with gz-sim's own parameter names.

Frames follow SDFormat: a link pose is relative to the model frame unless ``relative_to`` says otherwise; a joint
pose is relative to its child link; an inertial pose is relative to its link; an axis is expressed in the joint
frame, or in the model frame with ``<use_parent_model_frame>`` (SDF 1.5) or ``expressed_in="__model__"`` (1.7+).

Not read: visuals, collisions, sensors, materials (no effect on the dynamics simulated here). Anything else that
could matter (unsupported plugins, unresolved includes, links without ``<inertial>``) is listed in
:attr:`Model.warnings`.
"""

from __future__ import annotations

import math
import os
import warnings as _warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._vec import euler_to_quat

MODELS_DIR = Path(__file__).resolve().parent / "models"


# ---------------------------------------------------------------------------------------------- poses
@dataclass(frozen=True)
class Pose:
    """Rigid transform ``X_A_B``: ``R`` (3, 3) and ``p`` (3,) such that ``x_A = R @ x_B + p``."""

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    p: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __matmul__(self, other: "Pose") -> "Pose":
        return Pose(self.R @ other.R, self.R @ other.p + self.p)

    def inverse(self) -> "Pose":
        return Pose(self.R.T, -self.R.T @ self.p)

    def apply(self, x) -> np.ndarray:
        return self.R @ np.asarray(x, float) + self.p


def quat_matrix(w, x, y, z) -> np.ndarray:
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def parse_pose(el: ET.Element | None) -> tuple[Pose, str | None]:
    """``<pose>`` text and its ``relative_to`` attribute (None when absent)."""
    if el is None or el.text is None or not el.text.strip():
        return Pose(), (None if el is None else el.get("relative_to"))
    v = [float(s) for s in el.text.split()]
    fmt = el.get("rotation_format", "euler_rpy")
    if fmt == "quat_xyzw":
        if len(v) != 7:
            raise ValueError(f"<pose rotation_format='quat_xyzw'> needs 7 numbers, got {el.text!r}")
        R = quat_matrix(v[6], v[3], v[4], v[5])
    else:
        if len(v) != 6:
            raise ValueError(f"<pose> needs 6 numbers, got {el.text!r}")
        rpy = v[3:]
        if el.get("degrees", "false").strip().lower() in ("true", "1"):
            rpy = [math.radians(a) for a in rpy]
        R = quat_matrix(*euler_to_quat(*rpy))
    return Pose(R, np.array(v[:3])), el.get("relative_to")


def _vec(el: ET.Element | None, default) -> np.ndarray:
    if el is None or el.text is None:
        return np.array(default, float)
    return np.array([float(s) for s in el.text.split()])


def _float(el: ET.Element | None, default: float) -> float:
    return default if el is None or el.text is None else float(el.text)


def _bool(el: ET.Element | None, default: bool) -> bool:
    return default if el is None or el.text is None else el.text.strip().lower() in ("1", "true")


# ---------------------------------------------------------------------------------------------- elements
@dataclass
class Inertial:
    mass: float
    pose: Pose  # inertial frame in the link frame (origin = centre of mass)
    inertia: np.ndarray  # (3, 3) about the centre of mass, in the inertial frame


@dataclass
class Link:
    name: str
    pose: Pose  # link frame in the model frame
    inertial: Inertial
    gravity: bool = True


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    pose: Pose  # joint frame in the model frame
    axis: np.ndarray | None  # unit axis in the model frame
    lower: float = -1e16
    upper: float = 1e16
    damping: float = 0.0
    friction: float = 0.0


@dataclass
class Plugin:
    name: str
    filename: str
    element: ET.Element
    kind: str | None  # "AdvancedLiftDrag", "MulticopterMotorModel", "JointPositionController" or None
    scope: str = ""  # "outer::inner::" for a plugin of a nested or included model: its names are relative to it

    def scoped(self, name: str) -> str:
        """A link or joint name of this plugin, as a key of :attr:`Model.links` / :attr:`Model.joints`."""
        return self.scope + name


KNOWN_PLUGINS = {
    "AdvancedLiftDrag": ("advancedliftdrag", "advanced-lift-drag"),
    "MulticopterMotorModel": ("multicoptermotormodel", "multicopter-motor-model"),
    "JointPositionController": ("jointpositioncontroller", "joint-position-controller"),
}


def _plugin_kind(name: str, filename: str) -> str | None:
    key = (name + " " + filename).lower()
    for kind, tags in KNOWN_PLUGINS.items():
        if any(t in key for t in tags):
            return kind
    return None


@dataclass
class Model:
    name: str
    links: dict[str, Link]
    joints: dict[str, Joint]
    plugins: list[Plugin]
    canonical_link: str
    frames: dict[str, Pose] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    source: Path | None = None

    def link_pose(self, name: str) -> Pose:
        """``X_C_L``: link ``name`` in the canonical link frame."""
        return self.links[self.canonical_link].pose.inverse() @ self.links[name].pose

    def plugins_of(self, kind: str) -> list[Plugin]:
        return [p for p in self.plugins if p.kind == kind]


# ---------------------------------------------------------------------------------------------- reading
def resolve_uri(uri: str, base: Path | None, resource_paths=()) -> Path | None:
    """``model://NAME`` (or a path) to the directory or file it names, as gz-sim's resource search."""
    uri = uri.strip()
    if uri.startswith("model://"):
        rel = uri[len("model://"):]
        env = [p for p in os.environ.get("GZ_SIM_RESOURCE_PATH", "").split(os.pathsep) if p]
        for root in [*map(Path, resource_paths), *map(Path, env), MODELS_DIR]:
            cand = root / rel
            if cand.exists():
                return cand
        return None
    if uri.startswith(("http://", "https://")):
        return None
    cand = Path(uri) if Path(uri).is_absolute() or base is None else base / uri
    return cand if cand.exists() else None


def _model_file(path: Path) -> Path:
    if path.is_dir():
        config = path / "model.config"
        if config.exists():
            sdf = ET.parse(config).getroot().find("sdf")
            if sdf is not None and sdf.text:
                return path / sdf.text.strip()
        return path / "model.sdf"
    return path


def _version(root: ET.Element) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in root.get("version", "1.9").split("."))
    except ValueError:
        return (1, 9)


@dataclass
class _Content:
    """What a ``<model>`` contributes, names relative to its own scope, poses in its model frame."""

    links: dict[str, Link]
    joints: dict[str, Joint]
    frames: dict[str, Pose]  # <frame>s and the frames of nested models
    plugins: list[Plugin]
    order: list[str]  # link names in document order (merged includes in place)


class _Reader:
    def __init__(self, resource_paths=()):
        self.resource_paths = tuple(resource_paths)
        self.warnings: list[str] = []

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def read_model(self, el: ET.Element, base: Path | None, version: tuple[int, ...]) -> _Content:
        raw_links: dict[str, tuple[ET.Element, Pose, str | None]] = {}
        raw_frames: dict[str, tuple[Pose, str | None]] = {}
        raw_joints: dict[str, ET.Element] = {}
        plugins: list[Plugin] = []
        subs = []  # (prefix, content, pose, relative_to, model-frame name or None)
        order: list[tuple[str, object]] = []  # ("link", name) or ("sub", index)

        for child in el:
            tag = child.tag
            if tag == "link":
                name = child.get("name")
                if name in raw_links:
                    raise ValueError(f"duplicate link name {name!r}")
                pose, rel = parse_pose(child.find("pose"))
                raw_links[name] = (child, pose, rel)
                order.append(("link", name))
            elif tag == "frame":
                pose, rel = parse_pose(child.find("pose"))
                raw_frames[child.get("name")] = (pose, rel or child.get("attached_to"))
            elif tag == "joint":
                if child.get("name") in raw_joints:
                    raise ValueError(f"duplicate joint name {child.get('name')!r}")
                raw_joints[child.get("name")] = child
            elif tag == "plugin":
                name, filename = child.get("name", ""), child.get("filename", "")
                plugins.append(Plugin(name, filename, child, _plugin_kind(name, filename)))
            elif tag == "model":  # nested model
                pose, rel = parse_pose(child.find("pose"))
                content = self.read_model(child, base, version)
                subs.append((child.get("name") + "::", content, pose, rel, child.get("name")))
                order.append(("sub", len(subs) - 1))
            elif tag == "include":
                uri = child.findtext("uri", "").strip()
                path = resolve_uri(uri, base, self.resource_paths)
                if path is None:
                    self.warn(f"<include> {uri!r} not found: its links (mass) and plugins are ignored. Add its "
                              "directory to resource_paths or GZ_SIM_RESOURCE_PATH.")
                    continue
                file = _model_file(path)
                root = ET.parse(file).getroot()
                sub = root.find("model")
                content = self.read_model(sub, file.parent, _version(root))
                if child.find("pose") is not None:
                    pose, rel = parse_pose(child.find("pose"))
                else:  # the included model's own pose
                    pose, rel = parse_pose(sub.find("pose"))
                merge = child.get("merge", "false").strip().lower() == "true"
                name = (child.findtext("name") or sub.get("name")).strip()
                subs.append(("" if merge else name + "::", content, pose, rel, None if merge else name))
                order.append(("sub", len(subs) - 1))

        # names visible in this scope: local ones, and those of the sub-models (prefixed unless merged)
        sub_of: dict[str, tuple[int, str | None]] = {}
        for i, (pre, content, _, _, mname) in enumerate(subs):
            for table in (content.links, content.frames):
                for n in table:
                    if pre + n in sub_of or (not pre and (n in raw_links or n in raw_frames)):
                        raise ValueError(f"duplicate name {pre + n!r}")
                    sub_of[pre + n] = (i, n)
            if mname is not None:
                sub_of[mname] = (i, None)

        resolved: dict[str, Pose] = {"__model__": Pose()}
        sub_pose: dict[int, Pose] = {}

        def sub_frame(i: int, stack) -> Pose:
            if i not in sub_pose:
                _, _, pose, rel, _ = subs[i]
                sub_pose[i] = frame(rel, stack) @ pose if rel else pose
            return sub_pose[i]

        def frame(name: str, stack=()) -> Pose:
            if name in resolved:
                return resolved[name]
            if name in stack:
                raise ValueError(f"cycle in the pose graph at {name!r}")
            stack = stack + (name,)
            if name in raw_links:
                _, pose, rel = raw_links[name]
            elif name in raw_frames:
                pose, rel = raw_frames[name]
            elif name in raw_joints:
                j = raw_joints[name]
                pose, rel = parse_pose(j.find("pose"))
                rel = rel or j.findtext("child", "").strip()
            elif name in sub_of:
                i, inner = sub_of[name]
                X = sub_frame(i, stack)
                content = subs[i][1]
                if inner is not None:
                    X = X @ (content.links[inner].pose if inner in content.links else content.frames[inner])
                resolved[name] = X
                return X
            else:
                raise KeyError(f"frame {name!r} not found")
            X = frame(rel, stack) @ pose if rel else pose
            resolved[name] = X
            return X

        links: dict[str, Link] = {}
        joints: dict[str, Joint] = {}
        frames: dict[str, Pose] = {}
        for name, (lel, _, _) in raw_links.items():
            links[name] = self._link(name, lel, frame(name))
        for name in raw_frames:
            frames[name] = frame(name)
        for name, jel in raw_joints.items():
            joints[name] = self._joint(name, jel, frame, version)
        out_order: list[str] = []
        for kind, item in order:
            if kind == "link":
                out_order.append(item)
                continue
            pre, content, _, _, mname = subs[item]
            X = sub_frame(item, ())
            for n, link in content.links.items():
                links[pre + n] = Link(pre + n, X @ link.pose, link.inertial, link.gravity)
            for n, f in content.frames.items():
                frames[pre + n] = X @ f
            if mname is not None:
                frames[mname] = X
            for n, j in content.joints.items():
                if pre + n in joints:
                    raise ValueError(f"duplicate joint name {pre + n!r}")
                parent = j.parent if j.parent == "world" else pre + j.parent
                joints[pre + n] = Joint(pre + n, j.type, parent, pre + j.child, X @ j.pose,
                                        None if j.axis is None else X.R @ j.axis, j.lower, j.upper, j.damping,
                                        j.friction)
            for p in content.plugins:
                plugins.append(Plugin(p.name, p.filename, p.element, p.kind, pre + p.scope))
            out_order.extend(pre + n for n in content.order)
        return _Content(links, joints, frames, plugins, out_order)

    def _link(self, name: str, el: ET.Element, X: Pose) -> Link:
        iel = el.find("inertial")
        if iel is None:
            self.warn(f"link {name!r} has no <inertial>: SDFormat defaults used (mass 1 kg, inertia identity)")
            inertial = Inertial(1.0, Pose(), np.eye(3))
        else:
            ipose, irel = parse_pose(iel.find("pose"))
            if irel:
                self.warn(f"link {name!r}: inertial pose relative_to={irel!r} ignored (taken in the link frame)")
            I = iel.find("inertia")
            g = (lambda k, d=0.0: _float(None if I is None else I.find(k), d))
            inertia = np.array([[g("ixx", 1.0), g("ixy"), g("ixz")],
                                [g("ixy"), g("iyy", 1.0), g("iyz")],
                                [g("ixz"), g("iyz"), g("izz", 1.0)]])
            inertial = Inertial(_float(iel.find("mass"), 1.0), ipose, inertia)
        return Link(name, X, inertial, _bool(el.find("gravity"), True))

    def _joint(self, name, el, frame, version) -> Joint:
        jtype = el.get("type")
        parent = el.findtext("parent", "").strip()
        child = el.findtext("child", "").strip()
        X = frame(name)
        axis_el = el.find("axis")
        axis = lower = upper = None
        damping = friction = 0.0
        if axis_el is not None:
            xyz_el = axis_el.find("xyz")
            xyz = _vec(xyz_el, [0, 0, 1])
            norm = float(np.linalg.norm(xyz))
            if not norm > 0:
                raise ValueError(f"joint {name!r}: zero <axis><xyz>")
            xyz = xyz / norm
            expressed = None if xyz_el is None else xyz_el.get("expressed_in")
            # SDF <= 1.4 axes are in the model frame (libsdformat's 1.4 -> 1.5 conversion adds
            # use_parent_model_frame = true)
            if _bool(axis_el.find("use_parent_model_frame"), version < (1, 5)):
                expressed = "__model__"
            axis = (frame(expressed).R if expressed else X.R) @ xyz
            lim = axis_el.find("limit")
            if lim is not None:
                lower = _float(lim.find("lower"), None)
                upper = _float(lim.find("upper"), None)
            dyn = axis_el.find("dynamics")
            if dyn is not None:
                damping = _float(dyn.find("damping"), 0.0)
                friction = _float(dyn.find("friction"), 0.0)
        return Joint(name, jtype, parent, child, X, axis,
                     -1e16 if lower is None else lower, 1e16 if upper is None else upper, damping, friction)


def load_model(source: str | os.PathLike, resource_paths=()) -> Model:
    """Read a model: a ``model.sdf`` path, a model directory, or the name of a bundled model
    (``"advanced_plane"``)."""
    src = str(source)
    path = Path(src)
    if not path.exists() and (MODELS_DIR / src).exists():
        path = MODELS_DIR / src
    if not path.exists():
        raise FileNotFoundError(f"no model at {src!r} (bundled models: "
                                f"{sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir())})")
    file = _model_file(path)
    root = ET.parse(file).getroot()
    mel = root.find("model") if root.tag == "sdf" else root
    if mel is None or mel.tag != "model":
        raise ValueError(f"{file}: no <model> element (world files are read with load_world)")
    reader = _Reader((*resource_paths, file.parent.parent))
    c = reader.read_model(mel, file.parent, _version(root))
    if not c.links:
        raise ValueError(f"{file}: the model has no link")
    canonical = mel.get("canonical_link") or next((n for n in c.order if "::" not in n), c.order[0])
    for p in c.plugins:
        if p.kind is None:
            reader.warn(f"plugin {p.name or p.filename!r} is not simulated")
    for j in c.joints.values():
        if j.type not in ("fixed", "revolute", "continuous"):
            reader.warn(f"joint {j.name!r} of type {j.type!r} is treated as rigid")
    return Model(mel.get("name", file.stem), c.links, c.joints, c.plugins, canonical, c.frames, reader.warnings, file)


@dataclass
class World:
    """The world properties the aircraft feels: gravity, uniform wind, physics step."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.8)
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_step_size: float = 0.004


def load_world(path: str | os.PathLike) -> World:
    """Gravity, ``<wind><linear_velocity>`` and the physics ``<max_step_size>`` of a world file."""
    root = ET.parse(path).getroot()
    w = root.find("world")
    if w is None:
        raise ValueError(f"{path}: no <world>")
    world = World()
    g = w.find("gravity")
    if g is not None:
        world.gravity = tuple(_vec(g, world.gravity))
    wind = w.find("wind/linear_velocity")
    if wind is not None:
        world.wind = tuple(_vec(wind, world.wind))
    step = w.find("physics/max_step_size")
    if step is not None:
        world.max_step_size = float(step.text)
    return world


def warn_model(model: Model) -> None:
    for msg in model.warnings:
        _warnings.warn(f"{model.name}: {msg}", stacklevel=3)
