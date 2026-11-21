"""AdvancedLiftDrag against a second, independent world-frame transcription of the C++ (written separately from
src/wingbatch/reference.py), at edge states: alpha near 0, +-stall, +-pi/2, +-pi; beta near +-90 deg; backwards."""
import math
import numpy as np
import pytest
from independent_ald import gz_update, full_params, rotm
from wingbatch.systems.advanced_lift_drag import AdvancedLiftDrag, AdvancedLiftDragParams, ControlSurface

ANGLES = [1e-12, -1e-12, 0.3, 0.3000001, -0.3, 1.0, -1.0, math.pi / 2, -math.pi / 2, math.pi - 1e-9,
          -math.pi + 1e-9, 3.0, -3.0]


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("attitude", ["identity", "random"])
def test_port_matches_independent_transcription(overwrite, attitude):
    P = full_params()
    sysm = AdvancedLiftDrag(P, [0, 1], gz_cem_overwrite=overwrite)
    rng = np.random.default_rng(3)
    vs = [[10 * math.cos(a), 0.0, -10 * math.sin(a)] for a in ANGLES]
    vs += [[10 * math.cos(math.radians(b)), 10 * math.sin(math.radians(b)), 1.0] for b in (89.9, -89.9, 90.0, 120.0, -170.0)]
    for v in vs:
        for w in ([0.0, 0.0, 0.0], list(rng.uniform(-3, 3, 3))):
            R = np.eye(3) if attitude == "identity" else rotm(rng)
            vl, wl = np.array(v), np.array(w)
            F, T, _ = sysm.wrench(tuple(vl[:, None]), tuple(wl[:, None]), np.array([[0.2, -0.4]]))
            Fr, Tr, *_ = gz_update(P, R.tolist(), list(R @ vl), list(R @ wl), list(wl), {"j0": 0.2, "j1": -0.4},
                                   None, overwrite)
            np.testing.assert_allclose(np.ravel(F), R.T @ np.array(Fr), rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(np.ravel(T), R.T @ np.array(Tr), rtol=1e-10, atol=1e-10)


def test_missing_AR_or_eff_is_refused_with_a_clear_message():
    """gz-sim's defaults AR = eff = 0 give a non-finite force that Correct() drops: refused here instead."""
    with pytest.raises(ValueError, match="AR"):
        AdvancedLiftDrag(AdvancedLiftDragParams(link_name="l", AR=6.0))  # eff left at its gz default 0


def test_control_surface_without_any_servo_column():
    P = AdvancedLiftDragParams(link_name="l", AR=6, eff=0.9, CLa=5,
                               control_surfaces=[ControlSurface("j", 1, 0, 0, 0.01, 0, 0, 0)])
    v = (np.array([10.0]), np.array([0.0]), np.array([-1.0]))
    AdvancedLiftDrag(P, [-1]).wrench(v, (np.zeros(1),) * 3, np.zeros((1, 0)))
