"""Independent scalar transcription of AdvancedLiftDrag.cc::Update (gz-sim8 f380a94e), world frame,
compared against wingbatch.systems.advanced_lift_drag.AdvancedLiftDrag.wrench at edge states."""
import math, itertools
import numpy as np
from wingbatch.systems.advanced_lift_drag import AdvancedLiftDrag, AdvancedLiftDragParams, ControlSurface

def norm(v): return math.sqrt(sum(c*c for c in v))
def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def cross(a,b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def sc(s,a): return [s*x for x in a]
def add(*vs): return [sum(c) for c in zip(*vs)]
def correct(v): return [c if math.isfinite(c) else 0.0 for c in v]
def matv(R,v): return [sum(R[i][j]*v[j] for j in range(3)) for i in range(3)]

def gz_update(P, R, vlin_w, wang_w, wang_link, jpos, wind=None, overwrite=False):
    """C++ Update; R: link->world matrix. Returns (force_w, totalTorque_w) or None."""
    fwd = list(P.forward); n = norm(fwd); fwd = sc(1/n, fwd)
    up = list(P.upward); n = norm(up); up = sc(1/n, up) if n != 0 else up
    cpW = matv(R, P.cp)
    air = add(vlin_w, cross(wang_w, cpW))
    if wind is not None: air = add(air, sc(-1, wind))
    bx = matv(R, fwd); bz = sc(-1, matv(R, up)); by = cross(bz, bx)
    vin = add(air, sc(-dot(air, by), by)); s = norm(vin)
    if s <= 1e-9: return None
    sx = sc(1/s, vin); sy = by; sz = cross(sx, sy)
    span = math.sqrt(P.area*P.AR)
    mac = P.mac if abs(P.mac) > 1e-9 else P.area/span
    rr, pr, yr = wang_link[0], -wang_link[1], -wang_link[2]
    a = math.atan2(dot(sx,bz), dot(sx,bx)); sa, ca = math.sin(a), math.cos(a)
    b = math.atan2(dot(air,by), dot(air,bx))
    q = 0.5*P.air_density*s*s; h = 0.5*P.air_density*s
    M, As = 15.0, P.alpha_stall
    e1, e2 = math.exp(-M*(a-As)), math.exp(M*(a+As))
    sig = (1+e1+e2)/((1+e1)*(1+e2))
    sgn = a/abs(a) if a != 0 else float('nan')
    CL = (1-sig)*(P.CL0+P.CLa*a) + sig*(2*sgn*sa*sa*ca) + P.CLb*b
    tot = dict(CL=0., CD=0., CY=0., Cell=0., Cem=0., Cen=0.)
    for cs in P.control_surfaces:
        ang = jpos.get(cs.name, 0.0)*180/math.pi
        for k in tot: tot[k] += ang*getattr(cs, k+'_ctrl')*cs.direction
    CL += tot['CL']
    R3 = lambda cp_, cq_, cr_: cp_*(rr*span/2)*h + cq_*(pr*mac/2)*h + cr_*(yr*span/2)*h
    lift = sc((CL*q + R3(P.CLp,P.CLq,P.CLr))*P.area, sc(-1, sz))
    CDfp = 2/(1+math.exp(-0.224 + -0.115*max(P.AR, 1/P.AR)))
    CD = (1-sig)*(P.CD0 + CL*CL/(math.pi*P.AR*P.eff)) + sig*abs(CDfp*(0.5-0.5*math.cos(2*a))) + tot['CD']
    drag = sc((CD*q + R3(P.CDp,P.CDq,P.CDr))*P.area, sc(-1, sx))
    CY = P.CYa*a + P.CYb*b + tot['CY']
    side = sc((CY*q + R3(P.CYp,P.CYq,P.CYr))*P.area, sy)
    if a > As: Cem = P.Cem0 + (P.Cema*As + P.Cema_stall*(a-As))
    elif a < -As: Cem = P.Cem0 + (-P.Cema*As + P.Cema_stall*(a+As))
    else: Cem = P.Cem0 + P.Cema*a
    Cem = P.Cemb*b if overwrite else Cem + P.Cemb*b
    Cem += tot['Cem']
    pm = sc((Cem*q + R3(P.Cemp,P.Cemq,P.Cemr))*P.area*mac, by)
    Cell = P.Cella*a + P.Cellb*b + tot['Cell']
    rm = sc((Cell*q + R3(P.Cellp,P.Cellq,P.Cellr))*P.area*span, bx)
    Cen = P.Cena*a + P.Cenb*b + tot['Cen']
    ym = sc((Cen*q + R3(P.Cenp,P.Cenq,P.Cenr))*P.area*span, bz)
    F = correct(add(lift, drag, side)); T = correct(add(pm, rm, ym))
    return F, add(T, cross(cpW, F)), a, b

def rotm(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w,x,y,z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def full_params():
    P = AdvancedLiftDragParams(link_name="l")
    rng = np.random.default_rng(0)
    from wingbatch.systems.advanced_lift_drag import COEFFICIENTS
    for c in COEFFICIENTS: setattr(P, c, float(rng.uniform(-1, 1)))
    P.alpha_stall = 0.3; P.AR = 6.5; P.eff = 0.9; P.area = 0.4; P.mac = 0.0
    P.cp = (-0.1, 0.03, 0.02); P.forward = (2.0, 0.2, 0.0); P.upward = (0.0, 0.3, 1.5); P.air_density = 1.1
    P.control_surfaces = [ControlSurface("j0", -1, *rng.uniform(-0.01, 0.01, 6)),
                          ControlSurface("j1", 1, *rng.uniform(-0.01, 0.01, 6))]
    return P

IDENT = False
def check(P, overwrite=False):
    rng = np.random.default_rng(5)
    sysm = AdvancedLiftDrag(P, [0, 1], gz_cem_overwrite=overwrite)
    # link-frame air velocities at the link origin (wind folded in)
    vs = []
    for ang in [0.0, 1e-12, -1e-12, 0.3, 0.3000001, -0.3, 1.0, -1.0, math.pi/2, -math.pi/2,
                math.pi - 1e-9, -math.pi + 1e-9, math.pi, 3.0, -3.0]:
        vs.append([10*math.cos(ang), 0.0, -10*math.sin(ang)])
    for bdeg in [89.9, -89.9, 90.0, 120.0, -170.0]:
        b = math.radians(bdeg); vs.append([10*math.cos(b), 10*math.sin(b), 1.0])
    vs.append([-12.0, 0.0, 0.0]); vs.append([-12.0, 0.0, -0.0]); vs.append([1e-10, 3.0, 0.0])
    worst = 0.0; nan_cases = 0
    for v in vs:
        for w in ([0.0, 0.0, 0.0], list(rng.uniform(-3, 3, 3))):
            R = np.eye(3) if IDENT else rotm(rng)
            jp = {"j0": 0.2, "j1": -0.4}
            vl = np.array(v); wl = np.array(w)
            F, T, info = sysm.wrench(tuple(vl[:, None]), tuple(wl[:, None]), np.array([[0.2, -0.4]]))
            F = np.array([f[0] for f in F]); T = np.array([t[0] for t in T])
            # world-frame reference: link-origin world velocity R v, world ang vel R w
            ref = gz_update(P, R.tolist(), list(R @ vl), list(R @ wl), list(wl), jp, None, overwrite)
            if ref is None:
                assert not F.any() and not T.any(), (v, w)
                continue
            Fr = R.T @ np.array(ref[0]); Tr = R.T @ np.array(ref[1])
            err = max(np.max(np.abs(F - Fr)), np.max(np.abs(T - Tr))) / (1 + np.max(np.abs(Fr)) + np.max(np.abs(Tr)))
            if not np.all(Fr == 0) and err > 1e-10 or (np.all(Fr == 0) != np.all(F == 0)):
                print("MISMATCH", v, w, F, Fr, T, Tr, info["alpha"], ref[2])
            worst = max(worst, err)
            nan_cases += np.all(Fr == 0)
    print("worst rel err", worst, "force-zeroed cases", nan_cases, "overwrite", overwrite)

if __name__ == "__main__":
    P = full_params()
    check(P); check(P, True)
    IDENT = True
    print("identity attitude:"); check(P); check(P, True)
