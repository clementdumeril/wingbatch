"""Throughput of wingbatch: aircraft-steps per second against the batch size N, one CPU core.

    python benchmarks/throughput.py [--quick] [--json out.json]

Each aircraft is the bundled advanced_plane, trimmed at 12 m/s with a small perturbation, flying with fixed
commands. "rk4" is one Runge-Kutta 4 step (four evaluations of all the systems); "semi-implicit" one Gazebo-like
step (one evaluation). The baseline "scalar" is wingbatch.reference.ScalarAircraft: the same physics, one aircraft
at a time in plain Python, RK4. BLAS threads are pinned to 1 so that every number is single-core. Every number is
the best of ``--repeat`` trials (default 5; the median is kept in the JSON): on a shared or thermally throttled
machine the best trial is the closest to the code's own cost.
"""

from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import wingbatch as wb  # noqa: E402
from wingbatch.reference import ScalarAircraft  # noqa: E402


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def setup(n, integrator):
    ac = wb.Aircraft("advanced_plane", n, integrator=integrator, warn=False)
    one = wb.Aircraft("advanced_plane", 1, warn=False)
    r = wb.trim(one, 12.0)
    X = np.repeat(r["state"], n, axis=0)
    rng = np.random.default_rng(0)
    X[:, 10:13] += rng.uniform(-0.1, 0.1, (n, 3))
    ac.reset(X)
    servo = np.repeat(r["servo_cmd"], n, 0) + rng.uniform(-0.02, 0.02, (n, ac.n_servos))
    motor = np.repeat(r["motor_cmd"], n, 0)
    return ac, servo, motor


def _trials(run, repeat):
    rates = sorted(run() for _ in range(repeat))
    return rates[-1], rates[len(rates) // 2]


def time_batch(n, integrator, dt, min_time, repeat):
    ac, servo, motor = setup(n, integrator)
    ac.step(servo, motor, dt)  # warm-up

    def run():
        steps, t0 = 0, time.perf_counter()
        while True:
            for _ in range(5):
                ac.step(servo, motor, dt)
            steps += 5
            el = time.perf_counter() - t0
            if el >= min_time:
                return n * steps / el

    best, median = _trials(run, repeat)
    assert np.all(np.isfinite(ac.state))
    return best, median


def time_scalar(min_time, repeat):
    ac, servo, motor = setup(1, "rk4")
    ref = ScalarAircraft(ac)
    s, m = list(servo[0]), list(motor[0])

    def run():
        x = list(ac.state[0])
        steps, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < min_time:
            x = ref.step(x, s, m, 0.01)
            steps += 1
        return steps / (time.perf_counter() - t0)

    return _trials(run, repeat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--json")
    args = ap.parse_args()
    sizes = [1, 10, 100, 1000, 10000] if args.quick else [1, 10, 100, 1000, 3000, 10000, 30000, 100000]
    min_time = 0.3 if args.quick else 1.0
    repeat = 2 if args.quick else args.repeat
    info = dict(cpu=cpu_name(), os=f"{platform.system()} {platform.release()}", python=platform.python_version(),
                numpy=np.__version__, wingbatch=wb.__version__)
    print(" | ".join(f"{k} {v}" for k, v in info.items()), f"| best of {repeat}")
    scalar, scalar_med = time_scalar(min_time, repeat)
    print(f"scalar reference (plain Python, RK4): {scalar:,.0f} aircraft-steps/s ({1e3 / scalar:.2f} ms/step)")
    rows = []
    print(f"{'N':>7} | {'rk4 steps/s':>12} | {'x scalar':>8} | {'ms/step':>8} | {'semi-implicit steps/s':>21}")
    for n in sizes:
        rk, rk_med = time_batch(n, "rk4", 0.01, min_time, repeat)
        se, se_med = time_batch(n, "semi_implicit_euler", 0.004, min_time, repeat)
        rows.append(dict(n=n, rk4=rk, rk4_median=rk_med, semi_implicit=se, semi_implicit_median=se_med,
                         speedup_vs_scalar=rk / scalar))
        print(f"{n:>7} | {rk:>12,.0f} | {rk / scalar:>8.1f} | {n / rk * 1e3:>8.3f} | {se:>21,.0f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(machine=info, repeat=repeat, scalar=scalar, scalar_median=scalar_med, rows=rows), f,
                      indent=1)


if __name__ == "__main__":
    main()
