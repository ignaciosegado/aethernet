"""Monte Carlo evaluation suite.  Usage:  python -m aethernet.sim.mc --out docs/results [--quick]

Seeds 0-9 were the development/tuning set; seeds 100-119 gave the first (kappa=100) held-out run, archived; the final report uses fresh seeds 200+.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Dict, List

import numpy as np

from ..config import chi2_gate
from .run import run_scenario
from .scenario import SimConfig

MAIN = ["dead_reckoning", "v1_legacy", "v2", "v2_naive_independent", "v2_marginal_ci", "v2_frozen_bias",
        "v2_no_altitude", "v2_no_latency_comp"]
SUITES = {
    "nominal": dict(cfg=SimConfig(), algos=MAIN),
    "spoof_jump": dict(cfg=SimConfig(gnss_windows=[(0, 120)], attack={"type": "jump", "t0": 40.0, "offset": (30.0, 0.0)}),
                       algos=["v1_legacy", "v2"]),
    "spoof_drag": dict(cfg=SimConfig(gnss_windows=[(0, 120)], attack={"type": "drag", "t0": 40.0, "rate": 0.4}),
                       algos=["v1_legacy", "v2"]),
    "imu_stall": dict(cfg=SimConfig(imu_stall={"t0": 60.0, "dur": 3.0, "nodes": [3, 4, 5]}),
                      algos=["v1_legacy", "v2"]),
    "anchorless": dict(cfg=SimConfig(anchors=False, gnss_windows=[(0, 40)]),
                       algos=["dead_reckoning", "v1_legacy", "v2"]),
}


def _ci(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def run_suite(name: str, seeds: List[int], out: pathlib.Path, duration: float = 120.0) -> dict:
    spec = SUITES[name]
    cfg = spec["cfg"]
    cfg.duration = duration
    per: Dict[str, List[dict]] = {a: [] for a in spec["algos"]}
    t0 = time.time()
    for k, sd in enumerate(seeds):
        res = run_scenario(cfg, sd, spec["algos"])
        for a, m in res.items():
            per[a].append(m)
        print(f"[{name}] seed {sd} ({k + 1}/{len(seeds)}) {time.time() - t0:.0f}s", flush=True)
    summary = {}
    for a, runs in per.items():
        row = {}
        for key in ("rmse", "p95", "max", "final", "rel_rmse", "relay_rmse", "relay_max", "nees_median", "cpu_s"):
            row[key] = _ci([r[key] for r in runs])
        nees = np.concatenate([r["nees_samples_t"][:, 200::5] for r in runs], axis=0)      # (runs*nodes, samples)
        row["frac_nees_gt_99.9"] = float(np.mean(nees > chi2_gate(0.999, 2)))
        row["nees_mean_robust"] = float(np.mean(np.minimum(nees, 100.0)))
        K = nees.shape[0]
        lo, hi = chi2_gate(0.025, 2 * K) / K, chi2_gate(0.975, 2 * K) / K
        anees_t = nees.mean(axis=0)
        row["anees_median"] = float(np.median(anees_t))
        row["anees_frac_in_95"] = float(np.mean((anees_t >= lo) & (anees_t <= hi)))
        row["diag"] = {k: int(sum(r["diag"].get(k, 0) for r in runs)) for k in runs[0]["diag"]}
        summary[a] = row
    series = {a: {"err_t": np.mean([r["err_t"] for r in runs], axis=0).tolist(),
                  "relay_err_t": np.mean([r["relay_err_t"] for r in runs], axis=0).tolist(),
                  "relay_err_max_t": np.max([r["relay_err_t"] for r in runs], axis=0).tolist(),
                  "nees_t": np.mean([r["nees_t"] for r in runs], axis=0).tolist(),
                  "sigma_t": np.mean([r["sigma_t"] for r in runs], axis=0).tolist()} for a, runs in per.items()}
    out.mkdir(parents=True, exist_ok=True)
    doc = {"suite": name, "seeds": seeds, "duration": duration, "dt": cfg.dt, "summary": summary, "series": series}
    (out / f"{name}.json").write_text(json.dumps(doc))
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--suites", nargs="*", default=list(SUITES))
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    for name in a.suites:
        n = (3 if a.quick else (20 if name == "nominal" else 10))
        run_suite(name, list(range(200, 200 + n)), out, duration=60.0 if a.quick else 120.0)


if __name__ == "__main__":
    main()
