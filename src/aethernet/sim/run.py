from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np

from ..config import EKFConfig
from .scenario import Scenario, SimConfig
from .swarms import V1Swarm, V2Swarm

NAIVE = dict(ranging_rate_hz=1.0, peer_error_coherence_s=1.0)      # kappa = 1: peers treated as independent

ALGOS: Dict[str, Callable] = {
    "dead_reckoning": lambda sc: V2Swarm(sc, EKFConfig(), update=False, name="dead_reckoning"),
    "v1_legacy": lambda sc: V1Swarm(sc),
    "v2": lambda sc: V2Swarm(sc, EKFConfig(), name="v2"),
    # ablations ------------------------------------------------------------------------------------------
    "v2_naive_independent": lambda sc: V2Swarm(sc, EKFConfig(**NAIVE), name="v2_naive_independent"),
    "v2_marginal_ci": lambda sc: V2Swarm(sc, EKFConfig(mobile_peer_mode="ci", **NAIVE), name="v2_marginal_ci"),
    "v2_frozen_bias": lambda sc: V2Swarm(sc, EKFConfig(freeze_bias_on_range=True), name="v2_frozen_bias"),
    "v2_no_altitude": lambda sc: V2Swarm(sc, EKFConfig(use_altitude=False), name="v2_no_altitude"),
    "v2_no_latency_comp": lambda sc: V2Swarm(sc, EKFConfig(use_age=False), name="v2_no_latency_comp"),
}


def metrics(scen: Scenario, sw, warm: float = 10.0) -> dict:
    N, T, dt = scen.N, scen.T, scen.cfg.dt
    tact = list(range(2, N))
    err = np.linalg.norm(sw.pos - scen.pos, axis=2)
    k0 = int(warm / dt)
    e_t = err[tact]
    n_t = sw.nees[tact]
    # pairwise distance error among tactical nodes (relative geometry), every 20th step
    rel = []
    for s in range(k0, T, 20):
        pe, pt = sw.pos[tact, s], scen.pos[tact, s]
        de = np.linalg.norm(pe[:, None] - pe[None], axis=-1); dtru = np.linalg.norm(pt[:, None] - pt[None], axis=-1)
        iu = np.triu_indices(len(tact), 1)
        rel.append(np.mean((de[iu] - dtru[iu]) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(e_t[:, k0:] ** 2))),
        "final": float(np.mean(e_t[:, -20:])),
        "p95": float(np.percentile(e_t[:, k0:], 95)),
        "max": float(np.max(e_t[:, k0:])),
        "rel_rmse": float(np.sqrt(np.mean(rel))),
        "relay_rmse": float(np.sqrt(np.mean(err[1, k0:] ** 2))),
        "relay_max": float(np.max(err[1, k0:])),
        "nees_mean": float(np.nanmean(n_t[:, k0:])),
        "nees_median": float(np.nanmedian(n_t[:, k0:])),
        "err_t": e_t.mean(axis=0),
        "relay_err_t": err[1],
        "nees_t": np.nanmean(n_t, axis=0),
        "nees_samples_t": n_t,                      # (n_tact, T) for ANEES bounds
        "sigma_t": np.sqrt(sw.ptrace[tact].mean(axis=0)),
        "diag": sw.diag(),
    }


def run_scenario(sim_cfg: SimConfig, seed: int, algos=None, verbose: bool = False) -> Dict[str, dict]:
    scen = Scenario(sim_cfg, seed)
    names = list(algos or ALGOS)
    swarms = {n: ALGOS[n](scen) for n in names}
    cost = {n: 0.0 for n in names}
    for s in range(1, scen.T):
        for n, sw in swarms.items():
            t0 = time.perf_counter(); sw.step(s); cost[n] += time.perf_counter() - t0
    out = {n: metrics(scen, sw) for n, sw in swarms.items()}
    for n in names:
        out[n]["cpu_s"] = cost[n]
    if verbose:
        for n in names:
            m = out[n]
            print(f"  {n:24s} rmse {m['rmse']:7.2f} final {m['final']:7.2f} nees {m['nees_mean']:8.2f} cpu {m['cpu_s']:5.1f}s")
    return out
