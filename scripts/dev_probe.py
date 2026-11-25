"""Dev probe: quick per-config stats on DEV seeds (never the report seeds)."""
import sys, numpy as np
from aethernet.sim.scenario import SimConfig, Scenario
from aethernet.sim.swarms import V2Swarm
from aethernet.sim.run import metrics
from aethernet import EKFConfig

def probe(label, cfg, seeds=(0, 1, 2), sim=None):
    rows = []
    for seed in seeds:
        scen = Scenario(sim or SimConfig(duration=120.0), seed)
        sw = V2Swarm(scen, cfg)
        for s in range(1, scen.T): sw.step(s)
        err = np.linalg.norm(sw.pos - scen.pos, axis=2)[2:, 200:]; n = sw.nees[2:, 200:]
        d = sw.diag()
        m = metrics(scen, sw)
        rows.append((np.sqrt(np.mean(err**2)), np.median(n), np.mean(n > 13.8), err.max(), np.percentile(err, 95), d["recoveries"], d["inflations"], m["rel_rmse"]))
    o = np.array(rows).mean(0)
    print(f"{label:30s} rmse {o[0]:5.2f} p95 {o[4]:5.2f} max {o[3]:5.1f} | medNEES {o[1]:5.2f} frac>13.8 {o[2]:.3f} | rec {o[5]:.1f} infl {o[6]:.1f} | rel_rmse {o[7]:.2f}")

if __name__ == "__main__":
    for mode in ("ci", "kf"):
        for kappa in (1.0, 4.0, 16.0, 64.0):
            probe(f"{mode} kappa={kappa:g}", EKFConfig(mobile_peer_mode=mode, peer_cov_decorrelation=kappa))
