"""Replay a logged run through the estimator -- the bridge to real hardware data.

Log format: JSON Lines, first line a ``meta`` record, then events sorted by ``t`` (seconds, monotonic clock of the node):

  {"type":"meta","node_id":3,"key_hex":"<hex>"|null,"anchors":{"0":[x,y,z,std]},"init":{"pos":[x,y],"std":0.5,"alt":3.0}}
  {"t":..,"type":"imu","ax":..,"ay":..,"yaw":..}          world-frame horizontal accel (gravity removed) + yaw [rad]
  {"t":..,"type":"alt","z":..}                            altitude above mission origin [m]
  {"t":..,"type":"gnss","x":..,"y":..,"std":..}           optional absolute fix
  {"t":..,"type":"packet","hex":".."}                     raw 32-byte state packet, t = local ARRIVAL time
  {"t":..,"type":"range","peer":id,"r":..}                TWR range, t = local measurement time
  {"t":..,"type":"truth","x":..,"y":..}                   optional ground truth (RTK / total station / mocap)

Ground truth is used only for scoring; the NIS statistic needs none and is the first thing to check on real data.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import numpy as np

from .config import EKFConfig
from .datatypes import AnchorInfo, RangeObs
from .estimator import Navigator
from .protocol import Codec, PeerTable


def replay(path: str, cfg: Optional[EKFConfig] = None, batch_window_s: float = 0.02) -> dict:
    with open(path) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    meta = lines[0]
    assert meta["type"] == "meta", "first line must be a meta record"
    events = lines[1:]
    anchors = {int(k): AnchorInfo(*v) for k, v in meta.get("anchors", {}).items()}
    key = bytes.fromhex(meta["key_hex"]) if meta.get("key_hex") else None
    codec = Codec(key, insecure=key is None)
    table = PeerTable(codec, anchors)
    nav = Navigator(meta["node_id"], cfg, anchors=anchors)
    init = meta["init"]
    nav.initialize(init["pos"], init.get("vel"), init.get("std", 0.5), alt=init.get("alt"))

    traj: List[list] = []
    truth: List[list] = []
    batch: List[RangeObs] = []
    batch_t = None
    reports = {"accepted": 0, "gated": 0, "skipped": 0, "recovered": 0}

    def flush():
        nonlocal batch, batch_t
        if batch:
            r = nav.update_ranges(batch)
            reports["accepted"] += r.accepted; reports["gated"] += r.gated
            reports["skipped"] += r.skipped; reports["recovered"] += int(r.recovered)
        batch, batch_t = [], None

    last_acc = None
    for ev in events:
        t, ty = ev["t"], ev["type"]
        if ty in ("imu", "alt", "gnss", "truth") or (ty == "range" and batch_t is not None and t - batch_t > batch_window_s):
            flush()
        if ty == "imu":
            nav.predict(np.array([ev["ax"], ev["ay"]]), t, ev.get("yaw"))
            traj.append([t, *nav.x[:2], *nav.x[2:4], float(np.sqrt(nav.P[0, 0] + nav.P[1, 1]))])
        elif ty == "alt":
            nav.set_altitude(ev["z"], ev.get("std"))
        elif ty == "gnss":
            nav.update_gnss(np.array([ev["x"], ev["y"]]), ev["std"])
        elif ty == "packet":
            table.ingest(bytes.fromhex(ev["hex"]), t)
        elif ty == "range":
            o = table.observe(ev["peer"], ev["r"], t)
            if o is not None:
                if batch_t is None:
                    batch_t = t
                batch.append(o)
        elif ty == "truth":
            truth.append([t, ev["x"], ev["y"]])
    flush()
    traj_a = np.array(traj)
    out = {"traj": traj_a, "reports": reports, "rejected_packets": dict(table.rejected),
           "nis": nav.nis_consistency(), "diag": nav.diag.__dict__.copy(), "final_pos": nav.x[:2].copy()}
    if truth and len(traj_a):
        tr = np.array(truth)
        est = np.stack([np.interp(tr[:, 0], traj_a[:, 0], traj_a[:, 1]), np.interp(tr[:, 0], traj_a[:, 0], traj_a[:, 2])], 1)
        e = np.linalg.norm(est - tr[:, 1:3], axis=1)
        out["truth_rmse"] = float(np.sqrt(np.mean(e ** 2)))
        out["truth_p95"] = float(np.percentile(e, 95))
    return out


def main():
    ap = argparse.ArgumentParser(description="Replay an AETHERNET JSONL log")
    ap.add_argument("log"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = replay(a.log)
    print("range updates:", r["reports"], "| packets rejected:", r["rejected_packets"])
    if r["nis"]:
        m, lo, hi, n = r["nis"]
        verdict = "CONSISTENT" if lo <= m <= hi else ("OVERCONFIDENT (retune noise up)" if m > hi else "CONSERVATIVE")
        print(f"NIS mean {m:.2f} (95% band {lo:.2f}-{hi:.2f}, n={n}) -> {verdict}")
    if "truth_rmse" in r:
        print(f"vs truth: RMSE {r['truth_rmse']:.2f} m, p95 {r['truth_p95']:.2f} m")
    if a.out:
        np.savetxt(a.out, r["traj"], delimiter=",", header="t,x,y,vx,vy,sigma", comments="")


if __name__ == "__main__":
    main()
