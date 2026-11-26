#!/usr/bin/env python3
"""v2.2 Semi-Synthetic Cooperative Replay on NTU VIRAL Logs (Leakage-Free Multi-Seed Benchmark).

Replays eee_01, eee_02, and eee_03 in their shared Leica courtyard frame. During the blackout
window, Node 1 has 0 ground UWB anchors and 0 valid GNSS (or receives a slow GNSS spoofing ramp)
and localizes via peer-to-peer UWB slant ranges + authenticated 32-byte Codec telemetry from
airborne helpers Node 2 and Node 3.

Baselines & Estimators Compared:
  1. solo     : 0 ground anchors, 0 peers during blackout (pure IMU / spoofed-GNSS baseline)
  2. uwb_wls  : Epoch-by-epoch Gauss-Newton 2D UWB trilateration on peer ranges only (no IMU)
  3. kf_k1    : Naive cooperative EKF without coherence decorrelation (kappa = 1)
  4. kf_k200  : Decorrelated cooperative EKF (peer_error_coherence_s = 20.0 -> kappa = 200)
  5. ci       : AETHERNET Marginal Covariance Intersection + UWB-priority anti-spoof arbitration
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import aethernet
from aethernet.config import EKFConfig
from aethernet.datatypes import AnchorInfo, RangeObs
from aethernet.estimator import Navigator

_DUMMY_POS = np.zeros(2)
_DUMMY_P = np.eye(2)


def find_codec_class():
    candidates = [aethernet]
    for mod_info in pkgutil.iter_modules(aethernet.__path__):
        try:
            candidates.append(importlib.import_module(f"aethernet.{mod_info.name}"))
        except Exception:
            continue
    for mod in candidates:
        obj = getattr(mod, "Codec", None)
        if isinstance(obj, type):
            return obj
    return None


class _WireLink:
    def __init__(self, codec_cls, key: bytes):
        self.seq = 0
        self.impl = None
        if codec_cls is not None:
            sig = inspect.signature(codec_cls.__init__)
            if "node_id" in sig.parameters:
                self.impl = codec_cls(node_id=0, key=key)
            else:
                self.impl = codec_cls(key=key)

    def encode(self, snap):
        self.seq = (self.seq + 1) & 0xFFFF
        return snap if self.impl is None else self.impl.encode(snap, self.seq)

    def decode(self, frame):
        return frame if self.impl is None else self.impl.decode(frame)


@dataclass
class FlightLog:
    anchors: Dict[int, AnchorInfo]
    init_pos: List[float]
    init_std: float
    init_alt: float
    events: List[dict]
    duration: float


def load_jsonl_log(path: Path) -> FlightLog:
    with path.open("r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    meta = lines[0]
    assert meta.get("type") == "meta", f"Expected meta header in {path}"
    anchors = {
        int(k): AnchorInfo(x=float(v[0]), y=float(v[1]), z=float(v[2]), std=float(v[3]))
        for k, v in meta["anchors"].items()
    }
    events = lines[1:]
    duration = max((float(ev["t"]) for ev in events), default=0.0)
    init = meta.get("init", {})
    return FlightLog(
        anchors=anchors,
        init_pos=[float(init["pos"][0]), float(init["pos"][1])],
        init_std=float(init.get("std", 0.5)),
        init_alt=float(init.get("alt", 0.0)),
        events=events,
        duration=duration,
    )


def build_truth_interpolator(events: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_t: Dict[float, List[Optional[float]]] = {}
    for ev in events:
        t = float(ev["t"])
        if ev["type"] == "truth":
            slot = by_t.setdefault(t, [None, None, None])
            slot[0], slot[1] = float(ev["x"]), float(ev["y"])
        elif ev["type"] == "alt":
            slot = by_t.setdefault(t, [None, None, None])
            slot[2] = float(ev["z"])

    ts, xs, ys, zs = [], [], [], []
    last_z = 0.0
    for t in sorted(by_t):
        x, y, z = by_t[t]
        if z is not None:
            last_z = z
        if x is not None and y is not None:
            ts.append(t)
            xs.append(x)
            ys.append(y)
            zs.append(last_z)
    return np.array(ts), np.column_stack([xs, ys]), np.array(zs)


def solve_uwb_wls_2d(
    init_xy: np.ndarray,
    z_self: float,
    obs_list: List[RangeObs],
    iters: int = 8,
) -> np.ndarray:
    """Epoch-by-epoch Gauss-Newton 2D trilateration baseline (no IMU, no temporal filter)."""
    if len(obs_list) < 2:
        return init_xy.copy()
    x = init_xy.copy()
    for _ in range(iters):
        H, r, W = [], [], []
        for ob in obs_list:
            dx = x[0] - float(ob.peer_pos[0])
            dy = x[1] - float(ob.peer_pos[1])
            dz = z_self - float(ob.peer_alt or 0.0)
            pred = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if pred < 1e-3:
                continue
            H.append([dx / pred, dy / pred])
            r.append(float(ob.range_m) - pred)
            var = float(np.trace(ob.peer_P)) * 0.5 + 0.18 ** 2
            W.append(1.0 / max(var, 1e-3))
        if len(H) < 2:
            break
        H_arr = np.array(H)
        W_arr = np.diag(W)
        r_arr = np.array(r)
        A = H_arr.T @ W_arr @ H_arr + 1e-3 * np.eye(2)
        dx_step = np.linalg.solve(A, H_arr.T @ W_arr @ r_arr)
        x += dx_step
        if np.linalg.norm(dx_step) < 1e-4:
            break
    return x


def make_cfg_for_mode(mode: str, peer_rate_hz: float) -> EKFConfig:
    if mode == "kf_k1":
        return EKFConfig(
            ranging_rate_hz=peer_rate_hz,
            mobile_peer_mode="kf",
            peer_error_coherence_s=0.0,
        )
    if mode == "kf_k200":
        return EKFConfig(
            ranging_rate_hz=peer_rate_hz,
            mobile_peer_mode="kf",
            peer_error_coherence_s=20.0,
        )
    return EKFConfig(
        ranging_rate_hz=peer_rate_hz,
        mobile_peer_mode="ci",
        peer_error_coherence_s=10.0,
    )


def run_cooperative_trial(
    log1: FlightLog,
    log2: FlightLog,
    log3: FlightLog,
    mode: str,
    blackout_start: float = 35.0,
    blackout_end: float = 110.0,
    peer_rate_hz: float = 10.0,
    peer_noise_std: float = 0.18,
    peer_nlos_prob: float = 0.04,
    spoof_ramp_mps: float = 0.0,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)
    t_max = min(log1.duration, log2.duration, log3.duration)

    truths = {
        1: build_truth_interpolator(log1.events),
        2: build_truth_interpolator(log2.events),
        3: build_truth_interpolator(log3.events),
    }

    cfg = make_cfg_for_mode(mode, peer_rate_hz)
    navs = {
        1: Navigator(node_id=1, cfg=cfg, anchors=log1.anchors),
        2: Navigator(node_id=2, cfg=cfg, anchors=log2.anchors),
        3: Navigator(node_id=3, cfg=cfg, anchors=log3.anchors),
    }
    for vid, lg in ((1, log1), (2, log2), (3, log3)):
        navs[vid].initialize(lg.init_pos, pos_std=lg.init_std, alt=lg.init_alt)

    codec_cls = find_codec_class()
    key = bytes.fromhex("00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff")
    tx2 = _WireLink(codec_cls, key=key)
    tx3 = _WireLink(codec_cls, key=key)
    rx1 = _WireLink(codec_cls, key=key)

    stream: List[Tuple[float, int, dict]] = []
    for vid, lg in ((1, log1), (2, log2), (3, log3)):
        for ev in lg.events:
            if float(ev["t"]) <= t_max:
                stream.append((float(ev["t"]), vid, ev))

    dt_peer = 1.0 / peer_rate_hz
    t_tick = dt_peer
    while t_tick <= t_max:
        stream.append((t_tick, 0, {"t": t_tick, "type": "coop_tick"}))
        t_tick += dt_peer

    stream.sort(key=lambda item: (item[0], item[1]))

    latest_pkts: Dict[int, Tuple[object, float]] = {}
    wls_xy = np.array(log1.init_pos, dtype=float)
    z1_curr = float(log1.init_alt)

    errors_all: List[float] = []
    errors_blackout: List[float] = []
    nees_blackout: List[float] = []

    for t, vid, ev in stream:
        etype = ev["type"]

        if vid in (1, 2, 3):
            nav = navs[vid]
            if etype == "imu":
                nav.predict([ev["ax"], ev["ay"]], t=t, yaw=ev.get("yaw"))
            elif etype == "alt":
                nav.set_altitude(ev["z"])
                if vid == 1:
                    z1_curr = float(ev["z"])
            elif etype == "range":
                if vid == 1 and (blackout_start <= t <= blackout_end):
                    continue
                nav.update_ranges([
                    RangeObs(
                        peer_id=int(ev["peer"]),
                        range_m=float(ev["r"]),
                        peer_pos=_DUMMY_POS,
                        peer_P=_DUMMY_P,
                    )
                ])
                if vid == 1 and mode == "uwb_wls":
                    wls_xy = nav.x[:2].copy()
            elif etype == "truth" and vid == 1:
                gt_xy = np.array([ev["x"], ev["y"]])
                est_xy = wls_xy if (mode == "uwb_wls" and blackout_start <= t <= blackout_end) else nav.x[:2]
                err = float(np.linalg.norm(est_xy - gt_xy))
                errors_all.append(err)
                if blackout_start <= t <= blackout_end:
                    errors_blackout.append(err)
                    if mode != "uwb_wls":
                        nees_blackout.append(nav.nees(gt_xy))

        elif vid == 0 and etype == "coop_tick":
            for helper_id, tx in ((2, tx2), (3, tx3)):
                frame = tx.encode(navs[helper_id].snapshot())
                pkt = rx1.decode(frame)
                if pkt is not None:
                    latest_pkts[helper_id] = (pkt, t)

            peer_obs: List[RangeObs] = []
            if mode != "solo":
                t1, xy1, z1 = truths[1]
                p1 = np.array([
                    float(np.interp(t, t1, xy1[:, 0])),
                    float(np.interp(t, t1, xy1[:, 1])),
                    float(np.interp(t, t1, z1)),
                ])

                for helper_id in (2, 3):
                    if helper_id not in latest_pkts:
                        continue
                    pkt, pkt_t = latest_pkts[helper_id]
                    th, xyh, zh = truths[helper_id]
                    ph = np.array([
                        float(np.interp(t, th, xyh[:, 0])),
                        float(np.interp(t, th, xyh[:, 1])),
                        float(np.interp(t, th, zh)),
                    ])
                    true_rho = float(np.linalg.norm(p1 - ph))
                    if true_rho < 0.5:
                        continue

                    nlos_bias = rng.uniform(0.8, 2.2) if rng.random() < peer_nlos_prob else 0.0
                    meas_r = true_rho + rng.gauss(0.0, peer_noise_std) + nlos_bias

                    peer_obs.append(
                        RangeObs(
                            peer_id=helper_id,
                            range_m=meas_r,
                            peer_pos=pkt.pos,
                            peer_P=pkt.P_pos,
                            peer_alt=pkt.alt,
                            peer_alt_std=0.5,
                            peer_vel=pkt.vel,
                            peer_vel_std=pkt.vel_std,
                            age_s=t - pkt_t,
                        )
                    )

            if peer_obs:
                if mode == "uwb_wls":
                    if blackout_start <= t <= blackout_end:
                        wls_xy = solve_uwb_wls_2d(wls_xy, z1_curr, peer_obs)
                else:
                    navs[1].update_ranges(peer_obs)

            # Optional GNSS spoofing ramp injected during blackout on Node 1.
            # Under 'ci' mode, active UWB peer ranges cross-check GNSS and reject fixes that
            # deviate beyond the peer-constrained covariance ellipsoid (anti-spoof arbitration).
            if spoof_ramp_mps > 0.0 and (blackout_start <= t <= blackout_end) and mode != "uwb_wls":
                t1, xy1, _ = truths[1]
                true_x = float(np.interp(t, t1, xy1[:, 0]))
                true_y = float(np.interp(t, t1, xy1[:, 1]))
                ramp_err = spoof_ramp_mps * (t - blackout_start)
                gnss_xy = np.array([true_x + ramp_err, true_y + 0.5 * ramp_err])
                if mode == "ci" and len(peer_obs) >= 2:
                    innov = gnss_xy - navs[1].x[:2]
                    S_gnss = navs[1].P[:2, :2] + (2.5 ** 2) * np.eye(2)
                    nis_gnss = float(innov @ np.linalg.solve(S_gnss, innov))
                    if nis_gnss <= 5.99:  # 95% chi-square(2) gate against peer-locked state
                        navs[1].update_gnss(gnss_xy, std=2.5)
                else:
                    navs[1].update_gnss(gnss_xy, std=2.5)

    nis_res = navs[1].nis_consistency()
    nis_mean = nis_res[0] if (nis_res is not None and mode != "uwb_wls") else float("nan")

    err_arr = np.array(errors_all)
    blk_arr = np.array(errors_blackout) if errors_blackout else err_arr

    return {
        "rmse_blackout": float(np.sqrt(np.mean(blk_arr ** 2))),
        "p95_blackout": float(np.percentile(blk_arr, 95)),
        "rmse_all": float(np.sqrt(np.mean(err_arr ** 2))),
        "nees_median": float(np.median(nees_blackout)) if nees_blackout else float("nan"),
        "nis_mean": nis_mean,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="v2.2 Semi-Synthetic Cooperative Replay (Multi-Seed Baselines)")
    ap.add_argument("--log1", type=Path, required=True)
    ap.add_argument("--log2", type=Path, required=True)
    ap.add_argument("--log3", type=Path, required=True)
    ap.add_argument("--blackout-start", type=float, default=35.0)
    ap.add_argument("--blackout-end", type=float, default=110.0)
    ap.add_argument("--full-blackout", action="store_true")
    ap.add_argument("--peer-hz", type=float, default=10.0)
    ap.add_argument("--spoof-ramp", type=float, default=0.0, help="GNSS spoofing drift rate (m/s) during blackout")
    ap.add_argument("--seeds", type=int, default=20, help="Number of Monte Carlo seeds")
    args = ap.parse_args()

    log1 = load_jsonl_log(args.log1)
    log2 = load_jsonl_log(args.log2)
    log3 = load_jsonl_log(args.log3)

    t_shared = min(log1.duration, log2.duration, log3.duration)
    b_start = 0.0 if args.full_blackout else args.blackout_start
    b_end = t_shared if args.full_blackout else args.blackout_end

    print(f"v2.2 Semi-Synthetic Cooperative Replay ({args.seeds} seeds, {t_shared:.1f}s window, spoof_ramp={args.spoof_ramp} m/s)")
    print(f"Blackout on Node 1 ({args.log1.name}): {b_start:.0f}s..{b_end:.0f}s (0 ground anchors, 0 GNSS)\n")

    header = f"{'Method':<9} | {'Med Blk RMSE':>12} | {'Med Blk p95':>11} | {'Med Full RMSE':>13} | {'Med NEES (2D)':>13} | {'Mean NIS':>8}"
    print(header)
    print("-" * len(header))

    modes = ("solo", "uwb_wls", "kf_k1", "kf_k200", "ci")
    for mode in modes:
        runs = [
            run_cooperative_trial(
                log1=log1,
                log2=log2,
                log3=log3,
                mode=mode,
                blackout_start=b_start,
                blackout_end=b_end,
                peer_rate_hz=args.peer_hz,
                spoof_ramp_mps=args.spoof_ramp,
                seed=100 + s,
            )
            for s in range(args.seeds)
        ]
        med_blk_rmse = float(np.median([r["rmse_blackout"] for r in runs]))
        med_blk_p95 = float(np.median([r["p95_blackout"] for r in runs]))
        med_full_rmse = float(np.median([r["rmse_all"] for r in runs]))

        if mode == "uwb_wls":
            nees_str = f"{'N/A':>13}"
            nis_str = f"{'N/A':>8}"
        else:
            nees_str = f"{float(np.median([r['nees_median'] for r in runs])):>13.2f}"
            nis_str = f"{float(np.mean([r['nis_mean'] for r in runs])):>8.2f}"

        print(
            f"{mode:<9} | "
            f"{med_blk_rmse:>10.2f} m | "
            f"{med_blk_p95:>9.2f} m | "
            f"{med_full_rmse:>11.2f} m | "
            f"{nees_str} | "
            f"{nis_str}"
        )


if __name__ == "__main__":
    main()