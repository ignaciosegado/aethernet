#!/usr/bin/env python3
"""v2.2 Multi-Vehicle Cooperative Hardware Replay (0-Anchor Isolation on Node 1).

Pairs three real NTU VIRAL flight logs recorded in the same arena (e.g. eee_01, eee_02, eee_03)
in their shared Leica mission frame. During the blackout window (or the entire flight with
--full-blackout), Node 1 has 0 ground UWB anchors and 0 GNSS, localizing purely via peer-to-peer
UWB ranges and authenticated 32-byte Codec telemetry packets from airborne helpers Node 2 & Node 3.

Usage:
    python scripts/coop_ntu_viral.py --log1 logs/eee_01.jsonl --log2 logs/eee_02.jsonl --log3 logs/eee_03.jsonl
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
    """Locate Codec inside aethernet."""
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
    """Wrapper matching Codec(key).encode(snap, seq) and Codec(key).decode(frame)."""

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
        if self.impl is None:
            return snap
        return self.impl.encode(snap, self.seq)

    def decode(self, frame):
        if self.impl is None:
            return frame
        return self.impl.decode(frame)


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
    """Extract sorted (t, x, y, z) ground-truth arrays from converted NTU VIRAL JSONL."""
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
    seed: int = 42,
) -> dict:
    """Run 3-UAV cooperative trial where Node 1 drops to 0 ground anchors during blackout."""
    rng = random.Random(seed)
    t_max = min(log1.duration, log2.duration, log3.duration)

    truths = {
        1: build_truth_interpolator(log1.events),
        2: build_truth_interpolator(log2.events),
        3: build_truth_interpolator(log3.events),
    }

    base_cfg = EKFConfig(
        ranging_rate_hz=peer_rate_hz,
        mobile_peer_mode="ci" if mode == "ci" else "kf",
    )

    navs = {
        1: Navigator(node_id=1, cfg=base_cfg, anchors=log1.anchors),
        2: Navigator(node_id=2, cfg=base_cfg, anchors=log2.anchors),
        3: Navigator(node_id=3, cfg=base_cfg, anchors=log3.anchors),
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

    errors_all: List[float] = []
    errors_blackout: List[float] = []
    nees_blackout: List[float] = []
    coop_accepted = 0
    coop_gated = 0

    for t, vid, ev in stream:
        etype = ev["type"]

        if vid in (1, 2, 3):
            nav = navs[vid]
            if etype == "imu":
                nav.predict([ev["ax"], ev["ay"]], t=t, yaw=ev.get("yaw"))
            elif etype == "alt":
                nav.set_altitude(ev["z"])
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
            elif etype == "truth" and vid == 1:
                err = float(np.linalg.norm(nav.x[:2] - np.array([ev["x"], ev["y"]])))
                errors_all.append(err)
                if blackout_start <= t <= blackout_end:
                    errors_blackout.append(err)
                    nees_blackout.append(nav.nees([ev["x"], ev["y"]]))

        elif vid == 0 and etype == "coop_tick":
            for helper_id, tx in ((2, tx2), (3, tx3)):
                frame = tx.encode(navs[helper_id].snapshot())
                pkt = rx1.decode(frame)
                if pkt is not None:
                    latest_pkts[helper_id] = (pkt, t)

            if mode == "solo":
                continue

            t1, xy1, z1 = truths[1]
            p1 = np.array([
                float(np.interp(t, t1, xy1[:, 0])),
                float(np.interp(t, t1, xy1[:, 1])),
                float(np.interp(t, t1, z1)),
            ])

            peer_obs: List[RangeObs] = []
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
                rep = navs[1].update_ranges(peer_obs)
                coop_accepted += rep.accepted
                coop_gated += rep.gated

    nis_res = navs[1].nis_consistency()
    nis_mean = nis_res[0] if nis_res is not None else float("nan")

    err_arr = np.array(errors_all)
    blk_arr = np.array(errors_blackout) if errors_blackout else err_arr
    nees_arr = np.array(nees_blackout) if nees_blackout else np.array([float("nan")])

    return {
        "mode": mode,
        "duration_s": t_max,
        "rmse_all": float(np.sqrt(np.mean(err_arr ** 2))),
        "rmse_blackout": float(np.sqrt(np.mean(blk_arr ** 2))),
        "p95_blackout": float(np.percentile(blk_arr, 95)),
        "nees_blackout": float(np.mean(nees_arr)),
        "nis_mean": nis_mean,
        "coop_accepted": coop_accepted,
        "coop_gated": coop_gated,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="v2.2 3-UAV Cooperative Replay (0 Ground Anchors on Node 1)")
    ap.add_argument("--log1", type=Path, required=True, help="Primary UAV log with 0 anchors in blackout (e.g. logs/eee_01.jsonl)")
    ap.add_argument("--log2", type=Path, required=True, help="Helper UAV 2 log (e.g. logs/eee_02.jsonl)")
    ap.add_argument("--log3", type=Path, required=True, help="Helper UAV 3 log (e.g. logs/eee_03.jsonl)")
    ap.add_argument("--blackout-start", type=float, default=35.0)
    ap.add_argument("--blackout-end", type=float, default=110.0)
    ap.add_argument("--full-blackout", action="store_true", help="Node 1 has 0 ground anchors for the entire flight from t=0")
    ap.add_argument("--peer-hz", type=float, default=10.0)
    args = ap.parse_args()

    codec_cls = find_codec_class()
    codec_label = f"{codec_cls.__module__}.{codec_cls.__name__}" if codec_cls else "FallbackWire"

    log1 = load_jsonl_log(args.log1)
    log2 = load_jsonl_log(args.log2)
    log3 = load_jsonl_log(args.log3)

    t_shared = min(log1.duration, log2.duration, log3.duration)
    b_start = 0.0 if args.full_blackout else args.blackout_start
    b_end = t_shared if args.full_blackout else args.blackout_end

    print(f"Loaded 3 GNSS-denied UAVs over {t_shared:.1f}s shared flight window (wire codec: {codec_label}):")
    print(f"  Node 1 ({args.log1.name}): 0 GNSS, 0 ground anchors during {b_start:.0f}s..{b_end:.0f}s")
    print(f"  Node 2 ({args.log2.name}) & Node 3 ({args.log3.name}): 0 GNSS, airborne UWB anchor relays at {args.peer_hz:.0f} Hz\n")

    header = f"{'Mode':<6} | {'0-Anchor RMSE':>13} | {'0-Anchor p95':>12} | {'Full RMSE':>9} | {'Mean NEES (2D)':>14} | {'NIS Mean':>8} | {'Peer Acc/Gated':>14}"
    print(header)
    print("-" * len(header))

    for mode in ("solo", "kf", "ci"):
        res = run_cooperative_trial(
            log1=log1,
            log2=log2,
            log3=log3,
            mode=mode,
            blackout_start=b_start,
            blackout_end=b_end,
            peer_rate_hz=args.peer_hz,
        )
        print(
            f"{res['mode']:<6} | "
            f"{res['rmse_blackout']:>11.2f} m | "
            f"{res['p95_blackout']:>10.2f} m | "
            f"{res['rmse_all']:>7.2f} m | "
            f"{res['nees_blackout']:>14.2f} | "
            f"{res['nis_mean']:>8.2f} | "
            f"{res['coop_accepted']:>6d}/{res['coop_gated']:<7d}"
        )


if __name__ == "__main__":
    main()