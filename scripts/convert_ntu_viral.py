#!/usr/bin/env python3
"""Convert an NTU VIRAL ROS1 .bag + ground-truth CSV into AETHERNET replay.py JSONL format.

Separates onboard UAV UWB antennas (requester_id 200 vs 201) into distinct virtual anchor IDs
(e.g. 200100 vs 201100) so alternating body-frame lever-arm offsets do not inject step jumps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


def quat_to_rot_and_yaw(qx: float, qy: float, qz: float, qw: float) -> Tuple[np.ndarray, float]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        return np.eye(3), 0.0
    x, y, z, w = qx / norm, qy / norm, qz / norm, qw / norm
    R = np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return R, yaw


def load_gt_csv(csv_path: Path, prism_offset_body: np.ndarray) -> List[Tuple[float, np.ndarray, float]]:
    """Load NTU VIRAL ground-truth CSV using header names if present, else column count."""
    rows: List[Tuple[float, np.ndarray, float]] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        first_row = [c.strip() for c in next(reader)]

        header = None
        try:
            first_vals = [float(v) for v in first_row]
            raw_rows = [first_vals] + [[float(v) for v in r] for r in reader if r]
        except ValueError:
            header = [c.lstrip("%#").strip().lower() for c in first_row]
            raw_rows = [[float(v) for v in r] for r in reader if r]

    if not raw_rows:
        return rows

    for vals in raw_rows:
        if header and "field.header.stamp" in header:
            idx_t = header.index("field.header.stamp")
            idx_x = header.index("field.pose.position.x")
            t_raw = vals[idx_t]
            t_s = t_raw * 1e-9 if t_raw > 1e12 else t_raw
            p_prism = np.array(vals[idx_x:idx_x + 3], dtype=float)
            R_wb, yaw = quat_to_rot_and_yaw(vals[idx_x + 3], vals[idx_x + 4], vals[idx_x + 5], vals[idx_x + 6])
            p_imu = p_prism - R_wb @ prism_offset_body
        elif len(vals) >= 10:
            t_s = vals[1] + vals[2] * 1e-9
            p_prism = np.array(vals[3:6], dtype=float)
            R_wb, yaw = quat_to_rot_and_yaw(vals[6], vals[7], vals[8], vals[9])
            p_imu = p_prism - R_wb @ prism_offset_body
        else:
            t_raw = vals[0]
            t_s = t_raw * 1e-9 if t_raw > 1e12 else t_raw
            p_prism = np.array(vals[1:4], dtype=float)
            if len(vals) >= 8:
                R_wb, yaw = quat_to_rot_and_yaw(vals[4], vals[5], vals[6], vals[7])
                p_imu = p_prism - R_wb @ prism_offset_body
            else:
                yaw = 0.0
                p_imu = p_prism
        rows.append((t_s, p_imu, yaw))
    return rows


def solve_anchor_xyz(gt_t: np.ndarray, gt_xyz: np.ndarray, obs_t: np.ndarray, obs_r: np.ndarray) -> Tuple[np.ndarray, float]:
    """Estimate 3D anchor coordinates in Leica mission frame via Huber Gauss-Newton."""
    uav_xyz = np.column_stack([
        np.interp(obs_t, gt_t, gt_xyz[:, 0]),
        np.interp(obs_t, gt_t, gt_xyz[:, 1]),
        np.interp(obs_t, gt_t, gt_xyz[:, 2]),
    ])
    a = uav_xyz.mean(axis=0) + np.array([5.0, 5.0, 2.0])
    for it in range(60):
        diff = a - uav_xyz
        dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-6)
        res = obs_r - dist
        if it >= 10:
            med = max(0.10, float(np.median(np.abs(res))))
            w = np.where(np.abs(res) <= 2.5 * med, 1.0, (2.5 * med) / np.maximum(np.abs(res), 1e-6))
        else:
            w = np.ones_like(res)
        J = diff / dist[:, None]
        step = np.linalg.lstsq(J * w[:, None], res * w, rcond=None)[0]
        a = a + step
        if float(np.linalg.norm(step)) < 1e-5:
            break
    final_res = np.abs(obs_r - np.linalg.norm(a - uav_xyz, axis=1))
    return a, float(np.median(final_res))


def register_bag_msgdefs(reader: Reader, typestore) -> None:
    custom_types = {}
    for conn in reader.connections:
        if conn.msgtype in typestore.fielddefs:
            continue
        raw_def = getattr(conn.msgdef, "data", conn.msgdef)
        if isinstance(raw_def, bytes):
            raw_def = raw_def.decode("utf-8", errors="ignore")
        if raw_def:
            try:
                custom_types.update(get_types_from_msg(raw_def, conn.msgtype))
            except Exception:
                pass
    if custom_types:
        typestore.register(custom_types)


def convert(
    bag_path: Path,
    gt_csv_path: Path,
    out_path: Path,
    imu_topic: str = "/imu/imu",
    uwb_topic: str = "/uwb_endorange_info",
    gravity: float = 9.80665,
    anchor_std: float = 0.25,
    max_imu_hz: float = 200.0,
    magnetic_free_yaw: bool = False,
) -> None:
    prism_offset_body = np.array([-0.293656, -0.012288, -0.273095], dtype=float)
    gt_rows = load_gt_csv(gt_csv_path, prism_offset_body)
    if not gt_rows:
        raise RuntimeError(f"No ground-truth rows parsed from {gt_csv_path}")

    gt_abs_t = np.array([r[0] for r in gt_rows])
    gt_xyz_raw = np.array([r[1] for r in gt_rows])
    gt_yaws = np.unwrap(np.array([r[2] for r in gt_rows]))

    typestore = get_typestore(Stores.ROS1_NOETIC)
    imu_raw_list: List[Tuple[float, np.ndarray, np.ndarray, float]] = []
    uwb_raw_list: List[Tuple[float, int, int, float]] = []
    pair_counts: Counter = Counter()

    with Reader(bag_path) as reader:
        register_bag_msgdefs(reader, typestore)
        topic_map = {c.topic: c for c in reader.connections}
        if imu_topic not in topic_map:
            raise KeyError(f"IMU topic {imu_topic!r} not found. Available: {list(topic_map)}")
        if uwb_topic not in topic_map:
            candidates = [t for t in topic_map if "uwb" in t and "exo" not in t]
            if not candidates:
                raise KeyError(f"No UWB range topic found. Available: {list(topic_map)}")
            uwb_topic = candidates[0]

        for conn, timestamp_ns, rawdata in reader.messages(connections=[topic_map[imu_topic], topic_map[uwb_topic]]):
            t_abs = timestamp_ns * 1e-9
            if conn.topic == imu_topic:
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                R_wb, imu_yaw = quat_to_rot_and_yaw(
                    msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
                )
                acc_body = np.array([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z,
                ], dtype=float)
                imu_raw_list.append((t_abs, R_wb, acc_body, imu_yaw))

            elif conn.topic == uwb_topic:
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                req_id = int(getattr(msg, "requester_id", -1))
                resp_id = int(getattr(msg, "responder_id", -1))
                dist = float(getattr(msg, "distance", -1.0))
                if math.isfinite(dist) and dist > 0.0:
                    uwb_raw_list.append((t_abs, req_id, resp_id, dist))
                    pair_counts[(req_id, resp_id)] += 1

    if abs(imu_raw_list[0][0] - gt_abs_t[0]) > 1000.0:
        gt_abs_t = gt_abs_t - gt_abs_t[0] + imu_raw_list[0][0]

    t0 = max(imu_raw_list[0][0], gt_abs_t[0])
    t_end = min(imu_raw_list[-1][0], gt_abs_t[-1])

    yaw_diffs = []
    for t_abs, _, _, imu_yaw in imu_raw_list:
        if t0 <= t_abs <= t_end:
            leica_yaw = float(np.interp(t_abs, gt_abs_t, gt_yaws))
            yaw_diffs.append(math.atan2(math.sin(leica_yaw - imu_yaw), math.cos(leica_yaw - imu_yaw)))
    yaw_offset = float(np.median(yaw_diffs)) if yaw_diffs else 0.0
    c_off, s_off = math.cos(yaw_offset), math.sin(yaw_offset)
    R_leica_imu = np.array([[c_off, -s_off, 0.0], [s_off, c_off, 0.0], [0.0, 0.0, 1.0]])

    # Separate (requester_id, responder_id) links into distinct virtual anchor IDs (req * 1000 + resp)
    # so onboard UAV antennas 200 and 201 each get their own calibrated offset
    link_ranges: Dict[int, List[Tuple[float, float]]] = {}
    for t_abs, req_id, resp_id, dist in uwb_raw_list:
        if t0 <= t_abs <= t_end:
            virtual_id = req_id * 1000 + resp_id
            link_ranges.setdefault(virtual_id, []).append((t_abs, dist))

    gt_rel_t = gt_abs_t - t0
    anchors: Dict[int, List[float]] = {}
    anchor_ranges: Dict[int, List[Tuple[float, float]]] = {}
    for virtual_id, pairs in sorted(link_ranges.items()):
        dists = np.array([p[1] for p in pairs])
        if len(dists) < 50 or float(np.std(dists)) < 1.0:
            continue
        obs_t = np.array([p[0] - t0 for p in pairs])
        obs_r = np.array([p[1] for p in pairs])
        a_xyz, mad = solve_anchor_xyz(gt_rel_t, gt_xyz_raw, obs_t, obs_r)
        if mad < 0.8:
            anchors[virtual_id] = [round(float(a_xyz[0]), 4), round(float(a_xyz[1]), 4), round(float(a_xyz[2]), 4), anchor_std]
            anchor_ranges[virtual_id] = list(zip(obs_t.tolist(), obs_r.tolist()))
            print(f"Solved link {virtual_id}: xyz={anchors[virtual_id][:3]} (n={len(obs_r)}, median_res={mad:.3f} m)")

    if not anchors:
        raise RuntimeError("Could not identify static ground anchors from UWB ranges.")

    events: List[Tuple[float, dict]] = []
    last_imu_t = -1e9
    min_imu_dt = 1.0 / max_imu_hz if max_imu_hz > 0 else 0.0

    for t_abs, R_wb_imu, acc_body, imu_yaw in imu_raw_list:
        if t_abs < t0 or t_abs > t_end:
            continue
        t_rel = t_abs - t0
        if t_rel - last_imu_t < min_imu_dt:
            continue
        last_imu_t = t_rel

        if magnetic_free_yaw:
            yaw_true = float(np.interp(t_abs, gt_abs_t, gt_yaws))
            yaw_corr = yaw_true - imu_yaw
            c_c, s_c = math.cos(yaw_corr), math.sin(yaw_corr)
            R_align = np.array([[c_c, -s_c, 0.0], [s_c, c_c, 0.0], [0.0, 0.0, 1.0]])
            acc_world = R_align @ (R_wb_imu @ acc_body) - np.array([0.0, 0.0, gravity])
            yaw_out = yaw_true
        else:
            acc_world = R_leica_imu @ (R_wb_imu @ acc_body) - np.array([0.0, 0.0, gravity])
            yaw_out = imu_yaw + yaw_offset

        events.append((t_rel, {
            "t": round(t_rel, 6),
            "type": "imu",
            "ax": float(acc_world[0]),
            "ay": float(acc_world[1]),
            "yaw": float(yaw_out),
        }))

    for anchor_id, pairs in anchor_ranges.items():
        for t_rel, dist in pairs:
            events.append((t_rel, {
                "t": round(t_rel, 6),
                "type": "range",
                "peer": anchor_id,
                "r": float(dist),
            }))

    init_pos = None
    init_alt = 0.0
    for i, (t_abs, p_imu, _) in enumerate(gt_rows):
        t_use = float(gt_abs_t[i])
        if t_use < t0 or t_use > t_end:
            continue
        t_rel = t_use - t0
        if init_pos is None:
            init_pos = [float(p_imu[0]), float(p_imu[1])]
            init_alt = float(p_imu[2])
        events.append((t_rel, {"t": round(t_rel, 6), "type": "alt", "z": float(p_imu[2])}))
        events.append((t_rel, {"t": round(t_rel, 6), "type": "truth", "x": float(p_imu[0]), "y": float(p_imu[1])}))

    events.sort(key=lambda item: item[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "type": "meta",
        "node_id": 1,
        "key_hex": None,
        "anchors": {str(k): v for k, v in sorted(anchors.items())},
        "init": {
            "pos": init_pos or [0.0, 0.0],
            "std": 0.5,
            "alt": init_alt,
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
        for _, ev in events:
            f.write(json.dumps(ev) + "\n")

    print(f"Wrote {len(events)} events + {len(anchors)} anchor links to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert NTU VIRAL bag + GT CSV to AETHERNET replay JSONL")
    ap.add_argument("--bag", type=Path, required=True)
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imu-topic", default="/imu/imu")
    ap.add_argument("--uwb-topic", default="/uwb_endorange_info")
    ap.add_argument("--anchor-std", type=float, default=0.25)
    ap.add_argument("--max-imu-hz", type=float, default=200.0)
    ap.add_argument("--magnetic-free-yaw", action="store_true",
                        help="Use magnetic-free mission yaw while keeping raw VN-100 accelerometer specific force")
    args = ap.parse_args()
    convert(
        bag_path=args.bag,
        gt_csv_path=args.gt_csv,
        out_path=args.out,
        imu_topic=args.imu_topic,
        uwb_topic=args.uwb_topic,
        anchor_std=args.anchor_std,
        max_imu_hz=args.max_imu_hz,
        magnetic_free_yaw=args.magnetic_free_yaw,
    )


if __name__ == "__main__":
    main()