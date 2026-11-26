#!/usr/bin/env python3
"""Strict Out-of-Sample Calibration Split for NTU VIRAL (eee_01 train -> blind eee_02/eee_03 test).

Usage:
  1. Export frozen calibration from eee_01:
       python scripts/calibrate_split.py export --train-log logs/eee_01.jsonl --out-cal configs/eee_cal.json
  2. Apply frozen calibration blind to eee_02 and eee_03:
       python scripts/calibrate_split.py apply --in-log logs/eee_02.jsonl --cal configs/eee_cal.json --out-log logs/eee_02_blind.jsonl
       python scripts/calibrate_split.py apply --in-log logs/eee_03.jsonl --cal configs/eee_cal.json --out-log logs/eee_03_blind.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_cal(train_log: Path, out_cal: Path) -> None:
    with train_log.open("r", encoding="utf-8") as f:
        meta = json.loads(f.readline())
    assert meta.get("type") == "meta", f"Missing meta header in {train_log}"
    out_cal.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "calibrated_on": train_log.name,
        "anchors": meta["anchors"],
    }
    out_cal.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported {len(meta['anchors'])} frozen anchor links from {train_log.name} -> {out_cal}")


def apply_cal(in_log: Path, cal_path: Path, out_log: Path) -> None:
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    frozen_anchors = cal["anchors"]
    allowed_peers = {int(k) for k in frozen_anchors}

    with in_log.open("r", encoding="utf-8") as fin:
        lines = [json.loads(line) for line in fin if line.strip()]

    meta = lines[0]
    assert meta.get("type") == "meta"
    meta["anchors"] = frozen_anchors
    meta["calibration_source"] = cal["calibrated_on"]

    kept = [meta]
    dropped_ranges = 0
    for ev in lines[1:]:
        if ev.get("type") == "range" and int(ev["peer"]) not in allowed_peers:
            dropped_ranges += 1
            continue
        kept.append(ev)

    out_log.parent.mkdir(parents=True, exist_ok=True)
    with out_log.open("w", encoding="utf-8") as fout:
        for obj in kept:
            fout.write(json.dumps(obj) + "\n")

    print(f"Applied {cal_path.name} blind -> {out_log} (dropped {dropped_ranges} uncalibrated range events)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Out-of-sample anchor calibration split")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export")
    p_exp.add_argument("--train-log", type=Path, required=True)
    p_exp.add_argument("--out-cal", type=Path, required=True)

    p_app = sub.add_parser("apply")
    p_app.add_argument("--in-log", type=Path, required=True)
    p_app.add_argument("--cal", type=Path, required=True)
    p_app.add_argument("--out-log", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "export":
        export_cal(args.train_log, args.out_cal)
    else:
        apply_cal(args.in_log, args.cal, args.out_log)


if __name__ == "__main__":
    main()