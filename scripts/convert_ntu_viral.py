import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
from aethernet.datatypes import AnchorInfo


def save_calibration(
    out_path: Path,
    anchors: Dict[int, AnchorInfo],
    yaw_offset_rad: float,
    residual_scale: float = 1.0,
) -> None:
    """Freeze calibration parameters learned on the training sequence (e.g., eee_01)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "train_sequence_only",
        "yaw_offset_rad": float(yaw_offset_rad),
        "uwb_noise_scale": float(residual_scale),
        "anchors": {
            str(k): [float(a.x), float(a.y), float(a.z), float(a.std)]
            for k, a in sorted(anchors.items())
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(cal_path: Path) -> Tuple[Dict[int, AnchorInfo], float, float]:
    """Load frozen calibration parameters for blind evaluation on test sequences (eee_02, eee_03)."""
    payload = json.loads(cal_path.read_text(encoding="utf-8"))
    anchors = {
        int(k): AnchorInfo(x=float(v[0]), y=float(v[1]), z=float(v[2]), std=float(v[3]))
        for k, v in payload["anchors"].items()
    }
    return anchors, float(payload["yaw_offset_rad"]), float(payload.get("uwb_noise_scale", 1.0))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="NTU VIRAL calibration utilities",
    )
    ap.add_argument(
        "--save-cal",
        type=Path,
        default=None,
        help="Export calibrated anchors + yaw offset to JSON",
    )
    ap.add_argument(
        "--load-cal",
        type=Path,
        default=None,
        help="Load frozen anchors + yaw offset (blind test mode)",
    )
    args = ap.parse_args()

    if args.load_cal is not None:
        anchors, yaw_offset, residual_scale = load_calibration(args.load_cal)
        print(
            f"Loaded {len(anchors)} anchors from {args.load_cal} "
            f"(yaw_offset={yaw_offset:.6f} rad, uwb_noise_scale={residual_scale:.6g})"
        )


if __name__ == "__main__":
    main()