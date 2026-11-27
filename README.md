# AETHERNET

**Cooperative Relative Navigation & Anti-Spoofing Fusion SDK for Multi-Agent Operations**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)

AETHERNET is a lightweight state-estimation and wire-framing library designed to extend a trusted coordinate frame from a **mobile or local host** (vessel, USV, ground vehicle, or lead relay node) across a multi-drone team when GNSS is jammed or spoofed and visual odometry degrades due to darkness, sea spray, dust, or structural occlusion.

Rather than competing with infrastructure-free visual-inertial odometry (VIO) in open-sky flight, AETHERNET acts as a **cooperative ranging-fusion and anti-spoofing layer** that fuses inertial propagation with peer-to-peer UWB slant ranges—allowing occluded relay nodes with **zero direct host line-of-sight** and **zero GNSS** to maintain bounded relative positioning without covariance collapse.

---

## Core Capabilities

* **Marginal Covariance Intersection (`Navigator`, `mobile_peer_mode="ci"`):** Optimizes the convex fusion weight $\omega \in (0, 1)$ to minimize the trace of the fused covariance matrix, alongside coherence-time decorrelation (`peer_error_coherence_s`), when fusing uncertain mobile-peer positions without double-counting correlated trajectory errors.
* **Asymmetric Multipath & Spoofing Defense:** Combines per-peer EWMA NLOS gating (`nlos_ewma_beta`, `nlos_ewma_thresh`) against positive range bias with geometric range-consensus cross-checks (`_try_consensus` / Mahalanobis arbitration) to reject gradual GNSS position-ramp spoofing.
* **32-Byte Authenticated Binary Protocol (`Codec`):** Packs node position, velocity, altitude, uncertainty bounds, sequence counter, and truncated HMAC-SHA256 authentication into 32 bytes (`~2.5 kbps` at `10 Hz`) for low-bandwidth RF and telemetry links.
* **Drop-In Flight-Stack Integration:** Outputs covariance-calibrated state snapshots compatible with standard MAVLink external vision aiding (`VISION_POSITION_ESTIMATE` / `ODOMETRY`) on PX4 and ArduPilot without requiring hardware modifications to host mast infrastructure.

---

## Repository Layout

```text
aethernet/
├── .github/workflows/       # Automated unit and filter-consistency CI
├── configs/
│   └── eee_cal.json         # Frozen out-of-sample anchor calibration learned on eee_01
├── data/                    # Place downloaded NTU VIRAL .bag files here (git-ignored)
├── docs/
│   └── RESULTS.md           # Full out-of-sample & 20-seed benchmark methodology and tables
├── examples/                # Minimal runnable estimation and wire-codec examples
├── logs/                    # Generated .jsonl replay logs (git-ignored)
├── scripts/
│   ├── calibrate_split.py   # Strict train/test calibration split (eee_01 -> blind eee_02/03)
│   ├── convert_ntu_viral.py # ROS1 bag -> JSONL event stream converter
│   ├── coop_ntu_viral.py    # Multi-seed semi-synthetic cooperative replay benchmark
│   └── replay.py            # Single-node physical sensor replay evaluator
├── src/aethernet/           # Core Navigator, Codec, EKFConfig, and data structures
├── tests/                   # Unit, wire-codec, and covariance-consistency tests
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE                  # Apache License 2.0
├── NOTICE                   # Trademark and third-party attribution notice
└── pyproject.toml