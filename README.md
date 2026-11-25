# AETHERNET

**GNSS-denied, cooperative UWB + inertial navigation stack for multi-UAV swarms.**

AETHERNET fuses continuous-time strapdown IMU propagation, barometric/rangefinder altitude aiding, surveyed stationary UWB anchors, and peer-to-peer UWB ranging over an authenticated 32-byte binary wire protocol—without requiring GNSS, visual odometry, or external motion capture during flight.

---

## Core Architecture (`v2`)

* **6-State Error-State / Extended Kalman Filter (`src/aethernet/estimator.py`):** Tracks 2-D mission-frame position, velocity, and body-frame horizontal accelerometer bias (`[px, py, vx, vy, bx, by]`) with 3-D slant-range projection against altitude aiding.
* **Continuous-Time Process Noise (`src/aethernet/config.py`):** Parameterized by physical spectral densities (`acc_noise_density`, `bias_walk_density`, `acc_unmodeled_std/tau`) so covariance propagation remains rate-independent across varying IMU frequencies and gracefully inflates across telemetry gaps.
* **Consistent Multi-Node Fusion:**
  * **Surveyed Anchors:** Classical EKF updates with survey variance folded into measurement noise.
  * **Mobile Peers:** Marginal Covariance Intersection (CI) and coherence-time information decorrelation (`peer_kappa`) to prevent cross-correlation echo-chamber overconfidence when peers exchange repeated ranges.
* **Asymmetric Per-Peer EWMA NLOS Gating & Range-Consensus Recovery:**
  * Combines Huber measurement down-weighting with an asymmetric positive-innovation gate and a per-link Exponentially Weighted Moving Average (EWMA) NIS tracker (`nlos_ewma_beta`, `nlos_ewma_thresh`) to reject sustained wall reflections while decaying cleanly back to zero on line-of-sight recovery.
  * Multi-link nonlinear least-squares consensus (`_try_consensus`) detects and recovers from genuine filter displacement using physical ranges alone—immune to GNSS spoofing.
* **Authenticated 32-Byte Wire Protocol (`src/aethernet/protocol.py`):** Compact binary state broadcast packets protected by truncated HMAC-SHA256 authentication and sliding-window anti-replay filtering.

---

## Validation Status

* **Unit & Simulation Suite (`pytest`):** Verified across deterministic Monte Carlo scenarios covering IMU dropouts, GNSS spoofing rejection, cooperative peer anchor-sharing, and sustained NLOS multipath.
* **Physical Flight Hardware Logs (`NTU VIRAL`):** Validated on real-world UAV flight logs (VectorNav VN-100 IMU, Humatics P440 UWB radios, Leica MS60 laser-tracker ground truth) across outdoor courtyard (`eee_01`), indoor auditorium (`nya_01`), and building-canyon (`sbs_01`) environments. Custom embedded flight hardware integration is in progress.

### Hardware Replay Summary (`NTU VIRAL`)

| Sequence | Environment | Accepted / Gated Ranges | NIS Mean (`0.84–1.16` 95% band) | Position RMSE | p95 Error |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`eee_01`** | Outdoor courtyard / building exterior | `17,223` / `2,222` | `0.97` (`CONSISTENT`) | **0.51 m** | **0.78 m** |
| **`nya_01`** | Indoor steel/concrete auditorium (heavy multipath) | `15,424` / `2,969` | `1.50` (unmodeled lever-arm/NLOS spread) | **2.09 m** | **5.06 m** |
| **`sbs_01`** | Multi-building canyon (facade reflections) | `13,187` / `1,873` | `3.21` (facade NLOS + partial occlusion) | **2.94 m** | **7.89 m** |

Full benchmark methodology, ablation comparisons, and hardware scope caveats are documented in [`docs/RESULTS.md`](docs/RESULTS.md).


### Key Finding
Cooperative airborne relay reduces blackout position RMSE by **95.8%** (`71.57 m` $\rightarrow$ `3.01 m`). While naive cooperative Kalman fusion (`kf`) suffers severe covariance collapse (`Mean NEES = 23.53`) due to unmodeled cross-correlation across repeated peer updates, Marginal Covariance Intersection (`ci`) preserves both sub-3.1 m accuracy and statistical covariance consistency (`Mean NEES = 1.87`).

---

## Quickstart

### Installation

```bash
pip install -e .
# Optional (for converting ROS1 NTU VIRAL .bag logs):
pip install rosbags