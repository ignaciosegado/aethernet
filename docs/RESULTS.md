# AETHERNET Verification & Benchmark Results

This document records quantitative validation results for `Navigator` (`v2`) across synthetic stress tests and real-world physical flight datasets.

---

## 1. Real-World Flight Hardware Replay (`NTU VIRAL`)

To evaluate `Navigator` against genuine physical sensor noise—including VectorNav VN-100 accelerometer vibration/bias, Humatics P440 UWB ranging jitter, dual-antenna body geometry, and wall-induced Non-Line-of-Sight (NLOS) multipath—three public sequences from the **NTU VIRAL** UAV dataset were replayed against centimeter-accurate Leica MS60 laser-tracker ground truth.

### Preprocessing & Evaluation Methodology (`scripts/convert_ntu_viral.py`)
1. **Mission Frame & Lever-Arm Alignment:** Ground-truth prism coordinates are transformed to the onboard IMU center using the calibrated body-frame prism offset `[-0.293656, -0.012288, -0.273095] m`, and initial IMU heading is aligned to the Leica mission frame.
2. **Dual-Antenna Virtual Link Separation:** The NTU VIRAL UAV carries two onboard UWB transceivers (`requester_id` `200` and `201`) ranging to three ground stations (`100`, `101`, `102`). Each `(requester_id, responder_id)` pair is separated into a distinct virtual anchor link (`200100`–`201102`) and calibrated via Huber-weighted Gauss-Newton least squares so alternating left/right body-frame antenna offsets do not inject step jumps into a single coordinate.
3. **Unmodified Estimator Defaults:** All three logs were replayed using default `EKFConfig` parameters (`acc_noise_density = 0.01`, `uwb_noise_std = 0.12 m`, `nlos_ewma_beta = 0.85`, `nlos_ewma_thresh = 6.0`, `nlos_pos_gate_prob = 0.995`).

### Benchmark Summary

| Sequence | Physical Environment | Active Links | Range Updates (`accepted` / `gated`) | Gate Rate | NIS Mean (`0.84–1.16` 95% band) | Position RMSE | p95 Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`eee_01`** | Outdoor courtyard / building exterior | 6 | `17,223` / `2,222` | `11.4%` | **`0.97`** (`CONSISTENT`) | **`0.51 m`** | **`0.78 m`** |
| **`nya_01`** | Indoor steel/concrete auditorium (heavy multipath) | 6 | `15,424` / `2,969` | `16.1%` | `1.50` (`OVERCONFIDENT`) | **`2.09 m`** | **`5.06 m`** |
| **`sbs_01`** | Multi-building canyon (facade reflections) | 5 | `13,187` / `1,873` | `12.4%` | `3.21` (`OVERCONFIDENT`) | **`2.94 m`** | **`7.89 m`** |

*(Note: In `sbs_01`, link `200101` exceeded the converter's `0.8 m` median calibration residual threshold due to prolonged building-corner occlusion and was excluded prior to replay, leaving 5 active anchor links.)*

---

## 2. Multipath & Ablation Analysis

Replaying the indoor (`nya_01`) and building-canyon (`sbs_01`) logs across filter configurations highlights three physical behaviors:

1. **Mission Yaw Alignment Prevents Dead-Reckoning Divergence:** Unaligned onboard magnetometer/AHRS yaw offsets (up to `47.2°` between magnetic north and the Leica total-station frame) rotate horizontal specific-force vectors into the wrong direction during maneuvers. Aligning the mission frame immediately reduced baseline error from `>20 m` down to sub-meter (`0.51 m`) in open courtyard flight and `~2–3 m` in severe multipath.
2. **Dual-Antenna Link Separation Restores Covariance Calibration:** Separating onboard antennas `200` and `201` on `eee_01` brought the Normalized Innovation Squared (NIS) mean to **`0.97`**, squarely centered inside the `[0.84, 1.16]` 95% consistency interval (`n = 300`).
3. **Asymmetric EWMA NLOS Gating Bounds Sustained Wall Reflections:**
   * In `sbs_01`, separating antennas and enabling the zero-decay positive-innovation EWMA gate reduced position RMSE from **`3.92 m` (`p95 12.24 m`)** down to **`2.94 m` (`p95 7.89 m`)**—a **25% RMSE** and **35% p95** reduction under facade reflections.
   * Because static virtual anchor coordinates absorb only the *mean* antenna offset while rotating body-frame lever arms (`±0.35 m`) and indoor wall reflections remain in the residual, `NIS mean` rises to `1.50` (`nya_01`) and `3.21` (`sbs_01`) when using the `0.12 m` simulation default for `uwb_noise_std`. Raising `uwb_noise_std` to `0.25–0.35 m` for indoor/canyon hardware deployments aligns the filter's noise floor with physical multipath spread.

---

## 3. Scope & Known Limitations

* **Single-Vehicle Anchor Fusion in Hardware Logs:** NTU VIRAL evaluates a single UAV ranging against stationary ground anchors. Multi-vehicle peer-to-peer ranging (`mobile_peer_mode="ci"`) and 32-byte authenticated telemetry exchanges are verified in the simulation test suite (`tests/`) and remain to be validated on multi-drone physical hardware.
* **2D+Altitude Formulation:** `Navigator` estimates horizontal `[px, py, vx, vy, bx, by]` states using external barometric/rangefinder altitude aiding (`set_altitude`) and externally provided attitude/yaw (`predict(..., yaw=...)`). Full 6-DoF attitude and lever-arm state estimation is intentionally excluded to keep the filter deterministic and lightweight for embedded microcontrollers.


## 4. Multi-Vehicle Cooperative Hardware Replay (`v2.2` — Zero-Anchor Isolation)

To validate cooperative peer-to-peer localization (`mobile_peer_mode="ci"`) on physical sensor streams without GNSS or ground infrastructure, three simultaneous NTU VIRAL flight sequences recorded in the shared EEE courtyard (`eee_01`, `eee_02`, `eee_03`, `150.8 s` overlap) were replayed via `scripts/coop_ntu_viral.py`:

* **Scenario:** All three UAVs operate **100% GNSS-denied**. During `t = 35 s .. 110 s` (`75 s` outage), **Node 1 (`eee_01`)** loses **all** ground UWB anchors (`0` ground anchors, `0` GNSS) and relies exclusively on its VN-100 IMU plus `10 Hz` peer-to-peer UWB slant ranges and authenticated 32-byte `Codec` telemetry frames from airborne helpers **Node 2 (`eee_02`)** and **Node 3 (`eee_03`)**.

| Fusion Mode | 0-Anchor RMSE (`35–110 s`) | 0-Anchor p95 | Full-Flight RMSE (`150.8 s`) | Mean 2D NEES *(ideal = 2.00)* | NIS Mean | Peer Updates (`acc / gated`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`solo`** (0 anchors, 0 peers) | `71.57 m` | `117.61 m` | `50.98 m` | `5.13` *(diverged)* | `1.82` | `0 / 0` |
| **`kf`** (naive cooperative EKF) | `3.18 m` | `6.65 m` | `2.25 m` | `23.53` *(11.8x overconfident)* | `1.18` | `2,891 / 1` |
| **`ci`** (AETHERNET Marginal CI) | **`3.01 m`** | **`6.83 m`** | **`2.13 m`** | **`1.87`** *(consistent)* | **`1.19`** | `2,891 / 1` |

### Key Finding
Cooperative airborne relay reduces blackout position RMSE by **95.8%** (`71.57 m` $\rightarrow$ `3.01 m`). While naive cooperative Kalman fusion (`kf`) suffers severe covariance collapse (`Mean NEES = 23.53`) due to unmodeled cross-correlation across repeated peer updates, Marginal Covariance Intersection (`ci`) preserves both sub-3.1 m accuracy and statistical covariance consistency (`Mean NEES = 1.87`).