```markdown
# AETHERNET Benchmark Results & Validation Methodology

This document records the quantitative evaluation of AETHERNET (`v2.2`) on the public **NTU VIRAL** aerial robotics dataset, including the out-of-sample calibration split, baseline comparisons, and semi-synthetic multi-UAV cooperative replay experiments.

---

## 1. Dataset & Out-of-Sample Calibration Protocol

### Source Dataset
All sensor logs are derived from the [NTU VIRAL Dataset](https://ntu-aris.github.io/ntu_viral_dataset/) (Nguyen et al., *IJRR* 2022), collected on a DJI M600 hexacopter equipped with a VectorNav VN-100 IMU, Humatics P440 UWB ranging transceivers (four onboard nodes ranging to three ground anchors), and a Leica Nova MS60 total station providing millimeter-level ground truth.

### Leakage-Free Calibration Split (`calibrate_split.py`)
NTU VIRAL provides raw UWB ranges between onboard antenna IDs (`200`, `201`) and ground anchor IDs (`100`, `101`, `102`), without surveyed ground-truth coordinates for the static anchors. To prevent training/test calibration leakage:
1. **Training Sequence (`eee_01`):** Virtual anchor-pair coordinates (6 links) and initial mission-frame alignment are calibrated exclusively on `eee_01` and exported to [`configs/eee_cal.json`](../configs/eee_cal.json).
2. **Blind Test Sequences (`eee_02_blind`, `eee_03_blind`):** The frozen calibration file `configs/eee_cal.json` is applied directly to `eee_02` and `eee_03` with zero access to `eee_02` or `eee_03` Leica trajectories during conversion or parameter selection.

---

## 2. Semi-Synthetic Cooperative Replay (`v2.2`)

### Methodological Scope
Because NTU VIRAL is a single-UAV dataset, multi-drone cooperative experiments replay three flights (`eee_01`, `eee_02_blind`, `eee_03_blind`) simultaneously in their shared Leica courtyard frame over their `150.8 s` common time window:
* **Physical Sensor Streams:** Each node propagates its own real VN-100 IMU accelerations (`~385 Hz`) and barometric/laser altitude, while Nodes 2 and 3 fuse their real Humatics P440 ground-anchor ranges using the blind out-of-sample calibration (`configs/eee_cal.json`).
* **Synthesized Cooperative Layer:** At `10 Hz`, Nodes 2 and 3 encode their state snapshots into 32-byte HMAC-SHA256 authenticated binary frames (`Codec`) and transmit them to Node 1 alongside synthesized peer-to-peer UWB slant ranges (`0.18 m` Gaussian measurement noise plus a `4%` Bernoulli NLOS outlier rate adding `+0.8 m` to `+2.2 m` positive bias).
* **Outage & Fault Window (`t = 35.0–110.0 s`):** For `75.0 s`, Node 1 loses **all** ground UWB anchors (`0` ground anchors) and must localize exclusively through airborne peers Node 2 and Node 3.

### Estimators Evaluated (20 Monte Carlo Seeds)
1. **`solo`:** Node 1 with `0` ground anchors and `0` peer ranges during the `35–110 s` outage (pure VN-100 IMU dead reckoning in Scenario A; unguarded spoofed GNSS ingestion in Scenario B).
2. **`uwb_wls`:** Stateless epoch-by-epoch Gauss-Newton 2D weighted least-squares trilateration using only instantaneous peer UWB ranges (no IMU propagation, no covariance memory).
3. **`kf_k1`:** Standard cooperative EKF (`mobile_peer_mode="kf"`) with coherence decorrelation disabled (`peer_error_coherence_s = 0.0`, $\kappa = 1$).
4. **`kf_k200`:** Decorrelated cooperative EKF (`mobile_peer_mode="kf"`) with `peer_error_coherence_s = 20.0` ($\kappa = 200$ at `10 Hz`).
5. **`ci` (AETHERNET):** Marginal Covariance Intersection (`mobile_peer_mode="ci"`) with coherence decorrelation and UWB-priority GNSS cross-checking.

---

## 3. Quantitative Results

### Scenario A: Zero Ground Anchors & Zero GNSS (`35–110 s`, `spoof_ramp = 0.0 m/s`)

| Method | Med Blackout RMSE | Med Blackout p95 | Med Full RMSE (`150.8 s`) | Med 2D NEES *(ideal = 1.39)* | Mean NIS *(ideal = 1.00)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`solo`** | `71.57 m` | `117.61 m` | `50.98 m` | `3.95` *(diverged)* | `1.82` |
| **`uwb_wls`** | `9.62 m` | `14.86 m` | `9.63 m` | `N/A` | `N/A` |
| **`kf_k1`** | `2.70 m` | `6.25 m` | `1.92 m` | `12.04` *(~8.7x overconfident)* | `1.78` |
| **`kf_k200`** | `3.18 m` | `6.55 m` | `2.25 m` | `6.17` *(~4.4x overconfident)* | `1.19` |
| **`ci` (AETHERNET)** | **`3.15 m`** | **`6.63 m`** | **`2.23 m`** | **`0.31`** *(conservative)* | **`1.20`** |

#### Analysis (Scenario A)
* **Inertial-Ranging Fusion vs. Stateless Trilateration:** Fusing VN-100 inertial propagation with cooperative peer ranges reduces median blackout RMSE by **67.3%** compared to stateless 2-peer UWB trilateration (`9.62 m` $\rightarrow$ `3.15 m`), which suffers from airborne baseline GDOP sensitivity and NLOS range spikes.
* **Filter Consistency (`NEES`):** For a 2-DOF position state ($\chi^2_2$), the theoretical mean NEES is `2.00` and the theoretical median NEES is $2 \ln(2) \approx 1.39$. Even with heavy coherence decorrelation ($\kappa = 200$), `kf_k200` remains substantially overconfident (`Med NEES = 6.17`), meaning its reported covariance ellipse understates true position error. Marginal CI (`ci`) keeps covariance bounds conservative (`Med NEES = 0.31`) while matching or slightly outperforming `kf_k200` in position accuracy (`3.15 m` vs. `3.18 m`).

---

### Scenario B: Zero Ground Anchors + GNSS Ramp Spoofing (`0.5 m/s` Drift During `35–110 s`)
During the `75 s` ground-anchor outage, Node 1 receives a gradual GNSS position-spoofing ramp drifting at `0.5 m/s` (`37.5 m` cumulative offset by $t = 110\text{ s}$).

| Method | Med Blackout RMSE | Med Blackout p95 | Med Full RMSE (`150.8 s`) | Med 2D NEES *(ideal = 1.39)* | Mean NIS *(ideal = 1.00)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`solo`** | `24.15 m` | `39.70 m` | `26.73 m` | `1324.80` *(diverged)* | `2.08` |
| **`uwb_wls`**\* | `32.95 m` | `14.21 m` | `24.24 m` | `N/A` | `N/A` |
| **`kf_k1`** | `4.76 m` | `10.40 m` | `3.37 m` | `35.67` *(~25.7x overconfident)* | `1.37` |
| **`kf_k200`** | `4.73 m` | `8.71 m` | `3.35 m` | `20.79` *(~15.0x overconfident)* | `1.19` |
| **`ci` (AETHERNET)** | **`4.27 m`** | **`7.86 m`** | **`3.02 m`** | **`3.56`** *(~5.8x–10x safer)* | **`1.23`** |

*\*Note on `uwb_wls`: Stateless 2-peer trilateration exhibits geometric reflection ambiguity when two airborne peers become nearly collinear with Node 1, causing occasional single-epoch mirror flips that inflate squared-mean RMSE (`32.95 m`) relative to the `14.21 m` 95th percentile.*

#### Analysis (Scenario B)
* **Anti-Spoof Arbitration & Post-Outage Recovery:** Under simultaneous `0`-anchor isolation and a `0.5 m/s` GNSS spoofing ramp, `ci` achieves the lowest error across every metric (`4.27 m` blackout RMSE, `7.86 m` p95, `3.02 m` full-flight RMSE) and cleanly re-acquires ground anchors at $t = 110\text{ s}$ without covariance lockup.
* **Consistency Under Spoofing:** `ci` reduces median 2D NEES overconfidence by **5.8×** relative to `kf_k200` (`3.56` vs. `20.79`) and **10.0×** relative to `kf_k1` (`3.56` vs. `35.67`).

---

## 4. Known Evaluation Caveats

1. **One-Way Helper Topology:** In `coop_ntu_viral.py`, Nodes 2 and 3 fuse their own ground anchors and transmit snapshots one-way to Node 1. Bidirectional multi-hop loop closures (where Node 1 also feeds back into Nodes 2 and 3) require physical multi-node RF experiments.
2. **Coherent Swarm-Wide Spoofing:** Because pairwise UWB ranges constrain relative geometry ($\text{SE}(2)$ invariants), internal peer ranging alone cannot detect a rigid translation affecting all nodes simultaneously unless at least one host or anchor retains an unspoofed reference.
3. **Physical RF Channel Effects:** Synthesized peer ranges do not capture real UWB clock skew, TDMA slot contention, or antenna polarization loss; validating these effects on Qorvo DWM3000 hardware is the focus of TRL 5 testing.