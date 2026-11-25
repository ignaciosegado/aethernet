# From simulation to hardware

Nothing in this repository has been validated on hardware yet. This is the shortest path to evidence.

## 1. Minimal rig (about 3 nodes)
* UWB: DW3000-class modules with 802.15.4z STS (e.g. Qorvo DWM3000 / DWM3001C boards). Fixed anchor at a surveyed point.
* IMU+AHRS: use the flight controller's EKF output (PX4 `vehicle_local_position` / ArduPilot `LOCAL_POSITION_NED`):
  world acceleration (gravity removed), yaw, baro/rangefinder altitude. Log at >= 50 Hz.
* Compute: any Linux companion (Pi 4 / Jetson / Orange Pi). The filter costs ~0.2 ms per update in Python.
* Ground truth: **RTK GNSS** (cm-level) on every node, logged but *never fed to the filter*. This emulates GNSS denial
  in open sky while giving real truth. Alternatives: total station, mocap, tape-measured track.

## 2. Calibrations (do these first, they set the config)
1. UWB antenna delay and range bias: static ranges at 1, 2, 5, 10, 20 m, >= 1000 samples each. Fit bias(d); set
   `uwb_noise_std` to the residual std, `uwb_scale_std` to the slope error.
2. IMU Allan variance (>= 2 h static, vibration-isolated, then with motors running) -> `acc_noise_density`
   (use the *motors-on* value), `bias_walk_density`, `bias_init_std` (after a static calibration).
3. Altitude sensor noise -> `alt_std`.
4. Per-peer ranging rate you really achieve -> `ranging_rate_hz`. Coherence of peer error (try 5-20 s) ->
   `peer_error_coherence_s`.

## 3. Staged validation (stop at the first failed gate)
| Stage | Setup | Pass criterion |
|---|---|---|
| A | Bench, static, 3 nodes | Range error std within 1.5x spec; NIS mean in 95 % band |
| B | Walking, handheld, anchor + 2 nodes, RTK truth | NIS consistent; RMSE logged; no lockouts |
| C | Ground robot/cart, 4 nodes, forced NLOS (people, metal) | Gating rate sane; no divergence; recovery works |
| D | Tethered / short drone hops | Latency compensation verified (toggle `use_age`) |
| E | Free flight, GNSS intentionally ignored | Compare against RTK; publish RMSE + NIS plots |

## 4. Using real logs
Write the JSONL format described in `aethernet/replay.py`, then
`python -m aethernet.replay run.jsonl --out traj.csv`. The first check on any real log is the NIS line:
mean well above the printed band = overconfident (raise noise densities / coherence), well below = conservative.
Retune on one half of your logs, report on the other half.

## 5. Porting
Reference is NumPy float64. Before porting to float32 MCUs: keep Joseph-form updates, symmetrise P each step,
re-run the test-suite numerics in float32, and consider UD/square-root factorisation. The packet is fixed-point and
trivial to implement in C (`protocol.py` layout).
