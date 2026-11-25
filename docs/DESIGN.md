# Design notes (v2)

## State and frames
`x = [px, py, vx, vy, bx, by]` in a shared **mission frame** (metres, any origin, x/y horizontal).
Altitude is an *input* (baro/rangefinder), not a state; ranges are 3-D slant ranges, with the altitude
uncertainty of both nodes propagated into R and the Jacobian scaled by horizontal/slant geometry.
The accel input is the AHRS world-frame, gravity-compensated horizontal acceleration. If a yaw is supplied the
accel bias is a **body-frame** quantity rotated into the world each step (a world-frame constant bias is only
valid for a non-rotating vehicle). A common frame across nodes needs consistent yaw; this is a hard requirement.

## Process model
Continuous-time densities (rate independent): white accel PSD `q = density^2 + 2*sigma_u^2*tau_u`
(second term = white-equivalent of time-correlated tilt/yaw error), exact pos/vel block
`q*[[dt^3/3, dt^2/2],[dt^2/2, dt]]`, bias random-walk PSD. IMU gaps inflate covariance; they are never skipped.

## Updates
* **Fixed surveyed anchor** -> classical EKF update. Position comes from a local table, never from a packet.
* **Mobile peer** -> classical EKF update with the peer-uncertainty part of R multiplied by
  `kappa = ranging_rate * peer_error_coherence` (default 10 Hz x 20 s = 200). Rationale: repeated ranges to one
  peer inside its error-coherence time share one realisation of that peer's position error, so treating them as
  independent averages a *persistent* error down and the swarm becomes an echo chamber that drifts together while
  its covariance shrinks. This was the dominant failure found in simulation (naive: NEES ~10^4, v2: ~2-4).
  `kappa` is physically derived, but it is a **heuristic**, not a consistency proof, and it is sensitive:
  with kappa = 100 the held-out run showed median NEES 8.5 (echo-chamber episodes in some scenarios), with
  kappa >= 128 it was ~1-2 on unseen dev seeds (a cliff, see `docs/results/archive`). The default sits on the safe side
  of that cliff. Set `ranging_rate_hz` to the per-peer rate you really achieve and re-check NIS on your own logs.
* **Gating**: chi-square on normalised innovation (99.9 %), Huber down-weighting above 2 sigma. Gates are never
  widened by counters.
* **Displacement recovery** (RAIM-like): >= 3 links gated, jointly explained by one position (non-linear LS, chi-square
  fit test) for 3 consecutive epochs => re-centre. Independent NLOS spikes are mutually inconsistent and never qualify.
  Only physical ranges feed it, so a GNSS spoofer cannot trigger it.
* **Gate-streak inflation**: if >= 50 % of links are gated for >= 5 epochs and recovery is impossible, covariance is
  inflated geometrically (capped). Range evidence only.
* **GNSS**: 2-D NIS gate, no widening. A spoofed fix stays rejected while ranges anchor the filter.

## What was tried and rejected (kept honest)
| Idea | Result in simulation |
|---|---|
| Full-state Covariance Intersection per scalar range | Re-inflates every state 1/w per update (~80/s): filter diverges. |
| Marginal CI (inflate only along the measured direction) | Stable but leaks conservatism/optimism through cross-covariances; rmse ~4 m vs <1 m. Kept as `mobile_peer_mode="ci"` for ablation only. |
| Schmidt (frozen bias) with CI | Bias covariance diverges exponentially; now rejected by `EKFConfig.validate()`. |
| Batch-scaled R (v1: 1.6*m) | Makes total information independent of neighbour count; removed. |

## Known limits (do not paper over)
* Range-only networks cannot observe common translation/rotation. Without anchors or GNSS the swarm drifts as a
  rigid body; covariances grow but, in simulation, tails are still optimistic (see RESULTS.md).
* Planar filter: vertical channel is aided, not estimated. Large pitch/roll maneuvers are not modelled.
* No clock model: peer age comes from local packet arrival time, valid for UWB-latency links only.
* Pre-shared-key authentication protects against outsiders only.
