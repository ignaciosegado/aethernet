"""Configuration for the AETHERNET estimator.

Every noise parameter is a continuous-time *density* (or a per-measurement
standard deviation for exteroceptive sensors) so that tuning does not change
when the IMU/predict rate changes.  Units are SI throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import NormalDist


def chi2_gate(prob: float, dof: int) -> float:
    """Chi-square quantile (no SciPy dependency). Exact for dof 1, 2; Wilson-Hilferty otherwise."""
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1)")
    if dof == 1:
        z = NormalDist().inv_cdf(0.5 + prob / 2.0)
        return z * z
    if dof == 2:
        return -2.0 * math.log(1.0 - prob)
    z = NormalDist().inv_cdf(prob)
    a = 2.0 / (9.0 * dof)
    return dof * (1.0 - a + z * math.sqrt(a)) ** 3


@dataclass(frozen=True)
class EKFConfig:
    # --- IMU / process model -------------------------------------------------
    acc_noise_density: float = 0.01      # m/s^2/sqrt(Hz): white accel noise (include vibration; use Allan variance)
    bias_init_std: float = 0.04          # m/s^2: residual accel bias after static calibration
    bias_walk_density: float = 1.5e-3    # m/s^2/sqrt(s): bias random walk (also absorbs slow tilt-induced error)
    acc_unmodeled_std: float = 0.05      # m/s^2: time-correlated accel error (tilt/yaw error) ...
    acc_unmodeled_tau: float = 5.0       # s: ... and its correlation time (white-equivalent PSD = 2 sigma^2 tau)
    max_dt: float = 0.25                 # s: IMU gap beyond which covariance is inflated
    gap_acc_std: float = 3.0             # m/s^2: accel uncertainty assumed over an IMU gap

    # --- Ranging (UWB) --------------------------------------------------------
    uwb_noise_std: float = 0.12          # m: LOS ranging noise (calibrate per module pair)
    uwb_scale_std: float = 0.003         # fraction of range (distance-proportional error)
    alt_std: float = 0.5                 # m: altitude uncertainty (baro/rangefinder), per node
    peer_maneuver_std: float = 2.0       # m/s^2: peer accel uncertainty when extrapolating its packet to range time
    peer_max_age: float = 0.5            # s: drop peer states older than this
    min_range: float = 0.3               # m: ignore degenerate geometry
    anchor_extra_std: float = 0.03       # m: survey error of fixed anchors folded into R

    # --- Gating / robustness ---------------------------------------------------
    gate_prob_range: float = 0.999
    gate_prob_gnss: float = 0.999
    huber_k: float = 2.0                 # normalized-innovation level where R starts being inflated (Huber)
    nlos_ewma_beta: float = 0.85         # memory factor for per-peer sustained NLOS detection
    nlos_ewma_thresh: float = 6.0        # reject link only when sustained positive bias is unambiguous
    nlos_pos_gate_prob: float = 0.995    # single-step gate (7.88) for positive (NLOS-delayed) innovations
    gate_streak_batches: int = 5         # >= this many consecutive ranging epochs with >=50% of links gated ...
    gate_streak_inflate: float = 1.6     # ... => inflate pos/vel covariance by this factor per epoch (range evidence only)
    gate_streak_max_var: float = 400.0   # m^2 cap on position variance the inflation may reach
    consensus_min_links: int = 3         # >= this many gated links ...
    consensus_batches: int = 3           # ... explained by one common shift for this many consecutive batches => filter displaced

    # --- Cooperative fusion ----------------------------------------------------
    ranging_rate_hz: float = 10.0        # per-peer ranging rate actually achieved (set this to your hardware's rate!)
    peer_error_coherence_s: float = 20.0 # time over which a peer's position error is ~constant (repeat ranges share it)
    mobile_peer_mode: str = "kf"         # "ci": covariance intersection (consistent under unknown correlation) | "kf": naive independent
    freeze_bias_on_range: bool = False   # Schmidt-style (legacy v1 behaviour); only valid with mobile_peer_mode="kf"
    use_altitude: bool = True
    use_age: bool = True

    def validate(self) -> "EKFConfig":
        pos = ("acc_noise_density", "bias_init_std", "bias_walk_density", "acc_unmodeled_std", "acc_unmodeled_tau",
               "max_dt", "gap_acc_std", "uwb_noise_std", "alt_std", "peer_max_age", "min_range")
        for k in pos:
            if not getattr(self, k) > 0:
                raise ValueError(f"{k} must be > 0")
        if self.mobile_peer_mode not in ("ci", "kf"):
            raise ValueError("mobile_peer_mode must be 'ci' or 'kf'")
        if self.freeze_bias_on_range and self.mobile_peer_mode == "ci":
            raise ValueError("freeze_bias_on_range requires mobile_peer_mode='kf'")
        for k in ("gate_prob_range", "gate_prob_gnss", "nlos_pos_gate_prob"):
            if not 0.5 < getattr(self, k) < 1.0:
                raise ValueError(f"{k} must be in (0.5, 1)")
        if not 0.0 <= self.nlos_ewma_beta < 1.0:
            raise ValueError("nlos_ewma_beta must be in [0, 1)")
        if self.nlos_ewma_thresh <= 0.0:
            raise ValueError("nlos_ewma_thresh must be > 0")
        return self

    def with_(self, **kw) -> "EKFConfig":
        return replace(self, **kw).validate()

    @property
    def peer_kappa(self) -> float:
        """Information-decorrelation factor for mobile peers."""
        return max(1.0, self.ranging_rate_hz * self.peer_error_coherence_s)

    @property
    def q_acc(self) -> float:
        """White-equivalent accel PSD (m/s^2)^2/Hz."""
        return self.acc_noise_density ** 2 + 2.0 * self.acc_unmodeled_std ** 2 * self.acc_unmodeled_tau