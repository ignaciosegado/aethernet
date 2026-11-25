"""AETHERNET navigation filter (v2).

State  x = [px, py, vx, vy, bx, by]   (mission frame, metres; b = accel bias in the *body* horizontal frame
                                       when a yaw is supplied to ``predict``, else in the world frame)
Aiding altitude is an input (baro/rangefinder), not a state: ranges are 3-D slant ranges.

Design rules (see docs/DESIGN.md):
  * Process noise comes from continuous-time densities (rate independent, correct pos/vel cross-covariance).
  * Fixed surveyed anchors  -> classical EKF update (independent, known position).
  * Mobile peers            -> Covariance Intersection update (consistent under *unknown* cross-correlation).
  * No gate widening. Persistent inconsistency is handled by range-consensus displacement detection,
    which relies on physical ranges, so a GNSS spoofer cannot trigger it.
  * One RLock; no logging or allocation-heavy work inside the hot path beyond small numpy ops.
"""
from __future__ import annotations

import logging
import math
import threading
from collections import deque
from typing import Dict, Optional, Sequence

import numpy as np

from .config import EKFConfig, chi2_gate
from .datatypes import AnchorInfo, Diagnostics, NavSnapshot, RangeObs, RangeReport

log = logging.getLogger("aethernet")

_OMEGA = np.concatenate([np.linspace(0.04, 0.5, 24), np.linspace(0.52, 0.98, 24)])  # CI weight search grid
_I6 = np.eye(6)
_H_POS = np.zeros((2, 6))
_H_POS[0, 0] = 1.0
_H_POS[1, 1] = 1.0


class Navigator:
    def __init__(self, node_id: int, cfg: Optional[EKFConfig] = None, is_static_anchor: bool = False,
                 anchors: Optional[Dict[int, AnchorInfo]] = None):
        self.node_id = int(node_id)
        self.cfg = (cfg or EKFConfig()).validate()
        self.is_static_anchor = bool(is_static_anchor)
        self.anchors: Dict[int, AnchorInfo] = dict(anchors or {})
        self._lock = threading.RLock()

        self.x = np.zeros(6)
        self.P = np.diag([100.0, 100.0, 4.0, 4.0, self.cfg.bias_init_std ** 2, self.cfg.bias_init_std ** 2])
        self.alt = 0.0
        self.alt_std = self.cfg.alt_std
        self.initialized = False
        self.gnss_trusted = False
        self.diag = Diagnostics()
        self.nis_log: deque = deque(maxlen=300)

        self._last_t: Optional[float] = None
        self._last_acc = np.zeros(2)
        self._last_gap_t = -1e9
        self._now = 0.0
        self._gnss_consec = 0
        self._disp_votes = 0
        self._disp_last = np.zeros(2)
        self._gate_streak = 0

        c = self.cfg
        self._gate1 = chi2_gate(c.gate_prob_range, 1)
        self._gate2 = chi2_gate(c.gate_prob_gnss, 2)

    # ------------------------------------------------------------------ init
    def initialize(self, pos_xy, vel_xy=None, pos_std: float = 0.5, vel_std: Optional[float] = None,
                   alt: Optional[float] = None) -> None:
        with self._lock:
            self._init_locked(np.asarray(pos_xy, float), vel_xy, pos_std, vel_std, alt)

    def _init_locked(self, pos_xy, vel_xy, pos_std, vel_std=None, alt=None) -> None:
        self.x[:2] = pos_xy
        self.x[2:4] = 0.0 if vel_xy is None else np.asarray(vel_xy, float)
        self.x[4:6] = 0.0
        v_std = (1e-3 if self.is_static_anchor else 1.0) if vel_std is None else vel_std
        b_std = 1e-3 if self.is_static_anchor else self.cfg.bias_init_std
        self.P = np.diag([pos_std ** 2, pos_std ** 2, v_std ** 2, v_std ** 2, b_std ** 2, b_std ** 2])
        if alt is not None:
            self.alt = float(alt)
        self.initialized = True

    def set_altitude(self, alt: float, std: Optional[float] = None) -> None:
        with self._lock:
            self.alt = float(alt)
            if std is not None:
                self.alt_std = float(std)

    # --------------------------------------------------------------- predict
    def predict(self, acc_world_xy, t: float, yaw: Optional[float] = None) -> None:
        """Propagate with a gravity-compensated world-frame horizontal acceleration (from the AHRS).

        If ``yaw`` (rad, same frame as the acceleration) is given the accel bias is modelled in the body
        frame and rotated into the world frame, which is the physically correct behaviour for a turning
        vehicle.  ``t`` must be a monotonic clock (``time.monotonic()``).
        """
        with self._lock:
            self._now = t
            if not self.initialized or self.is_static_anchor:
                self._last_t = t
                return
            if self._last_t is None:
                self._last_t = t
                return
            dt = t - self._last_t
            if dt <= 0.0:
                self.diag.stale_imu += 1          # out-of-order / duplicate sample: ignore, keep clock
                return
            self._last_t = t
            a = np.asarray(acc_world_xy, float)
            gap = dt > self.cfg.max_dt
            if gap:
                self.diag.gaps += 1
                self._last_gap_t = t
                a_used = self._last_acc            # hold last known accel over the gap
            else:
                a_used = a
            self._last_acc = a

            if yaw is None:
                R = _I6[:2, :2]
            else:
                c, s = math.cos(yaw), math.sin(yaw)
                R = np.array([[c, -s], [s, c]])
            a_corr = a_used - R @ self.x[4:6]

            self.x[0:2] += self.x[2:4] * dt + 0.5 * a_corr * dt * dt
            self.x[2:4] += a_corr * dt

            F = _I6.copy()
            F[0, 2] = F[1, 3] = dt
            F[0:2, 4:6] = -0.5 * dt * dt * R
            F[2:4, 4:6] = -dt * R

            q = self.cfg.q_acc
            qb = self.cfg.bias_walk_density ** 2
            dt2, dt3 = dt * dt, dt ** 3
            Q = np.zeros((6, 6))
            for k in (0, 1):
                Q[k, k] = q * dt3 / 3.0
                Q[k + 2, k + 2] = q * dt
                Q[k, k + 2] = Q[k + 2, k] = q * dt2 / 2.0
                Q[k + 4, k + 4] = qb * dt
            if gap:                                  # unknown accel over the gap: inflate, never silently skip
                g2 = self.cfg.gap_acc_std ** 2
                for k in (0, 1):
                    Q[k, k] += g2 * dt2 * dt2 / 4.0
                    Q[k + 2, k + 2] += g2 * dt2
                    Q[k, k + 2] += g2 * dt3 / 2.0
                    Q[k + 2, k] += g2 * dt3 / 2.0
            P = F @ self.P @ F.T + Q
            self.P = 0.5 * (P + P.T)
            self.diag.predicts += 1

    # ------------------------------------------------------------- absolute
    def update_gnss(self, pos_xy, std: float) -> bool:
        """2-D absolute position (GNSS / surveyed fix). Plain NIS gate; the gate is NEVER widened.

        A fix the filter keeps rejecting stays rejected while the filter is anchored by ranges.  If the
        filter really is displaced, ``update_ranges`` detects it from physical ranges and re-centres it.
        """
        with self._lock:
            z = np.asarray(pos_xy, float)
            if not self.initialized:
                self._init_locked(z, None, std)
                self.gnss_trusted = True
                return True
            if self.is_static_anchor:
                return True
            y = z - self.x[:2]
            S = self.P[:2, :2] + np.eye(2) * std * std
            det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
            Sinv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
            nis = float(y @ Sinv @ y)
            if nis > self._gate2:
                self.diag.gnss_rejected += 1
                self._gnss_consec += 1
                self.gnss_trusted = False
                return False
            K = self.P[:, :2] @ Sinv
            self.x = self.x + K @ y
            IKH = _I6 - K @ _H_POS
            P = IKH @ self.P @ IKH.T + K @ (np.eye(2) * std * std) @ K.T
            self.P = 0.5 * (P + P.T)
            self.diag.gnss_accepted += 1
            self._gnss_consec = 0
            self.gnss_trusted = True
            self.nis_log.append(nis / 2.0)
            return True

    # --------------------------------------------------------------- ranges
    def update_ranges(self, observations: Sequence[RangeObs]) -> RangeReport:
        with self._lock:
            rep = RangeReport()
            if not self.initialized or self.is_static_anchor or not observations:
                return rep
            order = sorted(observations, key=lambda o: 0 if self._anchor_of(o) is not None else 1)
            gated_items = []
            for o in order:
                self._range_one(o, rep, gated_items)
            self.diag.range_accepted += rep.accepted
            self.diag.range_gated += rep.gated
            self.diag.range_skipped += rep.skipped
            tried = rep.accepted + rep.gated
            bad_epoch = tried >= 2 and rep.gated * 2 >= tried
            if bad_epoch:
                self._gate_streak += 1
                if len(gated_items) >= self.cfg.consensus_min_links:
                    rep.recovered = self._try_consensus(gated_items)
                if (not rep.recovered and self._gate_streak >= self.cfg.gate_streak_batches
                        and self.P[0, 0] + self.P[1, 1] < self.cfg.gate_streak_max_var):
                    f = self.cfg.gate_streak_inflate    # range-evidence-driven inflation (never triggered by GNSS)
                    self.P[:4, :4] *= f
                    self.P[:4, 4:] *= math.sqrt(f)
                    self.P[4:, :4] *= math.sqrt(f)
                    self.diag.inflations += 1
            else:
                self._gate_streak = 0
                self._disp_votes = 0
            if rep.recovered:
                self._gate_streak = 0
            return rep

    def _anchor_of(self, o: RangeObs) -> Optional[AnchorInfo]:
        return self.anchors.get(o.peer_id)

    def _range_one(self, o: RangeObs, rep: RangeReport, gated_items: list) -> None:
        c = self.cfg
        a_info = self._anchor_of(o)
        age = o.age_s if c.use_age else 0.0
        if age > c.peer_max_age or age < -c.peer_max_age or not math.isfinite(o.range_m) or o.range_m <= 0.0:
            rep.skipped += 1
            return
        if a_info is not None:                                   # surveyed anchor: trust the table, not the air
            pp = np.array([a_info.x, a_info.y])
            Pj = np.eye(2) * (a_info.std ** 2 + c.anchor_extra_std ** 2)
            pz, pz_std = a_info.z, 0.05
        else:
            pp = np.asarray(o.peer_pos, float)
            Pj = np.asarray(o.peer_P, float)
            pz, pz_std = o.peer_alt, o.peer_alt_std
            if age != 0.0 and o.peer_vel is not None:
                pp = pp + np.asarray(o.peer_vel, float) * age
                Pj = Pj + np.eye(2) * ((o.peer_vel_std * age) ** 2 + (0.5 * c.peer_maneuver_std * age * age) ** 2)
        d = self.x[:2] - pp
        dz = (self.alt - pz) if c.use_altitude else 0.0
        rho = math.sqrt(float(d @ d) + dz * dz)
        if rho < c.min_range:
            rep.skipped += 1
            return
        u = d / rho
        y = o.range_m - rho
        kappa = 1.0 if a_info is not None else c.peer_kappa
        r = c.uwb_noise_std ** 2 + (c.uwb_scale_std * rho) ** 2 + kappa * float(u @ Pj @ u)
        if c.use_altitude:
            r += (dz / rho) ** 2 * (self.alt_std ** 2 + pz_std ** 2)
        h = np.zeros(6)
        h[0], h[1] = u
        Ph = self.P @ h
        s = float(h @ Ph)
        S = s + r
        nis = y * y / S
        if nis > self._gate1:
            rep.gated += 1
            gated_items.append((pp, dz, o.range_m, r))
            return
        e = math.sqrt(nis)
        if e > c.huber_k:                                        # Huber: down-weight, don't discard
            r *= e / c.huber_k
        self.nis_log.append(nis)

        if a_info is not None or c.mobile_peer_mode == "kf":
            Pp, rr = self.P, r
        else:
            omega = self._ci_omega(Ph, s, r)
            if omega >= 1.0:
                rep.ci_no_gain += 1
                rep.accepted += 1
                return
            Pp = self.P + ((1.0 / omega) - 1.0) * np.outer(Ph, Ph) / s
            rr = r / (1.0 - omega)
        PpH = Pp @ h
        K = PpH / (float(h @ PpH) + rr)
        if c.freeze_bias_on_range:
            K[4:] = 0.0
        self.x = self.x + K * y
        IKH = _I6 - np.outer(K, h)
        P = IKH @ Pp @ IKH.T + rr * np.outer(K, K)
        self.P = 0.5 * (P + P.T)
        rep.accepted += 1

    def _ci_omega(self, Ph: np.ndarray, s: float, r: float) -> float:
        """Marginal-CI weight minimising the posterior position-variance trace.

        Full-state CI (P/w) re-inflates *every* direction on every scalar update; applied ~80x/s it blows the
        filter up (found in simulation, see docs/DESIGN.md).  Here the unknown cross-correlation is bounded only
        along the measured direction h:   P' = P + (1/w - 1) P h h' P / s ,   r' = r/(1-w).
        Resulting update:  K = P h /(s + w r/(1-w)),  P+ = P + [ (1-w)/(w s) - 1/(w^2 S') ] P h h' P,
        S' = s/w + r/(1-w).  Returns 1.0 when no w reduces the position-variance trace.
        """
        w = _OMEGA
        Sp = s / w + r / (1.0 - w)
        coef = (1.0 - w) / (w * s) - 1.0 / (w * w * Sp)
        p2 = float(Ph[0] * Ph[0] + Ph[1] * Ph[1])
        d = coef * p2                                   # change of position-variance trace (negative = gain)
        k = int(np.argmin(d))
        tr0 = float(self.P[0, 0] + self.P[1, 1])
        return float(w[k]) if d[k] < -1e-4 * tr0 else 1.0

    # ------------------------------------------------- displacement recovery
    def _try_consensus(self, items: list) -> bool:
        """>=3 gated links *jointly* explained by one position (nonlinear LS) => this filter is displaced.

        Single-link NLOS spikes are inconsistent with each other and never pass the fit test; a GNSS spoofer
        cannot trigger it either because only physical ranges are used.
        """
        c = self.cfg
        PP = np.array([it[0] for it in items])
        DZ = np.array([it[1] for it in items])
        Z = np.array([it[2] for it in items])
        W = 1.0 / np.array([it[3] for it in items])
        p = self.x[:2].copy()
        for _ in range(12):
            d = p - PP
            rho = np.maximum(np.sqrt(np.sum(d * d, axis=1) + DZ * DZ), 1e-6)
            J = d / rho[:, None]
            A = J.T @ (J * W[:, None])
            if np.linalg.cond(A) > 1e3:                      # collinear geometry: position not observable
                self._disp_votes = 0
                return False
            step = np.linalg.solve(A, J.T @ (W * (Z - rho)))
            p = p + step
            if float(np.linalg.norm(step)) < 1e-3:
                break
        d = p - PP
        rho = np.maximum(np.sqrt(np.sum(d * d, axis=1) + DZ * DZ), 1e-6)
        J = d / rho[:, None]
        res = Z - rho
        dof = len(Z) - 2
        Cd = np.linalg.inv(J.T @ (J * W[:, None]))
        if dof < 1 or float(np.sum(W * res * res)) > chi2_gate(0.999, dof):
            self._disp_votes = 0
            return False
        delta = p - self.x[:2]
        tol = 3.0 * math.sqrt(float(np.trace(Cd))) + 0.5
        if self._disp_votes > 0 and float(np.linalg.norm(delta - self._disp_last)) < tol:
            self._disp_votes += 1
        else:
            self._disp_votes = 1
        self._disp_last = delta
        if self._disp_votes < c.consensus_batches:
            return False
        self.x[:2] = p
        self.P[:2, :2] += np.outer(delta, delta) + Cd
        self.P = 0.5 * (self.P + self.P.T)
        self._disp_votes = 0
        self.diag.recoveries += 1
        log.warning("node %d: displacement recovery, shift=%s", self.node_id, np.round(delta, 2))
        return True

    # ------------------------------------------------------------ read-outs
    def nees(self, true_pos_xy) -> float:
        with self._lock:
            e = self.x[:2] - np.asarray(true_pos_xy, float)
            return float(e @ np.linalg.solve(self.P[:2, :2], e))

    def snapshot(self) -> NavSnapshot:
        with self._lock:
            degraded = (self._now - self._last_gap_t) < 2.0
            vs = math.sqrt(max(0.0, 0.5 * (self.P[2, 2] + self.P[3, 3])))
            return NavSnapshot(self.node_id, self.x[:2].copy(), self.x[2:4].copy(), self.P[:2, :2].copy(),
                               self.alt, self.alt_std, vs, self.is_static_anchor, self.gnss_trusted, degraded)

    def telemetry(self):
        s = self.snapshot()
        return s.pos, s.vel, float(math.sqrt(max(0.0, s.P_pos[0, 0] + s.P_pos[1, 1])))

    def nis_consistency(self, min_n: int = 50):
        """Mean normalized innovation squared of accepted updates (expected ~1). Needs no ground truth.
        Returns (mean, lo, hi, n); lo/hi are ~95% bounds under a Gaussian approximation. None if too few samples."""
        with self._lock:
            n = len(self.nis_log)
            if n < min_n:
                return None
            m = float(np.mean(self.nis_log))
            sd = math.sqrt(2.0 / n)
            return m, 1.0 - 1.96 * sd, 1.0 + 1.96 * sd, n

