import math
import threading
import numpy as np
import pytest
from aethernet import (AnchorInfo, EKFConfig, Navigator, RangeObs, multilaterate, chi2_gate)

ANCH = {0: AnchorInfo(0, 0, 1.0), 1: AnchorInfo(30, 0, 1.0), 2: AnchorInfo(0, 30, 1.0), 3: AnchorInfo(30, 30, 1.0)}


def anchor_obs(true_xy, alt, rng=None, sigma=0.12, bias=None):
    out = []
    for i, a in ANCH.items():
        rho = math.sqrt((true_xy[0]-a.x)**2 + (true_xy[1]-a.y)**2 + (alt-a.z)**2)
        z = rho + (0 if rng is None else rng.normal(0, sigma))
        if bias and i in bias:
            z += bias[i]
        out.append(RangeObs(i, z, np.array([a.x, a.y]), np.eye(2)*1e-3, a.z, 0.05, is_anchor=True))
    return out


def test_first_update_does_not_deadlock():          # v1 bug: non-reentrant Lock re-acquired in initialize_state
    n = Navigator(1)
    done = []
    th = threading.Thread(target=lambda: done.append(n.update_gnss(np.array([1.0, 2.0]), 0.5)))
    th.start(); th.join(timeout=2.0)
    assert done == [True] and n.initialized


def test_gap_inflates_instead_of_dropping():       # v1 bug: dt > max_dt silently skipped, P frozen
    a, b = Navigator(1), Navigator(2)
    for n in (a, b):
        n.initialize([0, 0], [1, 0], 0.5)
        n.predict(np.zeros(2), 0.0)
    a.predict(np.zeros(2), 0.1); b.predict(np.zeros(2), 2.0)      # 0.1 s step vs 2 s IMU stall
    assert b.x[0] > 1.5                                           # state advanced with held accel
    assert np.trace(b.P[:2, :2]) > np.trace(a.P[:2, :2]) * 5
    assert b.diag.gaps == 1 and b.snapshot().degraded


def test_out_of_order_imu_ignored():
    n = Navigator(1); n.initialize([0, 0], None, 0.5)
    n.predict(np.zeros(2), 1.0); n.predict(np.zeros(2), 1.1)
    P = n.P.copy(); n.predict(np.zeros(2), 1.05)
    assert np.allclose(P, n.P) and n.diag.stale_imu == 1


def test_process_noise_is_rate_independent():       # v1 bug: (sigma*dt)^2 per sample => tuning depended on IMU rate
    res = []
    for hz in (25, 100, 400):
        n = Navigator(1); n.initialize([0, 0], None, 0.0 + 1e-6, vel_std=1e-6)
        n.P[4:, 4:] *= 0
        dt = 1.0 / hz
        for k in range(int(10 * hz) + 1):
            n.predict(np.zeros(2), k * dt)
        res.append(n.P[2, 2])
    assert max(res) / min(res) < 1.02


def test_covariance_stays_psd_and_symmetric():
    rng = np.random.default_rng(1)
    n = Navigator(1); n.initialize([5, 5], [1, 1], 1.0, alt=3.0)
    for k in range(3000):
        n.predict(rng.normal(0, 1, 2), k * 0.01, yaw=0.3 * math.sin(k * 0.01))
        if k % 20 == 0:
            n.update_ranges(anchor_obs([5, 5], 3.0, rng))
    assert np.allclose(n.P, n.P.T) and np.min(np.linalg.eigvalsh(n.P)) > -1e-12


def test_body_frame_bias_jacobian():
    base = Navigator(1); base.initialize([0, 0], [0, 0], 1.0)
    base.predict(np.zeros(2), 0.0)
    yaw, dt, eps = 0.7, 0.1, 1e-3
    out = []
    for b in (np.zeros(2), np.array([eps, 0.0])):
        n = Navigator(1); n.initialize([0, 0], [0, 0], 1.0); n.x[4:6] = b
        n.predict(np.zeros(2), 0.0); n.predict(np.zeros(2), dt, yaw=yaw)
        out.append(n.x.copy())
    dpos = (out[1][:2] - out[0][:2]) / eps
    c, s = math.cos(yaw), math.sin(yaw)
    assert np.allclose(dpos, -0.5 * dt * dt * np.array([c, s]), atol=1e-9)   # bias rotates with yaw


def test_ranges_to_anchors_converge():
    rng = np.random.default_rng(2)
    n = Navigator(1, anchors=ANCH); n.initialize([14, 11], None, 3.0, alt=4.0)
    truth = np.array([12.0, 9.0])
    for k in range(80):
        n.predict(np.zeros(2), k * 0.05)
        n.update_ranges(anchor_obs(truth, 4.0, rng))
    assert np.linalg.norm(n.x[:2] - truth) < 0.1


def test_altitude_projection_matters():
    rng = np.random.default_rng(3)
    truth, alt = np.array([6.0, 5.0]), 12.0                    # 11 m above anchors: slant >> horizontal
    errs = {}
    for use in (True, False):
        n = Navigator(1, cfg=EKFConfig(use_altitude=use), anchors=ANCH); n.initialize([8, 8], None, 3.0, alt=alt)
        for k in range(150):
            n.predict(np.zeros(2), k * 0.05); n.update_ranges(anchor_obs(truth, alt, rng))
        errs[use] = np.linalg.norm(n.x[:2] - truth)
    assert errs[True] < 0.25 and errs[False] > 2 * errs[True]


def test_gnss_jump_is_never_accepted_with_anchored_filter():     # v1: accepted after 3 rejects (gate -> 150)
    rng = np.random.default_rng(4)
    n = Navigator(1, anchors=ANCH); n.initialize([12, 9], None, 1.0, alt=4.0)
    for k in range(60):
        n.predict(np.zeros(2), k * 0.05); n.update_ranges(anchor_obs([12, 9], 4.0, rng))
    accepted = 0
    for k in range(60, 400):
        n.predict(np.zeros(2), k * 0.05)
        n.update_ranges(anchor_obs([12, 9], 4.0, rng))
        if k % 10 == 0:
            accepted += n.update_gnss(np.array([42.0, 9.0]), 0.25)     # 30 m spoof jump
    assert accepted == 0 and np.linalg.norm(n.x[:2] - [12, 9]) < 0.2


def test_gnss_drag_limited_by_range_anchoring():
    rng = np.random.default_rng(5)
    n = Navigator(1, anchors=ANCH); n.initialize([12, 9], None, 1.0, alt=4.0)
    for k in range(60):
        n.predict(np.zeros(2), k * 0.05); n.update_ranges(anchor_obs([12, 9], 4.0, rng))
    for k in range(60, 1260):                                           # 60 s, spoof drifts 0.5 m/s
        n.predict(np.zeros(2), k * 0.05)
        n.update_ranges(anchor_obs([12, 9], 4.0, rng))
        if k % 10 == 0:
            n.update_gnss(np.array([12 + 0.5 * (k - 60) * 0.05, 9.0]), 0.25)
    assert np.linalg.norm(n.x[:2] - [12, 9]) < 1.0


def test_displacement_recovery_from_consensus_not_from_gate_widening():
    rng = np.random.default_rng(6)
    n = Navigator(1, anchors=ANCH); n.initialize([12, 9], None, 0.2, vel_std=0.1, alt=4.0)   # confidently wrong
    truth = np.array([24.0, 17.0])
    rec = 0
    for k in range(40):
        n.predict(np.zeros(2), k * 0.05)
        rec += n.update_ranges(anchor_obs(truth, 4.0, rng)).recovered
    assert rec == 1 and np.linalg.norm(n.x[:2] - truth) < 0.5


def test_single_nlos_link_does_not_trigger_recovery():
    rng = np.random.default_rng(7)
    n = Navigator(1, anchors=ANCH); n.initialize([12, 9], None, 0.5, alt=4.0)
    for k in range(200):
        n.predict(np.zeros(2), k * 0.05)
        n.update_ranges(anchor_obs([12, 9], 4.0, rng, bias={1: 6.0}))     # one anchor permanently NLOS
    assert n.diag.recoveries == 0 and np.linalg.norm(n.x[:2] - [12, 9]) < 0.3 and n.diag.range_gated > 100


def test_marginal_ci_never_more_confident_than_kf():
    """Same inputs: CI posterior covariance must dominate the (optimistic) independent-KF one."""
    def run(mode):
        n = Navigator(1, cfg=EKFConfig(mobile_peer_mode=mode, ranging_rate_hz=1.0, peer_error_coherence_s=1.0)); n.initialize([0, 0], None, 2.0, alt=0.0)
        o = RangeObs(9, 10.0, np.array([10.0, 0.0]), np.eye(2) * 0.5, 0.0, 0.5)
        n.update_ranges([o]); return n.P[0, 0]
    assert run("ci") >= run("kf")


def test_stale_peer_and_garbage_skipped():
    n = Navigator(1); n.initialize([0, 0], None, 1.0)
    bad = [RangeObs(5, float("nan"), np.array([5.0, 0]), np.eye(2), age_s=0.0),
           RangeObs(5, 5.0, np.array([5.0, 0]), np.eye(2), age_s=3.0),
           RangeObs(5, 5.0, np.array([0.0, 0.0]), np.eye(2), age_s=0.0)]          # zero distance
    rep = n.update_ranges(bad)
    assert rep.accepted == 0 and rep.skipped == 3


def test_multilateration():
    A = [[0, 0, 1], [30, 0, 1], [0, 30, 1], [30, 30, 1]]
    truth = np.array([11.0, 17.0]); alt = 5.0
    r = [math.sqrt((truth[0]-a[0])**2 + (truth[1]-a[1])**2 + (alt-a[2])**2) for a in A]
    p, P = multilaterate(A, r, alt, x0=[15, 15])
    assert np.linalg.norm(p - truth) < 1e-3 and np.all(np.linalg.eigvalsh(P) > 0)
    assert multilaterate([[0, 0, 1], [10, 0, 1], [20, 0, 1]], r[:3], alt) is None   # collinear anchors


def test_chi2_gate_values():
    assert abs(chi2_gate(0.999, 1) - 10.828) < 0.01 and abs(chi2_gate(0.99, 2) - 9.210) < 0.01
    assert abs(chi2_gate(0.95, 8) - 15.507) < 0.15


def test_config_validation():
    with pytest.raises(ValueError):
        EKFConfig(uwb_noise_std=-1).validate()
    with pytest.raises(ValueError):
        EKFConfig(mobile_peer_mode="x").validate()


def test_schmidt_freeze_only_with_kf():
    with pytest.raises(ValueError):
        Navigator(1, cfg=EKFConfig(freeze_bias_on_range=True, mobile_peer_mode="ci"))   # ci + freeze diverges
    Navigator(1, cfg=EKFConfig(freeze_bias_on_range=True))


def test_decorrelation_stops_averaging_down_persistent_peer_error():
    """100 ranges to ONE peer whose position is wrong by a constant: naive KF becomes confident, decorrelated does not."""
    def run(kappa_rate):
        n = Navigator(1, cfg=EKFConfig(ranging_rate_hz=kappa_rate, peer_error_coherence_s=1.0))
        n.initialize([0, 0], None, 5.0, alt=0.0)
        o = RangeObs(9, 10.0, np.array([10.0, 0.0]), np.eye(2) * 1.0, 0.0, 0.5)     # peer claims sigma 1 m
        for _ in range(100):
            n.update_ranges([o])
        return n.P[0, 0]
    assert run(100.0) > 30 * run(1.0)          # kappa=100 keeps variance near the peer's own 1 m^2; kappa=1 collapses it
