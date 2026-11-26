"""Physically-motivated scenario generator.

Deliberately NOT the filter's own model.  Mismatch sources: body-frame bias rotating with yaw, yaw error that
mis-rotates real acceleration, time-correlated tilt error, accel scale error, 3-D geometry, altitude sensor bias,
sparse connectivity, per-link latency, bursty positive NLOS, packet loss, correlated GNSS error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SimConfig:
    n_nodes: int = 10                 # node 0 = fixed base (anchor), node 1 = relay with GNSS, 2.. = tactical
    duration: float = 120.0
    dt: float = 0.05                  # filter/IMU-aggregate rate (20 Hz)
    uwb_every: int = 2                # ranging epoch every 2 steps (10 Hz)
    uwb_range: float = 28.0
    uwb_sigma: float = 0.12
    packet_loss: float = 0.15
    max_lag_steps: int = 2            # per-link state-packet latency 0..2 steps (0..100 ms)
    nlos_onset_per_s: float = 0.03    # per link
    nlos_mean_dur_s: float = 2.0
    nlos_bias: Tuple[float, float] = (0.3, 3.0)
    spike_prob: float = 0.01
    anchors: bool = True              # base station is a surveyed anchor
    swarm_center: Tuple[float, float] = (18.0, 14.0)
    # IMU truth
    acc_density: float = 0.01         # m/s^2/sqrt(Hz)
    bias_init_std: float = 0.04
    bias_walk: float = 5e-4
    tilt_std: float = 0.05            # m/s^2 time-correlated error (about 0.3 deg of tilt)
    tilt_tau: float = 15.0
    yaw_off_std_deg: float = 2.0
    yaw_noise_deg: float = 1.0
    scale_err_std: float = 0.005
    alt_bias_std: float = 0.3
    alt_noise: float = 0.15
    gnss_std: float = 0.25            # reported std of relay fixes
    gnss_corr_std: float = 0.15       # true correlated error (filter assumes independent)
    gnss_windows: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 40.0), (80.0, 120.0)])
    gnss_every: int = 10              # 2 Hz
    # attack (relay GNSS)
    attack: Optional[dict] = None     # {"type": "jump"|"drag", "t0": s, "offset": (dx,dy) | "rate": m/s}
    imu_stall: Optional[dict] = None  # {"t0": s, "dur": s, "nodes": [..]}


class Scenario:
    def __init__(self, cfg: SimConfig, seed: int):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        N, dt = cfg.n_nodes, cfg.dt
        T = int(round(cfg.duration / dt)) + 1
        self.N, self.T = N, T
        t = np.arange(T) * dt
        self.t = t

        pos = np.zeros((N, T, 2)); vel = np.zeros((N, T, 2)); acc = np.zeros((N, T, 2))
        cph = rng.uniform(0, 2 * np.pi, 2)
        cx = cfg.swarm_center[0] + 3.0 * np.sin(0.04 * t + cph[0])
        cy = cfg.swarm_center[1] + 3.0 * np.cos(0.05 * t + cph[1])
        dcx, dcy = 3.0 * 0.04 * np.cos(0.04 * t + cph[0]), -3.0 * 0.05 * np.sin(0.05 * t + cph[1])
        ddcx, ddcy = -3.0 * 0.04**2 * np.sin(0.04 * t + cph[0]), -3.0 * 0.05**2 * np.cos(0.05 * t + cph[1])
        for i in range(1, N):
            scale = 0.5 if i == 1 else 1.0
            for ax in (0, 1):
                p = v = a = 0.0
                for _ in range(2):
                    A, w, ph = scale * rng.uniform(3, 9), rng.uniform(0.12, 0.5), rng.uniform(0, 2 * np.pi)
                    p = p + A * np.sin(w * t + ph); v = v + A * w * np.cos(w * t + ph); a = a - A * w * w * np.sin(w * t + ph)
                pos[i, :, ax], vel[i, :, ax], acc[i, :, ax] = p, v, a
            pos[i, :, 0] += cx; pos[i, :, 1] += cy
            vel[i, :, 0] += dcx; vel[i, :, 1] += dcy
            acc[i, :, 0] += ddcx; acc[i, :, 1] += ddcy
        alt = np.zeros((N, T))
        alt[0] = 1.0
        for i in range(1, N):
            alt[i] = rng.uniform(2, 12) + 1.5 * np.sin(0.2 * t + rng.uniform(0, 6.3))
        self.pos, self.vel, self.acc, self.alt = pos, vel, acc, alt
        self.yaw_true = np.arctan2(vel[..., 1], vel[..., 0])

        # ---- IMU measurements (what the AHRS hands the filter) ----
        yaw_off = rng.normal(0, np.deg2rad(cfg.yaw_off_std_deg), N)
        yaw_meas = self.yaw_true + yaw_off[:, None] + rng.normal(0, np.deg2rad(cfg.yaw_noise_deg), (N, T))
        b_body = rng.normal(0, cfg.bias_init_std, (N, 2))
        bias = np.zeros((N, T, 2))
        walk = rng.normal(0, cfg.bias_walk * math.sqrt(dt), (N, T, 2))
        for k in range(T):
            b_body = b_body + walk[:, k]
            bias[:, k] = b_body
        phi = math.exp(-dt / cfg.tilt_tau)
        tilt = np.zeros((N, T, 2)); e = rng.normal(0, cfg.tilt_std, (N, 2))
        wn = rng.normal(0, cfg.tilt_std * math.sqrt(1 - phi * phi), (N, T, 2))
        for k in range(T):
            e = phi * e + wn[:, k]
            tilt[:, k] = e
        eyaw = yaw_off[:, None]                                    # yaw error as seen by the AHRS rotation
        ce, se = np.cos(eyaw), np.sin(eyaw)
        a_scaled = acc * (1.0 + rng.normal(0, cfg.scale_err_std, (N, 1, 1)))
        a_rot = np.stack([ce * a_scaled[..., 0] - se * a_scaled[..., 1],
                          se * a_scaled[..., 0] + ce * a_scaled[..., 1]], axis=-1)       # R(e_yaw) a_true
        cy_, sy_ = np.cos(yaw_meas), np.sin(yaw_meas)
        b_world = np.stack([cy_ * bias[..., 0] - sy_ * bias[..., 1], sy_ * bias[..., 0] + cy_ * bias[..., 1]], axis=-1)
        self.acc_meas = a_rot + b_world + tilt + rng.normal(0, cfg.acc_density / math.sqrt(dt), (N, T, 2))
        self.yaw_meas = yaw_meas
        self.alt_meas = alt + rng.normal(0, cfg.alt_bias_std, (N, 1)) + rng.normal(0, cfg.alt_noise, (N, T))

        # ---- initial errors ----
        self.init_err_p = rng.normal(0, 0.4, (N, 2)); self.init_err_p[0] = 0
        self.init_err_v = rng.normal(0, 0.15, (N, 2)); self.init_err_v[0] = 0

        # ---- UWB / links ----
        self.lag = rng.integers(0, cfg.max_lag_steps + 1, (N, N))
        nl = np.zeros((N, N), bool)
        onset = cfg.nlos_onset_per_s * dt * cfg.uwb_every
        exit_p = dt * cfg.uwb_every / cfg.nlos_mean_dur_s
        nl_bias = np.zeros((N, N))
        self.links: dict = {}
        iu = np.triu_indices(N, 1)
        for s in range(cfg.uwb_every, T, cfg.uwb_every):
            d3 = np.sqrt(np.sum((pos[:, s][:, None] - pos[:, s][None]) ** 2, -1) + (alt[:, s][:, None] - alt[:, s][None]) ** 2)
            ru = rng.random(len(iu[0])); new = (~nl[iu]) & (ru < onset); stay = nl[iu] & (ru > exit_p)
            nl_new = new | stay
            nb = rng.uniform(*cfg.nlos_bias, len(iu[0]))
            cur = nl_bias[iu]
            nl_bias[iu] = np.where(new, nb, np.where(nl_new, cur, 0.0))
            nl[iu] = nl_new
            ok = d3[iu] <= cfg.uwb_range
            if not cfg.anchors:
                ok &= (iu[0] != 0)
            z = d3[iu] + rng.normal(0, cfg.uwb_sigma, len(iu[0])) + nl_bias[iu]
            sp = rng.random(len(iu[0])) < cfg.spike_prob
            z = z + sp * rng.uniform(1.0, 6.0, len(iu[0]))
            gi = rng.random(len(iu[0])) > cfg.packet_loss
            gj = rng.random(len(iu[0])) > cfg.packet_loss
            sel = ok & (gi | gj)
            self.links[s] = (iu[0][sel], iu[1][sel], z[sel], gi[sel], gj[sel])

        # ---- GNSS fixes for relay (node 1) ----
        self.gnss: dict = {}
        ge = rng.normal(0, cfg.gnss_corr_std, 2)
        gphi = math.exp(-dt * cfg.gnss_every / 20.0)
        for s in range(cfg.gnss_every, T, cfg.gnss_every):
            ge = gphi * ge + rng.normal(0, cfg.gnss_corr_std * math.sqrt(1 - gphi * gphi), 2)
            ts = s * dt
            if not any(a <= ts <= b for a, b in cfg.gnss_windows):
                continue
            fix = pos[1, s] + ge + rng.normal(0, cfg.gnss_std * 0.6, 2)
            if cfg.attack:
                at = cfg.attack
                if ts >= at["t0"]:
                    if at["type"] == "jump":
                        fix = fix + np.array(at["offset"], float)
                    elif at["type"] == "drag":
                        fix = fix + np.array(at.get("dir", (1.0, 0.0))) * at["rate"] * (ts - at["t0"])
            self.gnss[s] = fix

        # ---- IMU stall: samples for these nodes/steps never arrive ----
        self.imu_missing = np.zeros((N, T), bool)
        if cfg.imu_stall:
            a, b = int(cfg.imu_stall["t0"] / dt), int((cfg.imu_stall["t0"] + cfg.imu_stall["dur"]) / dt)
            for n in cfg.imu_stall["nodes"]:
                self.imu_missing[n, a:b] = True
