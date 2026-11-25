"""Swarm adapters: drive a set of nodes from one shared Scenario so every algorithm sees identical measurements."""
from __future__ import annotations

import importlib.util
import logging
import pathlib
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np

from ..config import EKFConfig
from ..datatypes import AnchorInfo
from ..estimator import Navigator
from ..protocol import Codec, PeerTable
from .scenario import Scenario

SIM_KEY = b"sim-swarm-key-0123456789"


class _CachedCodec(Codec):
    """Broadcast decode cache (every receiver decodes the same bytes). Pure function => safe."""

    def __init__(self, key):
        super().__init__(key)
        self._c: Dict[bytes, object] = {}

    def decode(self, raw):
        r = self._c.get(raw, 0)
        if r == 0:
            r = super().decode(raw)
            if len(self._c) > 1024:
                self._c.clear()
            self._c[raw] = r
        return r


class _Base:
    name = "base"

    def __init__(self, scen: Scenario):
        self.scen = scen
        self.pos = np.zeros((scen.N, scen.T, 2))
        self.nees = np.full((scen.N, scen.T), np.nan)
        self.ptrace = np.zeros((scen.N, scen.T))

    def record(self, s: int):
        raise NotImplementedError


class V2Swarm(_Base):
    def __init__(self, scen: Scenario, cfg: EKFConfig, update: bool = True, name: str = "v2", log_node: Optional[int] = None):
        super().__init__(scen)
        self.name, self.update, self.cfg = name, update, cfg
        self.log_node = log_node
        self.log: list = []
        N = scen.N
        sc = scen.cfg
        self.anchors = {0: AnchorInfo(0.0, 0.0, 1.0, 0.03)} if sc.anchors else {}
        self.codec = _CachedCodec(SIM_KEY)
        self.nav: List[Navigator] = []
        for i in range(N):
            is_base = (i == 0 and sc.anchors)
            nv = Navigator(i, cfg, is_static_anchor=is_base, anchors=self.anchors)
            if is_base:
                nv.initialize([0.0, 0.0], pos_std=0.03, alt=1.0)
            else:
                nv.initialize(scen.pos[i, 0] + scen.init_err_p[i], scen.vel[i, 0] + scen.init_err_v[i], 0.5,
                              alt=scen.alt_meas[i, 0])
            nv.predict(np.zeros(2), 0.0)
            self.nav.append(nv)
        self.tables = [PeerTable(self.codec, self.anchors) for _ in range(N)]
        self.hist = [deque(maxlen=sc.max_lag_steps + 2) for _ in range(N)]
        self.seq = [0] * N
        self.record(0)

    def step(self, s: int):
        sc, scen = self.scen.cfg, self.scen
        t = s * sc.dt
        for i, nv in enumerate(self.nav):
            if scen.imu_missing[i, s]:
                continue
            nv.predict(scen.acc_meas[i, s], t, float(scen.yaw_meas[i, s]))
            nv.set_altitude(float(scen.alt_meas[i, s]))
            if i == self.log_node:
                self.log.append({"t": t, "type": "imu", "ax": float(scen.acc_meas[i, s][0]), "ay": float(scen.acc_meas[i, s][1]),
                                 "yaw": float(scen.yaw_meas[i, s])})
                self.log.append({"t": t, "type": "alt", "z": float(scen.alt_meas[i, s])})
        if self.update:
            fix = scen.gnss.get(s)
            if fix is not None:
                if self.log_node == 1:
                    self.log.append({"t": t, "type": "gnss", "x": float(fix[0]), "y": float(fix[1]), "std": sc.gnss_std})
                self.nav[1].update_gnss(fix, sc.gnss_std)
            for i, nv in enumerate(self.nav):
                self.seq[i] = (self.seq[i] + 1) & 0xFFFF
                self.hist[i].append(self.codec.encode(nv.snapshot(), self.seq[i]))
            if s in scen.links:
                self._ranging(s, t)
        self.record(s)

    def _ranging(self, s, t):
        scen, sc = self.scen, self.scen.cfg
        ia, ja, za, gi, gj = scen.links[s]
        inbox: Dict[int, list] = defaultdict(list)
        for k in range(len(ia)):
            i, j, z = int(ia[k]), int(ja[k]), float(za[k])
            if gi[k]:
                self._add(inbox, i, j, z, s, t)
            if gj[k]:
                self._add(inbox, j, i, z, s, t)
        for rx, obs in inbox.items():
            self.nav[rx].update_ranges(obs)

    def _add(self, inbox, rx, peer, z, s, t):
        if self.nav[rx].is_static_anchor:
            return
        tab = self.tables[rx]
        if peer in self.anchors:
            o = tab.observe(peer, z, t)
        else:
            lag = int(self.scen.lag[rx, peer]); h = self.hist[peer]
            if len(h) <= lag:
                return
            tab.ingest(h[-1 - lag], t - lag * self.scen.cfg.dt)
            if rx == self.log_node:
                self.log.append({"t": t - lag * self.scen.cfg.dt, "type": "packet", "hex": h[-1 - lag].hex()})
            o = tab.observe(peer, z, t)
        if rx == self.log_node and o is not None:
            self.log.append({"t": t, "type": "range", "peer": peer, "r": z})
        if o is not None:
            inbox[rx].append(o)

    def record(self, s):
        for i, nv in enumerate(self.nav):
            self.pos[i, s] = nv.x[:2]
            self.nees[i, s] = nv.nees(self.scen.pos[i, s]) if not nv.is_static_anchor else np.nan
            self.ptrace[i, s] = nv.P[0, 0] + nv.P[1, 1]

    def diag(self):
        keys = ("range_accepted", "range_gated", "gnss_accepted", "gnss_rejected", "recoveries", "inflations", "gaps")
        return {k: int(sum(getattr(n.diag, k) for n in self.nav)) for k in keys}


def _load_legacy():
    p = pathlib.Path(__file__).resolve().parents[3] / "legacy" / "aethernet_core_node_6.py"
    spec = importlib.util.spec_from_file_location("aethernet_legacy_v1", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    logging.getLogger().setLevel(logging.ERROR)
    return m


class V1Swarm(_Base):
    """The original filter exactly as shipped (world-frame bias, 2-D ranges, gate widening, 26 B CRC packets)."""
    name = "v1_legacy"

    def __init__(self, scen: Scenario):
        super().__init__(scen)
        self.m = _load_legacy()
        cfg = self.m.EKFConfig()                    # shipped defaults, no per-scenario tuning
        N, sc = scen.N, scen.cfg
        self.nav = []
        for i in range(N):
            is_base = (i == 0 and sc.anchors)
            nv = self.m.AethernetNavigator(i, is_static_anchor=is_base, config=cfg)
            if is_base:
                nv.initialize_state(np.zeros(2), pos_std=0.03)
            else:
                nv.initialize_state(scen.pos[i, 0] + scen.init_err_p[i], scen.vel[i, 0] + scen.init_err_v[i], 0.5)
            nv.predict_imu(np.zeros(2), 0.0)
            self.nav.append(nv)
        self.hist = [deque(maxlen=sc.max_lag_steps + 2) for _ in range(N)]
        self.record(0)

    def step(self, s):
        sc, scen = self.scen.cfg, self.scen
        t = s * sc.dt
        for i, nv in enumerate(self.nav):
            if not scen.imu_missing[i, s]:
                nv.predict_imu(scen.acc_meas[i, s], t)
        fix = scen.gnss.get(s)
        if fix is not None:
            self.nav[1].update_gnss_or_anchor(fix, sc.gnss_std)
        for i, nv in enumerate(self.nav):
            self.hist[i].append(self.m.AethernetNavigator.unpack_state_packet(nv.pack_state_packet()))
        if s in scen.links:
            ia, ja, za, gi, gj = scen.links[s]
            inbox = defaultdict(list)
            for k in range(len(ia)):
                i, j, z = int(ia[k]), int(ja[k]), float(za[k])
                for rx, peer, got in ((i, j, gi[k]), (j, i, gj[k])):
                    if not got:
                        continue
                    lag = int(scen.lag[rx, peer]); h = self.hist[peer]
                    if len(h) <= lag or h[-1 - lag] is None:
                        continue
                    pk = h[-1 - lag]
                    inbox[rx].append((pk["pos"], pk["P_pos"], z))
            for rx, obs in inbox.items():
                self.nav[rx].update_peers_uwb_batch(obs)
        self.record(s)

    def record(self, s):
        for i, nv in enumerate(self.nav):
            self.pos[i, s] = nv.x[:2]
            self.nees[i, s] = nv.nees_2d(self.scen.pos[i, s]) if not nv.is_static_anchor else np.nan
            self.ptrace[i, s] = nv.P[0, 0] + nv.P[1, 1]

    def diag(self):
        return {"gnss_rejected_hint": int(sum(n.consecutive_gnss_rejects for n in self.nav))}
