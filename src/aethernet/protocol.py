"""AETHERNET wire protocol v1 -- 32-byte authenticated state broadcast.

Layout (little endian, 24 B body + 8 B truncated HMAC-SHA256 tag):

  0  u8   magic/version   0xA1
  1  u8   node_id
  2  u8   flags           1 anchor | 2 gnss_trusted | 4 degraded | 8 saturated
  3  u16  seq             per-sender counter (replay protection)
  5  i32  x   [mm]        mission frame
  9  i32  y   [mm]
 13  i16  z   [cm]        altitude above mission origin
 15  i16  vx  [cm/s]
 17  i16  vy  [cm/s]
 19  u8   sigma_x code    log scale 1 mm .. 1 km, rounded UP (never over-confident)
 20  u8   sigma_y code
 21  i8   rho  (x127)     position correlation, |rho| <= 0.99
 22  u8   sigma_z code
 23  u8   sigma_v code    velocity std
 24  8 B  tag             HMAC-SHA256(key, bytes[0:24])[:8]

The old CRC-16 and the ambiguous ``range_to_target`` field are gone: the MAC subsumes integrity, and any
payload extension must be explicit.  Position uses int32 millimetres (no float32 precision loss).
Authentication is a pre-shared swarm key: it stops outsiders, NOT a compromised member (see docs/THREAT_MODEL.md).
"""
from __future__ import annotations

import hmac
import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .datatypes import AnchorInfo, NavSnapshot, RangeObs

MAGIC_VER = 0xA1
BODY_FMT = "<BBBHiihhhBBbBB"
BODY_SIZE = struct.calcsize(BODY_FMT)       # 24
TAG_SIZE = 8
PACKET_SIZE = BODY_SIZE + TAG_SIZE          # 32

FLAG_ANCHOR, FLAG_GNSS, FLAG_DEGRADED, FLAG_SATURATED = 0x01, 0x02, 0x04, 0x08

_SIG_MIN, _SIG_MAX = 1e-3, 1e3
_SIG_STEP = math.log(_SIG_MAX / _SIG_MIN) / 254.0
_COV_MARGIN = 1.02                          # decoder-side conservative inflation (covers rho/σ rounding)


def sigma_to_code(sigma: float) -> Tuple[int, bool]:
    """Encode std-dev on a log scale, rounding UP. Returns (code, saturated)."""
    if not math.isfinite(sigma) or sigma <= 0.0:
        return 255, True
    c = math.ceil(math.log(max(sigma, _SIG_MIN) / _SIG_MIN) / _SIG_STEP) + 1
    return (255, True) if c > 255 else (max(1, c), False)


def code_to_sigma(code: int) -> float:
    return _SIG_MIN * math.exp((code - 1) * _SIG_STEP)


def _sat(v: float, lo: int, hi: int) -> Tuple[int, bool]:
    r = int(round(v))
    return (hi, True) if r > hi else ((lo, True) if r < lo else (r, False))


@dataclass(frozen=True)
class StatePacket:
    node_id: int
    flags: int
    seq: int
    pos: np.ndarray
    alt: float
    vel: np.ndarray
    P_pos: np.ndarray
    alt_std: float
    vel_std: float

    @property
    def is_anchor_flag(self) -> bool:
        return bool(self.flags & FLAG_ANCHOR)

    @property
    def saturated(self) -> bool:
        return bool(self.flags & FLAG_SATURATED)


class Codec:
    """Encode/decode authenticated state packets."""

    def __init__(self, key: Optional[bytes], insecure: bool = False):
        if key is None:
            if not insecure:
                raise ValueError("a pre-shared key is required (pass insecure=True only for simulation)")
            self._key = None
        else:
            if len(key) < 16:
                raise ValueError("key must be >= 16 bytes")
            self._key = bytes(key)

    def _tag(self, body: bytes) -> bytes:
        if self._key is None:
            return b"\x00" * TAG_SIZE
        return hmac.new(self._key, body, hashlib.sha256).digest()[:TAG_SIZE]

    def encode(self, s: NavSnapshot, seq: int) -> bytes:
        sat = False
        x, a = _sat(s.pos[0] * 1000.0, -2**31, 2**31 - 1); sat |= a
        y, a = _sat(s.pos[1] * 1000.0, -2**31, 2**31 - 1); sat |= a
        z, a = _sat(s.alt * 100.0, -32768, 32767); sat |= a
        vx, a = _sat(s.vel[0] * 100.0, -32768, 32767); sat |= a
        vy, a = _sat(s.vel[1] * 100.0, -32768, 32767); sat |= a
        sx = math.sqrt(max(float(s.P_pos[0, 0]), 0.0))
        sy = math.sqrt(max(float(s.P_pos[1, 1]), 0.0))
        rho = 0.0 if sx * sy <= 0 else float(s.P_pos[0, 1]) / (sx * sy)
        rho = max(-0.99, min(0.99, rho))
        cx, a = sigma_to_code(sx); sat |= a
        cy, a = sigma_to_code(sy); sat |= a
        cz, a = sigma_to_code(s.alt_std); sat |= a
        cv, a = sigma_to_code(s.vel_std); sat |= a
        flags = ((FLAG_ANCHOR if s.is_anchor else 0) | (FLAG_GNSS if s.gnss_trusted else 0)
                 | (FLAG_DEGRADED if s.degraded else 0) | (FLAG_SATURATED if sat else 0))
        body = struct.pack(BODY_FMT, MAGIC_VER, s.node_id & 0xFF, flags, seq & 0xFFFF, x, y, z, vx, vy,
                           cx, cy, int(round(rho * 127)), cz, cv)
        return body + self._tag(body)

    def decode(self, raw: bytes) -> Optional[StatePacket]:
        if len(raw) != PACKET_SIZE:
            return None
        body, tag = raw[:BODY_SIZE], raw[BODY_SIZE:]
        if self._key is not None and not hmac.compare_digest(tag, self._tag(body)):
            return None
        (magic, node, flags, seq, x, y, z, vx, vy, cx, cy, rho, cz, cv) = struct.unpack(BODY_FMT, body)
        if magic != MAGIC_VER or 0 in (cx, cy, cz, cv):
            return None
        sx, sy = code_to_sigma(cx), code_to_sigma(cy)
        r = max(-127, rho) / 127.0
        P = np.array([[sx * sx, r * sx * sy], [r * sx * sy, sy * sy]]) * _COV_MARGIN
        return StatePacket(node, flags, seq, np.array([x, y]) / 1000.0, z / 100.0, np.array([vx, vy]) / 100.0,
                           P, code_to_sigma(cz) * math.sqrt(_COV_MARGIN), code_to_sigma(cv) * math.sqrt(_COV_MARGIN))


class ReplayWindow:
    """Per-sender monotonic sequence check with wrap-around and a reboot timeout."""

    def __init__(self, reset_after_s: float = 5.0):
        self.reset_after_s = reset_after_s
        self._last: Dict[int, Tuple[int, float]] = {}

    def accept(self, node_id: int, seq: int, t_rx: float) -> bool:
        prev = self._last.get(node_id)
        if prev is not None and (t_rx - prev[1]) <= self.reset_after_s:
            fwd = (seq - prev[0]) & 0xFFFF
            if fwd == 0 or fwd >= 0x8000:
                return False                       # duplicate / replay / reorder
        self._last[node_id] = (seq, t_rx)
        return True


class PeerTable:
    """Receive side: authenticate, de-replay, sanity-check, then turn (packet, TWR range) into a RangeObs."""

    def __init__(self, codec: Codec, anchors: Optional[Dict[int, AnchorInfo]] = None, max_speed: float = 60.0,
                 max_abs_pos: float = 5000.0, reset_after_s: float = 5.0):
        self.codec = codec
        self.anchors = dict(anchors or {})
        self.max_speed = max_speed
        self.max_abs_pos = max_abs_pos
        self._replay = ReplayWindow(reset_after_s)
        self._last: Dict[int, Tuple[StatePacket, float]] = {}
        self.rejected = {"format_or_mac": 0, "replay": 0, "implausible": 0, "saturated": 0}

    def ingest(self, raw: bytes, t_rx: float) -> Optional[StatePacket]:
        p = self.codec.decode(raw)
        if p is None:
            self.rejected["format_or_mac"] += 1
            return None
        if not self._replay.accept(p.node_id, p.seq, t_rx):
            self.rejected["replay"] += 1
            return None
        if (not np.all(np.isfinite(p.pos)) or float(np.max(np.abs(p.pos))) > self.max_abs_pos
                or float(np.linalg.norm(p.vel)) > self.max_speed):
            self.rejected["implausible"] += 1
            return None
        if p.saturated:                            # saturated covariance == "I don't know": never use as a reference
            self.rejected["saturated"] += 1
            return None
        self._last[p.node_id] = (p, t_rx)
        return p

    def observe(self, peer_id: int, range_m: float, t_range: float) -> Optional[RangeObs]:
        a = self.anchors.get(peer_id)
        if a is not None:                          # fixed anchor: position from survey table, not from the air
            return RangeObs(peer_id, range_m, np.array([a.x, a.y]), np.eye(2) * a.std ** 2, a.z, 0.05,
                            None, 0.0, 0.0, True)
        item = self._last.get(peer_id)
        if item is None:
            return None
        p, t_rx = item
        return RangeObs(peer_id, range_m, p.pos, p.P_pos, p.alt, p.alt_std, p.vel, p.vel_std,
                        t_range - t_rx, False)
