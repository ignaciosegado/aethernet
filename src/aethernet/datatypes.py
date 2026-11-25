from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AnchorInfo:
    """Surveyed fixed anchor. Position comes from the local table, never from the air."""
    x: float
    y: float
    z: float = 0.0
    std: float = 0.03


@dataclass(frozen=True)
class RangeObs:
    """One two-way-ranging measurement plus the peer state it refers to.

    age_s = (time the range was measured) - (time the peer state is valid). Computed locally from
    packet arrival timestamps; never needs clock sync between nodes.
    """
    peer_id: int
    range_m: float
    peer_pos: np.ndarray            # (2,) m, mission frame
    peer_P: np.ndarray              # (2,2) m^2
    peer_alt: float = 0.0
    peer_alt_std: float = 0.5
    peer_vel: Optional[np.ndarray] = None
    peer_vel_std: float = 1.0
    age_s: float = 0.0
    is_anchor: bool = False


@dataclass(frozen=True)
class NavSnapshot:
    """What a node publishes about itself."""
    node_id: int
    pos: np.ndarray
    vel: np.ndarray
    P_pos: np.ndarray
    alt: float
    alt_std: float
    vel_std: float
    is_anchor: bool = False
    gnss_trusted: bool = False
    degraded: bool = False


@dataclass
class RangeReport:
    accepted: int = 0
    gated: int = 0
    skipped: int = 0
    ci_no_gain: int = 0
    recovered: bool = False


@dataclass
class Diagnostics:
    predicts: int = 0
    gaps: int = 0
    stale_imu: int = 0
    gnss_accepted: int = 0
    gnss_rejected: int = 0
    range_accepted: int = 0
    range_gated: int = 0
    range_skipped: int = 0
    recoveries: int = 0
    inflations: int = 0
