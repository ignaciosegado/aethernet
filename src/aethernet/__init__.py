"""AETHERNET: decentralized cooperative navigation for GNSS-denied swarms (research prototype)."""
from .config import EKFConfig, chi2_gate
from .datatypes import AnchorInfo, Diagnostics, NavSnapshot, RangeObs, RangeReport
from .estimator import Navigator
from .initialization import multilaterate
from .protocol import PACKET_SIZE, Codec, PeerTable, ReplayWindow, StatePacket

__version__ = "2.0.0"
__all__ = ["EKFConfig", "chi2_gate", "AnchorInfo", "Diagnostics", "NavSnapshot", "RangeObs", "RangeReport",
           "Navigator", "multilaterate", "PACKET_SIZE", "Codec", "PeerTable", "ReplayWindow", "StatePacket"]
