"""Minimal end-to-end use of the public API: three fixed anchors, two mobile nodes, authenticated packets.

Run:  PYTHONPATH=src python examples/quickstart.py
"""
import math
import numpy as np
from aethernet import AnchorInfo, Codec, EKFConfig, Navigator, PeerTable

KEY = b"change-me-per-mission-0123456789"           # >= 16 bytes, loaded out-of-band, never committed
ANCHORS = {0: AnchorInfo(0.0, 0.0, 1.0, 0.03), 10: AnchorInfo(30.0, 0.0, 1.0, 0.03), 11: AnchorInfo(0.0, 30.0, 1.0, 0.03)}  # surveyed

codec = Codec(KEY)
nodes = {i: Navigator(i, EKFConfig(), anchors=ANCHORS) for i in (1, 2)}
tables = {i: PeerTable(codec, ANCHORS) for i in (1, 2)}
rng = np.random.default_rng(0)

truth = {1: np.array([10.0, 5.0]), 2: np.array([14.0, 9.0])}
alt = {1: 4.0, 2: 6.0}
for i, n in nodes.items():
    n.initialize(truth[i] + rng.normal(0, 0.5, 2), vel_xy=[0.5, 0.2], pos_std=0.6, alt=alt[i])

seq = {1: 0, 2: 0}
for k in range(1, 201):                               # 10 s at 20 Hz
    t = k * 0.05
    for i, n in nodes.items():
        truth[i] = truth[i] + np.array([0.5, 0.2]) * 0.05                   # slow drift of both nodes
        n.predict(np.array([0.0, 0.0]) + rng.normal(0, 0.05, 2), t, yaw=0.38)   # AHRS accel (world), yaw
        n.set_altitude(alt[i] + rng.normal(0, 0.1))
    if k % 2 == 0:                                    # 10 Hz ranging epoch
        for i, n in nodes.items():
            j = 3 - i
            seq[j] += 1
            tables[i].ingest(codec.encode(nodes[j].snapshot(), seq[j]), t_rx=t)       # peer broadcast received
            obs = []
            for aid, a in ANCHORS.items():                                             # TWR to each fixed anchor
                d = math.sqrt(float(np.sum((truth[i] - [a.x, a.y]) ** 2)) + (alt[i] - a.z) ** 2) + rng.normal(0, 0.12)
                obs.append(tables[i].observe(aid, d, t))
            d_peer = math.sqrt(float(np.sum((truth[i] - truth[j]) ** 2)) + (alt[i] - alt[j]) ** 2) + rng.normal(0, 0.12)
            obs.append(tables[i].observe(j, d_peer, t))                                # TWR to peer
            n.update_ranges([o for o in obs if o is not None])

for i, n in nodes.items():
    print(f"node {i}: error {np.linalg.norm(n.x[:2] - truth[i]):.2f} m, sigma {math.sqrt(n.P[0,0] + n.P[1,1]):.2f} m, "
          f"NIS {n.nis_consistency(min_n=20)}")
