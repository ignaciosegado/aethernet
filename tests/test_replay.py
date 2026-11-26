import json
import math
import numpy as np
from aethernet import AnchorInfo, Codec, EKFConfig, Navigator, RangeObs
from aethernet.replay import replay

KEY = b"replay-test-key-0123456789abcdef"


def test_replay_matches_live_navigator_exactly(tmp_path):
    anchors = {0: AnchorInfo(0.0, 0.0, 1.0, 0.03), 1: AnchorInfo(20.0, 0.0, 1.0, 0.03)}
    cfg = EKFConfig(mobile_peer_mode="ci")
    codec = Codec(KEY)

    # Live reference estimator (Node 3) and helper peer (Node 4)
    nav = Navigator(3, cfg, anchors=anchors)
    nav.initialize([4.0, 3.0], [0.5, 0.0], pos_std=0.5, alt=2.0)

    peer = Navigator(4, cfg)
    peer.initialize([10.0, 12.0], [0.0, 0.0], pos_std=0.4, alt=2.0)

    meta = {
        "type": "meta",
        "node_id": 3,
        "key_hex": KEY.hex(),
        "anchors": {str(k): [v.x, v.y, v.z, v.std] for k, v in anchors.items()},
        "init": {"pos": [4.0, 3.0], "vel": [0.5, 0.0], "std": 0.5, "alt": 2.0},
    }
    events = []
    true_pos = np.array([4.0, 3.0])

    for k in range(1, 51):
        t = round(k * 0.1, 4)
        true_pos = true_pos + np.array([0.5, 0.0]) * 0.1

        nav.predict(np.zeros(2), t)
        events.append({"t": t, "type": "imu", "ax": 0.0, "ay": 0.0})

        if k % 5 == 0:
            raw_pkt = codec.encode(peer.snapshot(), k // 5)
            pkt = codec.decode(raw_pkt)
            events.append({"t": t, "type": "packet", "hex": raw_pkt.hex()})

            r_anchor = math.hypot(true_pos[0] - 0.0, true_pos[1] - 0.0)
            r_anchor = math.sqrt(r_anchor**2 + (2.0 - 1.0) ** 2)
            r_peer = float(np.linalg.norm(true_pos - pkt.pos))

            nav.update_ranges([
                RangeObs(0, r_anchor, np.array([0.0, 0.0]), np.eye(2) * 0.03**2, 1.0, 0.05, is_anchor=True),
                RangeObs(4, r_peer, pkt.pos, pkt.P_pos, pkt.alt, pkt.alt_std, pkt.vel, pkt.vel_std, 0.0, False),
            ])
            events.append({"t": t, "type": "range", "peer": 0, "r": r_anchor})
            events.append({"t": t, "type": "range", "peer": 4, "r": r_peer})
            events.append({"t": t, "type": "truth", "x": float(true_pos[0]), "y": float(true_pos[1])})

    log_path = tmp_path / "log.jsonl"
    log_path.write_text("\n".join(json.dumps(x) for x in [meta] + events))

    res = replay(str(log_path), cfg=cfg, batch_window_s=1e-9)
    assert np.allclose(res["final_pos"], nav.x[:2], atol=1e-9)
    assert res["truth_rmse"] < 0.5
    assert res["rejected_packets"]["format_or_mac"] == 0