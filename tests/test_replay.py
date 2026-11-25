import json
import numpy as np
from aethernet.replay import replay
from aethernet.sim.scenario import Scenario, SimConfig
from aethernet.sim.swarms import V2Swarm, SIM_KEY
from aethernet import EKFConfig


def test_replay_reproduces_simulated_node_exactly(tmp_path):
    scen = Scenario(SimConfig(duration=25.0), seed=7)
    node = 3
    sw = V2Swarm(scen, EKFConfig(), log_node=node)
    for s in range(1, scen.T):
        sw.step(s)
    meta = {"type": "meta", "node_id": node, "key_hex": SIM_KEY.hex(), "anchors": {"0": [0.0, 0.0, 1.0, 0.03]},
            "init": {"pos": (scen.pos[node, 0] + scen.init_err_p[node]).tolist(), "vel": (scen.vel[node, 0] + scen.init_err_v[node]).tolist(),
                     "std": 0.5, "alt": float(scen.alt_meas[node, 0])}}
    p = tmp_path / "log.jsonl"
    first = {"t": 0.0, "type": "imu", "ax": 0.0, "ay": 0.0, "yaw": 0.0}          # sim calls predict(0, t=0) to start the clock
    p.write_text("\n".join(json.dumps(x) for x in [meta, first] + sw.log))
    r = replay(str(p), batch_window_s=1e-9)
    assert np.allclose(r["final_pos"], sw.nav[node].x[:2], atol=1e-9)
    assert r["nis"] is not None
