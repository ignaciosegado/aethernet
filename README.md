# AETHERNET

Decentralized cooperative navigation for GNSS-denied UAV swarms: a 6-state EKF per node (position, velocity,
body-frame accelerometer bias), 3-D UWB ranging to fixed surveyed anchors and to peers, and an authenticated
32-byte state broadcast. **Research prototype: validated in simulation under model mismatch, not on hardware.**

## Status
| Claim | Evidence |
|---|---|
| Runs without the v1 defects (deadlock, covariance clipping, silent IMU-gap drop, gate widening, rate-dependent Q) | 36 unit/regression tests |
| Packets authenticated, replay-protected, never over-confident | exhaustive bit-flip, forgery, replay, fuzz tests |
| Accuracy and consistency under realistic mismatch | Monte Carlo, 10-20 held-out scenarios per suite: [docs/RESULTS.md](docs/RESULTS.md) |
| Works on real hardware | **Not yet.** See [docs/HARDWARE.md](docs/HARDWARE.md) |

## Headline results (20 held-out scenarios, 120 s, 8 GNSS-denied nodes, simulator != filter model)
| | RMSE (m) | relative (pairwise) RMSE (m) | median NEES (ideal 1.39) | % NEES > 13.8 (ideal 0.1 %) |
|---|---|---|---|---|
| IMU only | 130 | 176 | 0.86 | 0 % |
| v1 (original, shipped defaults) | 199 ± 123 | 273 | 474 | 91 % |
| **v2** | **1.52 ± 0.34** | **0.98 ± 0.18** | **1.40** | **6.6 %** |

Honest reading: v2's *typical* consistency is right, its *tails* are still optimistic (6.6 % vs 0.1 %).
Without any fixed anchor the swarm still holds its shape (3.6 m relative error) but drifts as a rigid body
(21 m absolute after the relay loses GNSS): range-only networks cannot observe common translation/rotation.
A GNSS drag spoof on the relay is bounded, not eliminated (worst relay error 12.9 m in 10 runs).
v1 diverges here largely because the simulator is harsher than v1's implicit model; it is a fair statement of
"v1 under realistic mismatch", not a like-for-like tuned comparison.

## Install / use
```bash
pip install -e ".[dev]"
pytest -q
python examples/quickstart.py
python -m aethernet.sim.mc --quick            # simulation suite (full: ~30 min on one core)
python -m aethernet.replay my_run.jsonl       # score a real log: prints NIS consistency verdict
```

## Layout
`src/aethernet/` estimator, protocol, config, replay, sim/ | `tests/` | `docs/` design, threat model, hardware plan, results |
`legacy/` unmodified v1 (the baseline) | `examples/`

## What changed from v1
See [CHANGELOG.md](CHANGELOG.md) and [docs/DESIGN.md](docs/DESIGN.md) (including approaches that failed).

## Known limits
Planar filter with aided altitude; shared mission frame and consistent yaw required; no clock model; pre-shared-key
authentication does not stop a compromised member; the information-decorrelation factor (`ranging_rate_hz *
peer_error_coherence_s`) is a physically-motivated heuristic with a sensitivity cliff (see DESIGN.md); results are
simulation-only. Parameters were tuned on dev seeds 0-9; reported seeds (200+) were not used for tuning.
