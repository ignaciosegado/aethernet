# Changelog

## 2.0.0
Fixed: first-fix deadlock (non-reentrant lock); packet covariance clipped at 3.28 m^2; IMU gaps silently dropped;
GNSS gate widened to NIS<150 after 3 rejects (accepted spoofs); UWB gate widening after 5 batches; process noise
dependent on IMU rate and missing pos/vel cross-covariance; bias modelled in world frame; slant range treated as
horizontal; peer timestamp/velocity unused; batch factor 1.6*m made information independent of neighbour count;
CRC-only packets; `logging.basicConfig` at import; overloaded return type of the batch update.
Added: altitude-aware 3-D ranges, latency compensation, surveyed-anchor table, authenticated 32 B protocol with replay
protection, range-consensus displacement recovery, NIS self-consistency monitor, multilateration bootstrap, replay
harness, Monte Carlo suite with attacks and ablations, tests, CI, docs.
Removed: `range_to_target` packet field (ambiguous), CRC-16 (subsumed by MAC), Schmidt option with CI (diverges).
Renamed: `aethernet_core_node_6.py` -> `aethernet.estimator`; versions belong in git tags, not filenames.
