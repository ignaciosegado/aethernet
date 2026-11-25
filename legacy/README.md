Original v1 files, kept **unmodified** as the baseline the Monte Carlo suite compares against
(`aethernet.sim.swarms.V1Swarm` loads `aethernet_core_node_6.py` from here). Do not use for new work; known defects:
non-reentrant-lock deadlock on first fix, covariance clipped to 3.28 m^2 in packets, silent IMU-gap drop, gate
widening that defeats spoof rejection, rate-dependent process noise, world-frame bias, 2-D ranges, unauthenticated packets.
