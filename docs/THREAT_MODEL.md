# Threat model

| Threat | v1 | v2 mitigation | Residual risk |
|---|---|---|---|
| Forged / modified radio packet (outsider) | Accepted: CRC-16 is not authentication | 64-bit truncated HMAC-SHA256 over the 24 B body; wrong key / any bit flip rejected (tested exhaustively) | 2^-64 forgery chance per attempt |
| Replay of old packets | Accepted | Per-sender 16-bit sequence with wrap handling, reboot timeout, local arrival timestamps | Replay inside the timeout of a *rebooted* sender's counter reset |
| Compromised swarm member (has key) | Accepted | Anchor positions come from a local surveyed table; plausibility checks (speed, bounds, saturation); covariance-weighted trust | **Not mitigated**: an insider can lie slowly. Needs per-node asymmetric signatures or physical-consistency voting. |
| GNSS jump spoofing | Gate widened to NIS<150 after 3 rejects: spoof accepted | Gate never widened; ranges to anchors keep the filter tight, so the fix stays rejected | Filter with **no** range anchoring and large P will accept a fix that fits its P |
| GNSS slow drag (meaconing) | Passes the gate | Same range anchoring bounds the drag | Unanchored node can be dragged inside its own P |
| Coherent swarm-wide spoof | Undetectable | Undetectable by ranges (invariant to rigid translation); only surveyed anchors/IMU help | Fundamental physics |
| UWB distance reduction attack | Not considered | Use 802.15.4z STS-capable hardware; gate + Huber limit damage | Hardware dependent |
| Jamming / outage | IMU gap silently dropped | Gap inflates covariance, node flagged degraded | Dead-reckoning drift (tens of m in minutes) |

Key management: one 16+ byte key per mission, loaded out-of-band. Rotate per mission. Never commit keys.
