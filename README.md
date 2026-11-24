# AETHERNET — Decentralized Cooperative Navigation in GNSS-Denied Environments

> **Status:** Active R&D Prototype (Hardware-Decoupled Core & SIL Validation Suite)  
> **Stack:** Python 3.11 · NumPy · Matplotlib · Binary Wire Protocol (`struct` + CRC-16)

A lightweight, decentralized state-estimation engine for multi-agent UAV swarms operating in GNSS-denied or RF-contested environments. Rather than relying on centralized servers, heavy computer vision, or continuous satellite fixes, **AETHERNET** fuses local high-rate **IMU kinematics**, peer-to-peer **Ultra-Wideband (UWB) ranging**, and **intermittent absolute anchor/relay updates** using a consistency-preserving Schmidt-Kalman Extended Kalman Filter (EKF).

---

## 1. Core Architectural Principles

* **Purely Relational & Decentralized:** Every node maintains its own local 6-state filter $\mathbf{x} = [p_x, p_y, v_x, v_y, b_{ax}, b_{ay}]^\top$. No central fusion server or global map is required.
* **Schmidt-Kalman Bias Decoupling:** Relative range measurements between moving peers lack full observability of local accelerometer biases $(b_{ax}, b_{ay})$. During cooperative UWB updates, the Kalman gain rows corresponding to bias states are zeroed ($K_{4:, :} = 0$), preventing range noise or collinear swarm geometries from corrupting inertial bias estimates.
* **Hierarchical Trust & Covariance Inflation:** Peer state broadcasts include their $2 \times 2$ position covariance submatrix $P_{\text{pos}}$. Receiving nodes inflate neighbor uncertainty along the line-of-sight unit vector $\mathbf{u}_{ij}$ to absorb unknown cross-correlations between agents.
* **Anti-Hijack $\chi^2$ Gating (NIS):** Both 1D UWB range innovations and 2D intermittent GNSS/anchor updates are gated via Normalized Innovation Squared (NIS) thresholds to reject Non-Line-of-Sight (NLOS) multipath spikes and GNSS spoofing, featuring an adaptive recovery window after consecutive rejections[cite: 33, 34].
* **Low-Bandwidth Binary Telemetry (26 Bytes/Packet):** Custom little-endian wire format (`AE` magic header, node ID, status flags, millisecond timestamp, float32 position, quantized covariance terms, UWB range, and CRC-16-CCITT checksum) designed for constrained UWB/RF tactical links.

---

## 2. System Dataflow

```text
+-----------------------+       +-------------------------+       +-----------------------+
|   Local IMU / AHRS    |       |  Peer UWB Broadcasts    |       |  Intermittent Anchor  |
|  (50–200 Hz Accel)    |       | (26B Binary Packet +    |       |  or Relay GNSS Fix    |
|                       |       |  Two-Way Ranging)       |       |      (1–2 Hz)         |
+-----------+-----------+       +------------+------------+       +-----------+-----------+
            |                                |                                |
            v                                v                                v
+-----------------------+       +-------------------------+       +-----------------------+
|   EKF Time Update     |       |  1D NIS Chi-Square Gate |       |  2D NIS Spoof Gate    |
|  (State & Covariance  |       |  + Covariance Inflation |       |  (Absolute Position   |
|     Propagation)      |       |  + Schmidt Bias Lock    |       |      Correction)      |
+-----------+-----------+       +------------+------------+       +-----------+-----------+
            |                                |                                |
            +--------------------------------+--------------------------------+
                                             |
                                             v
                            +---------------------------------+
                            |   Local 6-DOF Kinematic State   |
                            |  [px, py, vx, vy, bax, bay] & P |
                            +----------------+----------------+
                                             |
                                             v
                            +---------------------------------+
                            |  26-Byte CRC16 Packet Encoder   |
                            |  (Broadcast to Neighbor Nodes)  |
                            +---------------------------------+
