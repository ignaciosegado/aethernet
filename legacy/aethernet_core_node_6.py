"""
AETHERNET - Core Navigation Engine (Production / Hardware-Ready v1.1)
=====================================================================
Módulo desacoplado de hardware listo para desplegar en:
  - Companion Computer (Raspberry Pi / Jetson / Orange Pi en Python)
  - O como referencia directa para portar a C++ (STM32 / ESP32)

Características implementadas:
  1. Filtro EKF Descentralizado de 6 estados [px, py, vx, vy, bax, bay] con esquema Schmidt-Kalman.
  2. Desacoplo de sesgos IMU durante actualizaciones relativas (evita inestabilidad por geometría colineal).
  3. Gating estadístico Chi-cuadrado (NIS) con recuperación anti-secuestro contra rebotes NLOS y spoofing GNSS.
  4. Protocolo binario compacto (struct) para transmisión por radio/UWB/UDP de bajo ancho de banda (26 bytes/paquete).
  5. Bucle asíncrono orientado a eventos (IMU @ 50-200 Hz, UWB @ 10-20 Hz, Anchor @ 1-2 Hz).
"""

import time
import struct
import threading
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# --- Protocolo de Paquete de Red (26 bytes exactos - apto para enlace de baja tasa) ---
# Formato: Header(2B) | NodeID(1B) | Flags(1B) | Timestamp_ms(4B) | X(4B) | Y(4B) | Pxx(2B) | Pxy(2B) | Pyy(2B) | Range_mm(2B, uint16) | CRC16(2B)
PACKET_FORMAT = '<2sBBIffhhhHH'
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)
MAGIC_HEADER = b'AE'

FLAG_IS_ANCHOR = 0x01
FLAG_GNSS_VALID = 0x02


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


@dataclass
class EKFConfig:
    acc_noise_std: float = 0.08       # m/s^2 (calibrar con IMU real en reposo)
    bias_residual_std: float = 0.04   # m/s^2 (sesgo residual tras calibración estática)
    bias_walk_std: float = 0.0005     # m/s^2 / sqrt(s)
    uwb_noise_std: float = 0.12       # m (calibrar con módulos DW3000)
    chi2_gate_1d: float = 9.0         # Umbral nominal para rechazo de outliers UWB
    chi2_gate_2d: float = 9.21        # Umbral 99% para rechazo de spoofing GNSS
    peer_cov_inflation: float = 6.0   # Factor conservador Schmidt-EKF para covarianza de vecino
    batch_corr_factor: float = 1.60   # Factor de correlación de lote CI (mantiene NEES ~ 2)
    max_dt: float = 0.5               # Límite de seguridad si hay lag en el hilo IMU


class AethernetNavigator:
    """Núcleo matemático de navegación autónoma para un nodo individual."""

    def __init__(self, node_id: int, is_static_anchor: bool = False, config: Optional[EKFConfig] = None):
        self.node_id = node_id
        self.is_static_anchor = is_static_anchor
        self.cfg = config or EKFConfig()
        self.lock = threading.Lock()

        # Estado: [px, py, vx, vy, bax, bay]
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.diag([10.0, 10.0, 1.0, 1.0, 0.04, 0.04]).astype(np.float64)
        self.last_imu_time: Optional[float] = None
        self.gnss_healthy: bool = False
        self.initialized: bool = False
        self.consecutive_rejects: int = 0
        self.consecutive_gnss_rejects: int = 0

    def initialize_state(self, pos_xy: np.ndarray, vel_xy: Optional[np.ndarray] = None, pos_std: float = 0.5):
        with self.lock:
            self.x[:2] = pos_xy
            self.x[2:4] = vel_xy if vel_xy is not None else np.zeros(2)
            self.x[4:6] = np.zeros(2)
            v_std = 1e-3 if self.is_static_anchor else 0.2
            b_std = 1e-3 if self.is_static_anchor else self.cfg.bias_residual_std
            self.P = np.diag([pos_std**2, pos_std**2, v_std**2, v_std**2, b_std**2, b_std**2])
            self.initialized = True
            logging.info(f"Nodo {self.node_id} inicializado en {pos_xy} (Ancla={self.is_static_anchor})")

    def predict_imu(self, acc_world_xy: np.ndarray, timestamp: float):
        """Paso de predicción de alta frecuencia (50-200 Hz) alimentado por IMU + AHRS."""
        with self.lock:
            if not self.initialized or self.is_static_anchor:
                self.last_imu_time = timestamp
                return

            if self.last_imu_time is None:
                self.last_imu_time = timestamp
                return

            dt = timestamp - self.last_imu_time
            self.last_imu_time = timestamp
            if dt <= 0.0 or dt > self.cfg.max_dt:
                return

            F = np.array([
                [1, 0, dt, 0, -0.5 * dt**2, 0],
                [0, 1, 0, dt, 0, -0.5 * dt**2],
                [0, 0, 1,  0, -dt,          0],
                [0, 0, 0,  1,  0,          -dt],
                [0, 0, 0,  0,  1,           0],
                [0, 0, 0,  0,  0,           1]
            ], dtype=np.float64)

            B = np.array([
                [0.5 * dt**2, 0],
                [0, 0.5 * dt**2],
                [dt, 0],
                [0, dt],
                [0, 0],
                [0, 0]
            ], dtype=np.float64)

            acc_eff = np.sqrt(self.cfg.acc_noise_std**2 + self.cfg.bias_residual_std**2)
            Q = np.diag([
                1e-2 * dt, 1e-2 * dt,
                (acc_eff * dt)**2, (acc_eff * dt)**2,
                (self.cfg.bias_walk_std**2) * dt, (self.cfg.bias_walk_std**2) * dt
            ])

            self.x = F @ self.x + B @ acc_world_xy
            self.P = F @ self.P @ F.T + Q

    def update_gnss_or_anchor(self, pos_xy: np.ndarray, pos_std: float) -> bool:
        """
        Fusión de referencia absoluta (GNSS o estación base).
        Incluye test NIS 2D con recuperación anti-secuestro para detectar jamming/spoofing
        sin bloquearse permanentemente tras un salto legítimo de covarianza.
        """
        with self.lock:
            if not self.initialized:
                self.initialize_state(pos_xy, pos_std=pos_std)
                self.gnss_healthy = True
                return True

            H = np.zeros((2, 6))
            H[0, 0] = 1.0
            H[1, 1] = 1.0
            R = np.eye(2) * (pos_std**2)
            y = pos_xy - self.x[:2]

            S = H @ self.P @ H.T + R
            nis = float(y.T @ np.linalg.inv(S) @ y)
            gate_2d = self.cfg.chi2_gate_2d if self.consecutive_gnss_rejects < 3 else 150.0

            if nis > gate_2d:
                self.gnss_healthy = False
                self.consecutive_gnss_rejects += 1
                logging.warning(f"Nodo {self.node_id}: Anomalía GNSS/Spoofing detectada (NIS={nis:.2f}). Ignorando.")
                return False

            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            I_KH = np.eye(6) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            self.gnss_healthy = True
            self.consecutive_gnss_rejects = 0
            self.consecutive_rejects = 0
            return True

    def update_peers_uwb_batch(
        self,
        peer_observations: List[Tuple[np.ndarray, np.ndarray, float]],
        return_rejected: bool = False
    ) -> int:
        """
        Fusión cooperativa por lotes de todos los vecinos recibidos en la ventana actual.
        Aplica esquema Schmidt-Kalman (bloquea actualización de sesgo en medidas relativas)
        y gating Chi-cuadrado con recuperación anti-secuestro.
        """
        with self.lock:
            if not self.initialized or self.is_static_anchor or len(peer_observations) == 0:
                return 0

            valid_H, valid_y, valid_R = [], [], []
            rejected = 0
            chi2_thresh = self.cfg.chi2_gate_1d if self.consecutive_rejects < 5 else 25.0

            for (peer_pos, peer_P_pos, measured_range_m) in peer_observations:
                diff = self.x[:2] - peer_pos
                pred_dist = float(np.linalg.norm(diff))
                if pred_dist < 1e-3:
                    continue

                u_ij = diff / pred_dist
                H_row = np.zeros(6)
                H_row[0] = u_ij[0]
                H_row[1] = u_ij[1]

                r_peer = float(u_ij @ peer_P_pos @ u_ij.T)
                r_eff = (self.cfg.uwb_noise_std**2) + self.cfg.peer_cov_inflation * r_peer

                innov = measured_range_m - pred_dist
                S_1d = float(H_row @ self.P @ H_row.T) + r_eff
                nis = (innov**2) / S_1d

                if nis <= chi2_thresh:
                    valid_H.append(H_row)
                    valid_y.append(innov)
                    valid_R.append(r_eff)
                else:
                    rejected += 1

            m = len(valid_H)
            if m == 0:
                self.consecutive_rejects += 1
                return rejected if return_rejected else 0

            self.consecutive_rejects = 0
            H = np.array(valid_H)
            y = np.array(valid_y)
            R = np.diag(valid_R) * max(1.0, self.cfg.batch_corr_factor * m)

            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            K[4:, :] = 0.0  # Schmidt-Kalman: no perturbar sesgos IMU con rangos relativos

            self.x = self.x + K @ y
            I_KH = np.eye(6) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            return rejected if return_rejected else m

    def nees_2d(self, true_pos_xy: np.ndarray) -> float:
        """Calcula Normalized Estimation Error Squared (NEES) 2D para validación estadística."""
        with self.lock:
            err = self.x[:2] - true_pos_xy
            return float(err.T @ np.linalg.inv(self.P[:2, :2]) @ err)

    def pack_state_packet(self, range_to_target_m: float = 0.0) -> bytes:
        """Serializa estado y covarianza 2x2 en un paquete binario de 26 bytes."""
        with self.lock:
            flags = (FLAG_IS_ANCHOR if self.is_static_anchor else 0) | (FLAG_GNSS_VALID if self.gnss_healthy else 0)
            ts_ms = int(time.time() * 1000) & 0xFFFFFFFF
            pxx = int(np.clip(self.P[0, 0] * 10000, -32767, 32767))
            pxy = int(np.clip(self.P[0, 1] * 10000, -32767, 32767))
            pyy = int(np.clip(self.P[1, 1] * 10000, -32767, 32767))
            # uint16 (0..65535 mm = hasta 65.535 m, cubre alcance completo DW3000)
            rng_mm = int(np.clip(range_to_target_m * 1000, 0, 65535))

            payload = struct.pack(
                '<2sBBIffhhhH',
                MAGIC_HEADER, self.node_id, flags, ts_ms,
                float(self.x[0]), float(self.x[1]),
                pxx, pxy, pyy, rng_mm
            )
            checksum = crc16_ccitt(payload)
            return payload + struct.pack('<H', checksum)

    @staticmethod
    def unpack_state_packet(raw: bytes) -> Optional[Dict]:
        """Deserializa y valida integridad CRC16 de un paquete entrante."""
        if len(raw) != PACKET_SIZE:
            return None
        header, node_id, flags, ts_ms, px, py, pxx, pxy, pyy, rng_mm, rx_crc = struct.unpack(PACKET_FORMAT, raw)
        if header != MAGIC_HEADER:
            return None
        if crc16_ccitt(raw[:-2]) != rx_crc:
            return None

        P_pos = np.array([
            [pxx / 10000.0, pxy / 10000.0],
            [pxy / 10000.0, pyy / 10000.0]
        ], dtype=np.float64)

        return {
            'node_id': node_id,
            'is_anchor': bool(flags & FLAG_IS_ANCHOR),
            'gnss_valid': bool(flags & FLAG_GNSS_VALID),
            'timestamp_ms': ts_ms,
            'pos': np.array([px, py], dtype=np.float64),
            'P_pos': P_pos,
            'range_m': rng_mm / 1000.0
        }

    def get_telemetry(self) -> Tuple[np.ndarray, np.ndarray, float]:
        with self.lock:
            std_2d = float(np.sqrt(self.P[0, 0] + self.P[1, 1]))
            return self.x[:2].copy(), self.x[2:4].copy(), std_2d


# --- Ejemplo de Ejecución en Modo Nodo Real ---
if __name__ == "__main__":
    base_node = AethernetNavigator(node_id=0, is_static_anchor=True)
    base_node.initialize_state(np.array([0.0, 0.0]), pos_std=0.05)

    drone_node = AethernetNavigator(node_id=2, is_static_anchor=False)
    drone_node.initialize_state(np.array([5.0, 0.0]), vel_xy=np.array([0.0, 1.0]), pos_std=0.4)

    tx_packet = base_node.pack_state_packet(range_to_target_m=34.85)
    assert len(tx_packet) == 26, "Tamaño de paquete incorrecto"

    t0 = time.time()
    drone_node.predict_imu(np.array([0.0, 0.0]), t0)
    drone_node.predict_imu(np.array([0.0, 0.0]), t0 + 0.1)

    parsed = AethernetNavigator.unpack_state_packet(tx_packet)
    assert parsed is not None and abs(parsed['range_m'] - 34.85) < 1e-3, "Error de escalado uint16 en rango"

    accepted_cnt = drone_node.update_peers_uwb_batch([(parsed['pos'], parsed['P_pos'], 5.02)])
    pos, vel, uncertainty = drone_node.get_telemetry()
    logging.info(f"Fusión UWB aceptada={accepted_cnt} | Posición={pos.round(3)} | Incertidumbre 1-sigma={uncertainty:.3f} m")
