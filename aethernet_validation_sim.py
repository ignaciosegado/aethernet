"""
AETHERNET - Suite de Simulación y Validación (SIL - Software-in-the-Loop)
=========================================================================
Arquitectura: EKF Cooperativo Descentralizado Jerárquico (Schmidt-EKF / Trust-Weighted)
              + Protección anti-secuestro (Dynamic Chi-Square Gating) contra rebotes NLOS.
Escenario: 1 Estación Base fija (coordenadas conocidas) + 1 Nodo Relé (GNSS intermitente)
           + 8 Nodos Tácticos en zona 100% GNSS-Denied.

Métricas validadas:
1. Error RMSE 2D (m) de los nodos GNSS-denied a lo largo de 120 s.
2. Consistencia estadística mediante NEES (Normalized Estimation Error Squared ~ 2.0).
3. Rechazo automático de outliers UWB (NLOS / Multipath) mediante gating Chi-cuadrado.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- 1. Parámetros Físicos y de Escenario ---
DT = 0.1                  # 10 Hz frecuencia de predicción IMU
SIM_TIME = 120.0          # 120 segundos de misión
STEPS = int(SIM_TIME / DT)

NUM_NODES = 10            # Nodo 0: Base fija | Nodo 1: Relé | Nodos 2..9: GNSS-Denied
MAX_UWB_RANGE = 35.0      # Alcance efectivo enlace UWB (m)
PACKET_LOSS_PROB = 0.15   # 15% pérdida de paquetes en enlace táctico
NLOS_OUTLIER_PROB = 0.05  # 5% de lecturas UWB con rebote NLOS (+3 a +8 m)

# Ruido de sensores (basado en hardware comercial calibrado: BMI088 + DW3000)
IMU_ACC_NOISE = 0.08      # m/s^2
IMU_BIAS_INIT = 0.04      # m/s^2 (sesgo residual tras calibración estática)
IMU_BIAS_WALK = 0.0005    # m/s^2 / sqrt(step)
UWB_NOISE = 0.12          # m (ruido LOS)
ANCHOR_POS_NOISE = 0.25   # m (precisión estación base / relé)
ANCHOR_PERIOD = 5         # Actualización absoluta cada 0.5 s (2 Hz)

# --- 2. Generación de Trayectorias Reales (Ground Truth) ---
t = np.arange(STEPS) * DT
true_pos = np.zeros((NUM_NODES, STEPS, 2))
true_vel = np.zeros((NUM_NODES, STEPS, 2))
true_acc = np.zeros((NUM_NODES, STEPS, 2))
true_bias = np.random.randn(NUM_NODES, 2) * IMU_BIAS_INIT

# Nodo 0: Estación Base fija en (0, 0)
true_pos[0, :, :] = np.array([0.0, 0.0])

# Nodos 1..9: Enjambre maniobrando en formación dinámica cerca de la base
for i in range(1, NUM_NODES):
    phase = (i - 1) * (2 * np.pi / (NUM_NODES - 1))
    r_orbit = 7.0 + 2.5 * (i % 2)
    omega = 0.15
    cx, cy = 10.0 + 3.0 * np.sin(0.05 * t), 8.0 + 3.0 * np.cos(0.05 * t)
    
    true_pos[i, :, 0] = cx + r_orbit * np.cos(omega * t + phase)
    true_pos[i, :, 1] = cy + r_orbit * np.sin(omega * t + phase)
    
    dcx = 3.0 * 0.05 * np.cos(0.05 * t)
    dcy = -3.0 * 0.05 * np.sin(0.05 * t)
    ddcx = -3.0 * (0.05**2) * np.sin(0.05 * t)
    ddcy = -3.0 * (0.05**2) * np.cos(0.05 * t)
    
    true_vel[i, :, 0] = dcx - r_orbit * omega * np.sin(omega * t + phase)
    true_vel[i, :, 1] = dcy + r_orbit * omega * np.cos(omega * t + phase)
    
    true_acc[i, :, 0] = ddcx - r_orbit * (omega**2) * np.cos(omega * t + phase)
    true_acc[i, :, 1] = ddcy - r_orbit * (omega**2) * np.sin(omega * t + phase)

# --- 3. Estimador Descentralizado por Nodo (EKF Cooperativo Consistente) ---
class DecentralizedNodeEKF:
    """
    Filtro de navegación local de 6 estados: [px, py, vx, vy, bax, bay].
    Implementa:
    - Predicción cinemática IMU.
    - Fusión cooperativa por lotes estilo Schmidt-Kalman:
      * Bloquea el cross-talk sobre los estados de sesgo (b_ax, b_ay) durante actualizaciones
        relativas UWB para evitar divergencia por falta de observabilidad.
      * Pondera la información de cada vecino según su jerarquía de confianza (traza de P_pos).
    - Gating estadístico Chi-cuadrado con recuperación anti-secuestro.
    """
    def __init__(self, node_id, init_pos, init_vel, is_static_base=False):
        self.id = node_id
        self.is_static_base = is_static_base
        self.x = np.hstack([init_pos, init_vel, [0.0, 0.0]])
        self.consecutive_rejects = 0
        
        if is_static_base:
            self.P = np.diag([1e-3, 1e-3, 1e-6, 1e-6, 1e-6, 1e-6])
        else:
            self.P = np.diag([0.5**2, 0.5**2, 0.2**2, 0.2**2, 0.04**2, 0.04**2])

        self.F = np.array([
            [1, 0, DT, 0, -0.5*DT**2, 0],
            [0, 1, 0, DT, 0, -0.5*DT**2],
            [0, 0, 1,  0, -DT,        0],
            [0, 0, 0,  1,  0,        -DT],
            [0, 0, 0,  0,  1,         0],
            [0, 0, 0,  0,  0,         1]
        ])
        self.B = np.array([
            [0.5*DT**2, 0],
            [0, 0.5*DT**2],
            [DT, 0],
            [0, DT],
            [0, 0],
            [0, 0]
        ])
        # Ruido de proceso Q dimensionado para absorber el sesgo residual no observado
        acc_eff_noise = np.sqrt(IMU_ACC_NOISE**2 + IMU_BIAS_INIT**2)
        self.Q = np.diag([
            1e-3, 1e-3,
            (acc_eff_noise * DT)**2, (acc_eff_noise * DT)**2,
            IMU_BIAS_WALK**2, IMU_BIAS_WALK**2
        ])

    def predict(self, acc_meas):
        if self.is_static_base:
            return
        self.x = self.F @ self.x + self.B @ acc_meas
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_absolute_anchor(self, z_pos, r_std):
        """Actualización de posición absoluta (Base o Relé con GNSS seguro)."""
        H = np.zeros((2, 6))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        R = np.eye(2) * (r_std**2)
        y = z_pos - self.x[:2]
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.consecutive_rejects = 0

    def update_cooperative_batch(self, neighbor_observations):
        """
        Fusión cooperativa por lotes de las observaciones UWB recibidas en el ciclo.
        """
        if self.is_static_base or len(neighbor_observations) == 0:
            return 0

        valid_H = []
        valid_y = []
        valid_R = []
        rejected = 0

        # Umbral dinámico anti-secuestro: si llevamos varios rechazos seguidos, abrimos ventana
        chi2_thresh = 9.0 if self.consecutive_rejects < 5 else 25.0

        for (n_pos, n_P_pos, z_dist) in neighbor_observations:
            diff = self.x[:2] - n_pos
            pred_dist = np.linalg.norm(diff)
            if pred_dist < 1e-4:
                continue

            u_ij = diff / pred_dist
            H_row = np.zeros(6)
            H_row[0] = u_ij[0]
            H_row[1] = u_ij[1]

            # Covarianza efectiva conservadora (absorbe la correlación desconocida con el vecino)
            r_neighbor = float(u_ij @ n_P_pos @ u_ij.T)
            r_eff = (UWB_NOISE**2) + 4.0 * r_neighbor

            innov = z_dist - pred_dist
            S_1d = float(H_row @ self.P @ H_row.T) + r_eff
            nis = (innov**2) / S_1d

            if nis > chi2_thresh:
                rejected += 1
                continue

            valid_H.append(H_row)
            valid_y.append(innov)
            valid_R.append(r_eff)

        m = len(valid_H)
        if m == 0:
            self.consecutive_rejects += 1
            return rejected

        self.consecutive_rejects = 0
        H = np.array(valid_H)
        y = np.array(valid_y)
        # Factor de correlación de lote para mantener consistencia estadística (NEES ~ 2)
        R = np.diag(valid_R) * max(1.0, 0.8 * m)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Esquema Schmidt-Kalman: anular ganancia sobre los estados de sesgo (indices 4, 5)
        # cuando la medida es puramente relativa, evitando que el ruido de rango corrompa el sesgo IMU.
        K[4:, :] = 0.0

        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return rejected

    def nees(self, true_p):
        err = self.x[:2] - true_p
        return float(err.T @ np.linalg.inv(self.P[:2, :2]) @ err)


# --- 4. Ejecución de Escenarios Comparativos ---
modes = ['IMU Only', 'IMU + UWB (Sin Anclas)', 'AETHERNET (Descentralizado)']
results_pos = {m: np.zeros((NUM_NODES, STEPS, 2)) for m in modes}
results_nees = {m: np.zeros(STEPS) for m in modes}
rejected_outliers = 0
total_uwb_packets = 0

init_err_p = np.random.randn(NUM_NODES, 2) * 0.4
init_err_v = np.random.randn(NUM_NODES, 2) * 0.15
init_err_p[0] = 0.0
init_err_v[0] = 0.0

swarms = {}
for m in modes:
    swarms[m] = [
        DecentralizedNodeEKF(
            i,
            true_pos[i, 0] + init_err_p[i],
            true_vel[i, 0] + init_err_v[i],
            is_static_base=(i == 0)
        )
        for i in range(NUM_NODES)
    ]
    for i in range(NUM_NODES):
        results_pos[m][i, 0] = swarms[m][i].x[:2]

for step in range(1, STEPS):
    true_bias += np.random.randn(NUM_NODES, 2) * IMU_BIAS_WALK
    acc_meas = true_acc[:, step] + true_bias + np.random.randn(NUM_NODES, 2) * IMU_ACC_NOISE

    uwb_links = []
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            d_true = np.linalg.norm(true_pos[i, step] - true_pos[j, step])
            if d_true <= MAX_UWB_RANGE and np.random.rand() > PACKET_LOSS_PROB:
                noise = np.random.randn() * UWB_NOISE
                if np.random.rand() < NLOS_OUTLIER_PROB:
                    noise += np.random.uniform(3.0, 8.0)  # Rebote NLOS
                uwb_links.append((i, j, d_true + noise))

    anchor_tick = (step % ANCHOR_PERIOD == 0)
    relay_pos_meas = true_pos[1, step] + np.random.randn(2) * ANCHOR_POS_NOISE

    for m in modes:
        nodes = swarms[m]
        for i in range(NUM_NODES):
            nodes[i].predict(acc_meas[i])

        if 'AETHERNET' in m and anchor_tick:
            nodes[0].update_absolute_anchor(np.array([0.0, 0.0]), 0.03)
            nodes[1].update_absolute_anchor(relay_pos_meas, ANCHOR_POS_NOISE)

        if 'UWB' in m or 'AETHERNET' in m:
            broadcast = [(n.x[:2].copy(), n.P[:2, :2].copy()) for n in nodes]
            inbox = {i: [] for i in range(NUM_NODES)}
            for (i, j, z_d) in uwb_links:
                inbox[i].append((broadcast[j][0], broadcast[j][1], z_d))
                inbox[j].append((broadcast[i][0], broadcast[i][1], z_d))
                if m == 'AETHERNET (Descentralizado)':
                    total_uwb_packets += 2

            for i in range(NUM_NODES):
                rej = nodes[i].update_cooperative_batch(inbox[i])
                if m == 'AETHERNET (Descentralizado)':
                    rejected_outliers += rej

        for i in range(NUM_NODES):
            results_pos[m][i, step] = nodes[i].x[:2]
        results_nees[m][step] = np.mean([nodes[i].nees(true_pos[i, step]) for i in range(2, NUM_NODES)])

# --- 5. Informe de Validación y Gráficas ---
print("=== INFORME DE VALIDACIÓN SIL - AETHERNET ===")
print(f"Duración misión: {SIM_TIME:.0f} s | Nodos GNSS-Denied: {NUM_NODES - 2}")
print(f"Paquetes UWB procesados: {total_uwb_packets} | Outliers NLOS bloqueados (Chi2): {rejected_outliers}")
for m in modes:
    err_denied = np.linalg.norm(results_pos[m][2:] - true_pos[2:], axis=2)
    rmse = np.sqrt(np.mean(err_denied**2))
    final_err = np.mean(err_denied[:, -1])
    mean_nees = np.mean(results_nees[m][10:])
    print(f" -> {m:30s} | RMSE: {rmse:6.2f} m | Error Final: {final_err:6.2f} m | NEES medio: {mean_nees:5.2f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.set_title("Trayectorias Reales vs AETHERNET (120 s)", fontweight='bold')
ax.plot(0, 0, 'ks', markersize=10, label='Estación Base (Fija)')
ax.plot(true_pos[1, :, 0], true_pos[1, :, 1], 'b--', alpha=0.5, label='Nodo Relé (GNSS)')
for i in range(2, NUM_NODES):
    ax.plot(true_pos[i, :, 0], true_pos[i, :, 1], ':', color='gray', alpha=0.4)
    ax.plot(results_pos['AETHERNET (Descentralizado)'][i, :, 0],
            results_pos['AETHERNET (Descentralizado)'][i, :, 1], '-', alpha=0.8)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize='small')

ax = axes[1]
ax.set_title("Error Medio en Nodos GNSS-Denied", fontweight='bold')
for m in modes:
    err_t = np.linalg.norm(results_pos[m][2:] - true_pos[2:], axis=2).mean(axis=0)
    ax.plot(t, err_t, label=m, linewidth=2)
ax.set_ylim(0, 15)
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Error Posición (m)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize='small')

ax = axes[2]
ax.set_title("Consistencia Estadística (NEES)", fontweight='bold')
ax.plot(t[1:], results_nees['AETHERNET (Descentralizado)'][1:], color='green', label='NEES AETHERNET')
ax.axhline(2.0, color='k', linestyle='--', label='Óptimo teórico (2.0)')
ax.set_ylim(0, 10)
ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("NEES")
ax.grid(True, alpha=0.3)
ax.legend(fontsize='small')

plt.tight_layout()
plt.savefig("aethernet_validation_report.png", dpi=150)
print("Gráfica guardada en: aethernet_validation_report.png")
