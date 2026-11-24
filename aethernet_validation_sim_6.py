"""
AETHERNET - Suite de Simulación y Validación (SIL - Software-in-the-Loop)
=========================================================================
Ejecuta directamente el motor de producción `AethernetNavigator` (`aethernet_core_node_6.py`)
incluyendo serialización/deserialización binaria de 26 bytes con verificación CRC-16.

Escenario: 1 Estación Base fija (coordenadas conocidas) + 1 Nodo Relé (GNSS intermitente)
           + 8 Nodos Tácticos en zona 100% GNSS-Denied.

Métricas validadas:
1. Error RMSE 2D (m) de los nodos GNSS-denied a lo largo de 120 s.
2. Consistencia estadística mediante NEES (Normalized Estimation Error Squared ~ 2.0).
3. Rechazo automático de outliers UWB (NLOS / Multipath) mediante gating Chi-cuadrado.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from aethernet_core_node_6 import AethernetNavigator, EKFConfig

# Silenciar logs de inicialización por nodo durante los benchmarks comparativos
logging.getLogger().setLevel(logging.WARNING)
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
IMU_BIAS_WALK = 0.0005    # m/s^2 / sqrt(s)
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

# --- 3. Instanciación del Motor Real (AethernetNavigator) por Escenario ---
ekf_cfg = EKFConfig(
    acc_noise_std=IMU_ACC_NOISE,
    bias_residual_std=IMU_BIAS_INIT,
    bias_walk_std=IMU_BIAS_WALK,
    uwb_noise_std=UWB_NOISE
)

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
    swarms[m] = []
    for i in range(NUM_NODES):
        is_base = (i == 0)
        nav = AethernetNavigator(node_id=i, is_static_anchor=is_base, config=ekf_cfg)
        nav.initialize_state(
            pos_xy=true_pos[i, 0] + init_err_p[i],
            vel_xy=true_vel[i, 0] + init_err_v[i],
            pos_std=0.03 if is_base else 0.5
        )
        # Sincronizar reloj inicial del filtro con t = 0.0
        nav.predict_imu(np.zeros(2), timestamp=0.0)
        swarms[m].append(nav)
        results_pos[m][i, 0] = nav.x[:2]

# --- 4. Bucle Principal de Simulación SIL ---
for step in range(1, STEPS):
    now = step * DT
    # Random walk continuo escalado correctamente con sqrt(DT)
    true_bias += np.random.randn(NUM_NODES, 2) * (IMU_BIAS_WALK * np.sqrt(DT))
    acc_meas = true_acc[:, step] + true_bias + np.random.randn(NUM_NODES, 2) * IMU_ACC_NOISE

    uwb_links = []
    for i in range(NUM_NODES):
        for j in range(i + 1, NUM_NODES):
            d_true = np.linalg.norm(true_pos[i, step] - true_pos[j, step])
            if d_true <= MAX_UWB_RANGE and np.random.rand() > PACKET_LOSS_PROB:
                noise = np.random.randn() * UWB_NOISE
                if np.random.rand() < NLOS_OUTLIER_PROB:
                    noise += np.random.uniform(3.0, 8.0)  # Rebote NLOS
                uwb_links.append((i, j, float(d_true + noise)))

    anchor_tick = (step % ANCHOR_PERIOD == 0)
    relay_pos_meas = true_pos[1, step] + np.random.randn(2) * ANCHOR_POS_NOISE

    for m in modes:
        nodes = swarms[m]
        for i in range(NUM_NODES):
            nodes[i].predict_imu(acc_meas[i], timestamp=now)

        if 'AETHERNET' in m and anchor_tick:
            nodes[0].update_gnss_or_anchor(np.array([0.0, 0.0]), 0.03)
            nodes[1].update_gnss_or_anchor(relay_pos_meas, ANCHOR_POS_NOISE)

        if 'UWB' in m or 'AETHERNET' in m:
            # Simulación fiel del enlace radio: cada nodo empaqueta 26 bytes y el receptor valida CRC16
            raw_packets = [n.pack_state_packet() for n in nodes]
            decoded_peers = [AethernetNavigator.unpack_state_packet(pkt) for pkt in raw_packets]

            inbox = {i: [] for i in range(NUM_NODES)}
            for (i, j, z_d) in uwb_links:
                pkt_j = decoded_peers[j]
                pkt_i = decoded_peers[i]
                if pkt_j is not None:
                    inbox[i].append((pkt_j['pos'], pkt_j['P_pos'], z_d))
                if pkt_i is not None:
                    inbox[j].append((pkt_i['pos'], pkt_i['P_pos'], z_d))
                if m == 'AETHERNET (Descentralizado)':
                    total_uwb_packets += 2

            for i in range(NUM_NODES):
                rej = nodes[i].update_peers_uwb_batch(inbox[i], return_rejected=True)
                if m == 'AETHERNET (Descentralizado)':
                    rejected_outliers += rej

        for i in range(NUM_NODES):
            results_pos[m][i, step] = nodes[i].x[:2]
        results_nees[m][step] = np.mean([nodes[i].nees_2d(true_pos[i, step]) for i in range(2, NUM_NODES)])

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
