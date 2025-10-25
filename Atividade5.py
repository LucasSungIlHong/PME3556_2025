import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================
# PARÂMETROS FÍSICOS E GLOBAIS
# ============================
rho = 1.0             # densidade [kg/m³]
nu = 1e-6             # viscosidade cinemática [m²/s]
kappa = 0.41          # constante de von Kármán
B = 5.0               # constante empírica da lei logarítmica
U_inf = 1.0           # velocidade livre [m/s]
x_ref = 1.0           # posição de referência [m]

# ============================
# LEITURA DOS DADOS CFD
# ============================
data1 = pd.read_csv("data1.txt")  # perfil de velocidade
data2 = pd.read_csv("data2.txt")  # tensões cisalhantes

# ============================
# VERIFICAÇÕES DE CONSISTÊNCIA
# ============================
x_col = "Points:0"
if np.isclose(data2[x_col], x_ref, atol=1e-3).any():
    tau_w = abs(data2.loc[np.isclose(data2[x_col], x_ref, atol=1e-3), "wallShearStress:0"].values[0])
else:
    tau_interp = np.interp(x_ref, data2[x_col], abs(data2["wallShearStress:0"]))
    tau_w = tau_interp

y_raw = data1["Points:1"]
y_min = y_raw.min()
if y_min > 1e-6:
    print(f"Atenção: a coordenada y mínima é {y_min:.2e} m. Subtraindo deslocamento.")
y = y_raw - y_min

# ============================
# PASSO 23 – VELOCIDADE DE CISALHAMENTO E PERFIL
# ============================
u_star = np.sqrt(tau_w / rho)
print(f"\nVelocidade de cisalhamento calculada: u* = {u_star:.6e} m/s")

Ux = data1["U:0"]

# Curvas teóricas (U⁺)
y_plus_theor = np.logspace(0, 3, 300)
U_plus_visc = y_plus_theor[y_plus_theor < 5]
U_plus_log = (1/kappa) * np.log(y_plus_theor[y_plus_theor > 30]) + B

# Reconstrução em unidades físicas
U_theor_visc = U_plus_visc * u_star
U_theor_log = U_plus_log * u_star

# ============================
# PASSO 19 – PERFIL ADIMENSIONAL (U+ vs y+)
# ============================
y_plus = (u_star * y) / nu
U_plus = Ux / u_star

# ----------- VERSÃO 1: CFD + CURVAS TEÓRICAS -----------
plt.figure(figsize=(7,5))
plt.plot(y_plus, U_plus, 'o', markersize=1.5, color='tab:blue', alpha=0.7, label='CFD (OpenFOAM)')
plt.plot(U_plus_visc, U_plus_visc, 'k--', linewidth=1.5, label='Subcamada viscosa $U^+ = y^+$')
plt.plot(y_plus_theor[y_plus_theor > 30], U_plus_log, 'r--', linewidth=1.5, label='Camada logarítmica')
plt.xscale('log')
plt.xlabel(r'$y^+$')
plt.ylabel(r'$U^+$')
plt.legend()
plt.grid(True, which='both')
plt.title('Perfil adimensional de velocidade (U⁺ vs y⁺) — CFD + Teórico')
plt.tight_layout()
plt.show()

# ----------- VERSÃO 2: APENAS CURVAS TEÓRICAS -----------
plt.figure(figsize=(7,5))
plt.plot(U_plus_visc, U_plus_visc, 'k--', linewidth=1.5, label='Subcamada viscosa $U^+ = y^+$')
plt.plot(y_plus_theor[y_plus_theor > 30], U_plus_log, 'r--', linewidth=1.5, label='Camada logarítmica')
plt.xscale('log')
plt.xlabel(r'$y^+$')
plt.ylabel(r'$U^+$')
plt.legend()
plt.grid(True, which='both')
plt.title('Perfil adimensional de velocidade (U⁺ vs y⁺) — Teórico')
plt.tight_layout()
plt.show()

# ============================
# PASSO 23 – PERFIL RECONSTRUÍDO DE VELOCIDADES Uₓ(y)
# ============================
# ----------- VERSÃO 1: CFD + CURVAS TEÓRICAS -----------
plt.figure(figsize=(7,5))
plt.plot(y, Ux, 'o', markersize=1.5, color='tab:blue', alpha=0.7, label='CFD (OpenFOAM)')
plt.plot((U_plus_visc * nu / u_star), U_theor_visc, 'k--', linewidth=1.5, label='Subcamada viscosa')
plt.plot((y_plus_theor[y_plus_theor > 30] * nu / u_star), U_theor_log, 'r--', linewidth=1.5, label='Camada logarítmica')
plt.xlabel('y [m]')
plt.ylabel('Uₓ [m/s]')
plt.legend()
plt.grid(True)
plt.title('Perfil de velocidades Uₓ(y) — CFD + Teórico')
plt.tight_layout()
plt.show()

# ----------- VERSÃO 2: APENAS CURVAS TEÓRICAS -----------
plt.figure(figsize=(7,5))
plt.plot((U_plus_visc * nu / u_star), U_theor_visc, 'k--', linewidth=1.5, label='Subcamada viscosa')
plt.plot((y_plus_theor[y_plus_theor > 30] * nu / u_star), U_theor_log, 'r--', linewidth=1.5, label='Camada logarítmica')
plt.xlabel('y [m]')
plt.ylabel('Uₓ [m/s]')
plt.legend()
plt.grid(True)
plt.title('Perfil de velocidades Uₓ(y) — Teórico')
plt.tight_layout()
plt.show()

# ============================
# PASSO 24 – COEFICIENTE DE ATRITO (COMPARAÇÃO COMPLETA)
# ============================
Re_x = (U_inf * x_ref) / nu
cf_sim = 2 * tau_w / (rho * U_inf**2)
cf_teor = 0.059 * Re_x**(-0.2)

print(f"\nCoeficiente de atrito (simulado): c_f = {cf_sim:.6e}")
print(f"Coeficiente de atrito (von Kármán): c_f = {cf_teor:.6e}")
print(f"Re_x = {Re_x:.3e}")

# Calcula c_f(x) ao longo da placa
data2["cf_local"] = 2 * abs(data2["wallShearStress:0"]) / (rho * U_inf**2)
data2["Re_x"] = (U_inf * data2["Points:0"]) / nu
cf_vonK_curve = 0.059 * data2["Re_x"]**(-0.2)

# ----------- VERSÃO 1: CFD + CURVA TEÓRICA -----------
plt.figure(figsize=(7,5))
plt.plot(data2["Re_x"], data2["cf_local"], 'o-', markersize=1.5, color='tab:blue', alpha=0.7, label='CFD (OpenFOAM)')
plt.plot(data2["Re_x"], cf_vonK_curve, 'r--', linewidth=1.5, label='von Kármán (perfil 1/7)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$Re_x$')
plt.ylabel(r'$c_f$')
plt.legend()
plt.grid(True, which='both')
plt.title('Coeficiente de atrito — CFD + Teórico')
plt.tight_layout()
plt.show()

# ----------- VERSÃO 2: APENAS CURVA TEÓRICA -----------
plt.figure(figsize=(7,5))
plt.plot(data2["Re_x"], cf_vonK_curve, 'r--', linewidth=1.5, label='von Kármán (perfil 1/7)')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$Re_x$')
plt.ylabel(r'$c_f$')
plt.legend()
plt.grid(True, which='both')
plt.title('Coeficiente de atrito — Teórico')
plt.tight_layout()
plt.show()
