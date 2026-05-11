"""
VISTA Torsional Dynamics — Modal Frequency Script
=================================================

Torsional equivalent of the existing pendulum modal-frequency scripts
(`DSE_Structures_Pendulums.py`, `DSE_Structures_ModalFreqs.py`).

Conceptually aligned with Section 2.2 of:
    Kassarian et al., "Modeling and stability of balloon-borne gondolas
    with coupled pendulum-torsion dynamics."

Only yaw/torsion about the local vertical flight-chain axis is modelled.
The balloon yaw is left free, so one near-zero rigid-body yaw mode is
expected. Stiffness comes purely from the bifilar gravitational lift
mechanism (relative twist -> cable tilt -> vertical lift -> restoring
torque); no arbitrary torsional springs are used.

Equation of motion (after adding modal damping):
    Mz * z_ddot + Cz * z_dot + Kz * z = Tz
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import eigh


# ==================================
# 1. INPUT PARAMETERS
# ==================================

# --- Balloon ---
m_balloon = 200.0          # kg     balloon mass
L_balloon = 5.0            # m      balloon diameter (thin spherical shell)

# --- Gondola ---
m_gondola = 50.0           # kg     gondola mass
R_gondola = 0.2            # m      gondola radius (vertical thin cylinder)
L_gondola = 0.6            # m      gondola length (kept for reference)

# --- Suspension / cables ---
L_susp            = 3.0    # m      total suspension length (balloon -> gondola)
N_tether          = 5      # -      numerical torsion discretization fidelity - number of the internal nodes in the cable
                           #.       total number of nodes = N_tether + 2 (balloon yaw node + gondola yaw node) - nodes = degrees of freedom
N_cables          = 2      # -      physical number of cables (bifilar / multifilar)
m_one_cable_total = 0.2    # kg     mass of ONE complete cable over L_susp
r                 = 0.25   # m      identical cable radius from yaw axis

# --- Environment ---
g = 8.72                   # m/s^2  Venus gravity at ~52 km altitude (VISTA)

# --- Damping ---
zeta_modal = 10e-3          # -      modal damping ratio for flexible modes


# ==================================
# 2. DERIVED PARAMETERS
# ==================================

N_nodes    = N_tether + 2          # node 0 = balloon yaw, node -1 = gondola yaw
N_elements = N_nodes - 1           # number of torsional elements

# Uniform discretization of the suspension into elements
L_element = np.ones(N_elements) * (L_susp / N_elements)

# Total cable mass aggregated from all physical cables, then distributed
m_cables_total = N_cables * m_one_cable_total
m_element      = np.ones(N_elements) * (m_cables_total / N_elements)

# Identical cable radius for every element (assumption #4)
r_element = np.ones(N_elements) * r

# Rigid-body yaw inertias
I_balloon_z = (1.0 / 6.0) * m_balloon * L_balloon ** 2   # thin spherical shell
I_gondola_z = 0.5 * m_gondola * R_gondola ** 2           # vertical cylinder


# ==================================
# 3. MASS-BELOW FUNCTION
# ==================================

def M_below(e):
    """
    Mass below torsional element e (excluding the element's own half-mass,
    which is added separately in the stiffness formula).

    Element e connects node e to node e+1. Below it sit:
        - all cable elements with index > e
        - the gondola
    """
    return m_gondola + np.sum(m_element[e + 1:])


# ==================================
# 4. ASSEMBLE TORSIONAL MASS MATRIX Mz
# ==================================

Mz = np.zeros((N_nodes, N_nodes))

# Rigid-body yaw inertias on the end nodes
Mz[0, 0]   += I_balloon_z
Mz[-1, -1] += I_gondola_z

# Distributed cable yaw inertia (consistent mass matrix from cable KE)
for e in range(N_elements):
    i = e
    j = e + 1
    I_e = m_element[e] * r_element[e] ** 2

    Mz[i, i] += I_e / 3.0
    Mz[j, j] += I_e / 3.0
    Mz[i, j] += I_e / 6.0
    Mz[j, i] += I_e / 6.0


# ==================================
# 5. ASSEMBLE TORSIONAL STIFFNESS MATRIX Kz
# ==================================

Kz = np.zeros((N_nodes, N_nodes))

for e in range(N_elements):
    i = e
    j = e + 1

    k_e = (M_below(e) + 0.5 * m_element[e]) * g * r_element[e] ** 2 / L_element[e]

    Kz[i, i] += k_e
    Kz[j, j] += k_e
    Kz[i, j] -= k_e
    Kz[j, i] -= k_e


# ==================================
# 6. SOLVE GENERALIZED EIGENVALUE PROBLEM
# ==================================

eigvals, eigvecs = eigh(Kz, Mz)

eigvals = np.real(eigvals)
eigvecs = np.real(eigvecs)

# Sort ascending
idx     = np.argsort(eigvals)
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Separate rigid-body yaw mode(s) from flexible modes
tol      = 1e-10
flexible = eigvals > tol

omega = np.sqrt(eigvals[flexible])
freq  = omega / (2.0 * np.pi)


# ==================================
# 7. MODAL DAMPING (FLEXIBLE MODES ONLY)
# ==================================

Phi = eigvecs[:, flexible].copy()

# Mass-normalize the flexible mode shapes: Phi^T Mz Phi = I
for i in range(Phi.shape[1]):
    norm = np.sqrt(Phi[:, i].T @ Mz @ Phi[:, i])
    Phi[:, i] = Phi[:, i] / norm

Cm = np.diag(2.0 * zeta_modal * omega)
Cz = Mz @ Phi @ Cm @ Phi.T @ Mz


# ==================================
# 8. OUTPUT
# ==================================

np.set_printoptions(precision=5, suppress=True, linewidth=140)

print("=" * 60)
print("VISTA Torsional Dynamics — Modal Frequency Output")
print("=" * 60)

print("\nN_nodes    =", N_nodes)
print("N_elements =", N_elements)
print("L_element  =", L_element)
print("m_element  =", m_element)
print("r_element  =", r_element)
print("I_balloon_z = {:.5f} kg m^2".format(I_balloon_z))
print("I_gondola_z = {:.5f} kg m^2".format(I_gondola_z))

print("\nTorsional mass matrix Mz [kg m^2]:")
print(Mz)

print("\nTorsional stiffness matrix Kz [N m / rad]:")
print(Kz)

print("\nFlexible torsional natural frequencies [Hz]:")
print(freq)

print("\nFlexible torsional natural angular frequencies [rad/s]:")
print(omega)

print("\nRigid-body yaw eigenvalue(s) [rad^2/s^2]:")
print(eigvals[~flexible])

print("\nModal damping matrix Cz [N m s / rad]:")
print(Cz)


# ==================================
# 9. VALIDATION CHECKS
# ==================================

print("\n" + "=" * 60)
print("Validation checks")
print("=" * 60)

print("Mz symmetric:", np.allclose(Mz, Mz.T))
print("Kz symmetric:", np.allclose(Kz, Kz.T))

eig_Mz = np.linalg.eigvalsh((Mz + Mz.T) / 2.0)
eig_Kz = np.linalg.eigvalsh((Kz + Kz.T) / 2.0)

print("Eigenvalues of Mz:", eig_Mz)
print("All Mz eigenvalues > 0:", np.all(eig_Mz > 0))

print("Eigenvalues of Kz:", eig_Kz)
print("Min |Kz eigenvalue| (expected ~0 for free balloon yaw): {:.3e}".format(
    np.min(np.abs(eig_Kz))
))
print("Number of (near-)zero Kz eigenvalues:",
      int(np.sum(np.abs(eig_Kz) < tol)))


# ==================================
# 10. SINGLE-ELEMENT ANALYTICAL CHECK
# ==================================

print("\n" + "=" * 60)
print("Single-element bifilar stiffness check (N_elements = 1 reference)")
print("=" * 60)

# Analytical bifilar torsion stiffness for the entire suspension treated
# as one lumped element:
#     k_t = (m_gondola + 0.5 * m_cables_total) * g * r^2 / L_susp
k_t_analytical = (m_gondola + 0.5 * m_cables_total) * g * r ** 2 / L_susp

# Equivalent stiffness of the discretized chain (springs in series):
#     1/k_eq = sum(1/k_e)
k_e_list = np.array([
    (M_below(e) + 0.5 * m_element[e]) * g * r_element[e] ** 2 / L_element[e]
    for e in range(N_elements)
])
k_eq_series = 1.0 / np.sum(1.0 / k_e_list)

print("Element stiffnesses k_e [N m / rad]:", k_e_list)
print("Series-equivalent stiffness k_eq    : {:.6f} N m / rad".format(k_eq_series))
print("Analytical lumped k_t               : {:.6f} N m / rad".format(k_t_analytical))
print("Note: k_eq matches k_t exactly only for N_elements = 1; for finer")
print("      discretization the distributed mass shifts k_eq slightly.")

if N_elements == 1:
    print("Single-element match:",
          np.isclose(k_e_list[0], k_t_analytical))


# ==================================
# 11. PHYSICAL-COORDINATE TIME RESPONSE
# ==================================

# ============================================================
# CASE SELECTOR  —  set ACTIVE_CASE to 1, 2, or 3
# ============================================================
ACTIVE_CASE = 2       # <--- change this to select the analysis case
#   1 : initial stored twist (distributed linearly along suspension)
#   2 : initial gondola spin rate
#   3 : angular impulse at gondola (converted to initial velocity jump)
# ============================================================

# --- Case parameters (edit these values as needed) ---
initial_twist_deg  = 5.0    # deg    Case 1: total twist stored in cable
initial_spin_deg_s = 1.0    # deg/s  Case 2: gondola initial yaw rate
impulse_Nms        = 0.1    # N m s  Case 3: angular impulse magnitude

# --- Simulation time span ---
t_final = 50.0              # s
dt      = 0.1               # s
t_eval  = np.arange(0.0, t_final + dt, dt)

# --- Build initial conditions for the selected case ---
if ACTIVE_CASE == 1:
    case_name = "Case 1: initial stored twist ({:.1f} deg)".format(initial_twist_deg)
    z0    = np.linspace(0.0, np.deg2rad(initial_twist_deg), N_nodes)
    zdot0 = np.zeros(N_nodes)

elif ACTIVE_CASE == 2:
    case_name = "Case 2: initial gondola spin ({:.1f} deg/s)".format(initial_spin_deg_s)
    z0        = np.zeros(N_nodes)
    zdot0     = np.zeros(N_nodes)
    zdot0[-1] = np.deg2rad(initial_spin_deg_s)

elif ACTIVE_CASE == 3:
    case_name = "Case 3: angular impulse at gondola ({:.3f} N m s)".format(impulse_Nms)
    J         = np.zeros(N_nodes)
    J[-1]     = impulse_Nms
    z0        = np.zeros(N_nodes)
    zdot0     = np.linalg.solve(Mz, J)

else:
    raise ValueError("ACTIVE_CASE must be 1, 2, or 3.")


# --- Torque input (zero for passive deployment cases) ---
def torque_input(t):
    """External torque vector. Extend here to add aero/ADCS torques."""
    return np.zeros(N_nodes)


# --- State-space RHS:  x = [z, zdot] ---
def rhs(t, x):
    z     = x[:N_nodes]
    zdot  = x[N_nodes:]
    Tz    = torque_input(t)
    zddot = np.linalg.solve(Mz, Tz - Cz @ zdot - Kz @ z)
    return np.concatenate([zdot, zddot])


# --- Integrate ---
x0  = np.concatenate([z0, zdot0])
sol = solve_ivp(
    rhs,
    t_span=(0.0, t_final),
    y0=x0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-8,
    atol=1e-10,
)
if not sol.success:
    raise RuntimeError("solve_ivp failed: " + sol.message)

t_hist    = sol.t
z_hist    = sol.y[:N_nodes, :]
zdot_hist = sol.y[N_nodes:, :]

# Acceleration reconstructed from the ODE (Tz = 0 for these cases)
Mz_inv     = np.linalg.inv(Mz)
zddot_hist = Mz_inv @ (-Cz @ zdot_hist - Kz @ z_hist)


# --- Time histories for gondola and relative quantities ---
theta_g        = z_hist[-1, :]
theta_dot_g    = zdot_hist[-1, :]
theta_ddot_g   = zddot_hist[-1, :]

theta_rel_gb      = z_hist[-1, :]   - z_hist[0, :]
theta_dot_rel_gb  = zdot_hist[-1, :] - zdot_hist[0, :]
theta_ddot_rel_gb = zddot_hist[-1, :] - zddot_hist[0, :]

# --- Element-level quantities ---
eta_hist      = z_hist[1:, :] - z_hist[:-1, :]          # relative twist per element
tau_hist      = k_e_list[:, None] * eta_hist             # restoring torque per element

max_g_yaw_deg    = np.rad2deg(np.max(np.abs(theta_g)))
max_g_rate_dps   = np.rad2deg(np.max(np.abs(theta_dot_g)))
max_g_acc_dps2   = np.rad2deg(np.max(np.abs(theta_ddot_g)))

max_rel_yaw_deg  = np.rad2deg(np.max(np.abs(theta_rel_gb)))
max_rel_rate_dps = np.rad2deg(np.max(np.abs(theta_dot_rel_gb)))
max_rel_acc_dps2 = np.rad2deg(np.max(np.abs(theta_ddot_rel_gb)))

max_eta_deg      = np.rad2deg(np.max(np.abs(eta_hist), axis=1))
max_tau_elem     = np.max(np.abs(tau_hist), axis=1)

# --- Modal design quantities ---
periods        = 1.0 / freq                             # s, one per flexible mode
settling_times = 4.0 / (zeta_modal * omega)            # s, 2 % settling per mode


# ==================================
# 12. SUMMARY TABLE
# ==================================

W  = 68   # total table width
sep = "+" + "-" * (W - 2) + "+"
hdr = lambda s: "| " + s.ljust(W - 4) + " |"

def row2(label, val, unit=""):
    s = "{:<38s} {:>14s}  {:>10s}".format(label, val, unit)
    return "| " + s.ljust(W - 4) + " |"

print("\n" + sep)
print(hdr("TORSIONAL TIME-RESPONSE SUMMARY  —  " + case_name))
print(sep)

# --- A. Modal properties ---
print(hdr("A. Modal properties"))
print("| " + "{:<6s} {:>14s} {:>14s} {:>14s} {:>10s}".format(
    "Mode", "Freq [Hz]", "Period [s]", "t_settle [s]", "(2%)").ljust(W - 4) + " |")
print(sep)
for i in range(len(freq)):
    print("| " + "{:<6d} {:>14.5f} {:>14.5f} {:>14.1f} {:>10s}".format(
        i + 1, freq[i], periods[i], settling_times[i], "").ljust(W - 4) + " |")
print(sep)

# --- B. Gondola yaw extremes ---
print(hdr("B. Gondola yaw  (absolute  /  relative to balloon)"))
print(sep)
print(row2("Max yaw angle          [deg]",
           "{:.4f}  /  {:.4f}".format(max_g_yaw_deg,   max_rel_yaw_deg), ""))
print(row2("Max yaw rate           [deg/s]",
           "{:.4f}  /  {:.4f}".format(max_g_rate_dps,  max_rel_rate_dps), ""))
print(row2("Max yaw acceleration   [deg/s²]",
           "{:.4f}  /  {:.4f}".format(max_g_acc_dps2,  max_rel_acc_dps2), ""))
print(sep)

# --- C. Element loads ---
print(hdr("C. Element design loads"))
print("| " + "{:<10s} {:>20s} {:>20s} {:>10s}".format(
    "Element", "Max twist [deg]", "Max torque [N m]", "Notes").ljust(W - 4) + " |")
print(sep)
for e in range(N_elements):
    note = ""
    if e == 0:
        note = "balloon-side"
    elif e == N_elements - 1:
        note = "gondola-side"
    print("| " + "{:<10d} {:>20.5f} {:>20.6f} {:>10s}".format(
        e, max_eta_deg[e], max_tau_elem[e], note).ljust(W - 4) + " |")
print(sep)


# ==================================
# 13. PLOTS
# ==================================

# --- Plot 1: Gondola yaw angle ---
fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(t_hist, np.rad2deg(theta_g),      label="Absolute (gondola)")
ax1.plot(t_hist, np.rad2deg(theta_rel_gb), label="Relative to balloon", linestyle="--")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Yaw angle [deg]")
ax1.set_title("Gondola yaw angle — " + case_name)
ax1.grid(True)
ax1.legend()
fig1.tight_layout()

# --- Plot 2: Gondola yaw rate ---
fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(t_hist, np.rad2deg(theta_dot_g),      label="Absolute (gondola)")
ax2.plot(t_hist, np.rad2deg(theta_dot_rel_gb), label="Relative to balloon", linestyle="--")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Yaw rate [deg/s]")
ax2.set_title("Gondola yaw rate — " + case_name)
ax2.grid(True)
ax2.legend()
fig2.tight_layout()

# --- Plot 3: Gondola yaw acceleration ---
fig3, ax3 = plt.subplots(figsize=(9, 4))
ax3.plot(t_hist, np.rad2deg(theta_ddot_g),      label="Absolute (gondola)")
ax3.plot(t_hist, np.rad2deg(theta_ddot_rel_gb), label="Relative to balloon", linestyle="--")
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Yaw acceleration [deg/s²]")
ax3.set_title("Gondola yaw acceleration — " + case_name)
ax3.grid(True)
ax3.legend()
fig3.tight_layout()

plt.show()
