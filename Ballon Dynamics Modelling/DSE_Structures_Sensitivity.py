"""
VISTA — One-at-a-Time Sensitivity Study
========================================

Implements the sensitivity-analysis spec in
``context_sensitivity analysis.md``.  Self-contained: it does NOT modify
``DSE_Structures_Combined.py``; it re-implements the same physics inside a
parametric ``run_single_case(params)`` so each call is independent.

Outputs (in ``sensitivity_results/``):

    sensitivity_one_at_a_time.csv     -- every case, every output
    sensitivity_<param>.csv           -- one CSV per swept parameter
    sensitivity_trend_summary.csv     -- per-(param, output) monotonic trend
    plots/*.png                       -- simple trend plots

All tables are also printed to the terminal.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==================================================================
# 1. FIXED ENVIRONMENT / NUMERICAL SETTINGS
# ==================================================================
# Parameters that the sensitivity study does NOT vary.
m_balloon = 200.0        # kg     balloon mass (envelope + lift gas)
L_balloon = 9.0          # m      balloon characteristic length

g         = 8.72         # m/s^2  Venus gravity at ~52 km
lamb      = 0.1          # [-]    buoyancy CG/CP destabilisation coefficient

N_tether_pendulum = 8    # discretisation segments (numerical, not physical)
N_tether_torsion  = 8

t_final  = 50.0
dt       = 0.05
mode_tol = 1e-8

results_dir = "sensitivity_results"
plots_dir   = os.path.join(results_dir, "plots")


# ==================================================================
# 2. BASELINE + SWEEP DEFINITIONS  (per spec sections 3 & 4)
# ==================================================================
baseline_case = {
    "L_susp":            15.0,
    "N_cables":          4,
    "r":                 0.5,
    "m_gondola":         100.0,
    "m_one_cable_total": 0.2,
    "gondola_shape":     "cube",
    "zeta_pendulum":     0.005,
    "zeta_torsion":      0.001,
    "disturbance_case":  "high",
}

disturbance_cases = {
    "low": {
        "pend_initial_angle_deg":         2.0,
        "pend_initial_rate_deg_s":        0.0,
        "torsion_initial_twist_deg":      2.0,
        "torsion_initial_yaw_rate_deg_s": 0.0,
        "pend_impulse_Ns":                0.0,
        "torsion_impulse_Nms":            0.0,
    },
    "medium": {
        "pend_initial_angle_deg":         4.0,
        "pend_initial_rate_deg_s":        0.0,
        "torsion_initial_twist_deg":      4.0,
        "torsion_initial_yaw_rate_deg_s": 0.0,
        "pend_impulse_Ns":                0.0,
        "torsion_impulse_Nms":            0.0,
    },
    "high": {
    # Pendulum / swing initial conditions
    "pend_initial_angle_deg": 8.0,
    "pend_initial_rate_deg_s": 3.0,
    "pend_impulse_Ns": 0.0,

    # Torsion / yaw initial conditions
    "torsion_initial_twist_deg": 8.0,
    "torsion_initial_yaw_rate_deg_s": 3.0,
    "torsion_impulse_Nms": 0.0,
    }
}

# Representative geometry for each gondola shape (spec section 4.8).
# All shapes scaled to V = 1.0 m^3, preserving the baseline h/R and a:b:c ratios.
gondola_geometry_cases = {
    "solid_cylinder":  {"R": 0.5419, "h": 1.0838},        # V = pi*R^2*h = 1.000 m^3, h=2R (upright)
    "cube":            {"side": 1.0},                      # V = 1.000 m^3
    "solid_sphere":    {"R": 0.6204},                      # V = 4/3*pi*R^3 = 1.000 m^3
    "rectangular_box": {"a": 1.7784, "b": 1.1856, "c": 0.4743},  # V = 1.000 m^3
}

# Spec sections 7-15: every one-at-a-time sweep.
SWEEPS = {
    # Continuous parameters: ~9-10 points spanning the spec's design range
    "L_susp":            [round(v, 3) for v in np.linspace(8.0, 22.0, 9)],          # 8..22 m
    "N_cables":          [3, 4, 5, 6, 7, 8],                                         # integers
    "r":                 [round(v, 3) for v in np.linspace(0.3, 2.5, 10)],          # 0.3..2.5 m
    "zeta_pendulum":     [round(v, 5) for v in np.linspace(0.001, 0.02, 10)],       # 0.1%..2%
    "zeta_torsion":      [round(v, 5) for v in np.linspace(0.001, 0.02, 10)],       # 0.1%..2%
    "m_gondola":         [round(v, 1) for v in np.linspace(50.0, 300.0, 9)],        # 50..300 kg
    "m_one_cable_total": [round(v, 3) for v in np.linspace(0.05, 1.5, 10)],         # 0.05..1.5 kg
    "gondola_shape":     ["solid_cylinder", "cube", "solid_sphere", "rectangular_box"],
}

# Per-spec table column lists (sections 7-15).  For each sweep we print a
# focused table showing only the columns the spec asks for.
# NOTE: "feasible" is appended automatically to every table.
SPEC_TABLE_COLUMNS = {
    "L_susp": [
        "L_susp",
        "dominant_pendulum_period_s", "dominant_torsion_period_s",
        "max_gondola_swing_angle_deg", "max_gondola_swing_rate_deg_s",
        "max_gondola_yaw_deg", "max_gondola_yaw_rate_deg_s",
        "peak_per_cable_tension_N", "maximum_torsional_torque_Nm",
        "required_imu_sampling_10x_hz",
    ],
    "N_cables": [
        "N_cables",
        "dominant_pendulum_period_s", "dominant_torsion_period_s",
        "max_gondola_swing_angle_deg", "max_gondola_swing_rate_deg_s",
        "max_gondola_yaw_deg", "max_gondola_yaw_rate_deg_s",
        "peak_total_cable_tension_N", "peak_per_cable_tension_N",
        "dynamic_static_tension_ratio", "maximum_torsional_torque_Nm",
        "required_imu_sampling_10x_hz",
    ],
    "r": [
        "r_attach",
        "dominant_torsion_period_s", "dominant_torsion_frequency_hz",
        "max_gondola_yaw_deg", "max_gondola_yaw_rate_deg_s",
        "maximum_torsional_torque_Nm", "maximum_torsion_element_twist_deg",
        "dominant_pendulum_period_s", "max_gondola_swing_angle_deg",
        "required_imu_sampling_10x_hz",
    ],
    "zeta_pendulum": [
        "zeta_pendulum",
        "dominant_pendulum_period_s",
        "max_gondola_swing_angle_deg", "max_gondola_swing_rate_deg_s",
        "settling_time_pendulum_s",
        "peak_per_cable_tension_N",
        "required_imu_sampling_10x_hz",
    ],
    "zeta_torsion": [
        "zeta_torsion",
        "dominant_torsion_period_s",
        "max_gondola_yaw_deg", "max_gondola_yaw_rate_deg_s",
        "settling_time_torsion_s",
        "maximum_torsional_torque_Nm", "maximum_torsion_element_twist_deg",
        "required_imu_sampling_10x_hz",
    ],
    "m_gondola": [
        "m_gondola",
        "dominant_pendulum_period_s", "dominant_torsion_period_s",
        "max_gondola_swing_angle_deg", "max_gondola_swing_rate_deg_s",
        "max_gondola_yaw_deg", "max_gondola_yaw_rate_deg_s",
        "peak_total_cable_tension_N", "peak_per_cable_tension_N",
        "dynamic_static_tension_ratio", "maximum_torsional_torque_Nm",
        "required_imu_sampling_10x_hz",
    ],
    "m_one_cable_total": [
        "m_one_cable_total", "m_cables_total",
        "dominant_pendulum_period_s", "dominant_torsion_period_s",
        "max_gondola_swing_angle_deg", "max_gondola_swing_rate_deg_s",
        "max_gondola_yaw_deg", "max_gondola_yaw_rate_deg_s",
        "peak_total_cable_tension_N", "peak_per_cable_tension_N",
        "dynamic_static_tension_ratio", "maximum_torsional_torque_Nm",
        "required_imu_sampling_10x_hz",
    ],
    "gondola_shape": [
        "gondola_shape", "gondola_reference_size",
        "I_gondola_tilt", "I_gondola_yaw",
        "dominant_pendulum_period_s", "dominant_torsion_period_s",
        "max_gondola_swing_angle_deg", "max_gondola_swing_rate_deg_s",
        "max_gondola_yaw_deg", "max_gondola_yaw_rate_deg_s",
        "peak_per_cable_tension_N", "maximum_torsional_torque_Nm",
        "required_imu_sampling_10x_hz",
    ],
}


# ==================================================================
# 2b. DESIGN CONSTRAINTS
# ==================================================================
# Each entry: column_name -> (operator, limit, short_label)
#   operator: "<="  (output must be <= limit to pass)
#             ">="  (output must be >= limit to pass)
CONSTRAINTS = {
    # Structural / geometric
    "L_susp":                        (">=",  10.0,   "L_susp>=10m"),
    # Angular motion envelope
    "max_gondola_swing_angle_deg":    ("<=",  10.0,   "swing<=10deg"),
    "max_gondola_yaw_deg":            ("<=",  10.0,   "yaw<=10deg"),
    # IMU hardware limit: 2000 samples/s available
    "required_imu_sampling_10x_hz":  ("<=", 2000.0,  "IMU_10x<=2kHz"),
}


def check_constraints(row):
    """Return (feasible: bool, violations: str) for a result row (dict or Series)."""
    violated = []
    for col, (op, limit, label) in CONSTRAINTS.items():
        val = row.get(col) if isinstance(row, dict) else (
              row[col] if col in row.index else None)
        if val is None:     
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if op == "<=" and v > limit:
            violated.append(label)
        elif op == ">=" and v < limit:
            violated.append(label)
    if violated:
        return False, "; ".join(violated)
    return True, "OK"


# ==================================================================
# 3. GONDOLA INERTIA HELPER  (spec section 4.8)
# ==================================================================
def gondola_inertia(m, shape, geom):
    """
    Return (I_tilt, I_yaw, L_ref) for a gondola of mass m, given a shape
    name and a dict of representative dimensions ``geom``.

    L_ref is a representative vertical size used downstream as ``L_gondola``
    when estimating effective pendulum length and tension.
    """
    if shape == "solid_cylinder":
        R, h = geom["R"], geom["h"]
        I_tilt = (1.0 / 12.0) * m * (3.0 * R ** 2 + h ** 2)
        I_yaw  = 0.5 * m * R ** 2
        L_ref  = h
    elif shape == "solid_sphere":
        R = geom["R"]
        I_tilt = (2.0 / 5.0) * m * R ** 2
        I_yaw  = (2.0 / 5.0) * m * R ** 2
        L_ref  = 2.0 * R
    elif shape == "cube":
        s = geom["side"]
        I_tilt = (1.0 / 6.0) * m * s ** 2
        I_yaw  = (1.0 / 6.0) * m * s ** 2
        L_ref  = s
    elif shape == "rectangular_box":
        a, b, c = geom["a"], geom["b"], geom["c"]
        I_x = (1.0 / 12.0) * m * (b ** 2 + c ** 2)
        I_y = (1.0 / 12.0) * m * (a ** 2 + c ** 2)
        I_z = (1.0 / 12.0) * m * (a ** 2 + b ** 2)
        I_tilt = 0.5 * (I_x + I_y)
        I_yaw  = I_z
        L_ref  = c
    else:
        raise ValueError("Unknown gondola shape: {}".format(shape))

    # Compact, readable size string for the table
    if shape == "solid_cylinder":
        size_str = "R={:.2f}, h={:.2f}".format(geom["R"], geom["h"])
    elif shape == "solid_sphere":
        size_str = "R={:.2f}".format(geom["R"])
    elif shape == "cube":
        size_str = "side={:.2f}".format(geom["side"])
    else:
        size_str = "a={:.2f}, b={:.2f}, c={:.2f}".format(geom["a"], geom["b"], geom["c"])
    return I_tilt, I_yaw, L_ref, size_str


# ==================================================================
# 4. PHYSICS BUILDERS  (translation of DSE_Structures_Combined.py)
# ==================================================================
def build_pendulum(L_susp, N_cables, m_gondola, m_one_cable_total,
                   I_gondola_tilt, L_gondola_ref):
    """Assemble pendulum mass and stiffness matrices."""
    N_p   = N_tether_pendulum + 2
    DOF_p = N_p + 1

    m_pend = [m_balloon]
    L_pend = [L_balloon]
    I_pend = [(1.0 / 6.0) * m_balloon * L_balloon ** 2]

    m_one_tether_pend = (N_cables * m_one_cable_total) / N_tether_pendulum
    L_one_tether_pend = L_susp / N_tether_pendulum
    for _ in range(N_tether_pendulum):
        m_pend.append(m_one_tether_pend)
        L_pend.append(L_one_tether_pend)
        I_pend.append((1.0 / 12.0) * m_one_tether_pend * L_one_tether_pend ** 2)

    m_pend.append(m_gondola)
    L_pend.append(L_gondola_ref)
    I_pend.append(I_gondola_tilt)

    def mu_p(k):
        if k >= N_p:
            return 0.0
        return sum(m_pend[k:N_p])

    rho   = 0.5
    Llist = list(L_pend)
    Llist[0] = (1.0 - rho) * L_pend[0]
    mulist = [mu_p(i) for i in range(N_p + 1)]

    Mx = np.zeros((DOF_p, DOF_p))
    for i in range(DOF_p):
        if i == 0:
            for j in range(2, DOF_p):
                Mx[i, j] = (0.5 * m_pend[j - 1] + mulist[j]) * Llist[j - 1]
        else:
            Mx[i, i] = (I_pend[i - 1]
                        + ((rho ** 2) * m_pend[i - 1] + mulist[i]) * (Llist[i - 1] ** 2))
            for j in range(2, DOF_p):
                if j > i:
                    Mx[i, j] = (0.5 * m_pend[j - 1] + mulist[j]) * Llist[i - 1] * Llist[j - 1]

    Mx[0, 0] = mu_p(0)
    Mx[0, 1] = Llist[0] * mu_p(1)
    Mx[1, 1] = I_pend[0] + Llist[0] ** 2 * mu_p(1)
    Mx = (Mx + Mx.T) - np.diag(np.diag(Mx))

    Kx = np.zeros((DOF_p, DOF_p))
    for i in range(DOF_p):
        if i == 0:
            Kx[i, i] = 0.0
        elif i == 1:
            Kx[i, i] = ((1.0 - rho) * g * L_pend[0] * mulist[1]
                        - lamb * g * L_pend[0] * mulist[0])
        else:
            Kx[i, i] = (rho * m_pend[i - 1] + mulist[i]) * g * L_pend[i - 1]

    return Mx, Kx, DOF_p, (DOF_p - 1)


def build_torsion(L_susp, N_cables, r, m_gondola, m_one_cable_total, I_gondola_yaw):
    """Assemble torsion mass / stiffness matrices and per-element bifilar k_e."""
    N_nodes_t    = N_tether_torsion + 2
    N_elements_t = N_nodes_t - 1

    L_e = np.ones(N_elements_t) * (L_susp / N_elements_t)
    m_cables_total = N_cables * m_one_cable_total
    m_e = np.ones(N_elements_t) * (m_cables_total / N_elements_t)
    r_e = np.ones(N_elements_t) * r

    I_balloon_z = (1.0 / 6.0) * m_balloon * L_balloon ** 2

    def M_below(e):
        return m_gondola + np.sum(m_e[e + 1:])

    Mz = np.zeros((N_nodes_t, N_nodes_t))
    Mz[0, 0]   += I_balloon_z
    Mz[-1, -1] += I_gondola_yaw
    for e in range(N_elements_t):
        i, j = e, e + 1
        I_e = m_e[e] * r_e[e] ** 2
        Mz[i, i] += I_e / 3.0
        Mz[j, j] += I_e / 3.0
        Mz[i, j] += I_e / 6.0
        Mz[j, i] += I_e / 6.0

    Kz       = np.zeros((N_nodes_t, N_nodes_t))
    k_e_list = np.zeros(N_elements_t)
    for e in range(N_elements_t):
        i, j = e, e + 1
        k_e = (M_below(e) + 0.5 * m_e[e]) * g * r_e[e] ** 2 / L_e[e]
        k_e_list[e] = k_e
        Kz[i, i] += k_e
        Kz[j, j] += k_e
        Kz[i, j] -= k_e
        Kz[j, i] -= k_e

    return Mz, Kz, k_e_list, N_elements_t, m_cables_total


def modal_decomposition(M, K, zeta):
    """K Phi = lambda M Phi, mass-normalised; build physical-coord damping C."""
    Msym = 0.5 * (M + M.T)
    Ksym = 0.5 * (K + K.T)
    eigvals, eigvecs = eigh(Ksym, Msym)
    flexible = eigvals > mode_tol
    if not np.any(flexible):
        raise RuntimeError("No flexible modes found.")
    omega = np.sqrt(np.clip(eigvals[flexible], 0.0, None))
    freq  = omega / (2.0 * np.pi)
    Phi = eigvecs[:, flexible].copy()
    for i in range(Phi.shape[1]):
        Phi[:, i] /= np.sqrt(Phi[:, i].T @ Msym @ Phi[:, i])
    Cm = np.diag(2.0 * zeta * omega)
    C  = Msym @ Phi @ Cm @ Phi.T @ Msym
    return omega, freq, C


def simulate(M, C, K, q0, qdot0, t_eval):
    n = q0.size
    def rhs(t, y):
        q, qdot = y[:n], y[n:]
        qddot = np.linalg.solve(M, -C @ qdot - K @ q)
        return np.concatenate([qdot, qddot])
    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]),
                    np.concatenate([q0, qdot0]),
                    t_eval=t_eval, method="RK45", rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    q_hist    = sol.y[:n, :]
    qdot_hist = sol.y[n:, :]
    qddot_hist = np.linalg.solve(M, -C @ qdot_hist - K @ q_hist)
    return sol.t, q_hist, qdot_hist, qddot_hist


# ==================================================================
# 5. SINGLE-CASE DRIVER
# ==================================================================
def run_single_case(params):
    """Run one configuration and return a flat dict of outputs."""
    L_susp            = float(params["L_susp"])
    N_cables          = int(params["N_cables"])
    r                 = float(params["r"])
    m_gondola         = float(params["m_gondola"])
    m_one_cable_total = float(params["m_one_cable_total"])
    zeta_pendulum     = float(params["zeta_pendulum"])
    zeta_torsion      = float(params["zeta_torsion"])
    gondola_shape     = str(params["gondola_shape"])
    dist_name         = str(params["disturbance_case"])

    # Gondola geometry / inertias.  If params override R, L, etc., honour
    # them (only used for the baseline solid_cylinder case).  Otherwise
    # fall back to the per-shape representative geometry.
    geom = dict(gondola_geometry_cases[gondola_shape])
    if gondola_shape == "solid_cylinder":
        if "R_gondola" in params:
            geom["R"] = float(params["R_gondola"])
        if "L_gondola" in params:
            geom["h"] = float(params["L_gondola"])
    I_tilt, I_yaw, L_ref, size_str = gondola_inertia(m_gondola, gondola_shape, geom)

    dist = disturbance_cases[dist_name]
    pend_initial_angle_deg         = dist["pend_initial_angle_deg"]
    pend_initial_rate_deg_s        = dist["pend_initial_rate_deg_s"]
    torsion_initial_twist_deg      = dist["torsion_initial_twist_deg"]
    torsion_initial_yaw_rate_deg_s = dist["torsion_initial_yaw_rate_deg_s"]
    pend_impulse_Ns                = dist["pend_impulse_Ns"]
    torsion_impulse_Nms            = dist["torsion_impulse_Nms"]

    t_eval = np.arange(0.0, t_final + dt, dt)

    # --- Build & solve ---
    Mx, Kx, DOF_p, gondola_idx = build_pendulum(
        L_susp, N_cables, m_gondola, m_one_cable_total, I_tilt, L_ref)
    Mz, Kz, k_e_list, N_elem_t, m_cables_total = build_torsion(
        L_susp, N_cables, r, m_gondola, m_one_cable_total, I_yaw)

    omega_x, freq_x, Cx = modal_decomposition(Mx, Kx, zeta_pendulum)
    omega_z, freq_z, Cz = modal_decomposition(Mz, Kz, zeta_torsion)

    # ICs
    x0    = np.zeros(DOF_p)
    xdot0 = np.zeros(DOF_p)
    x0[1:]    = np.deg2rad(pend_initial_angle_deg)
    xdot0[1:] = np.deg2rad(pend_initial_rate_deg_s)
    if pend_impulse_Ns != 0.0:
        Jx = np.zeros(DOF_p); Jx[0] = pend_impulse_Ns
        xdot0 += np.linalg.solve(Mx, Jx)

    z0    = np.zeros(N_elem_t + 1)
    zdot0 = np.zeros(N_elem_t + 1)
    if torsion_initial_twist_deg != 0.0:
        z0 += np.linspace(0.0, np.deg2rad(torsion_initial_twist_deg), N_elem_t + 1)
    zdot0[-1] += np.deg2rad(torsion_initial_yaw_rate_deg_s)
    if torsion_impulse_Nms != 0.0:
        Jz = np.zeros(N_elem_t + 1); Jz[-1] = torsion_impulse_Nms
        zdot0 += np.linalg.solve(Mz, Jz)

    _, x_hist, xdot_hist, _ = simulate(Mx, Cx, Kx, x0, xdot0, t_eval)
    _, z_hist, zdot_hist, _ = simulate(Mz, Cz, Kz, z0, zdot0, t_eval)

    # --- Pendulum post-processing ---
    theta_g_p     = x_hist[gondola_idx, :]
    theta_dot_g_p = xdot_hist[gondola_idx, :]
    max_swing_deg     = float(np.rad2deg(np.max(np.abs(theta_g_p))))
    max_swing_rate    = float(np.rad2deg(np.max(np.abs(theta_dot_g_p))))

    T_pend_dom  = float(2.0 * np.pi / omega_x[0])
    f_pend_dom  = float(freq_x[0])
    t_set_pend  = float(4.0 / (zeta_pendulum * omega_x[0])) if zeta_pendulum > 0 else float("inf")

    # Cable tension (engineering estimate, same as combined script)
    L_eff = L_susp + 0.5 * L_ref
    M_susp = m_gondola + m_cables_total
    T_total = M_susp * (g * np.cos(theta_g_p) + L_eff * theta_dot_g_p ** 2)
    T_total_max = float(np.max(T_total))
    T_total_min = float(np.min(T_total))
    T_per_cable_max = T_total_max / max(N_cables, 1)
    T_per_cable_min = T_total_min / max(N_cables, 1)
    T_static_per_cable = (m_gondola + m_cables_total) * g / max(N_cables, 1)
    dyn_stat_ratio = T_per_cable_max / T_static_per_cable if T_static_per_cable > 0 else float("nan")

    # --- Torsion post-processing ---
    theta_g_z     = z_hist[-1, :]
    theta_dot_g_z = zdot_hist[-1, :]
    max_yaw_deg   = float(np.rad2deg(np.max(np.abs(theta_g_z))))
    max_yaw_rate  = float(np.rad2deg(np.max(np.abs(theta_dot_g_z))))

    T_tors_dom   = float(2.0 * np.pi / omega_z[0])
    f_tors_dom   = float(freq_z[0])
    t_set_tors   = float(4.0 / (zeta_torsion * omega_z[0])) if zeta_torsion > 0 else float("inf")

    eta_hist = z_hist[1:, :] - z_hist[:-1, :]              # per-element twist
    tau_hist = k_e_list[:, None] * eta_hist                # per-element torque
    max_torsional_torque   = float(np.max(np.abs(tau_hist)))
    max_torsion_elem_twist = float(np.rad2deg(np.max(np.abs(eta_hist))))

    # IMU sampling (quantitative)
    f_max_relevant = float(max(np.max(freq_x), np.max(freq_z)))
    imu_10x = 10.0 * f_max_relevant
    imu_20x = 20.0 * f_max_relevant

    out = {
        # case definition
        "L_susp": L_susp, "N_cables": N_cables, "r_attach": r,
        "m_gondola": m_gondola, "m_one_cable_total": m_one_cable_total,
        "m_cables_total": m_cables_total,
        "zeta_pendulum": zeta_pendulum, "zeta_torsion": zeta_torsion,
        "gondola_shape": gondola_shape,
        "I_gondola_tilt": I_tilt, "I_gondola_yaw": I_yaw,
        "gondola_reference_size": size_str,
        "disturbance_case": dist_name,
        "pend_initial_angle_deg":         pend_initial_angle_deg,
        "pend_initial_rate_deg_s":        pend_initial_rate_deg_s,
        "torsion_initial_twist_deg":      torsion_initial_twist_deg,
        "torsion_initial_yaw_rate_deg_s": torsion_initial_yaw_rate_deg_s,

        # dynamic outputs
        "dominant_pendulum_period_s":    T_pend_dom,
        "dominant_torsion_period_s":     T_tors_dom,
        "dominant_pendulum_frequency_hz": f_pend_dom,
        "dominant_torsion_frequency_hz":  f_tors_dom,
        "max_gondola_swing_angle_deg":   max_swing_deg,
        "max_gondola_swing_rate_deg_s":  max_swing_rate,
        "max_gondola_yaw_deg":           max_yaw_deg,
        "max_gondola_yaw_rate_deg_s":    max_yaw_rate,
        "settling_time_pendulum_s":      t_set_pend,
        "settling_time_torsion_s":       t_set_tors,

        # structural outputs
        "peak_total_cable_tension_N":     T_total_max,
        "peak_per_cable_tension_N":       T_per_cable_max,
        "minimum_per_cable_tension_N":    T_per_cable_min,
        "dynamic_static_tension_ratio":   dyn_stat_ratio,
        "maximum_torsional_torque_Nm":    max_torsional_torque,
        "maximum_torsion_element_twist_deg": max_torsion_elem_twist,

        # IMU
        "f_max_relevant_hz":            f_max_relevant,
        "required_imu_sampling_10x_hz": imu_10x,
        "required_imu_sampling_20x_hz": imu_20x,
    }
    feasible, violations = check_constraints(out)
    out["feasible"]   = "OK" if feasible else "FAIL"
    out["violations"] = violations
    return out


# ==================================================================
# 6. SWEEP DRIVER
# ==================================================================
def run_one_at_a_time(baseline, sweeps):
    """Vary one parameter at a time around baseline.  Returns rows list."""
    rows = []
    case_id = 0
    for sweep_name, values in sweeps.items():
        for v in values:
            params = dict(baseline)
            params[sweep_name] = v
            try:
                res = run_single_case(params)
                res["case_id"]          = case_id
                res["sweep_name"]       = sweep_name
                res["varied_parameter"] = sweep_name
                res["varied_value"]     = v
                rows.append(res)
                print("  [{:3d}] {:<18s} = {!s:<18s}  done".format(
                    case_id, sweep_name, v))
            except Exception as exc:
                print("  [{:3d}] {:<18s} = {!s:<18s}  ERROR: {}".format(
                    case_id, sweep_name, v, exc))
            case_id += 1
    return rows


# ==================================================================
# 7. TABLE FORMATTING (human-readable)
# ==================================================================
# Per-column display rules:  (short header, decimal places)
# decimals = -1 -> use general %g; integer columns auto-detected.
COLUMN_DISPLAY = {
    # case-definition columns
    "L_susp":                              ("L_susp [m]",          2),
    "N_cables":                            ("N_cables [-]",       -1),
    "r_attach":                            ("r [m]",               2),
    "m_gondola":                           ("m_gond [kg]",         1),
    "m_one_cable_total":                   ("m_cable [kg]",        2),
    "m_cables_total":                      ("m_cabl_tot [kg]",     2),
    "zeta_pendulum":                       ("zeta_pend [-]",       4),
    "zeta_torsion":                        ("zeta_tors [-]",       4),
    "gondola_shape":                       ("shape",              -1),
    "gondola_reference_size":              ("size",               -1),
    "I_gondola_tilt":                      ("I_tilt [kg.m2]",      1),
    "I_gondola_yaw":                       ("I_yaw  [kg.m2]",      1),
    "disturbance_case":                    ("disturbance",        -1),
    "pend_initial_angle_deg":              ("pend_a0 [deg]",       2),
    "pend_initial_rate_deg_s":             ("pend_w0 [deg/s]",     2),
    "torsion_initial_twist_deg":           ("tors_a0 [deg]",       2),
    "torsion_initial_yaw_rate_deg_s":      ("tors_w0 [deg/s]",     2),
    # dynamic outputs
    "dominant_pendulum_period_s":          ("T_pend [s]",          3),
    "dominant_torsion_period_s":           ("T_tors [s]",          3),
    "dominant_pendulum_frequency_hz":      ("f_pend [Hz]",         4),
    "dominant_torsion_frequency_hz":       ("f_tors [Hz]",         4),
    "max_gondola_swing_angle_deg":         ("max_swing [deg]",     2),
    "max_gondola_swing_rate_deg_s":        ("max_swing_rate [deg/s]", 2),
    "max_gondola_yaw_deg":                 ("max_yaw [deg]",       2),
    "max_gondola_yaw_rate_deg_s":          ("max_yaw_rate [deg/s]", 3),
    "settling_time_pendulum_s":            ("t_set_pend [s]",      1),
    "settling_time_torsion_s":             ("t_set_tors [s]",      1),
    # structural
    "peak_total_cable_tension_N":          ("T_cab_tot_max [N]",   1),
    "peak_per_cable_tension_N":            ("T_per_cab_max [N]",   1),
    "minimum_per_cable_tension_N":         ("T_per_cab_min [N]",   1),
    "dynamic_static_tension_ratio":        ("T_dyn/T_stat [-]",    3),
    "maximum_torsional_torque_Nm":         ("max_tau [N.m]",       3),
    "maximum_torsion_element_twist_deg":   ("max_eta [deg]",       4),
    # IMU
    "f_max_relevant_hz":                   ("f_max [Hz]",          4),
    "required_imu_sampling_10x_hz":        ("IMU_fs_10x [Hz]",     1),
    "required_imu_sampling_20x_hz":        ("IMU_fs_20x [Hz]",     1),
    # constraint columns
    "feasible":                            ("feasible",           -1),
    "violations":                          ("violations",         -1),
    # trend-summary columns
    "value_low":                           ("value_low",          -1),
    "value_high":                          ("value_high",         -1),
    "relative_change":                     ("rel_change",         -1),
    "trend":                               ("trend",              -1),
    "sweep_name":                          ("sweep",              -1),
    "output":                              ("output",             -1),
}


def _display_header(col):
    return COLUMN_DISPLAY.get(col, (col, -1))[0]


def _display_decimals(col):
    return COLUMN_DISPLAY.get(col, (col, -1))[1]


def _format_cell(v, decimals):
    """Format a single cell value as a string (no padding)."""
    if v is None:
        return "-"
    if isinstance(v, (int, np.integer)):
        return "{:d}".format(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "inf" if v > 0 else "-inf"
        if decimals == -1:
            # general format
            if v == 0 or 1e-3 <= abs(v) < 1e6:
                return "{:.4g}".format(v)
            return "{:.3e}".format(v)
        if abs(v) >= 1e6 or (v != 0 and abs(v) < 10 ** (-decimals - 2)):
            return "{:.3e}".format(v)
        return "{:.{}f}".format(v, decimals)
    return str(v)


def format_table(df, columns, title=None):
    """Return a list of strings forming a nicely aligned table."""
    columns = [c for c in columns if c in df.columns]
    headers = [_display_header(c) for c in columns]
    decimals = [_display_decimals(c) for c in columns]

    # build all cell strings first to compute column widths
    cell_grid = []
    for _, row in df.iterrows():
        cell_grid.append([_format_cell(row[c], d) for c, d in zip(columns, decimals)])

    widths = []
    for j, h in enumerate(headers):
        col_max = max([len(h)] + [len(cell_grid[i][j]) for i in range(len(cell_grid))])
        widths.append(col_max)

    sep = "  "
    lines = []
    if title is not None:
        bar = "=" * max(len(title), sum(widths) + len(sep) * (len(widths) - 1))
        lines.append(bar)
        lines.append(title)
        lines.append(bar)
    lines.append(sep.join("{:>{}s}".format(headers[j], widths[j]) for j in range(len(headers))))
    lines.append(sep.join("-" * w for w in widths))
    for row_cells in cell_grid:
        lines.append(sep.join("{:>{}s}".format(row_cells[j], widths[j]) for j in range(len(row_cells))))
    return lines


def print_spec_table(df_full, sweep_name, baseline):
    """Print and return the per-spec table for one swept parameter."""
    sub = df_full[df_full["sweep_name"] == sweep_name].copy()
    if sub.empty:
        return None, None

    try:
        sub = sub.sort_values("varied_value", kind="stable").reset_index(drop=True)
    except TypeError:
        sub = sub.reset_index(drop=True)

    cols = SPEC_TABLE_COLUMNS.get(sweep_name, ["varied_value"])
    cols = [c for c in cols if c in sub.columns]
    # Always append constraint status
    for extra in ("feasible", "violations"):
        if extra in sub.columns and extra not in cols:
            cols.append(extra)

    title = "Table: sensitivity to '{}'  (other parameters at baseline)".format(sweep_name)
    lines = format_table(sub, cols, title=title)
    print("\n" + "\n".join(lines))

    out_df = sub[cols + ["sweep_name", "varied_parameter", "varied_value"]].copy()
    return out_df, lines


def round_dataframe_for_csv(df):
    """Return a copy of df with numeric columns rounded per COLUMN_DISPLAY."""
    out = df.copy()
    for c in out.columns:
        d = _display_decimals(c)
        if d == -1:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(d)
    return out


# ==================================================================
# 8. TREND SUMMARY
# ==================================================================
def trend_label(x_values, y_values, rel_tol=0.05):
    y = np.asarray(y_values, dtype=float)
    if y.size < 2 or np.any(~np.isfinite(y)):
        return "n/a"
    y0, y1 = y[0], y[-1]
    if abs(y1 - y0) <= rel_tol * max(abs(y0), 1e-12):
        return "approximately unchanged"
    diffs = np.diff(y)
    if np.all(diffs >= 0):
        return "increases"
    if np.all(diffs <= 0):
        return "decreases"
    return "non-monotonic"


# Outputs to evaluate trends on (numeric only)
TREND_OUTPUTS = [
    "dominant_pendulum_period_s", "dominant_torsion_period_s",
    "dominant_pendulum_frequency_hz", "dominant_torsion_frequency_hz",
    "max_gondola_swing_angle_deg", "max_gondola_swing_rate_deg_s",
    "max_gondola_yaw_deg", "max_gondola_yaw_rate_deg_s",
    "settling_time_pendulum_s", "settling_time_torsion_s",
    "peak_total_cable_tension_N", "peak_per_cable_tension_N",
    "minimum_per_cable_tension_N", "dynamic_static_tension_ratio",
    "maximum_torsional_torque_Nm", "maximum_torsion_element_twist_deg",
    "required_imu_sampling_10x_hz", "required_imu_sampling_20x_hz",
]


def compute_trend_summary(df_full):
    """For each (sweep, output) compute monotonic trend label + delta."""
    rows = []
    for sweep_name in df_full["sweep_name"].unique():
        sub = df_full[df_full["sweep_name"] == sweep_name].copy()
        try:
            sub = sub.sort_values("varied_value", kind="stable").reset_index(drop=True)
        except TypeError:
            sub = sub.reset_index(drop=True)
        x = sub["varied_value"].tolist()
        for out in TREND_OUTPUTS:
            if out not in sub.columns:
                continue
            y = sub[out].tolist()
            label = trend_label(x, y)
            try:
                y0, y1 = float(y[0]), float(y[-1])
                rel = (y1 - y0) / y0 if y0 != 0 else float("nan")
            except (TypeError, ValueError):
                y0, y1, rel = float("nan"), float("nan"), float("nan")
            rows.append({
                "sweep_name":    sweep_name,
                "output":        out,
                "value_low":     y0,
                "value_high":    y1,
                "relative_change": rel,
                "trend":         label,
            })
    return pd.DataFrame(rows)


def print_trend_summary(trend_df):
    """Print one nicely aligned trend table per sweep.  Returns the joined
    text (so it can also be saved to disk)."""
    all_lines = []
    header = "TREND SUMMARY  (output at LOWEST vs HIGHEST swept value)"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    all_lines.append(header)
    all_lines.append("=" * len(header))
    for sweep_name in trend_df["sweep_name"].unique():
        sub = trend_df[trend_df["sweep_name"] == sweep_name].copy()
        # Format value_low/high using each output's display decimals
        def fmt_val(out_name, v):
            d = _display_decimals(out_name)
            return _format_cell(v, d if d != -1 else 4)
        sub["value_low_str"]  = [fmt_val(o, v) for o, v in zip(sub["output"], sub["value_low"])]
        sub["value_high_str"] = [fmt_val(o, v) for o, v in zip(sub["output"], sub["value_high"])]
        sub["rel_change_str"] = [
            ("{:+.1f}%".format(100.0 * v) if (isinstance(v, (int, float, np.floating)) and np.isfinite(v))
             else "-")
            for v in sub["relative_change"]
        ]
        sub["output_disp"] = [_display_header(o) for o in sub["output"]]

        title = "Sweep: {}".format(sweep_name)
        # Manual alignment (custom columns)
        out_w   = max(len("output"), sub["output_disp"].str.len().max())
        lo_w    = max(len("value_low"), sub["value_low_str"].str.len().max())
        hi_w    = max(len("value_high"), sub["value_high_str"].str.len().max())
        rc_w    = max(len("rel_change"), sub["rel_change_str"].str.len().max())
        tr_w    = max(len("trend"), sub["trend"].str.len().max())
        bar = "-" * (out_w + lo_w + hi_w + rc_w + tr_w + 8)
        block = []
        block.append("")
        block.append(title)
        block.append(bar)
        block.append("  {:<{}s}  {:>{}s}  {:>{}s}  {:>{}s}  {}".format(
            "output", out_w, "value_low", lo_w, "value_high", hi_w,
            "rel_change", rc_w, "trend"))
        block.append(bar)
        for _, row in sub.iterrows():
            block.append("  {:<{}s}  {:>{}s}  {:>{}s}  {:>{}s}  {}".format(
                row["output_disp"], out_w,
                row["value_low_str"], lo_w,
                row["value_high_str"], hi_w,
                row["rel_change_str"], rc_w,
                row["trend"]))
        text = "\n".join(block)
        print(text)
        all_lines.extend(block)
    return "\n".join(all_lines)


# ==================================================================
# 9. PLOTS  (spec section 18)
# ==================================================================
PLOTS = [
    ("L_susp",            "L_susp",            "dominant_pendulum_period_s"),
    ("L_susp",            "L_susp",            "max_gondola_swing_rate_deg_s"),
    ("L_susp",            "L_susp",            "peak_per_cable_tension_N"),
    ("r",                 "r_attach",          "dominant_torsion_period_s"),
    ("r",                 "r_attach",          "max_gondola_yaw_rate_deg_s"),
    ("zeta_pendulum",     "zeta_pendulum",     "settling_time_pendulum_s"),
    ("zeta_torsion",      "zeta_torsion",      "settling_time_torsion_s"),
    ("N_cables",          "N_cables",          "peak_per_cable_tension_N"),
    ("m_gondola",         "m_gondola",         "peak_per_cable_tension_N"),
    ("m_one_cable_total", "m_one_cable_total", "peak_per_cable_tension_N"),
    ("gondola_shape",     "gondola_shape",     "dominant_torsion_period_s"),
]


def make_plots(df_full):
    os.makedirs(plots_dir, exist_ok=True)
    for sweep_name, xcol, ycol in PLOTS:
        sub = df_full[df_full["sweep_name"] == sweep_name].copy()
        if sub.empty or xcol not in sub.columns or ycol not in sub.columns:
            continue
        try:
            sub = sub.sort_values("varied_value", kind="stable")
        except TypeError:
            pass
        fig, ax = plt.subplots(figsize=(7, 4))
        x = sub[xcol].tolist()
        y = sub[ycol].tolist()
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in x):
            ax.plot(x, y, "o-")
        else:
            ax.plot(range(len(x)), y, "o-")
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels([str(v) for v in x], rotation=20)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title("{}: {} vs {}".format(sweep_name, xcol, ycol))
        ax.grid(True)
        fig.tight_layout()
        fname = "{}__{}.png".format(sweep_name, ycol)
        fig.savefig(os.path.join(plots_dir, fname), dpi=120)
        plt.close(fig)


# ==================================================================
# 10. MAIN
# ==================================================================
def main():
    os.makedirs(results_dir, exist_ok=True)
    print("Baseline case:")
    for k, v in baseline_case.items():
        print("  {:<22s} = {}".format(k, v))
    print("\nRunning one-at-a-time sweeps...")
    rows = run_one_at_a_time(baseline_case, SWEEPS)
    df = pd.DataFrame(rows)

    # Save full table (rounded for readability)
    full_csv = os.path.join(results_dir, "sensitivity_one_at_a_time.csv")
    round_dataframe_for_csv(df).to_csv(full_csv, index=False)
    print("\nSaved full sweep -> {}".format(full_csv))

    # Per-parameter spec tables (printed + saved as both .csv and .txt)
    all_text_blocks = []
    for sweep_name in SWEEPS.keys():
        sub_table, lines = print_spec_table(df, sweep_name, baseline_case)
        if sub_table is None:
            continue
        out_csv = os.path.join(results_dir, "sensitivity_{}.csv".format(sweep_name))
        round_dataframe_for_csv(sub_table).to_csv(out_csv, index=False)
        out_txt = os.path.join(results_dir, "sensitivity_{}.txt".format(sweep_name))
        with open(out_txt, "w") as f:
            f.write("\n".join(lines) + "\n")
        all_text_blocks.append("\n".join(lines))

    # One combined .txt with every per-parameter table
    combined_txt = os.path.join(results_dir, "sensitivity_tables_all.txt")
    with open(combined_txt, "w") as f:
        f.write("\n\n".join(all_text_blocks) + "\n")

    # Constraint summary
    print("\n" + "=" * 70)
    print("CONSTRAINT SUMMARY")
    print("Limits: L_susp >= 10 m  |  swing/yaw <= 10 deg  |  IMU_10x <= 2000 Hz")
    print("=" * 70)
    if "feasible" in df.columns:
        fail_df = df[df["feasible"] == "FAIL"][
            ["sweep_name", "varied_parameter", "varied_value", "violations"]
        ].copy()
        if fail_df.empty:
            print("  All cases PASS all constraints.")
        else:
            n_pass = (df["feasible"] == "OK").sum()
            n_fail = (df["feasible"] == "FAIL").sum()
            print("  PASS: {}   FAIL: {}".format(n_pass, n_fail))
            print()
            for sweep_name in fail_df["sweep_name"].unique():
                s = fail_df[fail_df["sweep_name"] == sweep_name]
                print("  Sweep '{}':".format(sweep_name))
                for _, r in s.iterrows():
                    print("    {} = {} -> {}".format(
                        r["varied_parameter"], r["varied_value"], r["violations"]))
        # Save constraint failure table
        constr_csv = os.path.join(results_dir, "sensitivity_constraint_violations.csv")
        fail_df.to_csv(constr_csv, index=False)
        print("\nSaved constraint violations -> {}".format(constr_csv))

    # Trend summary
    trend_df = compute_trend_summary(df)
    trend_text = print_trend_summary(trend_df)
    trend_csv = os.path.join(results_dir, "sensitivity_trend_summary.csv")
    round_dataframe_for_csv(trend_df).to_csv(trend_csv, index=False)
    trend_txt = os.path.join(results_dir, "sensitivity_trend_summary.txt")
    with open(trend_txt, "w") as f:
        f.write(trend_text + "\n")
    print("\nSaved trend summary -> {} (and .txt)".format(trend_csv))

    # Plots
    make_plots(df)
    print("Saved plots -> {}/".format(plots_dir))
    print("\nAll outputs in: {}/".format(results_dir))


if __name__ == "__main__":
    main()
