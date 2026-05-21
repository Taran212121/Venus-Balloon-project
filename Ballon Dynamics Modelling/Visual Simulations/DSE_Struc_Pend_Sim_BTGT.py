"""
VISTA Combined Dynamics — Pendulum + Torsion (uncoupled)
=========================================================

What this script does
---------------------
This is an engineering design tool for the VISTA Venus balloon mission.
It models two independent structural dynamics problems side-by-side:

  1. PENDULUM  (lateral swing):  Mx·ẍ + Cx·ẋ + Kx·x = Fx(t)
     The balloon, suspension cables, and gondola are discretised as a
     chain of rigid segments connected by revolute joints.  The model
     predicts how the gondola tilts and swings under disturbances.

  2. TORSION   (yaw twist):      Mz·z̈ + Cz·ż + Kz·z = Tz(t)
     The same chain is loaded in torsion.  The bifilar (multi-cable)
     arrangement provides a restoring torque that keeps the gondola
     from spinning.  The model predicts gondola yaw and cable twist.

The two models share the same geometry and mass inputs, but are
NOT coupled — pendulum swinging and yaw twisting are treated as
independent motions.  Coupling would only matter for large amplitudes.

How to use it
-------------
  1. Edit section 2 (USER INPUTS) to match your design.
  2. Run the script.  It prints a V&V check log, then a summary table.
  3. Two plot windows appear: one for pendulum, one for torsion.
  4. If a WARNING is printed, read the explanation next to it.

Key assumptions / limitations
------------------------------
  - Small-angle linearisation (valid up to ~10°).
  - Uniform modal damping ratio applied to all flexible modes.
  - The lateral translation DOF (y1) has no restoring force —
    the whole chain can drift sideways freely (no aerodynamic drag).
  - Cable tension estimate is a first-order engineering approximation.

Source scripts
--------------
  - DSE_Structures_Pendulums.py  (pendulum model reference)
  - DSE_Structures_Torsion.py    (torsion model reference)
  - Kassarian et al. 2021        (bifilar stiffness derivation)
"""

# ==================================
# 1. IMPORTS
# ==================================
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# ==================================
# 2. USER INPUTS
# ==================================
# *** CHANGE THESE VALUES TO MATCH YOUR DESIGN ***
# All numbers below propagate automatically into both the pendulum
# and torsion models.  Physical parameters come first, followed by
# numerical settings and then the disturbance inputs for each run.

# --- Shared physical parameters ---
m_balloon = 300.0       # kg     balloon mass
L_balloon = 9.0         # m      balloon diameter (thin spherical shell)

m_gondola = 100.0       # kg     gondola mass
R_gondola = 0.3         # m      gondola radius (vertical thin cylinder)
L_gondola = 0.4         # m      gondola length

L_susp = 15.0           # m      total suspension length (balloon -> gondola)

m_barometer = 0.03      # kg     single barometer mass
L_barometer = 0.01      # m      single barometer length
m_towbody = 12.0          # kg     total towbody tether mass
L_towbody = [15.0, 85.0, 200.0]           # m  towbody lengths

# Discretization fidelity (kept independent for the two physics)
# Increase these for a finer mesh (more segments = more DOFs = slower, but
# more accurate for distributed cable mass effects).  1 is sufficient for
# preliminary design.  They do NOT need to be equal to each other.
N_tether_pendulum = 6     # number of internal tether segments in pendulum chain
N_towbody_pendulum = [8, 16, 48]   # number of internal tether segments in pendulum chain for towbody

N_tether_torsion  = 6      # number of internal tether segments in torsion chain

# Cable / suspension geometry
N_cables          = 4      # physical number of cables (bifilar / multifilar)
m_one_cable_total = 0.5    # kg     mass of ONE complete cable over L_susp
r                 = 0.5   # m      cable radius from yaw axis (torsion)

# Environment
g = 8.72                   # m/s^2  Venus gravity at ~52 km

# Modal damping ratios
# A single damping ratio is applied to ALL flexible modes (proportional
# modal damping).  Pendulum is mainly aerodynamic; torsion is very lightly
# damped in the Venus 52 km environment.  Both are conservative estimates.
zeta_pendulum = 5e-3      # [-]  pendulum modal damping (0.5 % of critical)
zeta_torsion  = 1e-3      # [-]  torsion  modal damping (0.1 % of critical)

# Balloon CG/CP offset coefficient (pendulum buoyancy effect)
# lamb > 0 makes balloon tilt destabilising (CG below CP) which LOWERS
# the effective pendulum stiffness of the balloon segment.  Set lamb=0
# to treat the balloon as a passive mass with no buoyancy contribution.
lamb = 0.1                 # [-]  buoyancy destabilisation coefficient

# --- Simulation time settings ---
t_final = 50.0             # s
dt      = 0.05             # s
t_eval  = np.arange(0.0, t_final + dt, dt)

# --- Pendulum disturbance inputs (combine freely) ---
# The three inputs below can be set simultaneously; their effects add in
# the initial-condition vector.  Common usage:
#   • pend_initial_angle_deg  – sudden wind gust that tilts the whole chain
#   • pend_initial_rate_deg_s – initial swing velocity
#   • pend_impulse_Ns         – sharp horizontal force applied at the top
#     NOTE: this impulse acts on the lateral DOF y1, NOT directly on the
#     gondola tilt angle.  To see a visible tilt response you need a large
#     impulse (~hundreds of N·s) because the system inertia is ~250 kg.
pend_initial_angle_deg  = 10.0    # deg     uniform initial chain swing angle
pend_initial_rate_deg_s = 0.0    # deg/s   uniform initial chain swing rate
pend_impulse_Ns         = 5.0    # N s     generalized lateral impulse on y1

# --- Torsion disturbance inputs (combine freely) ---
# torsion_initial_twist_deg applies a linearly distributed pre-twist from
# balloon (0°) to gondola (the specified value).  This mimics a wound-up
# cable scenario.  The angular impulse is applied directly to the gondola.
torsion_initial_twist_deg     = 10.0   # deg     stored twist (linearly distributed)
torsion_initial_yaw_rate_deg_s = 0.0  # deg/s   gondola initial yaw rate
torsion_impulse_Nms           = 0.0   # N m s   angular impulse at gondola

# Numerical tolerance for separating rigid-body modes
mode_tol = 1e-8


# ==================================
# 3. BUILD PENDULUM MODEL  (Mx, Kx)
# ==================================
# Physical picture:
#   Imagine a chain hanging from a fixed point (the balloon attachment).
#   Each link is a rigid body (balloon / cable segment / gondola) that can
#   tilt independently.  The whole chain can also slide sideways (y1).
#
# Discretization (matches DSE_Structures_Pendulums.py):
#   N_p     = N_tether_pendulum + 2  (balloon + tether segments + gondola)
#   DOF_p   = N_p + 1                (y1 lateral + N_p tilt angles)
#
# Coordinate vector:  x = [ y1, theta_0, theta_1, ..., theta_{N_p-1} ]
#   - y1            : absolute lateral displacement of the top of the chain
#   - theta_0       : tilt angle of the balloon body
#   - theta_1..-2   : tilt angles of the cable segments
#   - theta_{N_p-1} : tilt angle of the gondola  (the quantity we care about most)
#
# Equations of motion derived from Lagrange's equations:
#   Mx ẍ + Cx ẋ + Kx x = Fx(t)

N_p   = N_tether_pendulum + 2 + sum(N_towbody_pendulum) + 3
DOF_p = N_p + 1

idx_balloon = 0
idx_tether_end = N_tether_pendulum + 1
idx_gondola = idx_tether_end
idx_towbody_start = idx_gondola + 1
idx_chain_start = idx_gondola + 1
idx_barometer1 = idx_towbody_start + N_towbody_pendulum[0]
idx_barometer2 = idx_barometer1 + N_towbody_pendulum[1]
idx_barometer3 = idx_barometer2 + N_towbody_pendulum[2]


# Per-segment mass / length / inertia lists
m_pend = [m_balloon]
L_pend = [L_balloon]
I_pend = [(1.0 / 6.0) * m_balloon * L_balloon ** 2]   # thin spherical shell

# Divide the total cable mass equally among the tether segments.
# Each segment is a uniform rod, so its moment of inertia is (1/12) m L^2.
m_one_tether_pend = (N_cables * m_one_cable_total) / N_tether_pendulum
L_one_tether_pend = L_susp / N_tether_pendulum
for _ in range(N_tether_pendulum):
    m_pend.append(m_one_tether_pend)
    L_pend.append(L_one_tether_pend)
    I_pend.append((1.0 / 12.0) * m_one_tether_pend * L_one_tether_pend ** 2)

# Gondola is a short solid cylinder; its inertia combines a hoop term
# (0.5 m R^2 for rotation about the long axis) and a rod term (1/12 m L^2
# for rotation perpendicular to the axis, which is the relevant pendulum tilt).
m_pend.append(m_gondola)
L_pend.append(L_gondola)
I_pend.append(0.5 * m_gondola * R_gondola ** 2 + (1.0 / 12.0) * m_gondola * L_gondola ** 2)

# Divide the total cable mass equally among the towbody segments.
# Each segment is a uniform rod, so its moment of inertia is (1/12) m L^2.
m_one_towbody_pend = (m_towbody) / sum(N_towbody_pendulum)
for i in range(len(N_towbody_pendulum)):
    for _ in range(N_towbody_pendulum[i]):
        m_pend.append(m_one_towbody_pend)
        Lseg1 = L_towbody[i] / N_towbody_pendulum[i]
        L_pend.append(Lseg1)
        I_pend.append((1.0 / 12.0) * m_one_tether_pend * (Lseg1) ** 2)
        
    m_pend.append(m_barometer)
    L_pend.append(L_barometer)
    I_pend.append((1.0 / 12.0) * m_barometer * (L_barometer) ** 2)

def mu_p(k):
    """
    Cumulative mass from chain element k down to the gondola.

    The pendulum chain is indexed top -> bottom:
        0 = balloon, 1..N_tether_pendulum = tether segments, N_p-1 = gondola.
    mu_p(k) = sum of all body masses from index k to N_p-1 (inclusive).
    Used to build both Mx (kinetic-energy moment-arm contributions) and
    Kx (gravitational restoring torque on each angle is proportional to
    the mass HANGING BELOW that hinge).

    Special cases:
        mu_p(0)        = total system mass (balloon + tethers + gondola)
        mu_p(1)        = mass below the balloon (i.e. tether + gondola)
        mu_p(>=N_p)    = 0
    """
    if k >= N_p:
        return 0.0
    return sum(m_pend[k:N_p])

# rho = fractional CG position along the balloon body (0 = top, 1 = bottom).
# For a sphere, rho = 0.5 (CG at centre).  This sets the effective moment arm
# of the balloon's mass in the stiffness and mass matrix expressions.
rho   = 0.5
Llist = list(L_pend)
Llist[0] = (1.0 - rho) * L_pend[0]   # distance from balloon hinge to balloon CG

# Pre-compute cumulative hanging masses for all chain positions.
# mulist[i] = total mass at or below hinge i.  Used heavily in Mx and Kx.
mulist = [mu_p(i) for i in range(N_p + 1)]

# --- Mass matrix Mx ---------------------------------------------------
# Row/column 0 corresponds to lateral translation y1; rows 1..end are tilt
# angles. The structure follows the Lagrangian derivation in
# DSE_Structures_Pendulums.py:
#   - Diagonal angle terms i>=1 contain segment own-inertia plus the
#     parallel-axis contribution of all mass hanging below that hinge,
#     evaluated at the moment-arm Llist[i-1].
#   - Off-diagonal terms M[i,j] for i,j>=2 (j>i) couple two tilt angles
#     through their shared swinging chain (cross-arm products).
#   - The first row/column couples lateral translation y1 to each tilt
#     angle through (1/2*m_j + mu(j+1))*L_j  (translation of body CG
#     induced by tilt of body j).
# After assembly the upper triangle is mirrored to enforce symmetry.
Mx = np.zeros((DOF_p, DOF_p))
for i in range(DOF_p):
    if i == 0:
        # Row 0 = lateral translation y1.
        # The coupling between y1 and tilt angle theta_j (column j>=2) is the
        # horizontal CG displacement of all mass hanging below hinge j when
        # theta_j changes: (0.5*m_j + mu_p(j+1)) * L_j.
        for j in range(2, DOF_p):
            Mx[i, j] = (0.5 * m_pend[j - 1] + mulist[j]) * Llist[j - 1]
    else:
        # Diagonal: own rotational inertia I_i plus the parallel-axis term for
        # all mass hanging below hinge i, at moment-arm Llist[i-1].
        Mx[i, i] = (I_pend[i - 1] + ((rho ** 2) * m_pend[i - 1] + mulist[i]) * (Llist[i - 1] ** 2))
        # Off-diagonal (i<j): two different tilt angles share a portion of the
        # swinging chain.  The coupling term is the product of their moment arms
        # times the common hanging mass.
        for j in range(2, DOF_p):
            if j > i:
                Mx[i, j] = (0.5 * m_pend[j - 1] + mulist[j]) * Llist[i - 1] * Llist[j - 1]

# Special top-of-chain entries (balloon row/column handled separately
# because the balloon CG offset rho alters the standard expression).
Mx[0, 0] = mu_p(0)                             # total mass (y1 inertia)
Mx[0, 1] = Llist[0] * mu_p(1)                 # y1 <-> balloon-tilt coupling
Mx[1, 1] = I_pend[0] + Llist[0] ** 2 * mu_p(1)  # balloon-tilt diagonal

# Mirror the upper triangle to the lower triangle to enforce symmetry.
# The formula below copies upper->lower without double-counting the diagonal.
Mx = (Mx + Mx.T) - np.diag(np.diag(Mx))

# --- Stiffness matrix Kx ---------------------------------------------
# IMPORTANT (modelling note):
#   The original DSE_Structures_Pendulums.py uses a DIAGONAL stiffness
#   matrix in the small-angle linearization. Each tilt angle has its own
#   gravitational restoring torque proportional to (mass hanging below
#   that hinge) * g * (segment length). There is no off-diagonal angle
#   coupling in K, even though M is fully coupled.
#   Lateral translation y1 has Kx[0,0] = 0 -> the chain can drift sideways
#   freely (no horizontal restoring force).
#   - i = 1 (balloon tilt) carries an extra (-lamb * g * L0 * mu(0)) term
#     that captures the buoyancy CG/CP offset (destabilizing if lamb>0).
Kx = np.zeros((DOF_p, DOF_p))
for i in range(DOF_p):
    if i == 0:
        # y1 has NO restoring force.  The chain can drift laterally without
        # any horizontal spring pulling it back (no aerodynamic drag modelled).
        Kx[i, i] = 0.0
    elif i == 1:
        # Balloon tilt stiffness: gravitational restoring term (hanging mass
        # below the balloon hinge × g × moment arm) MINUS the destabilising
        # buoyancy term (lamb × total mass × g × moment arm).
        # If lamb > 0 the buoyancy centre is above the CG, which destabilises
        # the balloon tilt and reduces the effective stiffness.
        Kx[i, i] = (1.0 - rho) * g * L_pend[0] * mulist[1] - lamb * g * L_pend[0] * mulist[0]
    else:
        # Tether segment and gondola tilt stiffnesses: standard pendulum formula.
        # Each hinge i is restrained by the weight of everything hanging below it.
        Kx[i, i] = (rho * m_pend[i - 1] + mulist[i]) * g * L_pend[i - 1]


# ==================================
# 4. BUILD TORSION MODEL  (Mz, Kz)
# ==================================
# Physical picture:
#   The same chain now carries a TWIST (yaw rotation about the vertical axis).
#   The cables act like torsional springs.  The restoring torque comes from
#   gravity: tilting a twisted cable pair under gravity produces a net torque
#   that tries to unwind the twist (bifilar pendulum effect — see Kassarian 2021).
#
# Discretization (kept identical to DSE_Structures_Torsion.py):
#   N_nodes_torsion    = N_tether_torsion + 2  (balloon node + cable nodes + gondola node)
#   N_elements_torsion = N_nodes_torsion - 1   (one element per cable span)
#
# Coordinate vector:  z = [ theta_0, theta_1, ..., theta_{N-1} ]
#   - theta_0      : yaw angle of the balloon (can rotate freely)
#   - theta_1..-2  : yaw angles of the cable nodes
#   - theta_{N-1}  : yaw angle of the gondola (the quantity we care about most)
#
# Equations of motion:
#   Mz·z̈ + Cz·ż + Kz·z = Tz(t)

N_nodes_torsion    = N_tether_torsion + 2
N_elements_torsion = N_nodes_torsion - 1

L_element_t = np.ones(N_elements_torsion) * (L_susp / N_elements_torsion)
m_cables_total = N_cables * m_one_cable_total
m_element_t    = np.ones(N_elements_torsion) * (m_cables_total / N_elements_torsion)
r_element_t    = np.ones(N_elements_torsion) * r

I_balloon_z = (1.0 / 6.0) * m_balloon * L_balloon ** 2
I_gondola_z = 0.5 * m_gondola * R_gondola ** 2


def M_below_t(e):
    """
    Mass below torsion element e, EXCLUDING that element's own half-mass.

    Element e connects torsion nodes e and e+1.  Below it sit:
        - all cable elements with index > e
        - the gondola
    The element's own half-mass is NOT included here because the bifilar
    stiffness expression
            k_e = (M_below(e) + 0.5 * m_e) * g * r^2 / L_e
    adds it explicitly.  Splitting it this way keeps M_below additive
    when summed across elements.
    """
    return m_gondola + np.sum(m_element_t[e + 1:])


# --- Mass matrix Mz ---
# Each cable element contributes a consistent mass matrix for a bar rotating
# about its end.  For element e connecting nodes i and j, the element yaw
# inertia is I_e = m_e * r^2 (all mass treated as a thin ring at radius r).
# The 2×2 consistent element mass matrix is:
#   I_e * [[1/3, 1/6],   <- diagonal (self) terms: 1/3 of element inertia
#          [1/6, 1/3]]   <- off-diagonal (coupled) terms: 1/6 of element inertia
# The balloon and gondola rigid-body inertias are added to their own diagonal.
Mz = np.zeros((N_nodes_torsion, N_nodes_torsion))
Mz[0, 0]   += I_balloon_z    # balloon spin inertia (thin shell about axis)
Mz[-1, -1] += I_gondola_z    # gondola spin inertia (solid cylinder about axis)
for e in range(N_elements_torsion):
    i, j = e, e + 1
    I_e = m_element_t[e] * r_element_t[e] ** 2   # element yaw inertia
    Mz[i, i] += I_e / 3.0    # self-term at node i
    Mz[j, j] += I_e / 3.0    # self-term at node j (consistent mass: 1/3 of I_e)
    Mz[i, j] += I_e / 6.0    # coupling i<->j
    Mz[j, i] += I_e / 6.0    # (Mz is symmetric by construction)

# --- Stiffness matrix Kz and per-element stiffnesses ---
# The bifilar (multi-cable) torsional stiffness of element e comes from
# Kassarian et al. (2021), Section 2.2.  The formula is:
#
#   k_e = (M_below + 0.5 * m_element) * g * r^2 / L_element
#
# where M_below is the total mass hanging BELOW element e (gondola + lower
# cables), and r is the lateral distance from each cable to the yaw axis.
# The 2×2 element stiffness matrix is a standard spring between nodes i and j:
#   k_e * [[ 1, -1],
#          [-1,  1]]
Kz       = np.zeros((N_nodes_torsion, N_nodes_torsion))
k_e_list = np.zeros(N_elements_torsion)    # saved for verification check (v)
for e in range(N_elements_torsion):
    i, j = e, e + 1
    k_e = (M_below_t(e) + 0.5 * m_element_t[e]) * g * r_element_t[e] ** 2 / L_element_t[e]
    k_e_list[e] = k_e
    Kz[i, i] += k_e     # spring pulls node i back
    Kz[j, j] += k_e     # spring pulls node j back
    Kz[i, j] -= k_e     # coupling: twisting i also pulls j
    Kz[j, i] -= k_e     # (Kz is symmetric)


# ==================================
# 5. MODAL ANALYSIS AND DAMPING
# ==================================
# Modal analysis solves K·Φ = λ·M·Φ, which gives the natural frequencies
# (ω_i = √λ_i) and mode shapes (columns of Φ) of the linearised system.
# We then build a physical-coordinate damping matrix C so that each flexible
# mode decays at the specified damping ratio ζ, while rigid-body modes
# (zero-frequency, e.g. free lateral drift of y1 or free balloon yaw) are
# left undamped.

def modal_decomposition(M, K, zeta, tol, label=""):
    """
    Solve the symmetric generalized eigenvalue problem  K Phi = lambda M Phi
    and build a modal-damping matrix in physical coordinates.

    Steps:
      1. Symmetrize M and K (defensive against tiny numerical asymmetry).
      2. Use scipy.linalg.eigh(K, M), which is the appropriate solver for
         symmetric K and symmetric positive-definite M. It returns:
             eigvals : sorted ascending generalized eigenvalues lambda_i
             eigvecs : columns are eigenvectors v_i satisfying
                       v_i^T M v_j = delta_ij  (M-orthonormal already).
      3. Warn if any lambda_i < -tol (sign/convention issue or instability).
      4. Identify FLEXIBLE modes by lambda_i > tol; raise if none exist.
         Modes with lambda <= tol are rigid-body / zero-frequency modes
         (e.g. free balloon yaw, lateral translation y1 with Kx[0,0]=0).
      5. omega_i = sqrt(lambda_i),  f_i = omega_i / (2 pi).
      6. Re-mass-normalize Phi (defensive; eigh should already deliver this).
      7. Build modal damping in physical coordinates:
             C = M Phi diag(2 zeta omega) Phi^T M
         which produces the same uncoupled modal damping ratio zeta on
         every flexible mode while preserving the rigid-body modes
         undamped (their column is excluded from Phi).

    Returns omega, freq, Phi, eigvals, flexible-mask, C.
    """
    Msym = 0.5 * (M + M.T)
    Ksym = 0.5 * (K + K.T)
    eigvals, eigvecs = eigh(Ksym, Msym)

    # Negative-eigenvalue check (sign / convention / instability indicator)
    if np.any(eigvals < -tol):
        print("WARNING [{}]: negative generalized eigenvalues detected (min = {:.3e}).".format(
            label or "modal", float(np.min(eigvals))))
        print("         May indicate instability, sign/convention issues, or an unstable equilibrium.")

    flexible = eigvals > tol
    if not np.any(flexible):
        raise RuntimeError(
            "modal_decomposition [{}]: no flexible modes found above tol={:.3e}. "
            "All generalized eigenvalues are <= tol.".format(label or "modal", tol)
        )

    omega = np.sqrt(np.clip(eigvals[flexible], 0.0, None))
    freq  = omega / (2.0 * np.pi)

    Phi = eigvecs[:, flexible].copy()
    for i in range(Phi.shape[1]):
        Phi[:, i] /= np.sqrt(Phi[:, i].T @ Msym @ Phi[:, i])

    Cm = np.diag(2.0 * zeta * omega)
    C  = Msym @ Phi @ Cm @ Phi.T @ Msym
    return omega, freq, Phi, eigvals, flexible, C


# Run modal analysis for both subsystems.
# Returns: natural frequencies (Hz), mode shapes, all eigenvalues,
#          flexible-mode mask, and the physical damping matrix C.
omega_x, freq_x, Phi_x, eigvals_x, flex_x, Cx = modal_decomposition(Mx, Kx, zeta_pendulum, mode_tol, label="pendulum")
omega_z, freq_z, Phi_z, eigvals_z, flex_z, Cz = modal_decomposition(Mz, Kz, zeta_torsion,  mode_tol, label="torsion")


# ==================================
# 6. INITIAL CONDITIONS AND DISTURBANCES
# ==================================
# The initial state vector is built by combining three independent inputs
# for each subsystem.  Think of them as three separate "scenarios" that
# can be superimposed:
#   (a) Initial displacement  – chain already tilted / twisted at t=0
#   (b) Initial velocity      – chain already moving at t=0
#   (c) Impulse               – instantaneous force/torque applied at t=0
#       (an impulse J changes the initial velocity by qdot0 += M^{-1} J)

# --- Pendulum: x = [y1, theta_0, ..., theta_{N_p-1}] ---
# Initial-condition convention:
#   The whole pendulum chain starts with a UNIFORM swing angle / rate.
#   pend_initial_angle_deg and pend_initial_rate_deg_s are applied to every
#   angular DOF (indices 1..end).  The lateral translation y1 (index 0) stays
#   at zero unless you set it explicitly below.
gondola_angle_idx = DOF_p - 1   # last theta is gondola tilt
y1_idx            = 0

x0_pend    = np.zeros(DOF_p)
xdot0_pend = np.zeros(DOF_p)

x0_pend[1:]    = np.deg2rad(pend_initial_angle_deg)
xdot0_pend[1:] = np.deg2rad(pend_initial_rate_deg_s)

# pend_impulse_Ns is a lateral impulse on y1 (think: a sudden horizontal
# wind gust integrated over its short duration).  It changes the initial
# VELOCITY of all DOFs through the full inverse mass matrix, so some tilt
# velocity is injected into the angle DOFs via the off-diagonal Mx terms.
# IMPORTANT: Because the system is heavy (~250 kg), even a 1000 N·s impulse
# only produces ~4 m/s lateral velocity — most of which stays in y1 drift.
if pend_impulse_Ns != 0.0:
    Jx = np.zeros(DOF_p)
    Jx[y1_idx] = pend_impulse_Ns
    xdot0_pend += np.linalg.solve(Mx, Jx)   # qdot0 += M^{-1} J

# --- Torsion: z = [theta_0, ..., theta_{N-1}] ---
z0_tors    = np.zeros(N_nodes_torsion)
zdot0_tors = np.zeros(N_nodes_torsion)

if torsion_initial_twist_deg != 0.0:
    # Linear distribution: balloon starts at 0°, gondola starts at the
    # specified angle.  Intermediate nodes are interpolated proportionally.
    z0_tors += np.linspace(0.0, np.deg2rad(torsion_initial_twist_deg), N_nodes_torsion)

zdot0_tors[-1] += np.deg2rad(torsion_initial_yaw_rate_deg_s)  # gondola spin

if torsion_impulse_Nms != 0.0:
    # Angular impulse applied at the gondola (bottom node only).
    Jz = np.zeros(N_nodes_torsion)
    Jz[-1] = torsion_impulse_Nms
    zdot0_tors += np.linalg.solve(Mz, Jz)   # qdot0 += M^{-1} J


# --- Continuous disturbance functions (zero by default) ---
# These functions define CONTINUOUS external loads that persist during the
# simulation.  They are called at every integrator time step.
# To add a wind gust or a sinusoidal torque, modify these functions.
# Examples:
#   return np.array([F_lateral, 0, 0, 0])   # constant horizontal force on y1
#   return np.array([0, 0, A*np.sin(2*pi*f*t)])   # sinusoidal torque on gondola

def pendulum_force_input(t):
    """
    Generalized force vector for the pendulum system at time t.
    Length = DOF_p = N_tether_pendulum + 3.
    Index 0 = force on y1 (lateral), indices 1..end = torques on tilt angles.
    Set to zero for free-decay / impulse-only simulations.
    """
    return np.zeros(DOF_p)


def torsion_torque_input(t):
    """
    Generalized torque vector for the torsion system at time t.
    Length = N_nodes_torsion = N_tether_torsion + 2.
    Index 0 = torque at balloon node, index -1 = torque at gondola.
    Set to zero for free-decay / impulse-only simulations.
    """
    return np.zeros(N_nodes_torsion)


# ==================================
# 7. TIME INTEGRATION
# ==================================
# Both systems are integrated in physical coordinates (no modal reduction).
# This means the full M, C, K matrices are used at each time step, which
# is fine for the small DOF counts used here (typically 3–10 DOFs per system).

def simulate_second_order(M, C, K, force_fn, q0, qdot0, t_eval):
    """
    Integrate a linear second-order MDOF system
            M qddot + C qdot + K q = f(t)
    in PHYSICAL coordinates (no modal projection).

    Approach:
      - Cast as a first-order system  y = [q ; qdot],
        ydot = [ qdot ; M^{-1} (f - C qdot - K q) ].
      - At every step, accelerations are obtained via np.linalg.solve(M, ...),
        i.e. by solving a linear system rather than forming an explicit
        matrix inverse. This is more accurate when M is ill-conditioned.
      - solve_ivp with RK45 + tight tolerances (rtol=1e-8, atol=1e-10)
        is used. RK45 is non-stiff, which is appropriate here because
        the highest eigen-frequency (~few tens of Hz) is well within
        explicit-integrator territory at the chosen dt.
      - After integration, ACCELERATIONS are reconstructed from the
        equation of motion at every stored time step (NOT by finite
        differencing the velocity history). This avoids derivative noise.

    Returns t, q_hist, qdot_hist, qddot_hist (all sampled at t_eval).
    """
    n = q0.size

    def rhs(t, y):
        q    = y[:n]
        qdot = y[n:]
        f    = force_fn(t)
        qddot = np.linalg.solve(M, f - C @ qdot - K @ q)
        return np.concatenate([qdot, qddot])

    y0 = np.concatenate([q0, qdot0])
    sol = solve_ivp(
        rhs, (t_eval[0], t_eval[-1]), y0,
        t_eval=t_eval, method="RK45",
        rtol=1e-8, atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    q_hist    = sol.y[:n, :]
    qdot_hist = sol.y[n:, :]

    # Reconstruct acceleration from the equation of motion (no finite differences)
    f_hist = np.column_stack([force_fn(ti) for ti in sol.t])
    qddot_hist = np.linalg.solve(M, f_hist - C @ qdot_hist - K @ q_hist)

    return sol.t, q_hist, qdot_hist, qddot_hist


t_p, x_hist, xdot_hist, xddot_hist = simulate_second_order(
    Mx, Cx, Kx, pendulum_force_input, x0_pend, xdot0_pend, t_eval
)
t_z, z_hist, zdot_hist, zddot_hist = simulate_second_order(
    Mz, Cz, Kz, torsion_torque_input, z0_tors, zdot0_tors, t_eval
)


# ==================================
# 8. POST-PROCESSING
# ==================================
# Extract the engineering quantities that matter for the design:
#   - Pendulum: gondola tilt angle, rate, acceleration, peak cable tension
#   - Torsion:  gondola yaw (absolute and relative to balloon), element loads

# --- Pendulum: gondola tilt is the main coordinate ---
theta_pend       = x_hist[gondola_angle_idx, :]
theta_dot_pend   = xdot_hist[gondola_angle_idx, :]
theta_ddot_pend  = xddot_hist[gondola_angle_idx, :]

max_pend_angle_deg = np.rad2deg(np.max(np.abs(theta_pend)))
max_pend_rate_dps  = np.rad2deg(np.max(np.abs(theta_dot_pend)))
max_pend_acc_dps2  = np.rad2deg(np.max(np.abs(theta_ddot_pend)))

# Maximum angle across ANY angular pendulum coordinate (excluding y1)
theta_all = x_hist[1:, :]
max_any_pend_angle_deg = np.rad2deg(np.max(np.abs(theta_all)))

period_pend       = 1.0 / freq_x
settling_pend     = 4.0 / (zeta_pendulum * omega_x)
T_pend_dom        = period_pend[0]
ts_pend_approx    = settling_pend[0]

# --- Peak cable tension (FIRST-ORDER ENGINEERING ESTIMATE) ---
# Simple-pendulum tension on the suspended assembly:
#     T(t) = M_susp * ( g * cos(theta) + L_eff * theta_dot^2 )
# This is NOT a detailed cable load model. It ignores cable mass kinematics,
# coupling with torsion, gondola tilt inertia, and lateral translation y1.
M_susp_pend = m_gondola + m_cables_total
L_eff_pend  = L_susp + 0.5 * L_gondola      # ~ distance from suspension top to gondola CG
T_cable_hist = M_susp_pend * (g * np.cos(theta_pend) + L_eff_pend * theta_dot_pend ** 2)
T_cable_max  = float(np.max(T_cable_hist))      # signed max (not abs)
T_cable_min  = float(np.min(T_cable_hist))
T_per_cable      = T_cable_max / max(N_cables, 1)
T_per_cable_min  = T_cable_min / max(N_cables, 1)

# Static reference for sanity check
T_static_per_cable = (m_gondola + m_cables_total) * g / max(N_cables, 1)
T_dyn_static_ratio = T_per_cable / T_static_per_cable if T_static_per_cable > 0 else float("nan")

# --- Torsion: gondola yaw + relative-to-balloon yaw ---
theta_g       = z_hist[-1, :]
theta_dot_g   = zdot_hist[-1, :]
theta_ddot_g  = zddot_hist[-1, :]

theta_rel_gb      = z_hist[-1, :]      - z_hist[0, :]
theta_dot_rel_gb  = zdot_hist[-1, :]   - zdot_hist[0, :]
theta_ddot_rel_gb = zddot_hist[-1, :]  - zddot_hist[0, :]

max_g_yaw_deg    = np.rad2deg(np.max(np.abs(theta_g)))
max_g_rate_dps   = np.rad2deg(np.max(np.abs(theta_dot_g)))
max_g_acc_dps2   = np.rad2deg(np.max(np.abs(theta_ddot_g)))
max_rel_yaw_deg  = np.rad2deg(np.max(np.abs(theta_rel_gb)))
max_rel_rate_dps = np.rad2deg(np.max(np.abs(theta_dot_rel_gb)))
max_rel_acc_dps2 = np.rad2deg(np.max(np.abs(theta_ddot_rel_gb)))

eta_hist     = z_hist[1:, :] - z_hist[:-1, :]
tau_hist     = k_e_list[:, None] * eta_hist
max_eta_deg  = np.rad2deg(np.max(np.abs(eta_hist), axis=1))
max_tau_elem = np.max(np.abs(tau_hist), axis=1)

period_tors    = 1.0 / freq_z
settling_tors  = 4.0 / (zeta_torsion * omega_z)
T_tors_dom     = period_tors[0]
ts_tors_approx = settling_tors[0]


# ==================================
# 9. VERIFICATION AND VALIDATION CHECKS
# ==================================
# These checks run automatically every time the script executes.
# They verify that the assembled matrices are physically sensible and that
# the numerical results are trustworthy.  Any WARNING printed here should
# be investigated before using the results for design decisions.

print("=" * 72)
print("VERIFICATION AND VALIDATION CHECKS")
print("=" * 72)

# ----- Pendulum model -----
print("--- Pendulum ---")
print("Mx symmetric:", np.allclose(Mx, Mx.T))
print("Kx symmetric:", np.allclose(Kx, Kx.T))

eig_Mx = np.linalg.eigvalsh(0.5 * (Mx + Mx.T))
print("All Mx eigenvalues > 0       :", bool(np.all(eig_Mx > 0)))
print("Min Mx eigenvalue            : {:.5e}".format(float(np.min(eig_Mx))))
print("Min Kx generalized eigenvalue: {:.5e}".format(float(np.min(eigvals_x))))
print("# near-zero generalized eigenvalues (|lambda|<tol):",
      int(np.sum(np.abs(eigvals_x) < mode_tol)))
print("Any negative generalized eigenvalues:",
      bool(np.any(eigvals_x < -mode_tol)),
      "(min = {:.3e})".format(float(np.min(eigvals_x))))
print("# flexible pendulum modes    :", int(np.sum(flex_x)))
if int(np.sum(flex_x)) == 0:
    print("WARNING: no flexible pendulum modes found.")

print("Pendulum natural frequencies [Hz]:", np.round(freq_x, 5))

if dt > T_pend_dom / 20.0:
    print("WARNING: dt = {:.4g} s > T_pend_dom/20 = {:.4g} s. Time resolution may be too coarse.".format(
        dt, T_pend_dom / 20.0))
else:
    print("Time-step resolution (pendulum): OK  (dt = {:.4g} s, T_dom/20 = {:.4g} s)".format(
        dt, T_pend_dom / 20.0))

if abs(pend_initial_angle_deg) > 10.0:
    print("WARNING: |pend_initial_angle_deg| = {:.2f} deg exceeds 10 deg small-angle assumption.".format(
        abs(pend_initial_angle_deg)))
else:
    print("Small-angle assumption (pendulum IC): OK  (|angle| = {:.2f} deg)".format(
        abs(pend_initial_angle_deg)))

# ----- Torsion model -----
print("\n--- Torsion ---")
print("Mz symmetric:", np.allclose(Mz, Mz.T))
print("Kz symmetric:", np.allclose(Kz, Kz.T))

eig_Mz = np.linalg.eigvalsh(0.5 * (Mz + Mz.T))
eig_Kz = np.linalg.eigvalsh(0.5 * (Kz + Kz.T))
print("All Mz eigenvalues > 0       :", bool(np.all(eig_Mz > 0)))
print("Min Mz eigenvalue            : {:.5e}".format(float(np.min(eig_Mz))))
print("Min Kz eigenvalue            : {:.5e}".format(float(np.min(eig_Kz))))
n_near_zero_Kz = int(np.sum(np.abs(eig_Kz) < mode_tol))
print("# near-zero Kz eigenvalues   :", n_near_zero_Kz)
print("Rigid-body yaw eigenvalue(s) [rad^2/s^2]:", np.round(eigvals_z[~flex_z], 5))

if n_near_zero_Kz < 1:
    print("WARNING: no near-zero torsional eigenvalue found, but free balloon yaw should produce one.")
else:
    print("Free-balloon-yaw rigid-body mode: OK  (1 expected, {} found).".format(n_near_zero_Kz))

flex_eigs_z = eigvals_z[flex_z]
if not bool(np.all(flex_eigs_z > 0)):
    print("WARNING: not all non-rigid torsional generalized eigenvalues are positive.")
else:
    print("All non-rigid torsional generalized eigenvalues > 0: OK")

if dt > T_tors_dom / 20.0:
    print("WARNING: dt = {:.4g} s > T_tors_dom/20 = {:.4g} s. Time resolution may be too coarse.".format(
        dt, T_tors_dom / 20.0))
else:
    print("Time-step resolution (torsion): OK  (dt = {:.4g} s, T_dom/20 = {:.4g} s)".format(
        dt, T_tors_dom / 20.0))

if abs(torsion_initial_twist_deg) > 10.0:
    print("WARNING: |torsion_initial_twist_deg| = {:.2f} deg exceeds 10 deg small-angle assumption.".format(
        abs(torsion_initial_twist_deg)))
else:
    print("Small-angle assumption (torsion IC): OK  (|twist| = {:.2f} deg)".format(
        abs(torsion_initial_twist_deg)))

# ----- Physical sanity checks -----
print("\n--- Physical sanity checks ---")
print("Static per-cable load        : {:.2f} N".format(T_static_per_cable))
print("Peak per-cable tension       : {:.2f} N".format(T_per_cable))
print("Min  per-cable tension       : {:.2f} N".format(T_per_cable_min))
print("Dynamic / static per-cable ratio: {:.3f}".format(T_dyn_static_ratio))
if T_dyn_static_ratio > 2.0:
    print("WARNING: peak per-cable tension is more than 2x the static estimate.")
if T_cable_min < 0.0:
    print("WARNING: minimum simple-estimate cable tension is negative ({:.2f} N).".format(T_cable_min))
    print("         This indicates predicted slack / nonphysical tension; case needs review.")

# ----- Output consistency checks -----
print("\n--- Output consistency checks ---")
print("Pendulum/torsion time arrays equal length:", t_p.shape == t_z.shape and len(t_p) == len(t_z))
for name, arr in [("x_hist", x_hist), ("xdot_hist", xdot_hist), ("xddot_hist", xddot_hist),
                  ("z_hist", z_hist), ("zdot_hist", zdot_hist), ("zddot_hist", zddot_hist)]:
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("NaN or Inf detected in {}.".format(name))
print("All state histories finite (no NaN/Inf): OK")


# ==================================
# 9b. RIGOROUS VERIFICATION
# ==================================
# Stronger cross-checks that compare model results against analytical
# references or mathematical identities.  If any of these fail, something
# is fundamentally wrong with the model — not just a tuning issue.

print("\n" + "=" * 72)
print("RIGOROUS VERIFICATION  (independent analytical cross-checks)")
print("=" * 72)

# --- (i) Cholesky positive-definiteness of M -----------------------------
# Mx and Mz must be SYMMETRIC POSITIVE-DEFINITE for the second-order
# system to be physically meaningful. A successful Cholesky factorization
# is a stronger test than "all eigenvalues > 0" because it also detects
# numerical ill-conditioning that eigvalsh might smooth over.
def cholesky_pd_check(M, name):
    """
    Verify that mass matrix M is symmetric positive-definite using a
    Cholesky factorisation.  A positive-definite M is required for the
    equations of motion to be physically meaningful (kinetic energy must
    be positive for any non-zero velocity).
    Returns True if the check passes, False otherwise.
    """
    try:
        np.linalg.cholesky(0.5 * (M + M.T))
        print("  [OK] {} is symmetric positive-definite (Cholesky succeeded).".format(name))
        return True
    except np.linalg.LinAlgError as e:
        print("  [FAIL] {} Cholesky failed: {}".format(name, e))
        return False

cholesky_pd_check(Mx, "Mx")
cholesky_pd_check(Mz, "Mz")

# --- (ii) Simple-pendulum analytical limit -------------------------------
# A point mass M_susp hung at length L_eff under gravity g should swing
# with angular frequency  omega = sqrt(g / L_eff)  and period
# T = 2*pi*sqrt(L_eff / g). Compare with the lowest pendulum-model mode.
L_eff_simple   = L_susp + 0.5 * L_gondola
omega_simple   = np.sqrt(g / L_eff_simple)
T_simple       = 2.0 * np.pi / omega_simple
T_model_dom    = 1.0 / freq_x[0]
rel_err_period = abs(T_model_dom - T_simple) / T_simple
print("\n  (ii) Simple-pendulum limit (point mass, L_eff = L_susp + L_gondola/2):")
print("       Analytical period T = 2*pi*sqrt(L_eff/g)        : {:8.4f} s".format(T_simple))
print("       Lowest pendulum-model period (1/freq_x[0])      : {:8.4f} s".format(T_model_dom))
print("       Relative error                                  : {:7.2%}".format(rel_err_period))
if rel_err_period > 0.30:
    print("       NOTE: discrepancy >30%. Expected: the chain pendulum lowest mode")
    print("             corresponds to the WHOLE chain swinging about y1 (free lateral")
    print("             drift partially mixed with the buoyancy-CG/CP term lamb).")
    print("             For a clean simple-pendulum check, set lamb=0 and use a single")
    print("             rigid body (N_tether_pendulum=0).")

# --- (iii) Energy conservation (zero damping) ----------------------------
# With zeta -> 0 and zero forcing, total mechanical energy
#   E(t) = 0.5 * qdot^T M qdot + 0.5 * q^T K q
# should be exactly conserved. Modal damping makes E decay monotonically
# at a rate set by zeta. We check both: the decay rate of E for each
# subsystem must be NEGATIVE (energy never grows) and small relative to
# the dominant period (no spurious growth from the integrator).
def energy_check(M, K, q_hist, qdot_hist, zeta, name):
    """
    Compute total mechanical energy E(t) = 0.5 qdot^T M qdot + 0.5 q^T K q
    at every stored time step and check that it behaves correctly:
      - With damping (zeta > 0): E should decrease monotonically.
      - Without damping (zeta = 0): E should be conserved (constant).
    An unexpected INCREASE in energy would indicate a bug in the integrator
    or a sign error in M/K.  The check is skipped if E(0) <= 0 (no initial
    energy to track).
    """
    E = (0.5 * np.einsum("it,ij,jt->t", qdot_hist, M, qdot_hist)
         + 0.5 * np.einsum("it,ij,jt->t", q_hist,  K, q_hist))
    if E[0] <= 0:
        print("  [{}] initial energy is non-positive ({:.3e}); skipping decay check.".format(name, E[0]))
        return
    dE_rel = (E[-1] - E[0]) / E[0]
    if zeta > 0:
        verdict = "decay expected" if dE_rel < 0 else "WARNING: energy grew"
    else:
        verdict = "should be ~0 (conservative)"
    print("  [{}] E(0) = {:.4e},  E(end) = {:.4e},  (E_end-E_0)/E_0 = {:+.3%}  -> {}".format(
        name, E[0], E[-1], dE_rel, verdict))

print("\n  (iii) Energy bookkeeping (mechanical energy along the trajectory):")
energy_check(Mx, Kx, x_hist, xdot_hist, zeta_pendulum, "pendulum")
energy_check(Mz, Kz, z_hist, zdot_hist, zeta_torsion,  "torsion")

# --- (iv) Free lateral drift sanity check (pendulum) ---------------------
# The lateral translation DOF y1 has Kx[0,0] = 0 and zero coupling to
# other angles in K. Without a restoring force, an impulse on y1 should
# cause uniform drift in y1 (ballistic). Here we verify Kx really has a
# zero row/column for y1.
row_norm = np.linalg.norm(Kx[0, :])
col_norm = np.linalg.norm(Kx[:, 0])
print("\n  (iv) Lateral free-drift property of y1 (pendulum):")
print("       ||Kx[0, :]|| = {:.3e}    ||Kx[:, 0]|| = {:.3e}".format(row_norm, col_norm))
if row_norm < 1e-12 and col_norm < 1e-12:
    print("       OK: y1 has no restoring force and no K-coupling -> free lateral drift.")
else:
    print("       WARNING: y1 column/row of Kx is not zero; lateral drift not free.")

# --- (v) Torsion single-element analytical check -------------------------
# Discretized chain stiffness in series should match the analytical
# bifilar lumped stiffness within numerical error.
k_t_lumped  = (m_gondola + 0.5 * m_cables_total) * g * r ** 2 / L_susp
k_eq_series = 1.0 / np.sum(1.0 / k_e_list)
print("\n  (v) Torsion bifilar stiffness consistency:")
print("       Lumped analytical k_t  = (M_g + m_c/2) g r^2 / L : {:.6f} N m/rad".format(k_t_lumped))
print("       Series-equivalent of element stiffnesses          : {:.6f} N m/rad".format(k_eq_series))
print("       Relative error                                    : {:7.3%}".format(
    abs(k_eq_series - k_t_lumped) / k_t_lumped))


# ==================================
#%% 10. SUMMARY OUTPUT
# ==================================
# Formatted summary table that collects the most important design outputs
# from both the pendulum and torsion analyses into a single printout.
# Sections A–F are printed, followed by the raw M and K matrices.

W   = 78
sep = "+" + "-" * (W - 2) + "+"
hdr = lambda s: "| " + s.ljust(W - 4) + " |"

def row2(label, val_str):
    s = "{:<48s} {:>22s}".format(label, val_str)
    return "| " + s.ljust(W - 4) + " |"

print("\n" + sep)
print(hdr("VISTA COMBINED DYNAMICS — DESIGN SUMMARY"))
print(sep)
print(hdr("Disturbance inputs (user-defined combination)"))
print(sep)
print(row2("Pendulum uniform-chain init angle [deg]",  "{:.4f}".format(pend_initial_angle_deg)))
print(row2("Pendulum uniform-chain init rate  [deg/s]","{:.4f}".format(pend_initial_rate_deg_s)))
print(row2("Pendulum lateral impulse on y1 [N s]",     "{:.4f}".format(pend_impulse_Ns)))
print(row2("Torsion initial stored twist [deg]",       "{:.4f}".format(torsion_initial_twist_deg)))
print(row2("Torsion initial gondola yaw rate [deg/s]", "{:.4f}".format(torsion_initial_yaw_rate_deg_s)))
print(row2("Torsion angular impulse [N m s]",          "{:.4f}".format(torsion_impulse_Nms)))
print(hdr("(Pendulum IC applied uniformly to ALL angular DOFs; y1 left at 0."))
print(hdr(" pend_impulse_Ns is a generalized lateral impulse on y1.)"))
print(sep)

# --- Pendulum modal table ---
print(hdr("A. Pendulum modal properties (zeta = {:.3e})".format(zeta_pendulum)))
print("| " + "{:<8s} {:>16s} {:>16s} {:>20s}".format(
    "Mode", "Freq [Hz]", "Period [s]", "t_settle (~2%) [s]").ljust(W - 4) + " |")
print(sep)
for i in range(len(freq_x)):
    print("| " + "{:<8d} {:>16.5f} {:>16.5f} {:>20.2f}".format(
        i + 1, freq_x[i], period_pend[i], settling_pend[i]).ljust(W - 4) + " |")
print(sep)

# --- Pendulum extremes ---
print(hdr("B. Pendulum gondola response extremes"))
print(sep)
print(row2("Max gondola pendulum angle [deg]",       "{:.4f}".format(max_pend_angle_deg)))
print(row2("Max ANY chain angle [deg]",              "{:.4f}".format(max_any_pend_angle_deg)))
print(row2("Max pendulum rate  [deg/s]",             "{:.4f}".format(max_pend_rate_dps)))
print(row2("Max pendulum accel [deg/s^2]",           "{:.4f}".format(max_pend_acc_dps2)))
print(row2("Peak cable tension (total) [N]  *est.",  "{:.2f}".format(T_cable_max)))
print(row2("Min  cable tension (total) [N]  *est.",  "{:.2f}".format(T_cable_min)))
print(row2("Peak cable tension per cable [N] *est.", "{:.2f}".format(T_per_cable)))
print(row2("Min  cable tension per cable [N] *est.", "{:.2f}".format(T_per_cable_min)))
print(row2("Static per-cable load [N]",              "{:.2f}".format(T_static_per_cable)))
print(row2("Dynamic / static per-cable ratio [-]",   "{:.3f}".format(T_dyn_static_ratio)))
print(hdr("(* first-order engineering estimate, NOT a detailed cable load model)"))
print(sep)

# --- Torsion modal table ---
print(hdr("C. Torsion modal properties (zeta = {:.3e})".format(zeta_torsion)))
print("| " + "{:<8s} {:>16s} {:>16s} {:>20s}".format(
    "Mode", "Freq [Hz]", "Period [s]", "t_settle (~2%) [s]").ljust(W - 4) + " |")
print(sep)
for i in range(len(freq_z)):
    print("| " + "{:<8d} {:>16.5f} {:>16.5f} {:>20.2f}".format(
        i + 1, freq_z[i], period_tors[i], settling_tors[i]).ljust(W - 4) + " |")
print(sep)

# --- Torsion extremes ---
print(hdr("D. Torsion gondola response extremes  (absolute  /  relative-to-balloon)"))
print(sep)
print(row2("Max gondola yaw         [deg]",
           "{:.4f}  /  {:.4f}".format(max_g_yaw_deg,  max_rel_yaw_deg)))
print(row2("Max gondola yaw rate    [deg/s]",
           "{:.4f}  /  {:.4f}".format(max_g_rate_dps, max_rel_rate_dps)))
print(row2("Max gondola yaw accel   [deg/s^2]",
           "{:.4f}  /  {:.4f}".format(max_g_acc_dps2, max_rel_acc_dps2)))
print(sep)

# --- Element loads ---
print(hdr("E. Torsion element design loads"))
print("| " + "{:<10s} {:>22s} {:>22s} {:>14s}".format(
    "Element", "Max twist [deg]", "Max torque [N m]", "Notes").ljust(W - 4) + " |")
print(sep)
for e in range(N_elements_torsion):
    note = ""
    if e == 0:
        note = "balloon-side"
    elif e == N_elements_torsion - 1:
        note = "gondola-side"
    print("| " + "{:<10d} {:>22.5f} {:>22.6f} {:>14s}".format(
        e, max_eta_deg[e], max_tau_elem[e], note).ljust(W - 4) + " |")
print(sep)

# --- Combined interpretation ---
print(hdr("F. Combined interpretation"))
print(sep)
print(row2("Dominant pendulum period [s]",        "{:.4f}".format(T_pend_dom)))
print(row2("Dominant torsion period  [s]",        "{:.4f}".format(T_tors_dom)))
print(row2("Dominant pendulum frequency [Hz]",    "{:.4f}".format(freq_x[0])))
print(row2("Dominant torsion frequency  [Hz]",    "{:.4f}".format(freq_z[0])))

f_min_pair = min(freq_x[0], freq_z[0])
freq_close = abs(freq_x[0] - freq_z[0]) / f_min_pair < 0.2
print(row2("Pendulum/torsion freq separation [%]",
           "{:.1f}".format(100.0 * abs(freq_x[0] - freq_z[0]) / f_min_pair)))
print(sep)
if freq_close:
    print(hdr("WARNING: dominant pendulum and torsion frequencies are within 20%."))
    print(hdr("         Coupled effects may be significant; revisit when the coupled"))
    print(hdr("         model is implemented."))
    print(sep)

# --- Matrices ---
np.set_printoptions(precision=5, suppress=True, linewidth=160)
print("\n" + "=" * 72)
print("PENDULUM MASS MATRIX  Mx  [kg / kg m / kg m^2]")
print("=" * 72)
print(Mx)

print("\n" + "=" * 72)
print("PENDULUM STIFFNESS MATRIX  Kx  [N/m or N m/rad on diagonal]")
print("=" * 72)
print(Kx)

print("\n" + "=" * 72)
print("TORSION MASS MATRIX  Mz  [kg m^2]")
print("=" * 72)
print(Mz)

print("\n" + "=" * 72)
print("TORSION STIFFNESS MATRIX  Kz  [N m / rad]")
print("=" * 72)
print(Kz)

# ==================================
#%% 11. PLOTS
# ==================================
# Two figure windows:
#   Figure 1 – Pendulum: gondola tilt angle, rate, and acceleration vs time.
#   Figure 2 – Torsion:  gondola yaw (absolute and relative to balloon) vs time.

# --- Pendulum (3 plots: angle / rate / acceleration) ---
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axes[0].plot(t_p, np.rad2deg(theta_pend),      label="Gondola pendulum angle")
axes[0].set_ylabel("Angle [deg]")
axes[0].set_title("Pendulum response")
axes[0].grid(True); axes[0].legend()

axes[1].plot(t_p, np.rad2deg(theta_dot_pend),  label="Angular velocity")
axes[1].set_ylabel("Rate [deg/s]")
axes[1].grid(True); axes[1].legend()

axes[2].plot(t_p, np.rad2deg(theta_ddot_pend), label="Angular acceleration")
axes[2].set_xlabel("Time [s]")
axes[2].set_ylabel("Accel [deg/s$^2$]")
axes[2].grid(True); axes[2].legend()
fig.tight_layout()

# --- Torsion (3 plots: angle / rate / acceleration), each with abs + relative ---
figT, axesT = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
axesT[0].plot(t_z, np.rad2deg(theta_g),       label="Absolute (gondola)")
axesT[0].plot(t_z, np.rad2deg(theta_rel_gb),  label="Relative to balloon", linestyle="--")
axesT[0].set_ylabel("Yaw [deg]")
axesT[0].set_title("Torsion response (gondola)")
axesT[0].grid(True); axesT[0].legend()

axesT[1].plot(t_z, np.rad2deg(theta_dot_g),       label="Absolute (gondola)")
axesT[1].plot(t_z, np.rad2deg(theta_dot_rel_gb),  label="Relative to balloon", linestyle="--")
axesT[1].set_ylabel("Yaw rate [deg/s]")
axesT[1].grid(True); axesT[1].legend()

axesT[2].plot(t_z, np.rad2deg(theta_ddot_g),      label="Absolute (gondola)")
axesT[2].plot(t_z, np.rad2deg(theta_ddot_rel_gb), label="Relative to balloon", linestyle="--")
axesT[2].set_xlabel("Time [s]")
axesT[2].set_ylabel("Yaw accel [deg/s$^2$]")
axesT[2].grid(True); axesT[2].legend()
figT.tight_layout()

plt.show()


#%% Simulation
# ==================================
# y-z TRAJECTORY VISUALISATION
# ==================================

from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

# ----------------------------------
# y-coordinate function
# ----------------------------------
def y_coord(y1, k, theta):
    """
    Horizontal coordinate of body k CG.
    """

    y = y1

    # -------------------------
    # BALLOON
    # -------------------------
    y += (1.0 - rho) * L_balloon * np.sin(theta[0])

    # -------------------------
    # MAIN TETHER
    # -------------------------
    if k > idx_balloon and k <= idx_tether_end:

        for i in range(1, k):
            y += L_pend[i] * np.sin(theta[i])

        y += 0.5 * L_pend[k] * np.sin(theta[k])

    # -------------------------
    # GONDOLA
    # -------------------------
    elif k == idx_gondola:

        for i in range(1, idx_tether_end):
            y += L_pend[i] * np.sin(theta[i])

        y += 0.5 * L_pend[k] * np.sin(theta[k])

    # -------------------------
    # TOWBODY + BAROMETERS
    # -------------------------
    elif k > idx_gondola:

        # full motion up to gondola
        for i in range(1, idx_tether_end):
            y += L_pend[i] * np.sin(theta[i])

        y += 0.5 * L_pend[idx_gondola] * np.sin(theta[idx_gondola])

        # towbody propagation (after gondola)
        for i in range(idx_gondola + 1, k):
            y += L_pend[i] * np.sin(theta[i])

        # current towbody segment CG
        y += 0.5 * L_pend[k] * np.sin(theta[k])

    return y

# ----------------------------------
# z-coordinate function
# ----------------------------------
def z_coord(k, theta):
    """
    Vertical coordinate of body k CG.
    z=0 at top attachment point.
    Negative z is downward.
    """

    z = 0.0

    # ---------------------------------
    # BALLOON
    # ---------------------------------
    z += -(1.0 - rho) * L_balloon * np.cos(theta[0])

    # ---------------------------------
    # TETHER (balloon → gondola)
    # ---------------------------------
    for i in range(1, idx_tether_end):
        if i < k:
            z += -L_one_tether_pend * np.cos(theta[i])
        elif i == k:
            z += -0.5 * L_one_tether_pend * np.cos(theta[i])
            return z

    # ---------------------------------
    # GONDOLA
    # ---------------------------------
    if k == idx_gondola:
        z += -0.5 * L_gondola * np.cos(theta[k])
        return z

    # full tether contribution before towbody
    for i in range(1, idx_tether_end):
        z += -L_one_tether_pend * np.cos(theta[i])

    z += -L_gondola * np.cos(theta[idx_gondola])  # full gondola offset

    # ---------------------------------
    # TOWBODY (multi-section)
    # ---------------------------------
    tow_start = idx_towbody_start

    # helper counters
    theta_i = tow_start

    for section_id, Nseg in enumerate(N_towbody_pendulum):

        Lsec = L_towbody[section_id] / Nseg

        for s in range(Nseg):

            if theta_i > k:
                return z

            if theta_i == k:
                z += -0.5 * Lsec * np.cos(theta[k])
                return z

            z += -Lsec * np.cos(theta[theta_i])
            theta_i += 1

        # ---------------------------------
        # BAROMETER (point mass, no length)
        # ---------------------------------
        if theta_i > k:
            return z

        if theta_i == k:
            return z  # barometer has no geometric length

        theta_i += 1  # skip barometer node

    return z


# ==================================
# COMPUTE ALL y,z POSITIONS
# ==================================

# Shape:
# y_vals[k, t]
# z_vals[k, t]

y_vals = np.zeros((N_p, len(t_p)))
z_vals = np.zeros((N_p, len(t_p)))

for t_idx in range(len(t_p)):

    y1_t = x_hist[0, t_idx]
    theta_t = x_hist[1:, t_idx]

    for k in range(N_p):

        y_vals[k, t_idx] = y_coord(y1_t, k, theta_t)
        z_vals[k, t_idx] = z_coord(k, theta_t)


# ==================================
# TRUE MULTI-BODY ANIMATION
# ==================================

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 10))

ax.set_aspect("equal")
ax.invert_yaxis()

ax.set_xlabel("y position [m]")
ax.set_ylabel("z position [m]")

ax.set_xlim(-200, 200)
ax.set_ylim(-sum(L_pend)-5, 2)

ax.grid(True)

title = ax.set_title("VISTA Structural Motion")

# ---------------------------------
# BALLOON CIRCLE VISUAL
# ---------------------------------

balloon_radius_vis = L_balloon / 2

balloon_patch = Circle(
    (0, 0),
    balloon_radius_vis,
    fill=False,
    linewidth=1
)

gondola_patch = Rectangle(
    (0, 0),
    2 * R_gondola,
    L_gondola,
    fill=False,
    linewidth=1
)

ax.add_patch(gondola_patch)

ax.add_patch(balloon_patch)

# One line object per body
body_lines = []

for _ in range(N_p):

    ln, = ax.plot([], [], lw=1)

    body_lines.append(ln)


def body_endpoints(y_base, z_base, L, theta):
    """
    Compute segment endpoints from base point.
    """

    y_tip = y_base + L * np.sin(theta)
    z_tip = z_base - L * np.cos(theta)

    return y_tip, z_tip


def update(frame):

    theta = x_hist[1:, frame]

    y0 = x_hist[0, frame]
    z0 = 0.0

    # =================================================
    # BALLOON
    # =================================================
    y_balloon_center = y0 + 0.5 * L_balloon * np.sin(theta[0])
    z_balloon_center = z0 - 0.5 * L_balloon * np.cos(theta[0])

    balloon_patch.center = (y_balloon_center, z_balloon_center)

    y1, z1 = body_endpoints(y0, z0, L_balloon, theta[0])
    body_lines[0].set_data([y0, y1], [z0, z1])

    yc, zc = y1, z1

    # =================================================
    # TETHER (uniform)
    # =================================================
    for i in range(1, idx_tether_end):

        yn, zn = body_endpoints(
            yc, zc,
            L_one_tether_pend,
            theta[i]
        )

        body_lines[i].set_data([yc, yn], [zc, zn])

        yc, zc = yn, zn

    # =================================================
    # GONDOLA
    # =================================================
    yg_center = yc + 0.5 * L_gondola * np.sin(theta[idx_gondola])
    zg_center = zc - 0.5 * L_gondola * np.cos(theta[idx_gondola])

    gondola_patch.set_width(2 * R_gondola)
    gondola_patch.set_height(L_gondola)

    gondola_patch.set_xy((
        yg_center - R_gondola,
        zg_center - 0.5 * L_gondola
    ))

    gondola_patch.angle = np.rad2deg(theta[idx_gondola])

    yc, zc = body_endpoints(yc, zc, L_gondola, theta[idx_gondola])

    # =================================================
    # TOWBODY + BAROMETERS (CRITICAL FIX)
    # =================================================
    i_theta = idx_chain_start

    section_lengths = L_towbody
    segment_counts = N_towbody_pendulum

    for sec in range(len(section_lengths)):

        Lsec = section_lengths[sec] / segment_counts[sec]

        # ---- towbody segments ----
        for s in range(segment_counts[sec]):

            if i_theta >= len(theta):
                break

            yn, zn = body_endpoints(
                yc, zc,
                Lsec,
                theta[i_theta]
            )

            body_lines[i_theta].set_data([yc, yn], [zc, zn])

            yc, zc = yn, zn
            i_theta += 1

        # ---- barometer (point mass: NO LENGTH) ----
        if i_theta >= len(theta):
            break

        # barometer has no geometry line (optional marker only)
        yc, zc = yc, zc
        i_theta += 1

    # =================================================
    # TITLE
    # =================================================
    ax.set_title(f"VISTA Structural Motion — t = {t_p[frame]:.2f} s")

    return body_lines + [balloon_patch, gondola_patch]
ani = FuncAnimation(
    fig,
    update,
    frames=len(t_p),
    interval=30,
    blit=False
)

plt.show()
