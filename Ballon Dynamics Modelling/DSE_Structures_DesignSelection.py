"""
VISTA — Final Design Selection
==============================

Implements the selection logic from the design-objectives table:

  1. REJECT configurations that violate hard constraints:
        - L_susp           <  10 m
        - max swing        >= 10 deg
        - max yaw          >= 10 deg
        - max element twist>= 10 deg
        - cable slack      (min per-cable tension <= 0)
        - IMU 10x rule     >  2000 Hz   (hardware limit)

  2. AMONG passing configurations PREFER (in order):
        - lowest mass            (m_gondola fixed; rank by m_cables_total)
        - lowest complexity      (fewest cables, simplest gondola shape)
        - shortest feasible cable (smallest L_susp)
        - simplest cable layout  (proxy: smaller N_cables, smaller r)
        - passive damping        (lowest zeta still passing wins on simplicity,
                                  but higher zeta gives shorter settling — we
                                  rank shorter settling next, so the overall
                                  balance is captured)

  3. TIEBREAKERS:
        - lower measurement-correction uncertainty
            (lower max angular rates / accelerations / amplitudes)
        - lower angular rates
        - shorter settling time
        - better structural margins (higher T_min, lower T_max)

Fixed inputs from the user:
    V_gondola = 1.0 m^3   (all shapes scaled to this volume)
    m_gondola = 100.0 kg

The baseline disturbance is the spec's "high" case (worst-case design driver).
"""

import os
import itertools
import numpy as np
import pandas as pd

# Re-use the validated physics from the sensitivity script
from DSE_Structures_Sensitivity import (
    run_single_case,
    gondola_geometry_cases,
    disturbance_cases,
)

# ------------------------------------------------------------------
# 1. FIXED USER INPUTS
# ------------------------------------------------------------------
M_GONDOLA_FIXED = 100.0          # kg  (user spec)
V_GONDOLA_FIXED = 1.0            # m^3 (user spec; gondola_geometry_cases already scaled)
DISTURBANCE     = "high"         # worst-case design driver

results_dir = "design_selection_results"
os.makedirs(results_dir, exist_ok=True)

# ------------------------------------------------------------------
# 2. HARD CONSTRAINTS (figures)
# ------------------------------------------------------------------
HARD_LIMITS = {
    "L_susp_min_m":            10.0,
    "swing_max_deg":           10.0,   # max_gondola_swing_angle_deg
    "yaw_max_deg":             10.0,   # max_gondola_yaw_deg
    "elem_twist_max_deg":      10.0,   # maximum_torsion_element_twist_deg
    "min_cable_tension_N":      0.0,   # strictly > 0 for no slack
    "imu_10x_max_hz":        2000.0,   # IMU sampling hardware limit
}


def passes_hard_constraints(row):
    """Return (ok, list_of_violation_labels)."""
    v = []
    if row["L_susp"] < HARD_LIMITS["L_susp_min_m"]:
        v.append("L_susp<10m")
    if row["max_gondola_swing_angle_deg"] >= HARD_LIMITS["swing_max_deg"]:
        v.append("swing>=10deg")
    if row["max_gondola_yaw_deg"] >= HARD_LIMITS["yaw_max_deg"]:
        v.append("yaw>=10deg")
    if row["maximum_torsion_element_twist_deg"] >= HARD_LIMITS["elem_twist_max_deg"]:
        v.append("twist>=10deg")
    if row["minimum_per_cable_tension_N"] <= HARD_LIMITS["min_cable_tension_N"]:
        v.append("slack")
    if row["required_imu_sampling_10x_hz"] > HARD_LIMITS["imu_10x_max_hz"]:
        v.append("IMU>2kHz")
    return (len(v) == 0), v


# ------------------------------------------------------------------
# 3. DESIGN GRID
# ------------------------------------------------------------------
# Targeted grid informed by the sensitivity study:
#   - cube and sphere give the lowest yaw amplitudes (lowest I_yaw)
#   - higher zeta_pendulum reduces swing
#   - higher zeta_torsion reduces yaw
#   - smaller r reduces torsional torque (but lengthens torsion period)
#   - more N_cables reduces per-cable tension (slack avoidance)
#   - L_susp must be >= 10 m; shorter is preferred for mass/complexity
DESIGN_GRID = {
    "L_susp":             [10.0, 12.0, 15.0, 18.0],
    "N_cables":           [3, 4, 6],
    "r":                  [0.5, 0.75, 1.0, 1.25],
    "m_one_cable_total":  [0.10],                      # light, simple
    "gondola_shape":      ["solid_cylinder", "cube",
                           "solid_sphere", "rectangular_box"],
    "zeta_pendulum":      [0.010, 0.020],
    "zeta_torsion":       [0.005, 0.010],
}


def grid_iter(grid):
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


# ------------------------------------------------------------------
# 4. RUN GRID
# ------------------------------------------------------------------
def run_design_grid():
    rows = []
    combos = list(grid_iter(DESIGN_GRID))
    n = len(combos)
    print("Running design grid: {} configurations".format(n))
    for i, combo in enumerate(combos):
        params = {
            "L_susp":            combo["L_susp"],
            "N_cables":          combo["N_cables"],
            "r":                 combo["r"],
            "m_gondola":         M_GONDOLA_FIXED,
            "m_one_cable_total": combo["m_one_cable_total"],
            "gondola_shape":     combo["gondola_shape"],
            "zeta_pendulum":     combo["zeta_pendulum"],
            "zeta_torsion":      combo["zeta_torsion"],
            "disturbance_case":  DISTURBANCE,
        }
        try:
            res = run_single_case(params)
        except Exception as exc:
            print("  [{:4d}/{:4d}] ERROR: {}".format(i + 1, n, exc))
            continue
        ok, vio = passes_hard_constraints(res)
        res["passes"]      = ok
        res["violations"]  = ";".join(vio) if vio else "OK"
        res["config_id"]   = i
        rows.append(res)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            n_pass = sum(1 for r in rows if r["passes"])
            print("  [{:4d}/{:4d}]  passing so far: {}".format(i + 1, n, n_pass))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 5. RANKING (selection logic, in order)
# ------------------------------------------------------------------
SHAPE_COMPLEXITY = {     # lower = simpler
    "cube":            1,
    "solid_sphere":    2,
    "solid_cylinder":  3,
    "rectangular_box": 4,
}


def rank_passing(df_pass):
    """Apply the multi-objective sort from the figures' selection logic."""
    df = df_pass.copy()
    df["shape_complexity"] = df["gondola_shape"].map(SHAPE_COMPLEXITY)
    # Composite "max angular response" used for measurement-uncertainty tier
    df["max_ang_rate_deg_s"] = df[[
        "max_gondola_swing_rate_deg_s",
        "max_gondola_yaw_rate_deg_s",
    ]].max(axis=1)
    df["max_ang_amp_deg"] = df[[
        "max_gondola_swing_angle_deg",
        "max_gondola_yaw_deg",
    ]].max(axis=1)
    df["max_settling_s"] = df[[
        "settling_time_pendulum_s",
        "settling_time_torsion_s",
    ]].max(axis=1)

    sort_cols = [
        # Step 2: mass / complexity / shortness / simplicity
        "m_cables_total",        # lowest mass
        "N_cables",              # fewest cables (complexity)
        "shape_complexity",      # simplest gondola shape
        "L_susp",                # shortest cable
        "r_attach",              # simplest layout (smaller spread)
        # Step 3: tiebreakers
        "max_ang_amp_deg",       # measurement-correction uncertainty
        "max_ang_rate_deg_s",    # angular rates
        "max_settling_s",        # settling time
        "peak_per_cable_tension_N",   # structural margins (lower peak)
    ]
    df = df.sort_values(sort_cols, ascending=True).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df, sort_cols


# ------------------------------------------------------------------
# 6. PRESENTATION
# ------------------------------------------------------------------
DISPLAY_COLS = [
    "rank",
    "L_susp", "N_cables", "r_attach",
    "gondola_shape",
    "zeta_pendulum", "zeta_torsion",
    "m_cables_total",
    "max_gondola_swing_angle_deg", "max_gondola_yaw_deg",
    "maximum_torsion_element_twist_deg",
    "max_gondola_swing_rate_deg_s", "max_gondola_yaw_rate_deg_s",
    "settling_time_pendulum_s", "settling_time_torsion_s",
    "peak_per_cable_tension_N", "minimum_per_cable_tension_N",
    "maximum_torsional_torque_Nm",
    "required_imu_sampling_10x_hz",
]

DISPLAY_HEADERS = {
    "rank":                                "rank",
    "L_susp":                              "L [m]",
    "N_cables":                            "Nc",
    "r_attach":                            "r [m]",
    "gondola_shape":                       "shape",
    "zeta_pendulum":                       "z_p",
    "zeta_torsion":                        "z_t",
    "m_cables_total":                      "m_cab [kg]",
    "max_gondola_swing_angle_deg":         "swing [deg]",
    "max_gondola_yaw_deg":                 "yaw [deg]",
    "maximum_torsion_element_twist_deg":   "twist [deg]",
    "max_gondola_swing_rate_deg_s":        "sw_rt [deg/s]",
    "max_gondola_yaw_rate_deg_s":          "yw_rt [deg/s]",
    "settling_time_pendulum_s":            "tset_p [s]",
    "settling_time_torsion_s":             "tset_t [s]",
    "peak_per_cable_tension_N":            "T_max [N]",
    "minimum_per_cable_tension_N":         "T_min [N]",
    "maximum_torsional_torque_Nm":         "tau_max [Nm]",
    "required_imu_sampling_10x_hz":        "IMU10x [Hz]",
}


def format_table(df, cols, headers):
    sub = df[cols].copy()
    rename = {c: headers.get(c, c) for c in cols}
    sub = sub.rename(columns=rename)
    return sub.to_string(index=False, float_format=lambda v: "{:8.3f}".format(v))


# ------------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------------
def main():
    print("=" * 76)
    print("VISTA Final Design Selection")
    print("=" * 76)
    print("Fixed inputs:")
    print("  m_gondola = {:.1f} kg".format(M_GONDOLA_FIXED))
    print("  V_gondola = {:.2f} m^3 (all shape sizes scaled to this volume)".format(
        V_GONDOLA_FIXED))
    print("  Disturbance scenario = '{}' (worst-case)".format(DISTURBANCE))
    d = disturbance_cases[DISTURBANCE]
    print("    pendulum IC: {:.1f} deg, {:.1f} deg/s".format(
        d["pend_initial_angle_deg"], d["pend_initial_rate_deg_s"]))
    print("    torsion  IC: {:.1f} deg, {:.1f} deg/s".format(
        d["torsion_initial_twist_deg"], d["torsion_initial_yaw_rate_deg_s"]))
    print()
    print("Hard constraints:")
    for k, v in HARD_LIMITS.items():
        print("  {:25s} = {}".format(k, v))
    print()

    df_all = run_design_grid()
    df_all.to_csv(os.path.join(results_dir, "design_grid_all.csv"), index=False)

    n_total = len(df_all)
    df_pass = df_all[df_all["passes"]].copy()
    n_pass  = len(df_pass)
    print()
    print("=" * 76)
    print("RESULT:  {} / {} configurations pass all hard constraints".format(
        n_pass, n_total))
    print("=" * 76)

    if n_pass == 0:
        print()
        print("No configuration passes the hard constraints with these inputs.")
        print("Top 10 closest-to-feasible (fewest violations, then smallest swing+yaw):")
        df_all["nv"] = df_all["violations"].apply(
            lambda s: 0 if s == "OK" else len(s.split(";")))
        df_all["worst_amp"] = df_all[[
            "max_gondola_swing_angle_deg", "max_gondola_yaw_deg"
        ]].max(axis=1)
        near = df_all.sort_values(["nv", "worst_amp"]).head(10)
        print(near[[
            "L_susp", "N_cables", "r_attach", "gondola_shape",
            "zeta_pendulum", "zeta_torsion",
            "max_gondola_swing_angle_deg", "max_gondola_yaw_deg",
            "maximum_torsion_element_twist_deg",
            "minimum_per_cable_tension_N",
            "required_imu_sampling_10x_hz",
            "violations",
        ]].to_string(index=False))
        return

    df_ranked, sort_cols = rank_passing(df_pass)
    df_ranked.to_csv(os.path.join(results_dir, "design_grid_passing_ranked.csv"),
                     index=False)

    print()
    print("Sorting hierarchy applied (ascending priority left -> right):")
    for c in sort_cols:
        print("  - {}".format(c))
    print()

    print("Top 10 ranked configurations:")
    print(format_table(df_ranked.head(10), DISPLAY_COLS, DISPLAY_HEADERS))
    print()

    best = df_ranked.iloc[0]
    print("=" * 76)
    print("RECOMMENDED FINAL DESIGN  (rank 1)")
    print("=" * 76)
    print("Configuration parameters:")
    print("  L_susp              = {:.2f} m".format(best["L_susp"]))
    print("  N_cables            = {}".format(int(best["N_cables"])))
    print("  r (attachment)      = {:.2f} m".format(best["r_attach"]))
    print("  gondola_shape       = {}".format(best["gondola_shape"]))
    print("  gondola_size        = {}".format(best["gondola_reference_size"]))
    print("  m_gondola           = {:.1f} kg  (fixed)".format(best["m_gondola"]))
    print("  m_one_cable_total   = {:.3f} kg".format(best["m_one_cable_total"]))
    print("  m_cables_total      = {:.3f} kg".format(best["m_cables_total"]))
    print("  zeta_pendulum       = {:.4f}".format(best["zeta_pendulum"]))
    print("  zeta_torsion        = {:.4f}".format(best["zeta_torsion"]))
    print()
    print("Predicted dynamic response (worst-case '{}' disturbance):".format(DISTURBANCE))
    print("  T_pend (dominant)   = {:.2f} s".format(best["dominant_pendulum_period_s"]))
    print("  T_tors (dominant)   = {:.2f} s".format(best["dominant_torsion_period_s"]))
    print("  max swing           = {:.2f} deg".format(best["max_gondola_swing_angle_deg"]))
    print("  max yaw             = {:.2f} deg".format(best["max_gondola_yaw_deg"]))
    print("  max element twist   = {:.2f} deg".format(best["maximum_torsion_element_twist_deg"]))
    print("  max swing rate      = {:.2f} deg/s".format(best["max_gondola_swing_rate_deg_s"]))
    print("  max yaw rate        = {:.2f} deg/s".format(best["max_gondola_yaw_rate_deg_s"]))
    print("  settling pend       = {:.1f} s".format(best["settling_time_pendulum_s"]))
    print("  settling tors       = {:.1f} s".format(best["settling_time_torsion_s"]))
    print()
    print("Structural envelope:")
    print("  T per cable, max    = {:.1f} N".format(best["peak_per_cable_tension_N"]))
    print("  T per cable, min    = {:.1f} N (slack-margin > 0)".format(
        best["minimum_per_cable_tension_N"]))
    print("  T_dyn / T_stat      = {:.3f}".format(best["dynamic_static_tension_ratio"]))
    print("  max torsional torque= {:.2f} N.m".format(best["maximum_torsional_torque_Nm"]))
    print()
    print("Sensing:")
    print("  f_max relevant      = {:.3f} Hz".format(best["f_max_relevant_hz"]))
    print("  IMU 10x rule        = {:.1f} Hz   (limit 2000 Hz)".format(
        best["required_imu_sampling_10x_hz"]))
    print("  IMU 20x rule        = {:.1f} Hz".format(best["required_imu_sampling_20x_hz"]))
    print()
    print("Files written to '{}':".format(results_dir))
    print("  design_grid_all.csv             - every configuration tried")
    print("  design_grid_passing_ranked.csv  - feasible configurations, ranked")


if __name__ == "__main__":
    main()
