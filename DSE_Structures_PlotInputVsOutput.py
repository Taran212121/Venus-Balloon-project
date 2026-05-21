"""
Generate input-vs-output plots from the sensitivity sweep.

Reads sensitivity_results/sensitivity_one_at_a_time.csv and produces, for
every swept input, one PNG per output column. Plots are organised as:

    sensitivity_results/plots_input_vs_output/
        L_susp/
            L_susp_vs_max_gondola_swing_angle_deg.png
            ...
        N_cables/
            ...
        r/
            ...
        ...

Numeric inputs -> line + marker plot, x-axis = swept value.
Categorical inputs (gondola_shape) -> bar chart.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV   = "sensitivity_results/sensitivity_one_at_a_time.csv"
ROOT  = "sensitivity_results/plots_input_vs_output"
os.makedirs(ROOT, exist_ok=True)

# Pretty axis labels (units) for both inputs and outputs.
LABELS = {
    "L_susp":                              "L_susp [m]",
    "N_cables":                            "N_cables [-]",
    "r_attach":                            "r [m]",
    "r":                                   "r [m]",
    "m_gondola":                           "m_gondola [kg]",
    "m_one_cable_total":                   "m_one_cable_total [kg]",
    "m_cables_total":                      "m_cables_total [kg]",
    "zeta_pendulum":                       "zeta_pendulum [-]",
    "zeta_torsion":                        "zeta_torsion [-]",
    "gondola_shape":                       "gondola_shape",
    "I_gondola_tilt":                      "I_gondola_tilt [kg.m^2]",
    "I_gondola_yaw":                       "I_gondola_yaw [kg.m^2]",
    "dominant_pendulum_period_s":          "T_pend [s]",
    "dominant_torsion_period_s":           "T_tors [s]",
    "dominant_pendulum_frequency_hz":      "f_pend [Hz]",
    "dominant_torsion_frequency_hz":       "f_tors [Hz]",
    "max_gondola_swing_angle_deg":         "max swing [deg]",
    "max_gondola_swing_rate_deg_s":        "max swing rate [deg/s]",
    "max_gondola_yaw_deg":                 "max yaw [deg]",
    "max_gondola_yaw_rate_deg_s":          "max yaw rate [deg/s]",
    "settling_time_pendulum_s":            "settling time pend [s]",
    "settling_time_torsion_s":             "settling time tors [s]",
    "peak_total_cable_tension_N":          "peak total cable tension [N]",
    "peak_per_cable_tension_N":            "peak per-cable tension [N]",
    "minimum_per_cable_tension_N":         "min per-cable tension [N]",
    "dynamic_static_tension_ratio":        "T_dyn / T_stat [-]",
    "maximum_torsional_torque_Nm":         "max torsional torque [N.m]",
    "maximum_torsion_element_twist_deg":   "max element twist [deg]",
    "f_max_relevant_hz":                   "f_max relevant [Hz]",
    "required_imu_sampling_10x_hz":        "IMU 10x sampling [Hz]",
    "required_imu_sampling_20x_hz":        "IMU 20x sampling [Hz]",
}

# Hard-constraint reference lines drawn on relevant output plots.
CONSTRAINT_LINES = {
    "max_gondola_swing_angle_deg":       (10.0, "limit 10 deg"),
    "max_gondola_yaw_deg":               (10.0, "limit 10 deg"),
    "maximum_torsion_element_twist_deg": (10.0, "limit 10 deg"),
    "minimum_per_cable_tension_N":       (0.0,  "slack limit"),
    "required_imu_sampling_10x_hz":      (2000.0, "IMU 2000 Hz"),
}

# Outputs we never plot (they are case-definition / metadata, not responses).
SKIP_AS_OUTPUT = {
    "L_susp", "N_cables", "r_attach", "m_gondola", "m_one_cable_total",
    "m_cables_total", "zeta_pendulum", "zeta_torsion", "gondola_shape",
    "I_gondola_tilt", "I_gondola_yaw", "gondola_reference_size",
    "disturbance_case",
    "pend_initial_angle_deg", "pend_initial_rate_deg_s",
    "torsion_initial_twist_deg", "torsion_initial_yaw_rate_deg_s",
    "feasible", "violations",
    "case_id", "sweep_name", "varied_parameter", "varied_value",
}


def safe_filename(s):
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))


def plot_numeric(df_sweep, sweep_name, x_col, y_col, out_dir):
    """Line + marker plot for a numeric input sweep."""
    df = df_sweep.dropna(subset=[x_col, y_col]).sort_values(x_col)
    if df.empty:
        return
    x = df[x_col].to_numpy()
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    if not np.any(np.isfinite(y)):
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(x, y, marker="o", linewidth=1.5, color="#1f77b4")

    # Constraint reference line if applicable
    if y_col in CONSTRAINT_LINES:
        lim, label = CONSTRAINT_LINES[y_col]
        ax.axhline(lim, color="red", linestyle="--", linewidth=1.0, label=label)
        ax.legend(loc="best", fontsize=9)

    ax.set_xlabel(LABELS.get(x_col, x_col))
    ax.set_ylabel(LABELS.get(y_col, y_col))
    ax.set_title("{}  vs  {}".format(
        LABELS.get(x_col, x_col), LABELS.get(y_col, y_col)))
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fname = "{}_vs_{}.png".format(safe_filename(sweep_name), safe_filename(y_col))
    fig.savefig(os.path.join(out_dir, fname), dpi=130)
    plt.close(fig)


def plot_categorical(df_sweep, sweep_name, x_col, y_col, out_dir):
    """Bar chart for a categorical input sweep (gondola_shape)."""
    df = df_sweep.dropna(subset=[x_col, y_col])
    if df.empty:
        return
    cats = df[x_col].astype(str).tolist()
    vals = pd.to_numeric(df[y_col], errors="coerce").to_numpy()
    if not np.any(np.isfinite(vals)):
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xpos = np.arange(len(cats))
    ax.bar(xpos, vals, color="#1f77b4", edgecolor="black")
    ax.set_xticks(xpos)
    ax.set_xticklabels(cats, rotation=20, ha="right")

    if y_col in CONSTRAINT_LINES:
        lim, label = CONSTRAINT_LINES[y_col]
        ax.axhline(lim, color="red", linestyle="--", linewidth=1.0, label=label)
        ax.legend(loc="best", fontsize=9)

    ax.set_xlabel(LABELS.get(x_col, x_col))
    ax.set_ylabel(LABELS.get(y_col, y_col))
    ax.set_title("{}  vs  {}".format(
        LABELS.get(x_col, x_col), LABELS.get(y_col, y_col)))
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fname = "{}_vs_{}.png".format(safe_filename(sweep_name), safe_filename(y_col))
    fig.savefig(os.path.join(out_dir, fname), dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    if not os.path.exists(CSV):
        raise SystemExit("Missing {}. Run DSE_Structures_Sensitivity.py first.".format(CSV))
    df = pd.read_csv(CSV)
    print("Loaded {} rows, {} columns".format(len(df), len(df.columns)))

    # The "input axis" for each sweep is whatever 'varied_parameter' is,
    # but in the dataframe the input lives in its own column (e.g. 'L_susp',
    # 'N_cables'). For 'r' the column is 'r_attach'.
    sweep_to_xcol = {
        "L_susp":            "L_susp",
        "N_cables":          "N_cables",
        "r":                 "r_attach",
        "zeta_pendulum":     "zeta_pendulum",
        "zeta_torsion":      "zeta_torsion",
        "m_gondola":         "m_gondola",
        "m_one_cable_total": "m_one_cable_total",
        "gondola_shape":     "gondola_shape",
    }

    # Outputs to plot = every column NOT in SKIP_AS_OUTPUT.
    outputs = [c for c in df.columns if c not in SKIP_AS_OUTPUT]
    print("Plotting {} outputs per input.".format(len(outputs)))

    sweeps = df["sweep_name"].dropna().unique().tolist()
    for sweep in sweeps:
        x_col = sweep_to_xcol.get(sweep, sweep)
        sub = df[df["sweep_name"] == sweep].copy()
        out_dir = os.path.join(ROOT, safe_filename(sweep))
        os.makedirs(out_dir, exist_ok=True)
        is_categorical = (sweep == "gondola_shape")
        n = 0
        for y_col in outputs:
            if y_col == x_col:
                continue
            if is_categorical:
                plot_categorical(sub, sweep, x_col, y_col, out_dir)
            else:
                plot_numeric(sub, sweep, x_col, y_col, out_dir)
            n += 1
        print("  {:<20s} -> {} plots in {}".format(sweep, n, out_dir))

    print("\nDone. Browse plots under: {}".format(ROOT))


if __name__ == "__main__":
    main()
