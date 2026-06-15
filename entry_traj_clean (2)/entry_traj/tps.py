"""Phase-0 TPS sizing basis: PV-calibrated best estimate plus a Level-0
uncertainty-multiplied design point checked against the HEEET-class envelope."""

from __future__ import annotations

from typing import Any

from .models import Config, Summary, Trajectory

PV_HEAT_CALIBRATION = {
    "convective_scale": 1.1544130018281797,
    "radiative_scale": 6.554848151448616,
    "convective_coefficient_si": 0.00012698543020109976,
    "radiative_coefficient_si": 1.9664544454345846e-24,
    "source_id": "PV-4PROBE-CAL-CO2",
}

LEVEL0_TPS_BOUNDS = {
    "heritage_heat_flux_cap_w_m2": 20_000_000.0,
    "phase0_uncertainty_multiplier": 1.25,
    "basis": (
        "Level 1 best estimate uses PV-calibrated CO2 convective/radiative correlations. "
        "Level 0 sizing basis applies a 1.25x residual multiplier from the post-calibration "
        "PV total-flux/load comparison and checks against the 20 MW/m^2 HEEET-class envelope."
    ),
}


def tps_sizing_basis(cfg: Config, nominal: Trajectory, summary: Summary) -> dict[str, Any]:
    peak_heat_stats = summary.metric_intervals["peak_heat"]
    heat_load_stats = summary.metric_intervals["heat_load"]
    multiplier = LEVEL0_TPS_BOUNDS["phase0_uncertainty_multiplier"]
    heritage_cap = LEVEL0_TPS_BOUNDS["heritage_heat_flux_cap_w_m2"]
    design_peak_heat_w_m2 = float(peak_heat_stats["p97_5"]) * multiplier
    design_heat_load_j_m2 = float(heat_load_stats["p97_5"]) * multiplier

    return {
        "method": "Phase-0 Level 1 + Level 0 TPS heating basis",
        "calibration": PV_HEAT_CALIBRATION,
        "level0_bounds": LEVEL0_TPS_BOUNDS,
        "best_estimate_nominal": {
            "peak_heat_w_m2": max(nominal.heat_flux_total_w_m2),
            "heat_load_j_m2": nominal.heat_load_total_j_m2[-1],
            "source_id": cfg.source_ids.get("peak_heat", "FA1-D41-entry-traj"),
        },
        "population_95_percentile": {
            "peak_heat_w_m2": float(peak_heat_stats["p97_5"]),
            "heat_load_j_m2": float(heat_load_stats["p97_5"]),
        },
        "tps_sizing_design_point": {
            "peak_heat_w_m2": design_peak_heat_w_m2,
            "heat_load_j_m2": design_heat_load_j_m2,
            "uncertainty_multiplier": multiplier,
            "peak_heat_margin_to_heritage_cap": heritage_cap - design_peak_heat_w_m2,
            "peak_heat_margin_fraction": (heritage_cap - design_peak_heat_w_m2) / heritage_cap,
            "passes_heritage_heat_flux_cap": design_peak_heat_w_m2 <= heritage_cap,
        },
        "notes": [
            "Use the design point for FA8/TPS sizing; use best-estimate values for trajectory narrative.",
            "This is a conservative boundary condition, not a material response or ablation calculation.",
            "First-principles CO2 radiation transport remains a G5 residual for later-phase closure.",
        ],
    }
