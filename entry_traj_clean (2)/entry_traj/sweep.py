"""EFPA feasibility sweep with an entry-system mass trade.

Runs three 3-sigma dispersion tails (shallow / nominal / steep) per candidate
EFPA and checks each against capture, load, heating, payload-energy and
deploy-box constraints.

Feasible candidates are then ranked on a Phase-0 entry-system mass model that
closes the steep-vs-shallow trade:

- steeper EFPA  -> higher worst-tail peak g    -> heavier aeroshell structure
- shallower EFPA -> higher integrated heat load -> heavier TPS (larger heatshield)

The recommended EFPA minimises TPS + structure mass; the shallowest feasible
EFPA (maximum corridor margin against skip-out) is reported alongside it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .integrator import RunSample, integrate
from .models import Config, Trajectory
from .tps import LEVEL0_TPS_BOUNDS

TAILS = ("shallow_3sigma", "nominal", "steep_3sigma")


def sweep_fpa(
    cfg: Config,
    fpa_values_deg: list[float],
    design_heat_load_cap_j_m2: float | None = None,
) -> dict[str, Any]:
    mass_cfg = cfg.raw.get("mass_model", {})
    mass_enabled = bool(mass_cfg.get("enabled", False))

    rows = [_evaluate_fpa(cfg, fpa_deg, design_heat_load_cap_j_m2) for fpa_deg in fpa_values_deg]
    if mass_enabled:
        for row in rows:
            row["mass"] = _mass_terms(cfg, mass_cfg, row["worst_case"])

    feasible = [row for row in rows if row["feasible"]]
    shallowest = min(feasible, key=lambda row: abs(row["fpa_deg"])) if feasible else None
    mass_optimal = None
    if mass_enabled and feasible:
        ranked = [row for row in feasible if row["mass"]["entry_system_mass_kg"] is not None]
        if ranked:
            # Tie-break toward shallower entry (more corridor margin).
            mass_optimal = min(
                ranked,
                key=lambda row: (row["mass"]["entry_system_mass_kg"], abs(row["fpa_deg"])),
            )

    recommended = mass_optimal if mass_optimal is not None else shallowest
    selection_rule = (
        "mass-optimal feasible EFPA (minimum TPS + structure mass; "
        "shallower preferred on ties)"
        if mass_optimal is not None
        else "shallowest feasible EFPA (maximum corridor margin against skip-out)"
    )

    return {
        "fpa_values_deg": list(fpa_values_deg),
        "dispersion_fpa_deg": cfg.dispersion_fpa_deg,
        "rows": rows,
        "n_feasible": len(feasible),
        "recommended_fpa_deg": recommended["fpa_deg"] if recommended else None,
        "shallowest_feasible_fpa_deg": shallowest["fpa_deg"] if shallowest else None,
        "mass_optimal_fpa_deg": mass_optimal["fpa_deg"] if mass_optimal else None,
        "selection_rule": selection_rule,
        "constraints": _constraint_descriptions(cfg, design_heat_load_cap_j_m2),
        "mass_model": _mass_model_description(cfg, mass_cfg) if mass_enabled else {"enabled": False},
    }


def _mass_terms(cfg: Config, mass_cfg: dict[str, Any], worst_case: dict[str, Any]) -> dict[str, Any]:
    """TPS + structure mass for one EFPA candidate from worst-tail design values."""
    design_heat_load = worst_case.get("design_heat_load_j_m2")
    peak_g = worst_case.get("peak_g")
    if design_heat_load is None or peak_g is None:
        return {
            "tps_mass_kg": None,
            "tps_mass_fraction": None,
            "structure_mass_kg": None,
            "structure_mass_fraction": None,
            "entry_system_mass_kg": None,
            "entry_system_mass_fraction": None,
        }

    tps_cfg = mass_cfg.get("tps", {})
    struct_cfg = mass_cfg.get("structure", {})

    heat_load_j_cm2 = design_heat_load / 1e4
    tps_fraction = (
        float(tps_cfg.get("coefficient_percent", 0.091))
        / 100.0
        * heat_load_j_cm2 ** float(tps_cfg.get("exponent", 0.51575))
    )
    tps_mass_kg = tps_fraction * cfg.m_entry_kg

    struct_fraction = (
        float(struct_cfg.get("reference_mass_fraction", 0.14))
        * (peak_g / float(struct_cfg.get("reference_peak_g", 100.0)))
        ** float(struct_cfg.get("exponent", 1.0))
    )
    structure_mass_kg = struct_fraction * cfg.m_entry_kg

    total = tps_mass_kg + structure_mass_kg
    return {
        "tps_mass_kg": tps_mass_kg,
        "tps_mass_fraction": tps_fraction,
        "structure_mass_kg": structure_mass_kg,
        "structure_mass_fraction": struct_fraction,
        "entry_system_mass_kg": total,
        "entry_system_mass_fraction": total / cfg.m_entry_kg,
    }


def _mass_model_description(cfg: Config, mass_cfg: dict[str, Any]) -> dict[str, Any]:
    tps_cfg = mass_cfg.get("tps", {})
    struct_cfg = mass_cfg.get("structure", {})
    return {
        "enabled": True,
        "m_entry_kg": cfg.m_entry_kg,
        "tps": {
            "relation": "m_tps/m_entry [%] = coefficient_percent * (design heat load [J/cm^2])^exponent",
            "coefficient_percent": float(tps_cfg.get("coefficient_percent", 0.091)),
            "exponent": float(tps_cfg.get("exponent", 0.51575)),
            "input": "worst-tail heat load x Level-0 uncertainty multiplier",
            "source_id": str(tps_cfg.get("source_id", "TPS-MF-HEATLOAD-LV2003")),
        },
        "structure": {
            "relation": "m_struct = reference_mass_fraction * m_entry * (worst-tail peak g / reference_peak_g)^exponent",
            "reference_mass_fraction": float(struct_cfg.get("reference_mass_fraction", 0.14)),
            "reference_peak_g": float(struct_cfg.get("reference_peak_g", 100.0)),
            "exponent": float(struct_cfg.get("exponent", 1.0)),
            "source_id": str(struct_cfg.get("source_id", "PHASE0-STRUCT-GLOAD-LINEAR")),
        },
        "notes": [
            "Phase-0 ranking model for the EFPA trade only; not a mass budget.",
            "TPS term uses the empirical ablator mass-fraction vs heat-load correlation "
            "(carbon-phenolic-class heritage fit).",
            "Structure term is a placeholder g-load scaling; calibrate reference_mass_fraction "
            "and reference_peak_g against the FA mass budget before using for closure.",
        ],
    }


def _evaluate_fpa(
    cfg: Config, fpa_deg: float, design_heat_load_cap_j_m2: float | None
) -> dict[str, Any]:
    base = dataclasses.replace(cfg, entry_fpa_deg=fpa_deg)
    tails: dict[str, Trajectory] = {
        "shallow_3sigma": integrate(base, RunSample(cfg.m_entry_kg, fpa_deg + cfg.dispersion_fpa_deg, label="shallow")),
        "nominal": integrate(base, RunSample(cfg.m_entry_kg, fpa_deg, label="nominal")),
        "steep_3sigma": integrate(base, RunSample(cfg.m_entry_kg, fpa_deg - cfg.dispersion_fpa_deg, label="steep")),
    }
    multiplier = LEVEL0_TPS_BOUNDS["phase0_uncertainty_multiplier"]
    heritage_cap = LEVEL0_TPS_BOUNDS["heritage_heat_flux_cap_w_m2"]

    captured_all = all(run.success for run in tails.values())
    worst_peak_g = max((max(run.deceleration_g) for run in tails.values() if run.success), default=float("inf"))
    worst_peak_heat = max((max(run.heat_flux_total_w_m2) for run in tails.values() if run.success), default=float("inf"))
    worst_heat_load = max((run.heat_load_total_j_m2[-1] for run in tails.values() if run.success), default=float("inf"))
    worst_payload_energy = max(
        (run.payload_absorbed_energy_j[-1] for run in tails.values() if run.success), default=float("inf")
    )

    constraints = {
        "captured_all_tails": captured_all,
        "peak_g_cap": captured_all and worst_peak_g <= cfg.validation.peak_g_cap_g,
        "design_peak_heat_cap": captured_all and worst_peak_heat * multiplier <= heritage_cap,
        "design_heat_load_cap": (
            captured_all
            and (design_heat_load_cap_j_m2 is None or worst_heat_load * multiplier <= design_heat_load_cap_j_m2)
        ),
        "payload_energy_cap": (
            captured_all
            and (
                not cfg.payload_thermal.enabled
                or worst_payload_energy <= cfg.payload_thermal.energy_capacity_j
            )
        ),
        "mach_handoff_box": captured_all
        and all(run.mach[-1] <= cfg.deploy.mach_ceiling for run in tails.values() if run.success),
        "deploy_trigger_box": captured_all
        and all(_handoff_in_deploy_box(cfg, run) for run in tails.values() if run.success),
    }

    return {
        "fpa_deg": fpa_deg,
        "feasible": all(constraints.values()),
        "constraints": constraints,
        "tails": {
            label: {
                "success": run.success,
                "failure_reason": run.failure_reason,
                "fpa_deg": float(run.input_sample["fpa_deg"]),
                "peak_g": max(run.deceleration_g) if run.success else None,
                "peak_heat_w_m2": max(run.heat_flux_total_w_m2) if run.success else None,
                "heat_load_j_m2": run.heat_load_total_j_m2[-1] if run.success else None,
                "payload_absorbed_energy_j": run.payload_absorbed_energy_j[-1] if run.success else None,
                "mach_handoff": run.mach[-1] if run.success else None,
                "q_handoff_pa": run.dynamic_pressure_pa[-1] if run.success else None,
                "handoff_altitude_m": run.altitude_m[-1] if run.success else None,
            }
            for label, run in tails.items()
        },
        "worst_case": {
            "peak_g": worst_peak_g if captured_all else None,
            "design_peak_heat_w_m2": worst_peak_heat * multiplier if captured_all else None,
            "design_heat_load_j_m2": worst_heat_load * multiplier if captured_all else None,
            "payload_absorbed_energy_j": worst_payload_energy if captured_all else None,
        },
    }


def _handoff_in_deploy_box(cfg: Config, run: Trajectory) -> bool:
    if cfg.handoff_trigger != "deploy_event":
        return True
    q_pa = run.dynamic_pressure_pa[-1]
    return (
        cfg.deploy.q_floor_pa <= q_pa <= cfg.deploy.q_cap_pa
        and run.altitude_m[-1] >= cfg.deploy.min_altitude_m
        and run.mach[-1] <= cfg.deploy.mach_ceiling
    )


def _constraint_descriptions(cfg: Config, design_heat_load_cap_j_m2: float | None) -> dict[str, str]:
    multiplier = LEVEL0_TPS_BOUNDS["phase0_uncertainty_multiplier"]
    return {
        "captured_all_tails": "All three dispersion tails reach handoff (no skip-out / stall / floor violation).",
        "peak_g_cap": f"Worst-tail peak deceleration <= {cfg.validation.peak_g_cap_g:g} g.",
        "design_peak_heat_cap": (
            f"Worst-tail peak heat flux x {multiplier:g} <= "
            f"{LEVEL0_TPS_BOUNDS['heritage_heat_flux_cap_w_m2']:.3g} W/m^2 (HEEET-class envelope)."
        ),
        "design_heat_load_cap": (
            f"Worst-tail heat load x {multiplier:g} <= {design_heat_load_cap_j_m2:.3g} J/m^2."
            if design_heat_load_cap_j_m2 is not None
            else "No design heat-load cap supplied (constraint passes by default)."
        ),
        "payload_energy_cap": (
            f"Worst-tail payload absorbed energy <= {cfg.payload_thermal.energy_capacity_j:.3g} J."
            if cfg.payload_thermal.enabled
            else "Payload thermal gate disabled."
        ),
        "mach_handoff_box": f"Hand-off Mach <= {cfg.deploy.mach_ceiling:g} for all tails.",
        "deploy_trigger_box": "Hand-off state inside the deploy q/Mach/altitude box (deploy_event trigger only).",
    }
