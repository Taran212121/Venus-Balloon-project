"""VIRA profile ensemble: rerun the nominal trajectory across the full
VCD-VIRA latitude x solar-zenith-angle selector grid and report the spread."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from .atmosphere import AtmosphereModel, vira_selectors
from .integrator import integrate
from .models import Config

METRICS = ("peak_g", "peak_q_pa", "peak_heat_w_m2", "heat_load_j_m2", "mach_handoff", "handoff_altitude_m")


def run_profile_ensemble(cfg: Config) -> dict[str, Any]:
    if cfg.atmosphere.density_kind != "profile_file":
        raise ValueError("Profile ensemble requires a profile_file atmosphere config")

    rows: list[dict[str, Any]] = []
    for selector in vira_selectors():
        atm = dataclasses.replace(cfg.atmosphere, profile_selector=selector)
        run_cfg = dataclasses.replace(cfg, atmosphere=atm)
        try:
            trajectory = integrate(run_cfg)
        except (ValueError, FileNotFoundError) as exc:
            rows.append({"selector": selector, "success": False, "failure_reason": str(exc)})
            continue
        row: dict[str, Any] = {
            "selector": selector,
            "success": trajectory.success,
            "failure_reason": trajectory.failure_reason,
        }
        if trajectory.success:
            peak_g_index = max(
                range(len(trajectory.deceleration_g)), key=trajectory.deceleration_g.__getitem__
            )
            row.update(
                peak_g=max(trajectory.deceleration_g),
                peak_q_pa=max(trajectory.dynamic_pressure_pa),
                peak_heat_w_m2=max(trajectory.heat_flux_total_w_m2),
                heat_load_j_m2=trajectory.heat_load_total_j_m2[-1],
                mach_handoff=trajectory.mach[-1],
                handoff_altitude_m=trajectory.altitude_m[-1],
                peak_g_altitude_m=trajectory.altitude_m[peak_g_index],
                local_scale_height_at_peak_g_m=_local_scale_height_m(
                    run_cfg, trajectory.altitude_m[peak_g_index]
                ),
            )
        rows.append(row)

    ranges: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = sorted(row[metric] for row in rows if row.get("success") and metric in row)
        if values:
            ranges[metric] = {
                "min": values[0],
                "p50": values[len(values) // 2],
                "max": values[-1],
            }
    return {
        "baseline_selector": cfg.atmosphere.profile_selector,
        "n_selectors": len(rows),
        "n_success": sum(1 for row in rows if row.get("success")),
        "rows": rows,
        "metric_ranges": ranges,
    }


def _local_scale_height_m(cfg: Config, altitude_m: float) -> float:
    model = AtmosphereModel(cfg)
    alt_lo, alt_hi = altitude_m - 2000.0, altitude_m + 2000.0
    rho_lo, rho_hi = model.density_kg_m3(alt_lo), model.density_kg_m3(alt_hi)
    if rho_lo <= 0.0 or rho_hi <= 0.0 or rho_lo == rho_hi:
        return float("nan")
    return (alt_hi - alt_lo) / math.log(rho_lo / rho_hi)
