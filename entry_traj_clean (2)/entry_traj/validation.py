"""V&V checks against the nominal trajectory.

- Allen-Eggers closed-form peak deceleration (applicable only for the
  exponential-atmosphere, constant-FPA model).
- Peak-g and peak-heat-flux caps from the validation config.
- Payload absorbed-energy cap (lumped thermal gate).
- Optional VCD density envelope containment for profile atmospheres.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .atmosphere import AtmosphereModel
from .models import Config, Report, Trajectory
from .physics import G0_M_S2


def validate(cfg: Config, trajectory: Trajectory) -> Report:
    checks: list[dict[str, Any]] = []
    context = trajectory.validation_context

    checks.append(_allen_eggers_check(cfg, trajectory))
    checks.append(
        _cap_check(
            "peak_g_cap",
            value=max(trajectory.deceleration_g),
            cap=float(context["peak_g_cap_g"]),
            units="g",
        )
    )
    checks.append(
        _cap_check(
            "peak_heat_cap",
            value=max(trajectory.heat_flux_total_w_m2),
            cap=float(context["peak_heat_cap_w_m2"]),
            units="W/m^2",
        )
    )
    if bool(context.get("payload_thermal_enabled")):
        checks.append(
            _cap_check(
                "payload_energy_cap",
                value=trajectory.payload_absorbed_energy_j[-1],
                cap=float(context["payload_energy_capacity_j"]),
                units="J",
            )
        )
    envelope_check = _vcd_envelope_check(cfg)
    if envelope_check is not None:
        checks.append(envelope_check)

    return Report(
        passed=all(check["passed"] for check in checks),
        checks=tuple(checks),
        context=dict(context),
    )


def allen_eggers_peak_deceleration_g(
    velocity_m_s: float, fpa_deg: float, scale_height_m: float
) -> float:
    """Allen-Eggers peak deceleration for ballistic entry, in Earth g."""
    gamma = math.radians(abs(fpa_deg))
    return velocity_m_s**2 * math.sin(gamma) / (2.0 * math.e * scale_height_m) / G0_M_S2


def _allen_eggers_check(cfg: Config, trajectory: Trajectory) -> dict[str, Any]:
    context = trajectory.validation_context
    applicable = (
        cfg.atmosphere.density_kind == "exponential"
        and str(context.get("trajectory_model")) == "constant_fpa"
    )
    scale_height_m = _benchmark_scale_height_m(cfg)
    predicted_g = allen_eggers_peak_deceleration_g(
        float(context["entry_velocity_m_s"]), float(context["entry_fpa_deg"]), scale_height_m
    )
    simulated_g = max(trajectory.deceleration_g)
    relative_error = abs(simulated_g - predicted_g) / predicted_g if predicted_g > 0 else math.inf
    tolerance = float(context["allen_eggers_tolerance_fraction"])
    return {
        "name": "allen_eggers_peak_g",
        "passed": (relative_error <= tolerance) if applicable else True,
        "allen_eggers_applicable": "true" if applicable else "false",
        "predicted_peak_g": predicted_g,
        "simulated_peak_g": simulated_g,
        "relative_error": relative_error,
        "tolerance_fraction": tolerance,
        "benchmark_scale_height_m": scale_height_m,
        "note": (
            "Analytic benchmark valid for exponential-atmosphere constant-FPA ballistic entry."
            if applicable
            else "Not applicable for this atmosphere/trajectory model; check passes by definition. "
            "Reported numbers use a local scale-height fit near 78 km for reference only."
        ),
    }


def _benchmark_scale_height_m(cfg: Config) -> float:
    """Local density scale height near peak deceleration altitude (~78 km)."""
    if cfg.atmosphere.density_kind == "exponential":
        return cfg.atmosphere.scale_height_m
    model = AtmosphereModel(cfg)
    alt_lo, alt_hi = 74_000.0, 82_000.0
    rho_lo, rho_hi = model.density_kg_m3(alt_lo), model.density_kg_m3(alt_hi)
    if rho_lo <= 0.0 or rho_hi <= 0.0 or rho_lo == rho_hi:
        return cfg.atmosphere.scale_height_m
    return (alt_hi - alt_lo) / math.log(rho_lo / rho_hi)


def _cap_check(name: str, value: float, cap: float, units: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": value <= cap,
        "value": value,
        "cap": cap,
        "margin": cap - value,
        "margin_fraction": (cap - value) / cap if cap > 0 else math.nan,
        "units": units,
    }


def _vcd_envelope_check(cfg: Config) -> dict[str, Any] | None:
    """Check the configured density model stays inside the VCD lo/hi envelope."""
    envelope_path = cfg.atmosphere.vcd_envelope_path
    if envelope_path is None:
        return None
    rows = _read_envelope_rows(Path(envelope_path))
    model = AtmosphereModel(cfg)
    violations: list[dict[str, float]] = []
    for altitude_m, rho_low, rho_high in rows:
        rho = model.density_kg_m3(altitude_m)
        if not (rho_low <= rho <= rho_high):
            violations.append(
                {
                    "altitude_m": altitude_m,
                    "density_kg_m3": rho,
                    "rho_low_kg_m3": rho_low,
                    "rho_high_kg_m3": rho_high,
                }
            )
    return {
        "name": "vcd_density_envelope",
        "passed": not violations,
        "n_points": len(rows),
        "n_violations": len(violations),
        "violations": violations[:20],
        "envelope_path": str(envelope_path),
    }


def _read_envelope_rows(path: Path) -> list[tuple[float, float, float]]:
    """CSV columns: altitude_m, rho_low_kg_m3, rho_high_kg_m3 (header optional)."""
    rows: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            parts = [part.strip() for part in line.replace(";", ",").split(",")]
            if len(parts) < 3:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue  # header or comment line
    if not rows:
        raise ValueError(f"No usable envelope rows found in {path}")
    return rows


def report_text(report: Report) -> str:
    """Human-readable validation report."""
    lines = [
        "entry_traj validation report",
        f"overall: {'PASS' if report.passed else 'FAIL'}",
        "",
    ]
    for check in report.checks:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"[{status}] {check['name']}")
        for key, value in check.items():
            if key in ("name", "passed", "violations"):
                continue
            lines.append(f"    {key}: {value}")
        if check.get("violations"):
            lines.append(f"    first violations: {check['violations'][:3]}")
    return "\n".join(lines) + "\n"
