"""Parachute deploy trigger resolution within one integration step.

Two trigger modes:

- ``mach_q``: legacy gate -- Mach at/below trigger_mach AND dynamic pressure
  inside [q_floor, q_cap] AND altitude above the floor.
- ``q_window``: first crossing into the dynamic-pressure window above the
  altitude floor with Mach at/below the ceiling.

Both return an event dict (time, altitude, velocity, q, Mach) or a failure
event with ``failure_reason`` set; ``None`` means no trigger this step.
"""

from __future__ import annotations

from typing import Any

from .atmosphere import AtmosphereModel
from .models import Config
from .physics import flow_metrics


def deploy_trigger_crossing(
    cfg: Config,
    atmosphere: AtmosphereModel,
    m_entry_kg: float,
    prev_t_s: float,
    dt_s: float,
    prev_alt_m: float,
    new_alt_m: float,
    prev_vel_m_s: float,
    new_vel_m_s: float,
) -> dict[str, Any] | None:
    args = (cfg, atmosphere, m_entry_kg, prev_t_s, dt_s, prev_alt_m, new_alt_m, prev_vel_m_s, new_vel_m_s)
    if cfg.deploy.trigger_mode == "q_window":
        return _q_window_crossing(*args)
    return _mach_q_crossing(*args)


def _event_at_fraction(
    cfg: Config,
    atmosphere: AtmosphereModel,
    m_entry_kg: float,
    prev_t_s: float,
    dt_s: float,
    prev_alt_m: float,
    new_alt_m: float,
    prev_vel_m_s: float,
    new_vel_m_s: float,
    fraction: float,
) -> dict[str, Any]:
    t = prev_t_s + fraction * dt_s
    alt = prev_alt_m + fraction * (new_alt_m - prev_alt_m)
    vel = prev_vel_m_s + fraction * (new_vel_m_s - prev_vel_m_s)
    metrics = flow_metrics(cfg, atmosphere, m_entry_kg, alt, vel)
    return {
        "time_s": t,
        "altitude_m": alt,
        "velocity_m_s": vel,
        "dynamic_pressure_pa": metrics["dynamic_pressure_pa"],
        "mach": metrics["mach"],
    }


def _crossing_fraction(previous: float, current: float, target: float) -> float:
    denom = current - previous
    if denom == 0.0:
        return 0.0
    return min(1.0, max(0.0, (target - previous) / denom))


def _mach_q_crossing(
    cfg: Config,
    atmosphere: AtmosphereModel,
    m_entry_kg: float,
    prev_t_s: float,
    dt_s: float,
    prev_alt_m: float,
    new_alt_m: float,
    prev_vel_m_s: float,
    new_vel_m_s: float,
) -> dict[str, Any] | None:
    deploy = cfg.deploy
    prev = flow_metrics(cfg, atmosphere, m_entry_kg, prev_alt_m, prev_vel_m_s)
    new = flow_metrics(cfg, atmosphere, m_entry_kg, new_alt_m, new_vel_m_s)
    step = (cfg, atmosphere, m_entry_kg, prev_t_s, dt_s, prev_alt_m, new_alt_m, prev_vel_m_s, new_vel_m_s)

    window_met = (
        new["mach"] <= deploy.trigger_mach
        and deploy.q_floor_pa <= new["dynamic_pressure_pa"] <= deploy.q_cap_pa
        and new_alt_m >= deploy.min_altitude_m
    )
    if window_met:
        fractions = [0.0]
        if prev["mach"] > deploy.trigger_mach >= new["mach"]:
            fractions.append(_crossing_fraction(prev["mach"], new["mach"], deploy.trigger_mach))
        if prev["dynamic_pressure_pa"] < deploy.q_floor_pa <= new["dynamic_pressure_pa"]:
            fractions.append(
                _crossing_fraction(prev["dynamic_pressure_pa"], new["dynamic_pressure_pa"], deploy.q_floor_pa)
            )
        return _event_at_fraction(*step, min(1.0, max(0.0, max(fractions))))

    if new["mach"] <= deploy.trigger_mach:
        fraction = _crossing_fraction(prev["mach"], new["mach"], deploy.trigger_mach)
        if new_alt_m >= deploy.min_altitude_m and new["dynamic_pressure_pa"] > deploy.q_cap_pa:
            event = _event_at_fraction(*step, fraction)
            event["failure_reason"] = "deploy_trigger_q_cap_exceeded"
            return event
        if new_alt_m < deploy.min_altitude_m:
            event = _event_at_fraction(*step, fraction)
            event["failure_reason"] = "deploy_trigger_altitude_below_min"
            return event
    return None


def _q_window_crossing(
    cfg: Config,
    atmosphere: AtmosphereModel,
    m_entry_kg: float,
    prev_t_s: float,
    dt_s: float,
    prev_alt_m: float,
    new_alt_m: float,
    prev_vel_m_s: float,
    new_vel_m_s: float,
) -> dict[str, Any] | None:
    deploy = cfg.deploy
    prev = flow_metrics(cfg, atmosphere, m_entry_kg, prev_alt_m, prev_vel_m_s)
    new = flow_metrics(cfg, atmosphere, m_entry_kg, new_alt_m, new_vel_m_s)
    step = (cfg, atmosphere, m_entry_kg, prev_t_s, dt_s, prev_alt_m, new_alt_m, prev_vel_m_s, new_vel_m_s)

    # Candidate fractions: every boundary crossing within the step.
    fractions = {0.0, 1.0}
    for prev_value, new_value, target in (
        (prev["dynamic_pressure_pa"], new["dynamic_pressure_pa"], deploy.q_floor_pa),
        (prev["dynamic_pressure_pa"], new["dynamic_pressure_pa"], deploy.q_cap_pa),
        (prev["mach"], new["mach"], deploy.mach_ceiling),
        (prev_alt_m, new_alt_m, deploy.min_altitude_m),
    ):
        if (prev_value - target) * (new_value - target) <= 0.0:
            fractions.add(_crossing_fraction(prev_value, new_value, target))

    for fraction in sorted(fractions):
        event = _event_at_fraction(*step, fraction)
        if (
            deploy.q_floor_pa <= float(event["dynamic_pressure_pa"]) <= deploy.q_cap_pa
            and float(event["altitude_m"]) >= deploy.min_altitude_m
            and float(event["mach"]) <= deploy.mach_ceiling
        ):
            return event

    if new_alt_m < deploy.min_altitude_m:
        fraction = (
            _crossing_fraction(prev_alt_m, new_alt_m, deploy.min_altitude_m)
            if prev_alt_m >= deploy.min_altitude_m
            else 1.0
        )
        event = _event_at_fraction(*step, fraction)
        event["failure_reason"] = "deploy_q_window_altitude_below_min"
        return event
    return None
