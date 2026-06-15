"""Post-deploy descent: staged parachute drag, jettison, buoyancy ramp.

A 1D vertical-descent model from the deploy event down to float hand-off
(D33/FA4 timing and power windows, not parachute FSI). Enabled only when the
raw config contains ``mission_timeline.post_deploy_model`` with
``enabled: true``; otherwise ``simulate_post_deploy`` returns ``None`` and
downstream outputs skip it.

Stage sequence (config keys in ``mission_timeline.post_deploy_model``):

- ``dgb_filling``: CdS ramps 0 -> ``dgb_pilot_cds_m2`` over ``dgb_fill_time_s``.
- ``reefed``: ``reefed_ringsail_cds_m2`` until ``disreef_delay_s`` has elapsed
  AND Mach <= ``disreef_mach`` (defaults to ``deploy.disreef_mach_max``).
- ``full_ringsail``: CdS ramps reefed -> ``full_ringsail_cds_m2`` over
  ``ringsail_fill_time_s``.

Mass steps from ``entry_stack_mass_kg`` to ``post_jettison_mass_kg`` at
``jettison_delay_s``. Buoyancy ramps linearly from ``inflation_start_delay_s``
over ``inflation_duration_s`` with displaced volume
``n2_gas_mass_kg / n2_density_float_kg_m3``. Float hand-off occurs when the
stack reaches ``target_altitude_m``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .atmosphere import AtmosphereModel
from .models import Config, Trajectory


@dataclass(frozen=True)
class PostDeployResult:
    time_s: tuple[float, ...]
    altitude_m: tuple[float, ...]
    velocity_m_s: tuple[float, ...]
    mass_kg: tuple[float, ...]
    cds_m2: tuple[float, ...]
    buoyancy_n: tuple[float, ...]
    stage: tuple[str, ...]
    events_s: dict[str, float | None]
    anchors: dict[str, float]
    success: bool
    failure_reason: str | None

    def to_plain(self) -> dict[str, Any]:
        return {
            "events_s": self.events_s,
            "anchors": self.anchors,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "n_points": len(self.time_s),
        }


def simulate_post_deploy(cfg: Config, trajectory: Trajectory) -> PostDeployResult | None:
    raw = cfg.raw.get("mission_timeline", {}).get("post_deploy_model")
    if not isinstance(raw, dict) or not raw.get("enabled", False):
        return None
    deploy_time = trajectory.event_times_s.get("drogue_deploy")
    if deploy_time is None:
        deploy_time = trajectory.event_times_s.get("handoff")
    if deploy_time is None or not trajectory.time_s:
        return None

    atmosphere = AtmosphereModel(cfg)
    target_alt = float(raw.get("target_altitude_m", 55000.0))
    dt = float(raw.get("time_step_s", 0.25))
    max_time = float(raw.get("max_time_s", 1800.0))
    if dt <= 0.0 or max_time <= 0.0:
        raise ValueError("post_deploy_model time_step_s and max_time_s must be positive")

    deploy_t = float(deploy_time)
    altitude = float(trajectory.altitude_m[-1])
    velocity = max(0.0, float(trajectory.velocity_m_s[-1]))

    events: dict[str, float | None] = {
        "deploy": deploy_t,
        "dgb_inflated": deploy_t + float(raw.get("dgb_fill_time_s", 1.0)),
        "disreef": None,
        "jettison": None,
        "inflation_start": None,
        "inflation_end": None,
        "float_handoff": None,
    }
    channels: dict[str, list] = {
        key: []
        for key in ("time_s", "altitude_m", "velocity_m_s", "mass_kg", "cds_m2", "buoyancy_n", "stage")
    }

    def record(t_rel: float, alt: float, vel: float) -> None:
        sound = atmosphere.speed_of_sound_m_s(alt)
        stage, cds = _stage(raw, cfg, t_rel, vel, sound)
        rho = atmosphere.density_kg_m3(alt)
        channels["time_s"].append(deploy_t + t_rel)
        channels["altitude_m"].append(alt)
        channels["velocity_m_s"].append(vel)
        channels["mass_kg"].append(_mass_kg(raw, t_rel))
        channels["cds_m2"].append(cds)
        channels["buoyancy_n"].append(_buoyancy_n(raw, cfg, rho, _gas_fraction(raw, t_rel), alt))
        channels["stage"].append(stage)

    t_rel = 0.0
    record(t_rel, altitude, velocity)
    while altitude > target_alt and t_rel < max_time and velocity >= 0.0:
        prev_t, prev_alt, prev_vel = t_rel, altitude, velocity
        altitude, velocity = _rk4_step(raw, cfg, atmosphere, prev_t, prev_alt, prev_vel, dt)
        t_rel = prev_t + dt
        _capture_events(raw, cfg, atmosphere, events, deploy_t, prev_t, t_rel, prev_alt, altitude, prev_vel, velocity)
        record(t_rel, altitude, velocity)

    if channels["altitude_m"] and channels["altitude_m"][-1] <= target_alt:
        events["float_handoff"] = channels["time_s"][-1]

    success = events["float_handoff"] is not None
    return PostDeployResult(
        time_s=tuple(channels["time_s"]),
        altitude_m=tuple(channels["altitude_m"]),
        velocity_m_s=tuple(channels["velocity_m_s"]),
        mass_kg=tuple(channels["mass_kg"]),
        cds_m2=tuple(channels["cds_m2"]),
        buoyancy_n=tuple(channels["buoyancy_n"]),
        stage=tuple(channels["stage"]),
        events_s=events,
        anchors=_anchors(raw),
        success=success,
        failure_reason=None if success else "target_altitude_not_reached",
    )


def post_deploy_duration_s(result: PostDeployResult | None) -> float | None:
    if result is None:
        return None
    deploy = result.events_s.get("deploy")
    float_t = result.events_s.get("float_handoff")
    if deploy is None or float_t is None:
        return None
    return float(float_t) - float(deploy)


def _stage(raw: dict[str, Any], cfg: Config, t_rel: float, velocity: float, speed_of_sound: float) -> tuple[str, float]:
    mach = velocity / max(speed_of_sound, 1.0e-9)
    dgb_fill = float(raw.get("dgb_fill_time_s", 1.0))
    disreef_delay = float(raw.get("disreef_delay_s", 5.0))
    disreef_mach = float(raw.get("disreef_mach", cfg.deploy.disreef_mach_max))
    if t_rel < dgb_fill:
        return "dgb_filling", _ramp(0.0, float(raw.get("dgb_pilot_cds_m2", 7.20)), t_rel / max(dgb_fill, 1.0e-9))
    if t_rel < disreef_delay or mach > disreef_mach:
        return "reefed", float(raw.get("reefed_ringsail_cds_m2", 5.15))
    fill_time = float(raw.get("ringsail_fill_time_s", 2.0))
    reefed_cds = float(raw.get("reefed_ringsail_cds_m2", 5.15))
    full_cds = float(raw.get("full_ringsail_cds_m2", 40.2))
    fraction = min(1.0, max(0.0, (t_rel - disreef_delay) / max(fill_time, 1.0e-9)))
    return "full_ringsail", _ramp(reefed_cds, full_cds, fraction)


def _mass_kg(raw: dict[str, Any], t_rel: float) -> float:
    initial = float(raw.get("entry_stack_mass_kg", 689.130))
    post_jettison = float(raw.get("post_jettison_mass_kg", initial))
    return post_jettison if t_rel >= float(raw.get("jettison_delay_s", 18.0)) else initial


def _gas_fraction(raw: dict[str, Any], t_rel: float) -> float:
    start = float(raw.get("inflation_start_delay_s", 21.0))
    duration = float(raw.get("inflation_duration_s", 30.0))
    if t_rel <= start:
        return 0.0
    return min(1.0, max(0.0, (t_rel - start) / max(duration, 1.0e-9)))


def _buoyancy_n(raw: dict[str, Any], cfg: Config, rho_atm: float, gas_fraction: float, altitude_m: float) -> float:
    gas_mass = float(raw.get("n2_gas_mass_kg", 260.0))
    gas_density_float = float(raw.get("n2_density_float_kg_m3", 0.59))
    volume_m3 = gas_mass / max(gas_density_float, 1.0e-9)
    return rho_atm * volume_m3 * gas_fraction * _gravity_m_s2(cfg, altitude_m)


def _rk4_step(
    raw: dict[str, Any],
    cfg: Config,
    atmosphere: AtmosphereModel,
    t_rel: float,
    altitude_m: float,
    velocity_m_s: float,
    dt: float,
) -> tuple[float, float]:
    def deriv(ti: float, hi: float, vi: float) -> tuple[float, float]:
        rho = atmosphere.density_kg_m3(hi)
        sound = atmosphere.speed_of_sound_m_s(hi)
        _, cds = _stage(raw, cfg, ti, max(0.0, vi), sound)
        mass = _mass_kg(raw, ti)
        drag = 0.5 * rho * max(0.0, vi) ** 2 * cds
        buoyancy = _buoyancy_n(raw, cfg, rho, _gas_fraction(raw, ti), hi)
        dhdt = -max(0.0, vi)
        dvdt = _gravity_m_s2(cfg, hi) - drag / max(mass, 1.0e-9) - buoyancy / max(mass, 1.0e-9)
        return dhdt, dvdt

    k1_h, k1_v = deriv(t_rel, altitude_m, velocity_m_s)
    k2_h, k2_v = deriv(t_rel + 0.5 * dt, altitude_m + 0.5 * dt * k1_h, velocity_m_s + 0.5 * dt * k1_v)
    k3_h, k3_v = deriv(t_rel + 0.5 * dt, altitude_m + 0.5 * dt * k2_h, velocity_m_s + 0.5 * dt * k2_v)
    k4_h, k4_v = deriv(t_rel + dt, altitude_m + dt * k3_h, velocity_m_s + dt * k3_v)
    next_h = altitude_m + dt * (k1_h + 2.0 * k2_h + 2.0 * k3_h + k4_h) / 6.0
    next_v = velocity_m_s + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    return next_h, max(0.0, next_v)


def _capture_events(
    raw: dict[str, Any],
    cfg: Config,
    atmosphere: AtmosphereModel,
    events: dict[str, float | None],
    deploy_t: float,
    prev_t: float,
    t_rel: float,
    prev_alt: float,
    alt: float,
    prev_vel: float,
    vel: float,
) -> None:
    jettison_delay = float(raw.get("jettison_delay_s", 18.0))
    if events["jettison"] is None and prev_t <= jettison_delay <= t_rel:
        events["jettison"] = deploy_t + jettison_delay
    inflation_start = float(raw.get("inflation_start_delay_s", 21.0))
    if events["inflation_start"] is None and prev_t <= inflation_start <= t_rel:
        events["inflation_start"] = deploy_t + inflation_start
    if events["inflation_end"] is None:
        end = inflation_start + float(raw.get("inflation_duration_s", 30.0))
        if prev_t <= end <= t_rel:
            events["inflation_end"] = deploy_t + end
    if events["disreef"] is None:
        prev_mach = prev_vel / max(atmosphere.speed_of_sound_m_s(prev_alt), 1.0e-9)
        mach = vel / max(atmosphere.speed_of_sound_m_s(alt), 1.0e-9)
        delay = float(raw.get("disreef_delay_s", 5.0))
        disreef_mach = float(raw.get("disreef_mach", cfg.deploy.disreef_mach_max))
        if t_rel >= delay and prev_mach >= disreef_mach >= mach:
            fraction = 0.0 if mach == prev_mach else (disreef_mach - prev_mach) / (mach - prev_mach)
            events["disreef"] = deploy_t + prev_t + min(1.0, max(0.0, fraction)) * (t_rel - prev_t)


def _gravity_m_s2(cfg: Config, altitude_m: float) -> float:
    return cfg.venus_gravity_m_s2 * (cfg.venus_radius_m / (cfg.venus_radius_m + max(0.0, altitude_m))) ** 2


def _anchors(raw: dict[str, Any]) -> dict[str, float]:
    return {
        "dgb_terminal_v_73km_m_s": float(raw.get("dgb_terminal_v_73km_m_s", 184.0)),
        "full_terminal_v_55km_m_s": float(raw.get("full_terminal_v_55km_m_s", 17.3)),
    }


def _ramp(start: float, stop: float, fraction: float) -> float:
    f = min(1.0, max(0.0, fraction))
    return start + (stop - start) * f
