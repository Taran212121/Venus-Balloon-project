"""1D ballistic entry integrator.

RK4 point-mass integration from the entry interface down to hand-off, with
two trajectory models (constant flight-path angle, or a no-lift spherical
model that propagates gamma) and two hand-off triggers (fixed altitude, or
the parachute deploy event resolved by ``deploy.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .atmosphere import AtmosphereModel, AtmosphereSample
from .deploy import deploy_trigger_crossing
from .models import Config, Trajectory
from .physics import G0_M_S2, cd_at_mach, flow_metrics, gravity_m_s2


@dataclass(frozen=True)
class RunSample:
    """Dispersed inputs for one trajectory run."""

    m_entry_kg: float
    fpa_deg: float
    atmosphere: AtmosphereSample | None = None
    label: str = "nominal"

    def as_mapping(self) -> dict[str, Any]:
        sample = self.atmosphere or AtmosphereSample()
        return {
            "label": self.label,
            "m_entry_kg": self.m_entry_kg,
            "fpa_deg": self.fpa_deg,
            "rho_sigma": sample.rho_sigma,
            "density_multiplier": sample.density_multiplier,
            "temperature_delta_k": sample.temperature_delta_k,
            "gram_sample_index": sample.gram_sample_index,
        }


class _Recorder:
    """Accumulates trajectory channels and trapezoidal heat loads."""

    CHANNELS = (
        "time_s",
        "altitude_m",
        "velocity_m_s",
        "mach",
        "deceleration_g",
        "dynamic_pressure_pa",
        "heat_flux_conv_w_m2",
        "heat_flux_rad_w_m2",
        "heat_flux_total_w_m2",
        "heat_load_conv_j_m2",
        "heat_load_rad_j_m2",
        "heat_load_total_j_m2",
        "payload_temperature_k",
        "payload_absorbed_energy_j",
        "density_kg_m3",
        "temperature_k",
    )

    def __init__(self, cfg: Config, atmosphere: AtmosphereModel, m_entry_kg: float) -> None:
        self._cfg = cfg
        self._atmosphere = atmosphere
        self._m_entry_kg = m_entry_kg
        self.data: dict[str, list[float]] = {name: [] for name in self.CHANNELS}

    def append(self, t_s: float, altitude_m: float, velocity_m_s: float) -> None:
        cfg = self._cfg
        d = self.data
        metrics = flow_metrics(cfg, self._atmosphere, self._m_entry_kg, altitude_m, max(0.0, velocity_m_s))
        dt_s = t_s - d["time_s"][-1] if d["time_s"] else 0.0

        d["time_s"].append(t_s)
        d["altitude_m"].append(altitude_m)
        d["velocity_m_s"].append(max(0.0, velocity_m_s))
        d["mach"].append(metrics["mach"])
        d["deceleration_g"].append(metrics["drag_accel_m_s2"] / G0_M_S2)
        d["dynamic_pressure_pa"].append(metrics["dynamic_pressure_pa"])
        for flux, load in (
            ("heat_flux_conv_w_m2", "heat_load_conv_j_m2"),
            ("heat_flux_rad_w_m2", "heat_load_rad_j_m2"),
            ("heat_flux_total_w_m2", "heat_load_total_j_m2"),
        ):
            d[flux].append(metrics[flux])
            previous = d[load][-1] if d[load] else 0.0
            increment = (
                0.5 * (d[flux][-2] + d[flux][-1]) * dt_s if len(d[flux]) >= 2 and dt_s > 0.0 else 0.0
            )
            d[load].append(previous + increment)
        self._append_payload_thermal(metrics["temperature_k"], dt_s)
        d["density_kg_m3"].append(metrics["density_kg_m3"])
        d["temperature_k"].append(metrics["temperature_k"])

    def _append_payload_thermal(self, env_temp_k: float, dt_s: float) -> None:
        cfg = self._cfg.payload_thermal
        temps = self.data["payload_temperature_k"]
        energies = self.data["payload_absorbed_energy_j"]
        if not temps:
            temps.append(cfg.initial_temp_k)
            energies.append(0.0)
            return
        if not cfg.enabled or dt_s <= 0.0:
            temps.append(temps[-1])
            energies.append(energies[-1])
            return
        heat_capacity_j_k = cfg.payload_mass_kg * cfg.payload_cp_j_kg_k
        prev_env_k = self.data["temperature_k"][-1]
        prev_power_w = _payload_power_w(cfg, prev_env_k, temps[-1])
        predicted_temp_k = temps[-1] + prev_power_w * dt_s / heat_capacity_j_k
        current_power_w = _payload_power_w(cfg, env_temp_k, predicted_temp_k)
        absorbed_j = energies[-1] + 0.5 * (prev_power_w + current_power_w) * dt_s
        energies.append(absorbed_j)
        temps.append(cfg.initial_temp_k + absorbed_j / heat_capacity_j_k)


def _payload_power_w(cfg, env_temp_k: float, payload_temp_k: float) -> float:
    return cfg.ua_eff_w_k * max(0.0, env_temp_k - payload_temp_k) + cfg.residual_radiation_w


def integrate(cfg: Config, sample: RunSample | None = None) -> Trajectory:
    run = sample or RunSample(m_entry_kg=cfg.m_entry_kg, fpa_deg=cfg.entry_fpa_deg)
    atmosphere = AtmosphereModel(cfg, run.atmosphere)
    recorder = _Recorder(cfg, atmosphere, run.m_entry_kg)
    fpa_rad = math.radians(run.fpa_deg)
    gamma_rad = fpa_rad
    dt = cfg.numerics.time_step_s

    t = 0.0
    altitude = cfg.interface_altitude_m
    velocity = cfg.entry_velocity_m_s
    failure_reason: str | None = None
    event_times: dict[str, float | None] = {
        "T0_interface": 0.0,
        "peak_g": None,
        "peak_q": None,
        "peak_heat": None,
        "drogue_deploy": None,
        "main_deploy": None,
        "handoff": None,
    }

    recorder.append(t, altitude, velocity)
    entry_cd = cd_at_mach(cfg, recorder.data["mach"][0])

    while t < cfg.numerics.max_time_s:
        if cfg.handoff_trigger == "altitude" and altitude <= cfg.handoff_alt_m:
            event_times["handoff"] = t
            break
        if altitude > cfg.numerics.max_altitude_m and t > 0.0:
            failure_reason = "skip_out_or_above_max_altitude"
            break
        if velocity <= 1.0:
            failure_reason = "velocity_stalled_before_handoff"
            break

        prev_t, prev_alt, prev_vel = t, altitude, velocity
        if cfg.numerics.trajectory_model == "spherical_ballistic":
            new_alt, new_vel, gamma_rad = _rk4_spherical(
                cfg, atmosphere, run.m_entry_kg, prev_alt, prev_vel, gamma_rad, dt
            )
        else:
            new_alt, new_vel = _rk4_constant_fpa(
                cfg, atmosphere, run.m_entry_kg, fpa_rad, prev_alt, prev_vel, dt
            )
        t = prev_t + dt

        for key, alt_threshold in (
            ("drogue_deploy", cfg.events.drogue_deploy_alt_m),
            ("main_deploy", cfg.events.main_deploy_alt_m),
        ):
            if alt_threshold is not None and prev_alt > alt_threshold >= new_alt:
                fraction = (alt_threshold - prev_alt) / (new_alt - prev_alt)
                event_times[key] = prev_t + fraction * dt

        if cfg.handoff_trigger == "deploy_event":
            trigger = deploy_trigger_crossing(
                cfg, atmosphere, run.m_entry_kg, prev_t, dt, prev_alt, new_alt, prev_vel, new_vel
            )
            if trigger is not None:
                t = float(trigger["time_s"])
                altitude = float(trigger["altitude_m"])
                velocity = float(trigger["velocity_m_s"])
                recorder.append(t, altitude, velocity)
                if trigger.get("failure_reason"):
                    failure_reason = str(trigger["failure_reason"])
                else:
                    event_times["drogue_deploy"] = t
                    event_times["handoff"] = t
                break

        if cfg.handoff_trigger == "altitude" and prev_alt > cfg.handoff_alt_m >= new_alt:
            fraction = (cfg.handoff_alt_m - prev_alt) / (new_alt - prev_alt)
            t = prev_t + fraction * dt
            altitude = cfg.handoff_alt_m
            velocity = prev_vel + fraction * (new_vel - prev_vel)
            recorder.append(t, altitude, velocity)
            event_times["handoff"] = t
            break

        altitude, velocity = new_alt, new_vel
        recorder.append(t, altitude, velocity)
    else:
        failure_reason = "max_time_exceeded_before_handoff"

    if event_times["handoff"] is None and failure_reason is None:
        failure_reason = "handoff_not_reached"

    data = recorder.data
    for key, channel in (("peak_g", "deceleration_g"), ("peak_q", "dynamic_pressure_pa"), ("peak_heat", "heat_flux_total_w_m2")):
        if data["time_s"]:
            event_times[key] = data["time_s"][_argmax(data[channel])]

    return Trajectory(
        success=failure_reason is None,
        failure_reason=failure_reason,
        event_times_s=event_times,
        input_sample=run.as_mapping(),
        validation_context=_validation_context(cfg, run, entry_cd),
        **{name: tuple(values) for name, values in data.items()},
    )


def _validation_context(cfg: Config, run: RunSample, entry_cd: float) -> dict[str, Any]:
    return {
        "entry_velocity_m_s": cfg.entry_velocity_m_s,
        "entry_fpa_deg": run.fpa_deg,
        "m_entry_kg": run.m_entry_kg,
        "entry_cd": entry_cd,
        "reference_area_m2": cfg.reference_area_m2,
        "scale_height_m": cfg.atmosphere.scale_height_m,
        "peak_g_cap_g": cfg.validation.peak_g_cap_g,
        "peak_heat_cap_w_m2": cfg.validation.peak_heat_cap_w_m2,
        "payload_energy_capacity_j": cfg.payload_thermal.energy_capacity_j,
        "payload_thermal_enabled": cfg.payload_thermal.enabled,
        "payload_thermal_ua_eff_w_k": cfg.payload_thermal.ua_eff_w_k,
        "payload_thermal_residual_radiation_w": cfg.payload_thermal.residual_radiation_w,
        "deploy_trigger_mode": cfg.deploy.trigger_mode,
        "deploy_q_floor_pa": cfg.deploy.q_floor_pa,
        "deploy_q_cap_pa": cfg.deploy.q_cap_pa,
        "deploy_min_altitude_m": cfg.deploy.min_altitude_m,
        "deploy_trigger_mach": cfg.deploy.trigger_mach,
        "allen_eggers_tolerance_fraction": cfg.validation.allen_eggers_tolerance_fraction,
        "include_gravity": cfg.numerics.include_gravity,
        "trajectory_model": cfg.numerics.trajectory_model,
    }


def _rk4_constant_fpa(
    cfg: Config,
    atmosphere: AtmosphereModel,
    m_entry_kg: float,
    fpa_rad: float,
    altitude_m: float,
    velocity_m_s: float,
    dt_s: float,
) -> tuple[float, float]:
    def deriv(h: float, v: float) -> tuple[float, float]:
        metrics = flow_metrics(cfg, atmosphere, m_entry_kg, h, max(0.0, v))
        gravity = gravity_m_s2(cfg, h) if cfg.numerics.include_gravity else 0.0
        return (
            max(0.0, v) * math.sin(fpa_rad),
            -metrics["drag_accel_m_s2"] - gravity * math.sin(fpa_rad),
        )

    k1 = deriv(altitude_m, velocity_m_s)
    k2 = deriv(altitude_m + 0.5 * dt_s * k1[0], velocity_m_s + 0.5 * dt_s * k1[1])
    k3 = deriv(altitude_m + 0.5 * dt_s * k2[0], velocity_m_s + 0.5 * dt_s * k2[1])
    k4 = deriv(altitude_m + dt_s * k3[0], velocity_m_s + dt_s * k3[1])
    next_h = altitude_m + dt_s * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
    next_v = velocity_m_s + dt_s * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
    return next_h, max(0.0, next_v)


def _rk4_spherical(
    cfg: Config,
    atmosphere: AtmosphereModel,
    m_entry_kg: float,
    altitude_m: float,
    velocity_m_s: float,
    gamma_rad: float,
    dt_s: float,
) -> tuple[float, float, float]:
    """No-lift spherical point-mass step with propagated flight-path angle."""

    def deriv(h: float, v: float, gamma: float) -> tuple[float, float, float]:
        metrics = flow_metrics(cfg, atmosphere, m_entry_kg, h, max(0.0, v))
        gravity = gravity_m_s2(cfg, h) if cfg.numerics.include_gravity else 0.0
        radius_m = cfg.venus_radius_m + max(0.0, h)
        safe_v = max(v, 1.0e-9)
        return (
            max(0.0, v) * math.sin(gamma),
            -metrics["drag_accel_m_s2"] - gravity * math.sin(gamma),
            (safe_v / radius_m - gravity / safe_v) * math.cos(gamma),
        )

    k1 = deriv(altitude_m, velocity_m_s, gamma_rad)
    k2 = deriv(
        altitude_m + 0.5 * dt_s * k1[0],
        velocity_m_s + 0.5 * dt_s * k1[1],
        gamma_rad + 0.5 * dt_s * k1[2],
    )
    k3 = deriv(
        altitude_m + 0.5 * dt_s * k2[0],
        velocity_m_s + 0.5 * dt_s * k2[1],
        gamma_rad + 0.5 * dt_s * k2[2],
    )
    k4 = deriv(altitude_m + dt_s * k3[0], velocity_m_s + dt_s * k3[1], gamma_rad + dt_s * k3[2])
    next_h = altitude_m + dt_s * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6.0
    next_v = velocity_m_s + dt_s * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6.0
    next_gamma = gamma_rad + dt_s * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6.0
    return next_h, max(0.0, next_v), next_gamma


def _argmax(values: list[float]) -> int:
    best_index = 0
    for idx, value in enumerate(values):
        if value > values[best_index]:
            best_index = idx
    return best_index
