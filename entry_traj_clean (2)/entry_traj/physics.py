"""Point-mass flow and heating relations shared by the integrator and checks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .atmosphere import AtmosphereModel
    from .models import Config

G0_M_S2 = 9.80665


def gravity_m_s2(cfg: "Config", altitude_m: float) -> float:
    radius = cfg.venus_radius_m
    return cfg.venus_gravity_m_s2 * (radius / (radius + max(0.0, altitude_m))) ** 2


def cd_at_mach(cfg: "Config", mach: float) -> float:
    """Constant Cd, or linear interpolation in the Cd(M) table when provided."""
    if not cfg.cd_mach_table:
        if cfg.cd is None:
            raise ValueError("Config requires cd or cd_mach_table")
        return cfg.cd
    table = sorted(cfg.cd_mach_table)
    if mach <= table[0][0]:
        return table[0][1]
    if mach >= table[-1][0]:
        return table[-1][1]
    for (m0, cd0), (m1, cd1) in zip(table, table[1:]):
        if mach <= m1:
            return cd0 + (mach - m0) / (m1 - m0) * (cd1 - cd0)
    return table[-1][1]


def flow_metrics(
    cfg: "Config",
    atmosphere: "AtmosphereModel",
    m_entry_kg: float,
    altitude_m: float,
    velocity_m_s: float,
) -> dict[str, float]:
    """Local density, Mach, dynamic pressure, drag and stagnation heat fluxes.

    Convective heating is Sutton-Graves; radiative heating is the
    PV-calibrated CO2 power law q_r = C * rho^a * v^b * Rn^c.
    """
    density = atmosphere.density_kg_m3(altitude_m)
    temperature = atmosphere.temperature_k(altitude_m)
    speed_of_sound = atmosphere.speed_of_sound_m_s(altitude_m)
    mach = velocity_m_s / speed_of_sound if speed_of_sound > 0 else 0.0
    dynamic_pressure = 0.5 * density * velocity_m_s**2
    drag_accel = dynamic_pressure * cd_at_mach(cfg, mach) * cfg.reference_area_m2 / m_entry_kg

    convective = (
        cfg.heating.convective.sutton_graves_coefficient_si
        * math.sqrt(max(density, 0.0) / cfg.nose_radius_m)
        * velocity_m_s**3
    )
    rad = cfg.heating.radiative
    radiative = (
        rad.coefficient_si
        * max(density, 0.0) ** rad.density_exponent
        * velocity_m_s**rad.velocity_exponent
        * cfg.nose_radius_m**rad.nose_radius_exponent
    )
    return {
        "density_kg_m3": density,
        "temperature_k": temperature,
        "mach": mach,
        "dynamic_pressure_pa": dynamic_pressure,
        "drag_accel_m_s2": drag_accel,
        "heat_flux_conv_w_m2": convective,
        "heat_flux_rad_w_m2": radiative,
        "heat_flux_total_w_m2": convective + radiative,
    }


def interp(x: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    """Piecewise-linear interpolation with flat extrapolation. xs must ascend."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    fraction = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + fraction * (ys[hi] - ys[lo])


def log_interp(x: float, xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    """Linear interpolation in log-space of y (for densities). xs must ascend."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    if ys[lo] <= 0.0 or ys[hi] <= 0.0:
        fraction = (x - xs[lo]) / (xs[hi] - xs[lo])
        return ys[lo] + fraction * (ys[hi] - ys[lo])
    fraction = (x - xs[lo]) / (xs[hi] - xs[lo])
    return math.exp(math.log(ys[lo]) + fraction * (math.log(ys[hi]) - math.log(ys[lo])))
