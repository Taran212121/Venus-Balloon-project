"""Configuration and result records.

Every record is a plain frozen dataclass. ``to_plain`` converts any record to
JSON-ready dicts and lists; there are no per-class serializers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping


def to_plain(obj: Any) -> Any:
    """Recursively convert dataclasses, mappings and sequences to JSON-safe types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {str(key): to_plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(value) for value in obj]
    return obj


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TemperatureConfig:
    kind: str = "table"  # table | constant | fallback_speed_of_sound
    altitude_m: tuple[float, ...] = ()
    temperature_k: tuple[float, ...] = ()
    constant_temperature_k: float | None = None
    fallback_speed_of_sound_m_s: float | None = None


@dataclass(frozen=True)
class DispersionConfig:
    """Atmosphere density/temperature dispersion source."""

    mode: str = "synthetic_gram_fixture"  # synthetic_gram_fixture | venus_gram_hdf5
    venus_gram_hdf5_path: str | None = None
    density_multiplier_dataset: str = "/density_multiplier"
    temperature_delta_dataset: str | None = "/temperature_delta_k"
    altitude_dataset: str | None = "/altitude_m"
    synthetic_density_ln_sigma: float = 0.15
    synthetic_temperature_sigma_k: float = 8.0


@dataclass(frozen=True)
class AtmosphereConfig:
    model: str = "exponential"  # exponential | profile
    vcd_envelope_path: str | None = None
    # Exponential parameters; retained for the Allen-Eggers benchmark even
    # when the run itself uses a profile.
    rho_ref_kg_m3: float = 0.01
    alt_ref_m: float = 80_000.0
    scale_height_m: float = 7_000.0
    density_kind: str = "exponential"  # exponential | table | profile_file
    profile_path: str | None = None
    profile_format: str | None = None  # vcd_vira | venusgram_vira
    profile_selector: str | None = None
    density_altitude_m: tuple[float, ...] = ()
    density_kg_m3: tuple[float, ...] = ()
    temperature: TemperatureConfig = field(default_factory=TemperatureConfig)
    dispersion: DispersionConfig = field(default_factory=DispersionConfig)


@dataclass(frozen=True)
class ConvectiveHeatingConfig:
    sutton_graves_coefficient_si: float
    source_id: str = "PV-4PROBE-CAL-CO2-CONV"


@dataclass(frozen=True)
class RadiativeHeatingConfig:
    coefficient_si: float
    density_exponent: float = 0.5
    velocity_exponent: float = 8.0
    nose_radius_exponent: float = 0.0
    source_id: str = "PV-4PROBE-CAL-CO2-RAD"


@dataclass(frozen=True)
class HeatingConfig:
    convective: ConvectiveHeatingConfig
    radiative: RadiativeHeatingConfig


@dataclass(frozen=True)
class PayloadThermalConfig:
    """Lumped pre-deploy payload thermal gate (active TMS assumed post-deploy)."""

    enabled: bool = True
    payload_mass_kg: float = 14.0
    payload_cp_j_kg_k: float = 800.0
    initial_temp_k: float = 273.15
    limit_temp_k: float = 333.15
    ua_eff_w_k: float = 0.0
    residual_radiation_w: float = 0.0

    @property
    def energy_capacity_j(self) -> float:
        return self.payload_mass_kg * self.payload_cp_j_kg_k * (
            self.limit_temp_k - self.initial_temp_k
        )


@dataclass(frozen=True)
class EventConfig:
    drogue_deploy_alt_m: float | None = None
    main_deploy_alt_m: float | None = None


@dataclass(frozen=True)
class DeployConfig:
    trigger_mode: str = "mach_q"  # mach_q | q_window
    trigger_mach: float = 2.0
    mach_ceiling: float = 2.2
    q_floor_pa: float = 200.0
    q_cap_pa: float = 3000.0
    min_altitude_m: float = 67_000.0
    disreef_mach_max: float = 0.8


@dataclass(frozen=True)
class NumericsConfig:
    time_step_s: float = 0.10
    max_time_s: float = 1200.0
    max_altitude_m: float = 250_000.0
    include_gravity: bool = True
    trajectory_model: str = "constant_fpa"  # constant_fpa | spherical_ballistic


@dataclass(frozen=True)
class ValidationConfig:
    peak_g_cap_g: float = 200.0
    peak_heat_cap_w_m2: float = 20_000_000.0
    allen_eggers_tolerance_fraction: float = 0.05


@dataclass(frozen=True)
class PersistenceConfig:
    workbook_path: str = "results/workbook_v1.xlsx"
    latex_macros_path: str = "results/macros.tex"
    require_existing_workbook: bool = False


@dataclass(frozen=True)
class Config:
    interface_altitude_m: float
    entry_velocity_m_s: float
    entry_fpa_deg: float
    m_entry_kg: float
    cd: float | None
    cd_mach_table: tuple[tuple[float, float], ...]
    diameter_m: float | None
    reference_area_m2: float
    nose_radius_m: float
    handoff_alt_m: float
    handoff_trigger: str  # altitude | deploy_event
    venus_radius_m: float
    venus_gravity_m_s2: float
    gas_gamma: float
    gas_constant_j_kg_k: float
    atmosphere: AtmosphereConfig
    heating: HeatingConfig
    payload_thermal: PayloadThermalConfig
    events: EventConfig
    deploy: DeployConfig
    dispersions_n_runs: int
    dispersions_seed: int
    dispersion_mass_fraction: float
    dispersion_fpa_deg: float
    dispersion_fpa_is_3sigma: bool
    numerics: NumericsConfig
    validation: ValidationConfig
    persistence: PersistenceConfig
    architecture: dict[str, Any] = field(default_factory=dict)
    source_ids: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ballistic_coefficient_kg_m2(self) -> float | None:
        if self.cd is None:
            return None
        return self.m_entry_kg / (self.cd * self.reference_area_m2)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Trajectory:
    success: bool
    failure_reason: str | None
    time_s: tuple[float, ...]
    altitude_m: tuple[float, ...]
    velocity_m_s: tuple[float, ...]
    mach: tuple[float, ...]
    deceleration_g: tuple[float, ...]
    dynamic_pressure_pa: tuple[float, ...]
    heat_flux_conv_w_m2: tuple[float, ...]
    heat_flux_rad_w_m2: tuple[float, ...]
    heat_flux_total_w_m2: tuple[float, ...]
    heat_load_conv_j_m2: tuple[float, ...]
    heat_load_rad_j_m2: tuple[float, ...]
    heat_load_total_j_m2: tuple[float, ...]
    payload_temperature_k: tuple[float, ...]
    payload_absorbed_energy_j: tuple[float, ...]
    density_kg_m3: tuple[float, ...]
    temperature_k: tuple[float, ...]
    event_times_s: dict[str, float | None]
    input_sample: dict[str, Any]
    validation_context: dict[str, Any]


@dataclass(frozen=True)
class Ensemble:
    config: Config
    n_requested: int
    seed: int
    trajectories: tuple[Trajectory, ...]
    samples: tuple[dict[str, Any], ...]

    @property
    def n_success(self) -> int:
        return sum(1 for run in self.trajectories if run.success)

    @property
    def n_failed(self) -> int:
        return len(self.trajectories) - self.n_success


@dataclass(frozen=True)
class Summary:
    n_requested: int
    n_success: int
    n_failed: int
    metric_intervals: dict[str, dict[str, Any]]
    sobol_s1: dict[str, dict[str, float]]
    sobol_sum_s1: dict[str, float]
    sobol_interaction_or_residual_share: dict[str, float]
    dominant_driver: dict[str, str]
    handoff_corridor: dict[str, Any]
    event_time_arrays_s: dict[str, tuple[float | None, ...]]
    source_ids: dict[str, str]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    """Validation report: one dict per check plus the run context."""

    passed: bool
    checks: tuple[dict[str, Any], ...]
    context: dict[str, Any]
