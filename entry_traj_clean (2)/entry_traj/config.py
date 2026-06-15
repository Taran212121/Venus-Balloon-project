"""YAML config loading: defaults, deep-merge, construction, validation.

The schema is unchanged from v0.1 -- existing config files load as-is.
Relative paths are resolved against the config file's directory.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import (
    AtmosphereConfig,
    Config,
    ConvectiveHeatingConfig,
    DeployConfig,
    DispersionConfig,
    EventConfig,
    HeatingConfig,
    NumericsConfig,
    PayloadThermalConfig,
    PersistenceConfig,
    RadiativeHeatingConfig,
    TemperatureConfig,
    ValidationConfig,
)

DEFAULTS: dict[str, Any] = {
    "architecture": {
        "tuple_id": "T2/A1/Mt3/S2/M2/epsilon-1",
        "workbook_id": "EDDI_parametric_mass_sizing_tool_v4_FINAL (1).xlsx",
    },
    "mass_basis": {
        "source_workbook": "EDDI_parametric_mass_sizing_tool_v4_FINAL (1).xlsx",
        "entry_mass_without_system_margin_kg": 689.130,
        "entry_mass_with_system_margin_kg": 755.653,
        "note": "EDDI audit basis: trajectory uses the without-system-margin mass; with-system-margin mass is carried as a closure check.",
    },
    "entry": {
        "interface_altitude_m": 140000.0,
        "velocity_m_s": 10000.0,
        "fpa_deg": None,  # required: set an exact published value
        "m_entry_kg": 689.130,
    },
    "vehicle": {
        "cd": 1.4,
        "cd_mach_table": None,
        "diameter_m": None,
        "reference_area_m2": None,
        "nose_radius_m": 1.0,
        "forebody_wetted_area_m2": 3.6,
        "backshell_wetted_area_m2": 3.4,
    },
    "planet": {"venus_radius_m": 6051800.0, "gravity_m_s2": 8.87},
    "gas": {"gamma": 1.30, "specific_gas_constant_j_kg_k": 188.92},
    "atmosphere": {
        "model": "exponential",
        "vcd_envelope_path": None,
        "rho_ref_kg_m3": 0.01,
        "alt_ref_m": 80000.0,
        "scale_height_m": 7000.0,
        "density": {
            "kind": "exponential",
            "profile_path": None,
            "profile_format": None,
            "profile_selector": None,
            "altitude_m": [],
            "density_kg_m3": [],
        },
        "temperature": {
            "kind": "table",
            "altitude_m": [0.0, 50000.0, 70000.0, 90000.0, 120000.0, 200000.0],
            "temperature_k": [735.0, 340.0, 240.0, 180.0, 160.0, 150.0],
        },
        "dispersion": {
            "mode": "synthetic_gram_fixture",
            "venus_gram_hdf5_path": None,
            "density_multiplier_dataset": "/density_multiplier",
            "temperature_delta_dataset": "/temperature_delta_k",
            "altitude_dataset": "/altitude_m",
            "synthetic_density_ln_sigma": 0.15,
            "synthetic_temperature_sigma_k": 8.0,
        },
    },
    "heating": {
        "convective": {
            "sutton_graves_coefficient_si": 0.00012698543020109976,
            "source_id": "PV-4PROBE-CAL-CO2-CONV",
        },
        "radiative": {
            "coefficient_si": 1.9664544454345846e-24,
            "density_exponent": 0.5,
            "velocity_exponent": 8.0,
            "nose_radius_exponent": 0.0,
            "source_id": "PV-4PROBE-CAL-CO2-RAD",
        },
    },
    "payload_thermal": {
        "enabled": True,
        "payload_mass_kg": 14.0,
        "payload_cp_j_kg_k": 800.0,
        "initial_temp_k": 273.15,
        "limit_temp_k": 333.15,
        "ua_eff_w_k": 0.0,
        "residual_radiation_w": 0.0,
    },
    "handoff": {"altitude_m": 70000.0, "trigger": "altitude"},
    "deploy": {
        "trigger_mode": "mach_q",
        "trigger_mach": 2.0,
        "mach_ceiling": 2.2,
        "q_floor_pa": 200.0,
        "q_cap_pa": 3000.0,
        "min_altitude_m": 67000.0,
        "disreef_mach_max": 0.8,
    },
    "events": {"drogue_deploy_alt_m": None, "main_deploy_alt_m": None},
    "dispersions": {
        "n_runs": 1000,
        "seed": 1001,
        "mass_fraction": 0.10,
        "fpa_deg": 2.0,
        "fpa_dispersion_is_3sigma": False,
    },
    "numerics": {
        "time_step_s": 0.10,
        "max_time_s": 1200.0,
        "max_altitude_m": 250000.0,
        "include_gravity": True,
        "trajectory_model": "constant_fpa",
    },
    "validation": {
        "peak_g_cap_g": 200.0,
        "peak_heat_cap_w_m2": 20_000_000.0,
        "allen_eggers_tolerance_fraction": 0.05,
    },
    # Phase-0 entry-system mass model used by the EFPA sweep to close the
    # steep-vs-shallow trade: steeper -> higher peak g -> heavier aeroshell
    # structure; shallower -> higher integrated heat load -> heavier TPS.
    "mass_model": {
        "enabled": True,
        "tps": {
            # Empirical ablator mass-fraction correlation:
            #   m_tps / m_entry [%] = coefficient_percent * (heat load [J/cm^2])^exponent
            # Applied to the worst-tail design heat load (x1.25 Level-0 multiplier).
            "coefficient_percent": 0.091,
            "exponent": 0.51575,
            "source_id": "TPS-MF-HEATLOAD-LV2003",
        },
        "structure": {
            # Aeroshell/structure mass sized by worst-tail axial load:
            #   m_struct = reference_mass_fraction * m_entry * (peak_g / reference_peak_g)^exponent
            # Placeholder Phase-0 scaling; calibrate reference values against the
            # FA mass budget before using for closure.
            "reference_mass_fraction": 0.14,
            "reference_peak_g": 100.0,
            "exponent": 1.0,
            "source_id": "PHASE0-STRUCT-GLOAD-LINEAR",
        },
    },
    "persistence": {
        "workbook_path": "results/workbook_v1.xlsx",
        "latex_macros_path": "results/macros.tex",
        "require_existing_workbook": False,
    },
    "source_ids": {
        "peak_g": "FA1-D41-entry-traj",
        "peak_q": "FA1-D41-entry-traj",
        "peak_heat": "FA1-D41-entry-traj",
        "heat_load": "FA1-D41-entry-traj",
        "payload_energy_absorbed": "FA8-payload-energy-gate",
        "mach_handoff": "FA1-D41-entry-traj",
    },
}


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load entry_traj YAML config files.") from exc

    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    raw = _deep_merge(DEFAULTS, loaded)
    if raw["atmosphere"].get("density", {}).get("kind") == "profile_file":
        raw["atmosphere"]["model"] = "profile"
    cfg = _build(raw, config_path.parent)
    _validate(cfg, config_path.parent)
    return cfg


def config_from_mapping(raw_in: Mapping[str, Any], base_dir: str | Path = ".") -> Config:
    """Build a validated Config from an in-memory mapping (used by the GUI)."""
    raw = _deep_merge(DEFAULTS, raw_in)
    if raw["atmosphere"].get("density", {}).get("kind") == "profile_file":
        raw["atmosphere"]["model"] = "profile"
    cfg = _build(raw, Path(base_dir))
    _validate(cfg, Path(base_dir))
    return cfg


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build(raw: Mapping[str, Any], base_dir: Path) -> Config:
    entry, vehicle, planet, gas = raw["entry"], raw["vehicle"], raw["planet"], raw["gas"]
    atm_raw = raw["atmosphere"]
    density_raw = atm_raw.get("density", {})
    temp_raw = atm_raw["temperature"]
    disp_raw = atm_raw["dispersion"]
    deploy_raw = raw["deploy"]
    payload_raw = raw["payload_thermal"]
    dispersions = raw["dispersions"]
    numerics = raw["numerics"]
    validation = raw["validation"]
    persistence = raw["persistence"]

    cd_table_raw = vehicle.get("cd_mach_table")
    cd_mach_table = (
        tuple((float(m), float(cd)) for m, cd in cd_table_raw) if cd_table_raw else ()
    )
    diameter_m = _opt_float(vehicle.get("diameter_m"))
    area = _opt_float(vehicle.get("reference_area_m2"))
    if area is None:
        if diameter_m is None:
            raise ValueError("vehicle.reference_area_m2 or vehicle.diameter_m is required")
        area = math.pi * diameter_m**2 / 4.0

    if entry.get("fpa_deg") is None:
        raise ValueError("entry.fpa_deg is required; set an exact published value before running this config")

    gram_path = _opt_path(disp_raw.get("venus_gram_hdf5_path"), base_dir)
    atmosphere = AtmosphereConfig(
        model=str(atm_raw.get("model", "exponential")),
        vcd_envelope_path=_opt_path(atm_raw.get("vcd_envelope_path"), base_dir),
        rho_ref_kg_m3=float(atm_raw["rho_ref_kg_m3"]),
        alt_ref_m=float(atm_raw["alt_ref_m"]),
        scale_height_m=float(atm_raw["scale_height_m"]),
        density_kind=str(density_raw.get("kind", "exponential")),
        profile_path=_opt_path(density_raw.get("profile_path"), base_dir),
        profile_format=_opt_str(density_raw.get("profile_format")),
        profile_selector=_opt_str(density_raw.get("profile_selector")),
        density_altitude_m=tuple(float(v) for v in density_raw.get("altitude_m", ())),
        density_kg_m3=tuple(float(v) for v in density_raw.get("density_kg_m3", ())),
        temperature=TemperatureConfig(
            kind=str(temp_raw.get("kind", "table")),
            altitude_m=tuple(float(v) for v in temp_raw.get("altitude_m", ())),
            temperature_k=tuple(float(v) for v in temp_raw.get("temperature_k", ())),
            constant_temperature_k=_opt_float(temp_raw.get("constant_temperature_k")),
            fallback_speed_of_sound_m_s=_opt_float(temp_raw.get("fallback_speed_of_sound_m_s")),
        ),
        dispersion=DispersionConfig(
            mode=str(disp_raw.get("mode", "synthetic_gram_fixture")),
            venus_gram_hdf5_path=gram_path,
            density_multiplier_dataset=str(disp_raw.get("density_multiplier_dataset", "/density_multiplier")),
            temperature_delta_dataset=_opt_str(disp_raw.get("temperature_delta_dataset")),
            altitude_dataset=_opt_str(disp_raw.get("altitude_dataset")),
            synthetic_density_ln_sigma=float(disp_raw.get("synthetic_density_ln_sigma", 0.15)),
            synthetic_temperature_sigma_k=float(disp_raw.get("synthetic_temperature_sigma_k", 8.0)),
        ),
    )
    heating = HeatingConfig(
        convective=ConvectiveHeatingConfig(
            sutton_graves_coefficient_si=float(raw["heating"]["convective"]["sutton_graves_coefficient_si"]),
            source_id=str(raw["heating"]["convective"]["source_id"]),
        ),
        radiative=RadiativeHeatingConfig(
            coefficient_si=float(raw["heating"]["radiative"]["coefficient_si"]),
            density_exponent=float(raw["heating"]["radiative"]["density_exponent"]),
            velocity_exponent=float(raw["heating"]["radiative"]["velocity_exponent"]),
            nose_radius_exponent=float(raw["heating"]["radiative"]["nose_radius_exponent"]),
            source_id=str(raw["heating"]["radiative"]["source_id"]),
        ),
    )

    return Config(
        interface_altitude_m=float(entry["interface_altitude_m"]),
        entry_velocity_m_s=float(entry["velocity_m_s"]),
        entry_fpa_deg=float(entry["fpa_deg"]),
        m_entry_kg=float(entry["m_entry_kg"]),
        cd=_opt_float(vehicle.get("cd")),
        cd_mach_table=cd_mach_table,
        diameter_m=diameter_m,
        reference_area_m2=area,
        nose_radius_m=float(vehicle["nose_radius_m"]),
        handoff_alt_m=float(raw["handoff"]["altitude_m"]),
        handoff_trigger=str(raw["handoff"].get("trigger", "altitude")),
        venus_radius_m=float(planet["venus_radius_m"]),
        venus_gravity_m_s2=float(planet["gravity_m_s2"]),
        gas_gamma=float(gas["gamma"]),
        gas_constant_j_kg_k=float(gas["specific_gas_constant_j_kg_k"]),
        atmosphere=atmosphere,
        heating=heating,
        payload_thermal=PayloadThermalConfig(
            enabled=bool(payload_raw.get("enabled", True)),
            payload_mass_kg=float(payload_raw.get("payload_mass_kg", 14.0)),
            payload_cp_j_kg_k=float(payload_raw.get("payload_cp_j_kg_k", 800.0)),
            initial_temp_k=float(payload_raw.get("initial_temp_k", 273.15)),
            limit_temp_k=float(payload_raw.get("limit_temp_k", 333.15)),
            ua_eff_w_k=float(payload_raw.get("ua_eff_w_k", 0.0)),
            residual_radiation_w=float(payload_raw.get("residual_radiation_w", 0.0)),
        ),
        events=EventConfig(
            drogue_deploy_alt_m=_opt_float(raw["events"].get("drogue_deploy_alt_m")),
            main_deploy_alt_m=_opt_float(raw["events"].get("main_deploy_alt_m")),
        ),
        deploy=DeployConfig(
            trigger_mode=str(deploy_raw.get("trigger_mode", "mach_q")),
            trigger_mach=float(deploy_raw.get("trigger_mach", 2.0)),
            mach_ceiling=float(deploy_raw.get("mach_ceiling", 2.2)),
            q_floor_pa=float(deploy_raw.get("q_floor_pa", 200.0)),
            q_cap_pa=float(deploy_raw.get("q_cap_pa", 3000.0)),
            min_altitude_m=float(deploy_raw.get("min_altitude_m", 67000.0)),
            disreef_mach_max=float(deploy_raw.get("disreef_mach_max", 0.8)),
        ),
        dispersions_n_runs=int(dispersions["n_runs"]),
        dispersions_seed=int(dispersions["seed"]),
        dispersion_mass_fraction=float(dispersions["mass_fraction"]),
        dispersion_fpa_deg=float(dispersions["fpa_deg"]),
        dispersion_fpa_is_3sigma=bool(dispersions.get("fpa_dispersion_is_3sigma", False)),
        numerics=NumericsConfig(
            time_step_s=float(numerics["time_step_s"]),
            max_time_s=float(numerics["max_time_s"]),
            max_altitude_m=float(numerics["max_altitude_m"]),
            include_gravity=bool(numerics["include_gravity"]),
            trajectory_model=str(numerics.get("trajectory_model", "constant_fpa")),
        ),
        validation=ValidationConfig(
            peak_g_cap_g=float(validation["peak_g_cap_g"]),
            peak_heat_cap_w_m2=float(validation["peak_heat_cap_w_m2"]),
            allen_eggers_tolerance_fraction=float(validation["allen_eggers_tolerance_fraction"]),
        ),
        persistence=PersistenceConfig(
            workbook_path=str(persistence["workbook_path"]),
            latex_macros_path=str(persistence["latex_macros_path"]),
            require_existing_workbook=bool(persistence["require_existing_workbook"]),
        ),
        architecture=dict(raw.get("architecture", {})),
        source_ids=dict(raw.get("source_ids", {})),
        raw=dict(raw),
    )


def _validate(cfg: Config, base_dir: Path) -> None:
    _require_positive(
        {
            "entry_velocity_m_s": cfg.entry_velocity_m_s,
            "m_entry_kg": cfg.m_entry_kg,
            "reference_area_m2": cfg.reference_area_m2,
            "nose_radius_m": cfg.nose_radius_m,
            "venus_radius_m": cfg.venus_radius_m,
            "gas_gamma": cfg.gas_gamma,
            "gas_constant_j_kg_k": cfg.gas_constant_j_kg_k,
            "rho_ref_kg_m3": cfg.atmosphere.rho_ref_kg_m3,
            "scale_height_m": cfg.atmosphere.scale_height_m,
            "time_step_s": cfg.numerics.time_step_s,
            "max_time_s": cfg.numerics.max_time_s,
        }
    )
    _require(cfg.interface_altitude_m > cfg.handoff_alt_m, "interface_altitude_m must be above handoff altitude_m")
    _require(cfg.handoff_trigger in {"altitude", "deploy_event"}, "handoff.trigger must be 'altitude' or 'deploy_event'")
    _require(
        cfg.numerics.trajectory_model in {"constant_fpa", "spherical_ballistic"},
        "numerics.trajectory_model must be 'constant_fpa' or 'spherical_ballistic'",
    )
    _require(cfg.deploy.trigger_mode in {"mach_q", "q_window"}, "deploy.trigger_mode must be 'mach_q' or 'q_window'")
    _require(cfg.cd is not None or bool(cfg.cd_mach_table), "vehicle.cd or vehicle.cd_mach_table is required")
    _require(cfg.cd is None or cfg.cd > 0, "vehicle.cd must be positive")
    _require(
        all(m >= 0 and cd > 0 for m, cd in cfg.cd_mach_table),
        "vehicle.cd_mach_table rows must be [mach >= 0, cd > 0]",
    )

    deploy = cfg.deploy
    _require(
        deploy.trigger_mach > 0
        and deploy.mach_ceiling > 0
        and deploy.q_floor_pa >= 0
        and deploy.q_cap_pa > 0
        and deploy.min_altitude_m > 0
        and deploy.disreef_mach_max > 0,
        "deploy trigger_mach, mach_ceiling, q bounds, min_altitude_m, and disreef_mach_max must be positive",
    )
    _require(deploy.trigger_mach <= deploy.mach_ceiling, "deploy.trigger_mach must be <= deploy.mach_ceiling")
    _require(deploy.q_floor_pa < deploy.q_cap_pa, "deploy.q_floor_pa must be less than deploy.q_cap_pa")

    atm = cfg.atmosphere
    _require(atm.model in {"exponential", "profile"}, "atmosphere.model must be 'exponential' or 'profile'")
    _require(
        atm.model != "profile" or atm.density_kind == "profile_file",
        "atmosphere.model='profile' requires density.kind='profile_file'",
    )
    if atm.density_kind == "table":
        _require(
            len(atm.density_altitude_m) == len(atm.density_kg_m3) and len(atm.density_altitude_m) >= 2,
            "density table requires matching altitude_m and density_kg_m3 arrays",
        )
        _require(all(v > 0 for v in atm.density_kg_m3), "density_kg_m3 values must be positive")
    elif atm.density_kind == "profile_file":
        _require(atm.profile_path is not None, "profile_file density requires profile_path")
        _require(atm.profile_format is not None, "profile_file density requires profile_format")
        _require(atm.profile_selector is not None, "profile_file density requires profile_selector")
        if not Path(atm.profile_path).exists():
            raise FileNotFoundError(f"atmosphere profile_path does not exist: {atm.profile_path}")
    else:
        _require(atm.density_kind == "exponential", "atmosphere.density.kind must be exponential, table, or profile_file")
    if atm.vcd_envelope_path is not None and not Path(atm.vcd_envelope_path).exists():
        raise FileNotFoundError(f"VCD envelope path does not exist: {atm.vcd_envelope_path}")

    temp = atm.temperature
    if temp.kind == "table":
        _require(
            len(temp.altitude_m) == len(temp.temperature_k) and len(temp.altitude_m) >= 2,
            "temperature table requires matching altitude_m and temperature_k arrays",
        )
        _require(all(v > 0 for v in temp.temperature_k), "temperature_k values must be positive")
    elif temp.kind == "constant":
        _require(
            temp.constant_temperature_k is not None and temp.constant_temperature_k > 0,
            "constant temperature requires constant_temperature_k > 0",
        )
    elif temp.kind == "fallback_speed_of_sound":
        _require(
            temp.fallback_speed_of_sound_m_s is not None and temp.fallback_speed_of_sound_m_s > 0,
            "fallback speed of sound requires fallback_speed_of_sound_m_s > 0",
        )
    else:
        raise ValueError(f"Unsupported temperature.kind {temp.kind!r}")

    if cfg.payload_thermal.enabled:
        payload = cfg.payload_thermal
        _require_positive(
            {
                "payload_thermal.payload_mass_kg": payload.payload_mass_kg,
                "payload_thermal.payload_cp_j_kg_k": payload.payload_cp_j_kg_k,
                "payload_thermal.initial_temp_k": payload.initial_temp_k,
                "payload_thermal.limit_temp_k": payload.limit_temp_k,
            }
        )
        _require(payload.limit_temp_k > payload.initial_temp_k, "payload_thermal.limit_temp_k must exceed initial_temp_k")
        _require(
            payload.ua_eff_w_k >= 0 and payload.residual_radiation_w >= 0,
            "payload_thermal.ua_eff_w_k and residual_radiation_w must be non-negative",
        )

    mode = atm.dispersion.mode
    if mode == "venus_gram_hdf5":
        path = atm.dispersion.venus_gram_hdf5_path
        _require(path is not None, "venus_gram_hdf5 mode requires venus_gram_hdf5_path")
        if not Path(path).exists():
            raise FileNotFoundError(f"Venus-GRAM HDF5 path does not exist: {path}")
    else:
        _require(mode == "synthetic_gram_fixture", "atmosphere.dispersion.mode must be venus_gram_hdf5 or synthetic_gram_fixture")

    if cfg.persistence.require_existing_workbook:
        workbook = Path(cfg.persistence.workbook_path)
        if not workbook.is_absolute():
            workbook = base_dir / workbook
        if not workbook.exists():
            raise FileNotFoundError(f"Workbook path does not exist: {workbook}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_positive(values: dict[str, float]) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value!r}")


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _opt_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _opt_path(value: Any, base_dir: Path) -> str | None:
    if value is None:
        return None
    candidate = Path(str(value))
    return str(candidate if candidate.is_absolute() else base_dir / candidate)
