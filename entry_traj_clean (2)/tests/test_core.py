"""Core regression tests: config loading, physics anchoring, deploy trigger,
Monte Carlo reproducibility, and validation checks."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from entry_traj import integrate, load_config, monte_carlo, summarize, validate
from entry_traj.validation import allen_eggers_peak_deceleration_g

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def example_cfg():
    return load_config(CONFIG_DIR / "config.example.yaml")


def _fast_cfg(cfg):
    numerics = dataclasses.replace(cfg.numerics, time_step_s=0.5)
    return dataclasses.replace(cfg, numerics=numerics)


def test_example_config_loads(example_cfg):
    assert example_cfg.entry_fpa_deg == -8.0
    assert example_cfg.m_entry_kg == 667.0
    assert example_cfg.handoff_trigger == "deploy_event"
    assert example_cfg.deploy.trigger_mode == "q_window"
    assert 200.0 < example_cfg.ballistic_coefficient_kg_m2 < 300.0


def test_nominal_trajectory_reaches_handoff(example_cfg):
    run = integrate(_fast_cfg(example_cfg))
    assert run.success, run.failure_reason
    assert run.event_times_s["handoff"] is not None
    # FA1 baseline: deploy in the low-70s km at low supersonic Mach.
    assert 60_000.0 < run.altitude_m[-1] < 80_000.0
    assert run.mach[-1] <= example_cfg.deploy.mach_ceiling
    q_pa = run.dynamic_pressure_pa[-1]
    assert example_cfg.deploy.q_floor_pa <= q_pa <= example_cfg.deploy.q_cap_pa


def test_allen_eggers_agreement_exponential_atmosphere(example_cfg):
    """With an exponential atmosphere + constant FPA, the simulated peak g
    must match Allen-Eggers within the configured tolerance."""
    atm = dataclasses.replace(
        example_cfg.atmosphere,
        model="exponential",
        density_kind="exponential",
        profile_path=None,
        profile_format=None,
        profile_selector=None,
    )
    numerics = dataclasses.replace(
        example_cfg.numerics, trajectory_model="constant_fpa", time_step_s=0.1, include_gravity=False
    )
    cfg = dataclasses.replace(example_cfg, atmosphere=atm, numerics=numerics)
    run = integrate(cfg)
    assert run.success, run.failure_reason
    predicted = allen_eggers_peak_deceleration_g(
        cfg.entry_velocity_m_s, cfg.entry_fpa_deg, cfg.atmosphere.scale_height_m
    )
    simulated = max(run.deceleration_g)
    assert abs(simulated - predicted) / predicted < cfg.validation.allen_eggers_tolerance_fraction


def test_monte_carlo_reproducible(example_cfg):
    cfg = _fast_cfg(example_cfg)
    first = summarize(monte_carlo(cfg, n=24, seed=42))
    second = summarize(monte_carlo(cfg, n=24, seed=42))
    assert first.metric_intervals["peak_g"] == second.metric_intervals["peak_g"]
    assert first.n_success == second.n_success
    assert first.n_success >= 20  # the FA1 box should capture nearly all runs


def test_validation_report(example_cfg):
    cfg = _fast_cfg(example_cfg)
    run = integrate(cfg)
    report = validate(cfg, run)
    names = {check["name"] for check in report.checks}
    assert "allen_eggers_peak_g" in names
    assert "peak_g_cap" in names
    assert "peak_heat_cap" in names
    assert report.passed, [check for check in report.checks if not check["passed"]]


def test_deploy_failure_below_altitude_floor(example_cfg):
    """Raising the altitude floor above the deploy window must fail the run
    with the documented failure reason."""
    deploy = dataclasses.replace(example_cfg.deploy, min_altitude_m=90_000.0)
    cfg = dataclasses.replace(_fast_cfg(example_cfg), deploy=deploy)
    run = integrate(cfg)
    assert not run.success
    assert run.failure_reason == "deploy_q_window_altitude_below_min"


def test_sweep_mass_trade(example_cfg):
    """The EFPA sweep must close the mass trade: structure mass grows toward
    steeper EFPA (higher peak g), TPS mass grows toward shallower EFPA
    (higher heat load), and the recommendation is mass-optimal feasible."""
    from entry_traj.sweep import sweep_fpa

    cfg = _fast_cfg(example_cfg)
    result = sweep_fpa(cfg, [-7.0, -8.0, -9.0, -10.0])
    assert result["mass_model"]["enabled"]

    rows = {row["fpa_deg"]: row for row in result["rows"] if row["feasible"]}
    assert len(rows) >= 2
    ordered = [rows[k] for k in sorted(rows, reverse=True)]  # shallow -> steep

    struct = [row["mass"]["structure_mass_kg"] for row in ordered]
    tps = [row["mass"]["tps_mass_kg"] for row in ordered]
    assert all(b > a for a, b in zip(struct, struct[1:]))  # steeper -> heavier structure
    assert all(b < a for a, b in zip(tps, tps[1:]))  # steeper -> lighter TPS

    optimal = result["mass_optimal_fpa_deg"]
    assert optimal is not None
    assert result["recommended_fpa_deg"] == optimal
    best = min(rows.values(), key=lambda row: row["mass"]["entry_system_mass_kg"])
    assert best["fpa_deg"] == optimal


def test_sweep_mass_model_disabled(example_cfg):
    """With the mass model off, the sweep falls back to shallowest-feasible."""
    from entry_traj.sweep import sweep_fpa

    cfg = _fast_cfg(example_cfg)
    raw = dict(cfg.raw)
    raw["mass_model"] = {"enabled": False}
    cfg = dataclasses.replace(cfg, raw=raw)
    result = sweep_fpa(cfg, [-8.0, -9.0])
    assert result["mass_model"] == {"enabled": False}
    assert result["recommended_fpa_deg"] == result["shallowest_feasible_fpa_deg"]
    assert "mass" not in result["rows"][0]
