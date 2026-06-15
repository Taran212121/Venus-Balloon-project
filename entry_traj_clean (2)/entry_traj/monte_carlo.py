"""Dispersed ensemble: Latin-hypercube sampling, percentiles, first-order
Sobol attribution (binned estimator), and the altitude-resolved hand-off
corridor.

Inputs dispersed per run: entry mass (uniform +/- fraction), EFPA (uniform
bound or 3-sigma normal), and the atmosphere sample from ``GramSource``.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

from .atmosphere import GramSource
from .integrator import RunSample, integrate
from .models import Config, Ensemble, Summary

SENSITIVITY_INPUTS = ("m_entry", "FPA", "rho_atm/GRAM")
OUTPUT_METRICS = {
    "peak_g": ("peak_g", "g"),
    "peak_q_pa": ("peak_q", "Pa"),
    "peak_heat_w_m2": ("peak_heat", "W/m^2"),
    "heat_load_j_m2": ("heat_load", "J/m^2"),
    "payload_absorbed_energy_j": ("payload_energy_absorbed", "J"),
    "mach_handoff": ("mach_handoff", ""),
}


def monte_carlo(cfg: Config, n: int = 1000, seed: int = 1001) -> Ensemble:
    gram_source = GramSource.from_config(cfg)
    trajectories = []
    samples = []
    for idx, unit in enumerate(_unit_samples(n, 3, seed)):
        mass_factor = 1.0 + cfg.dispersion_mass_fraction * (2.0 * unit[0] - 1.0)
        if cfg.dispersion_fpa_is_3sigma:
            u = min(1.0 - 1e-12, max(1e-12, unit[1]))
            fpa_delta_deg = (cfg.dispersion_fpa_deg / 3.0) * NormalDist().inv_cdf(u)
        else:
            fpa_delta_deg = cfg.dispersion_fpa_deg * (2.0 * unit[1] - 1.0)
        run = RunSample(
            m_entry_kg=cfg.m_entry_kg * mass_factor,
            fpa_deg=cfg.entry_fpa_deg + fpa_delta_deg,
            atmosphere=gram_source.sample_from_unit(unit[2]),
            label=f"mc_{idx:04d}",
        )
        trajectories.append(integrate(cfg, run))
        samples.append(run.as_mapping())
    return Ensemble(
        config=cfg,
        n_requested=n,
        seed=seed,
        trajectories=tuple(trajectories),
        samples=tuple(samples),
    )


def summarize(ens: Ensemble) -> Summary:
    metric_values = _metric_values(ens)
    metric_intervals = {}
    for metric, values in metric_values.items():
        label, units = OUTPUT_METRICS[metric]
        metric_intervals[label] = {
            "p2_5": percentile(values, 2.5),
            "p50": percentile(values, 50.0),
            "p97_5": percentile(values, 97.5),
            "units": units,
            "source_id": ens.config.source_ids.get(label, "FA1-D41-entry-traj"),
        }

    sample_columns = _sample_columns(ens)
    sobol = {
        OUTPUT_METRICS[metric][0]: _first_order_indices(sample_columns, values)
        for metric, values in metric_values.items()
    }
    sum_s1 = {metric: sum(values.values()) for metric, values in sobol.items()}
    return Summary(
        n_requested=ens.n_requested,
        n_success=ens.n_success,
        n_failed=ens.n_failed,
        metric_intervals=metric_intervals,
        sobol_s1=sobol,
        sobol_sum_s1=sum_s1,
        sobol_interaction_or_residual_share={k: 1.0 - v for k, v in sum_s1.items()},
        dominant_driver={
            metric: max(values.items(), key=lambda item: item[1])[0] if values else "none"
            for metric, values in sobol.items()
        },
        handoff_corridor=_handoff_corridor(ens),
        event_time_arrays_s=_event_time_arrays(ens),
        source_ids=ens.config.source_ids,
        notes=(
            "Intervals are 2.5/50/97.5 percentiles of the output population, not confidence intervals on the mean.",
            "rho_atm uncertainty is represented by the GRAM-derived atmosphere sample only.",
            "FA1 handoff is configured as either fixed altitude or a deploy event; q_window mode stops at the first deploy dynamic-pressure window above the altitude floor, while mach_q mode retains the legacy Mach/q gate.",
            "Payload absorbed energy is a lumped pre-deploy thermal gate; active TMS is assumed available only after deploy.",
        ),
    )


def sobol_indices(ens: Ensemble, output_metric: str = "peak_heat_w_m2") -> dict[str, float]:
    metric_values = _metric_values(ens)
    if output_metric not in metric_values:
        raise ValueError(
            f"Unknown output_metric {output_metric!r}; valid metrics: {', '.join(sorted(metric_values))}"
        )
    return _first_order_indices(_sample_columns(ens), metric_values[output_metric])


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low, high = int(math.floor(rank)), int(math.ceil(rank))
    if low == high:
        return ordered[low]
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


def _unit_samples(n: int, dim: int, seed: int) -> list[list[float]]:
    """Latin-hypercube unit samples via chaospy when available, stdlib otherwise."""
    try:
        import chaospy as cp

        distribution = cp.J(*(cp.Uniform(0.0, 1.0) for _ in range(dim)))
        raw = distribution.sample(n, rule="latin_hypercube", seed=seed)
        return [[float(raw[col][row]) for col in range(dim)] for row in range(n)]
    except Exception:
        rng = random.Random(seed)
        columns: list[list[float]] = []
        for _ in range(dim):
            values = [(idx + rng.random()) / n for idx in range(n)]
            rng.shuffle(values)
            columns.append(values)
        return [[columns[col][row] for col in range(dim)] for row in range(n)]


def _metric_values(ens: Ensemble) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {key: [] for key in OUTPUT_METRICS}
    for run in ens.trajectories:
        if not run.success:
            continue
        values["peak_g"].append(max(run.deceleration_g))
        values["peak_q_pa"].append(max(run.dynamic_pressure_pa))
        values["peak_heat_w_m2"].append(max(run.heat_flux_total_w_m2))
        values["heat_load_j_m2"].append(run.heat_load_total_j_m2[-1])
        values["payload_absorbed_energy_j"].append(run.payload_absorbed_energy_j[-1])
        values["mach_handoff"].append(run.mach[-1])
    return values


def _sample_columns(ens: Ensemble) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {name: [] for name in SENSITIVITY_INPUTS}
    for run in ens.trajectories:
        if not run.success:
            continue
        columns["m_entry"].append(float(run.input_sample["m_entry_kg"]))
        columns["FPA"].append(float(run.input_sample["fpa_deg"]))
        columns["rho_atm/GRAM"].append(float(run.input_sample["rho_sigma"]))
    return columns


def _first_order_indices(
    sample_columns: dict[str, list[float]], y_values: list[float]
) -> dict[str, float]:
    """Binned conditional-variance estimator of first-order Sobol indices."""
    if len(y_values) < 8:
        return {name: 0.0 for name in SENSITIVITY_INPUTS}
    mean_y = sum(y_values) / len(y_values)
    variance_y = sum((v - mean_y) ** 2 for v in y_values) / len(y_values)
    if variance_y <= 0.0:
        return {name: 0.0 for name in SENSITIVITY_INPUTS}

    bin_count = max(4, min(12, int(math.sqrt(len(y_values)))))
    chunk_size = max(1, len(y_values) // bin_count)
    result = {}
    for name in SENSITIVITY_INPUTS:
        pairs = sorted(zip(sample_columns[name], y_values), key=lambda item: item[0])
        conditional_var = 0.0
        for start in range(0, len(pairs), chunk_size):
            chunk_y = [y for _, y in pairs[start : start + chunk_size]]
            if chunk_y:
                chunk_mean = sum(chunk_y) / len(chunk_y)
                conditional_var += (len(chunk_y) / len(pairs)) * (chunk_mean - mean_y) ** 2
        result[name] = conditional_var / variance_y
    return result


def _handoff_corridor(ens: Ensemble) -> dict[str, object]:
    """2.5/50/97.5 percentile bands vs altitude across all successful runs."""
    runs = [run for run in ens.trajectories if run.success]
    if not runs:
        return {"altitude_m": []}
    cfg = ens.config
    altitudes = [
        cfg.handoff_alt_m + idx * (cfg.interface_altitude_m - cfg.handoff_alt_m) / 100.0
        for idx in range(101)
    ]
    corridor: dict[str, object] = {"altitude_m": tuple(altitudes)}
    for attr in ("mach", "dynamic_pressure_pa", "heat_flux_total_w_m2", "deceleration_g", "velocity_m_s"):
        profiles = [
            (tuple(reversed(run.altitude_m)), tuple(reversed(getattr(run, attr))))
            for run in runs
        ]
        bands: dict[str, list[float]] = {"p2_5": [], "p50": [], "p97_5": []}
        for altitude_m in altitudes:
            values = [_interp_ascending(alts, vals, altitude_m) for alts, vals in profiles]
            bands["p2_5"].append(percentile(values, 2.5))
            bands["p50"].append(percentile(values, 50.0))
            bands["p97_5"].append(percentile(values, 97.5))
        corridor[attr] = {key: tuple(vals) for key, vals in bands.items()}
    return corridor


def _interp_ascending(xs: tuple[float, ...], ys: tuple[float, ...], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] < x:
            lo = mid
        else:
            hi = mid
    if xs[hi] == xs[lo]:
        return ys[lo]
    fraction = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + fraction * (ys[hi] - ys[lo])


def _event_time_arrays(ens: Ensemble) -> dict[str, tuple[float | None, ...]]:
    names = ("T0_interface", "peak_g", "peak_q", "peak_heat", "drogue_deploy", "main_deploy", "handoff")
    return {
        name: tuple(run.event_times_s.get(name) for run in ens.trajectories) for name in names
    }
