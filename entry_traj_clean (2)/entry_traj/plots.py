"""Matplotlib figures with one shared, restrained style.

matplotlib is imported inside functions so the simulation core stays
importable without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import Summary, Trajectory

INK = "#20242a"
MUTED = "#6b7280"
ACCENT = "#b3552d"
ACCENT_2 = "#225b63"
HAIRLINE = "#d9d6cd"
PAPER = "#fdfdfb"

_EVENT_LABELS = {
    "peak_g": "peak g",
    "peak_q": "peak q",
    "peak_heat": "peak heat",
    "drogue_deploy": "deploy",
    "handoff": "hand-off",
}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "text.color": INK,
            "axes.edgecolor": HAIRLINE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": INK,
            "grid.alpha": 0.12,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )
    return plt


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def trajectory_figure(trajectory: Trajectory, path: str | Path, title: str = "Nominal entry trajectory") -> Path:
    plt = _style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharex=True)
    t = trajectory.time_s

    panels = (
        (axes[0][0], [v / 1000.0 for v in trajectory.altitude_m], "altitude [km]", ACCENT_2),
        (axes[0][1], [v / 1000.0 for v in trajectory.velocity_m_s], "velocity [km/s]", ACCENT_2),
        (axes[1][0], trajectory.deceleration_g, "deceleration [g]", ACCENT),
        (axes[1][1], [v / 1e6 for v in trajectory.heat_flux_total_w_m2], "heat flux [MW/m$^2$]", ACCENT),
    )
    for ax, values, label, color in panels:
        ax.plot(t, values, color=color)
        ax.set_ylabel(label)
    for ax in axes[1]:
        ax.set_xlabel("time from interface [s]")

    for key, label in _EVENT_LABELS.items():
        event_t = trajectory.event_times_s.get(key)
        if event_t is None:
            continue
        for row in axes:
            for ax in row:
                ax.axvline(event_t, color=MUTED, linewidth=0.7, alpha=0.45)
        axes[0][0].annotate(
            label,
            xy=(event_t, axes[0][0].get_ylim()[1]),
            xytext=(2, -2),
            textcoords="offset points",
            fontsize=7.5,
            color=MUTED,
            rotation=90,
            va="top",
        )
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold")
    return _save(fig, path)


def corridor_figure(summary: Summary, path: str | Path) -> Path:
    plt = _style()
    corridor = summary.handoff_corridor
    altitudes = [v / 1000.0 for v in corridor.get("altitude_m", ())]
    if not altitudes:
        raise ValueError("Summary has no handoff corridor data")

    metrics = (
        ("velocity_m_s", "velocity [km/s]", 1e-3),
        ("mach", "Mach", 1.0),
        ("dynamic_pressure_pa", "dynamic pressure [kPa]", 1e-3),
        ("heat_flux_total_w_m2", "heat flux [MW/m$^2$]", 1e-6),
    )
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.6), sharey=True)
    for ax, (metric, label, scale) in zip(axes, metrics):
        bands = corridor[metric]
        lo = [v * scale for v in bands["p2_5"]]
        mid = [v * scale for v in bands["p50"]]
        hi = [v * scale for v in bands["p97_5"]]
        ax.fill_betweenx(altitudes, lo, hi, color=ACCENT_2, alpha=0.16, linewidth=0)
        ax.plot(mid, altitudes, color=ACCENT_2)
        ax.set_xlabel(label)
    axes[0].set_ylabel("altitude [km]")
    fig.suptitle(
        f"Dispersed corridor, {summary.n_success}/{summary.n_requested} runs (2.5/50/97.5%)",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    return _save(fig, path)


def tornado_figure(summary: Summary, path: str | Path, metric: str = "peak_heat") -> Path:
    plt = _style()
    values = summary.sobol_s1.get(metric, {})
    if not values:
        raise ValueError(f"No Sobol indices for metric {metric!r}")
    ordered = sorted(values.items(), key=lambda item: item[1])
    labels = [name for name, _ in ordered]
    s1 = [value for _, value in ordered]

    fig, ax = plt.subplots(figsize=(6.4, 0.7 + 0.55 * len(labels)))
    ax.barh(labels, s1, color=ACCENT, height=0.55)
    residual = summary.sobol_interaction_or_residual_share.get(metric, 0.0)
    ax.set_xlabel("first-order Sobol index $S_1$")
    ax.set_xlim(0, 1)
    ax.set_title(f"Drivers of {metric} (interaction/residual share {residual:.2f})", loc="left", fontweight="bold")
    for idx, value in enumerate(s1):
        ax.annotate(f"{value:.2f}", xy=(value, idx), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED)
    return _save(fig, path)


def timeline_figure(trajectory: Trajectory, path: str | Path) -> Path:
    """Altitude vs time with labelled mission events -- the compact timeline."""
    plt = _style()
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.plot(trajectory.time_s, [v / 1000.0 for v in trajectory.altitude_m], color=ACCENT_2)
    ax.set_xlabel("time from interface [s]")
    ax.set_ylabel("altitude [km]")

    labels = {
        "T0_interface": "T0 interface",
        "peak_g": "peak g",
        "peak_q": "peak q",
        "peak_heat": "peak heat",
        "drogue_deploy": "deploy",
        "main_deploy": "main deploy",
        "handoff": "hand-off",
    }
    times = trajectory.time_s
    for key, label in labels.items():
        event_t = trajectory.event_times_s.get(key)
        if event_t is None:
            continue
        idx = min(range(len(times)), key=lambda i: abs(times[i] - event_t))
        alt_km = trajectory.altitude_m[idx] / 1000.0
        ax.plot([event_t], [alt_km], marker="o", markersize=4, color=ACCENT)
        ax.annotate(
            f"{label}\n{event_t:.0f} s / {alt_km:.0f} km",
            xy=(event_t, alt_km),
            xytext=(6, 10),
            textcoords="offset points",
            fontsize=7.5,
            color=INK,
        )
    ax.set_title("Entry timeline", loc="left", fontweight="bold")
    return _save(fig, path)


def fpa_sweep_figure(sweep_result: dict[str, Any], path: str | Path) -> Path:
    plt = _style()
    rows = sweep_result["rows"]
    fpa = [row["fpa_deg"] for row in rows]
    feasible = [row["feasible"] for row in rows]
    peak_g = [
        row["worst_case"]["peak_g"] if row["worst_case"]["peak_g"] is not None else float("nan")
        for row in rows
    ]
    recommended = sweep_result.get("recommended_fpa_deg")
    has_mass = sweep_result.get("mass_model", {}).get("enabled", False) and all(
        "mass" in row for row in rows
    )

    def _mark_recommended(ax) -> None:
        if recommended is None:
            return
        ax.axvline(recommended, color=ACCENT_2, linewidth=0.9, alpha=0.6)
        ax.annotate(
            f"recommended {recommended:g}\u00b0",
            xy=(recommended, ax.get_ylim()[1]),
            xytext=(4, -4),
            textcoords="offset points",
            fontsize=8,
            color=ACCENT_2,
            va="top",
        )

    n_panels = 2 if has_mass else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7.6 * n_panels, 4.0))
    axes = [axes] if n_panels == 1 else list(axes)

    ax = axes[0]
    ax.plot(fpa, peak_g, color=MUTED, linewidth=1.0, zorder=1)
    for x, y, ok in zip(fpa, peak_g, feasible):
        ax.plot([x], [y], marker="o", markersize=5, color=ACCENT_2 if ok else ACCENT, zorder=2)
    _mark_recommended(ax)
    ax.set_xlabel("entry flight-path angle [deg]")
    ax.set_ylabel("worst-tail peak deceleration [g]")
    ax.set_title("EFPA feasibility sweep (teal feasible, sienna infeasible)", loc="left", fontweight="bold")

    if has_mass:
        def _col(name: str) -> list[float]:
            return [
                row["mass"][name] if row["mass"][name] is not None else float("nan")
                for row in rows
            ]

        ax2 = axes[1]
        ax2.plot(fpa, _col("tps_mass_kg"), color=ACCENT, linewidth=1.2, label="TPS (heat load)")
        ax2.plot(fpa, _col("structure_mass_kg"), color=MUTED, linewidth=1.2, label="structure (peak g)")
        ax2.plot(fpa, _col("entry_system_mass_kg"), color=INK, linewidth=1.6, label="total entry system")
        for x, y, ok in zip(fpa, _col("entry_system_mass_kg"), feasible):
            ax2.plot([x], [y], marker="o", markersize=5, color=ACCENT_2 if ok else ACCENT, zorder=3)
        _mark_recommended(ax2)
        ax2.set_xlabel("entry flight-path angle [deg]")
        ax2.set_ylabel("entry-system mass [kg]")
        ax2.set_title("Mass trade: TPS vs structure", loc="left", fontweight="bold")
        ax2.legend(frameon=False, fontsize=8, loc="best")

    return _save(fig, path)


def comparison_bars(labels: list[str], series: dict[str, list[float]], title: str, path: str | Path) -> Path:
    plt = _style()
    fig, axes = plt.subplots(1, len(series), figsize=(4.4 * len(series), 3.4))
    if len(series) == 1:
        axes = [axes]
    colors = (ACCENT_2, ACCENT, MUTED)
    for ax, (name, values), color in zip(axes, series.items(), colors):
        ax.barh(labels, values, color=color, height=0.55)
        ax.set_xlabel(name)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold")
    return _save(fig, path)


def profile_ensemble_figure(result: dict[str, Any], path: str | Path) -> Path:
    plt = _style()
    rows = [row for row in result["rows"] if row.get("success")]
    if not rows:
        raise ValueError("No successful profile-ensemble rows to plot")
    rows = sorted(rows, key=lambda row: row["peak_g"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, max(4.2, 0.16 * len(rows))))
    left.barh([row["selector"] for row in rows], [row["peak_g"] for row in rows], color=ACCENT_2, height=0.6)
    left.set_xlabel("peak deceleration [g]")
    left.tick_params(axis="y", labelsize=6)
    left.axvline(32.0, color=ACCENT, linewidth=0.9)
    left.annotate("DAVINCI ref 32 g", xy=(32.0, len(rows) - 1), fontsize=7.5, color=ACCENT, rotation=90, va="top", xytext=(-10, 0), textcoords="offset points")

    right.scatter(
        [row["local_scale_height_at_peak_g_m"] / 1000.0 for row in rows],
        [row["peak_g"] for row in rows],
        s=16,
        color=ACCENT,
        alpha=0.8,
    )
    right.set_xlabel("local scale height at peak g [km]")
    right.set_ylabel("peak deceleration [g]")
    fig.suptitle(
        f"VIRA profile ensemble, {result['n_success']}/{result['n_selectors']} selectors",
        x=0.02,
        ha="left",
        fontweight="bold",
    )
    return _save(fig, path)
