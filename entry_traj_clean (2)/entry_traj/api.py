"""High-level run orchestration: nominal + Monte Carlo + validation + TPS
basis + post-deploy descent, with all file outputs written to one directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config
from .integrator import integrate
from .models import Config, Ensemble, Report, Summary, Trajectory, to_plain
from .monte_carlo import monte_carlo, summarize
from .outputs import (
    corridor_rows,
    ensemble_metric_rows,
    event_time_rows,
    mission_phase_markdown,
    mission_phase_rows,
    trajectory_rows,
    write_csv,
    write_json,
    write_latex_macros,
    write_workbook,
)
from .post_deploy import simulate_post_deploy
from .tps import tps_sizing_basis
from .validation import report_text, validate


def run(
    config_path: str | Path,
    out_dir: str | Path = "results",
    n_runs: int | None = None,
    seed: int | None = None,
    plots: bool = True,
) -> dict[str, Any]:
    """Full pipeline for one config. Returns a dict of in-memory results."""
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    nominal = integrate(cfg)
    report = validate(cfg, nominal)
    ensemble = monte_carlo(
        cfg,
        n=n_runs if n_runs is not None else cfg.dispersions_n_runs,
        seed=seed if seed is not None else cfg.dispersions_seed,
    )
    summary = summarize(ensemble)
    tps_basis = tps_sizing_basis(cfg, nominal, summary)
    post_deploy = simulate_post_deploy(cfg, nominal)

    _write_outputs(cfg, out, nominal, report, ensemble, summary, tps_basis, post_deploy)
    if plots:
        _write_plots(out, nominal, summary)

    return {
        "config": cfg,
        "nominal": nominal,
        "report": report,
        "ensemble": ensemble,
        "summary": summary,
        "tps_basis": tps_basis,
        "post_deploy": post_deploy,
        "out_dir": out,
    }


def _write_outputs(
    cfg: Config,
    out: Path,
    nominal: Trajectory,
    report: Report,
    ensemble: Ensemble,
    summary: Summary,
    tps_basis: dict[str, Any],
    post_deploy,
) -> None:
    write_json(out / "summary.json", to_plain(summary))
    write_json(out / "validation_report.json", to_plain(report))
    (out / "validation_report.txt").write_text(report_text(report), encoding="utf-8")
    write_csv(out / "nominal_trajectory.csv", *trajectory_rows(nominal))
    write_csv(out / "ensemble_metrics.csv", *ensemble_metric_rows(ensemble))
    write_csv(out / "event_times.csv", *event_time_rows(summary))
    write_csv(out / "handoff_corridor.csv", *corridor_rows(summary))
    write_csv(out / "mission_phase_table.csv", *mission_phase_rows(nominal))
    (out / "mission_phase_table.md").write_text(mission_phase_markdown(nominal), encoding="utf-8")
    write_json(out / "tps_sizing_basis.json", tps_basis)

    if post_deploy is not None:
        write_json(out / "post_deploy_descent.json", {
            **post_deploy.to_plain(),
        })
        write_csv(
            out / "post_deploy_descent.csv",
            ["time_s", "altitude_m", "velocity_m_s", "mass_kg", "cds_m2", "buoyancy_n", "stage"],
            [
                list(row)
                for row in zip(
                    post_deploy.time_s,
                    post_deploy.altitude_m,
                    post_deploy.velocity_m_s,
                    post_deploy.mass_kg,
                    post_deploy.cds_m2,
                    post_deploy.buoyancy_n,
                    post_deploy.stage,
                )
            ],
        )

    workbook_path = out / Path(cfg.persistence.workbook_path).name
    if write_workbook(workbook_path, cfg, summary, report) is None:
        (out / "workbook_skipped.txt").write_text(
            "openpyxl is not installed; workbook output skipped. pip install openpyxl\n",
            encoding="utf-8",
        )
    write_latex_macros(out / Path(cfg.persistence.latex_macros_path).name, cfg, summary)


def _write_plots(out: Path, nominal: Trajectory, summary: Summary) -> None:
    from . import plots

    plots.trajectory_figure(nominal, out / "nominal_trajectory.png")
    plots.timeline_figure(nominal, out / "mission_timeline.png")
    plots.corridor_figure(summary, out / "handoff_corridor.png")
    plots.tornado_figure(summary, out / "sobol_tornado.png", metric="peak_heat")
