"""Command-line interface.

Commands (unchanged from v0.1):
  run             full pipeline for one config
  batch           Pioneer Venus anchoring batch over a config directory
  sweep-fpa       EFPA feasibility sweep
  sweep-profiles  VIRA selector profile ensemble
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="entry_traj", description="Phase-0 Venus entry trajectory tool")
    parser.add_argument("--version", action="version", version=f"entry_traj {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run nominal + Monte Carlo pipeline for one config")
    run.add_argument("config", help="Path to YAML config")
    run.add_argument("--out", default="results", help="Output directory (default: results)")
    run.add_argument("--n-runs", type=int, default=None, help="Override dispersions.n_runs")
    run.add_argument("--seed", type=int, default=None, help="Override dispersions.seed")
    run.add_argument("--no-plots", action="store_true", help="Skip PNG figure generation")

    batch = sub.add_parser("batch", help="Run every config in a directory (Pioneer Venus anchoring)")
    batch.add_argument("config_dir", help="Directory containing *.yaml configs")
    batch.add_argument("--out", default="results/pioneer_venus", help="Output directory")

    sweep = sub.add_parser("sweep-fpa", help="EFPA feasibility sweep with 3-sigma dispersion tails")
    sweep.add_argument("config", help="Path to YAML config")
    sweep.add_argument("--out", default="results", help="Output directory")
    sweep.add_argument("--fpa-min", type=float, required=True, help="Shallowest EFPA [deg], e.g. -7.0")
    sweep.add_argument("--fpa-max", type=float, required=True, help="Steepest EFPA [deg], e.g. -11.0")
    sweep.add_argument("--fpa-step", type=float, default=0.25, help="Step magnitude [deg] (default 0.25)")
    sweep.add_argument("--design-heat-load-cap", type=float, default=None, help="Optional design heat-load cap [J/m^2]")

    profiles = sub.add_parser("sweep-profiles", help="Rerun nominal across the VIRA selector grid")
    profiles.add_argument("config", help="Path to YAML config (profile atmosphere)")
    profiles.add_argument("--out", default="results", help="Output directory")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "batch":
        return _cmd_batch(args)
    if args.command == "sweep-fpa":
        return _cmd_sweep_fpa(args)
    if args.command == "sweep-profiles":
        return _cmd_sweep_profiles(args)
    return 2


def _cmd_run(args) -> int:
    from .api import run

    result = run(
        args.config,
        out_dir=args.out,
        n_runs=args.n_runs,
        seed=args.seed,
        plots=not args.no_plots,
    )
    summary = result["summary"]
    report = result["report"]
    print(f"entry_traj run complete -> {result['out_dir']}")
    print(f"  runs: {summary.n_success}/{summary.n_requested} successful")
    for metric, stats in summary.metric_intervals.items():
        units = f" {stats['units']}" if stats["units"] else ""
        print(f"  {metric}: p50 {stats['p50']:.4g}{units}  (95% {stats['p2_5']:.4g} .. {stats['p97_5']:.4g})")
    print(f"  validation: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


def _cmd_batch(args) -> int:
    from .batch import run_batch

    summary = run_batch(args.config_dir, args.out)
    print(f"batch complete -> {args.out}")
    for row in summary["rows"]:
        peak_g = row.get("peak_g")
        peak_g_text = f"peak_g {peak_g:.1f}" if isinstance(peak_g, float) else (row.get("failure_reason") or "")
        print(f"  {row['probe']}: {row['status']} {peak_g_text}")
    return 0 if all(row["status"] == "ok" for row in summary["rows"]) else 1


def _cmd_sweep_fpa(args) -> int:
    from .config import load_config
    from .outputs import write_csv, write_json
    from .plots import fpa_sweep_figure
    from .sweep import sweep_fpa

    cfg = load_config(args.config)
    lo, hi = sorted((args.fpa_min, args.fpa_max))
    step = abs(args.fpa_step) or 0.25
    values = []
    value = lo
    while value <= hi + 1e-9:
        values.append(round(value, 6))
        value += step

    result = sweep_fpa(cfg, values, design_heat_load_cap_j_m2=args.design_heat_load_cap)
    out = Path(args.out)
    write_json(out / "fpa_sweep.json", result)

    mass_enabled = result["mass_model"].get("enabled", False)
    mass_cols = ["tps_mass_kg", "structure_mass_kg", "entry_system_mass_kg"] if mass_enabled else []
    write_csv(
        out / "fpa_sweep.csv",
        ["fpa_deg", "feasible"] + mass_cols + list(result["rows"][0]["constraints"]),
        [
            [row["fpa_deg"], row["feasible"]]
            + [row["mass"][col] for col in mass_cols]
            + list(row["constraints"].values())
            for row in result["rows"]
        ],
    )
    fpa_sweep_figure(result, out / "fpa_sweep.png")
    recommended = result["recommended_fpa_deg"]
    print(f"sweep-fpa complete -> {out} ({result['n_feasible']}/{len(values)} feasible)")
    print(f"  recommended EFPA: {recommended if recommended is not None else 'none feasible'}")
    print(f"  selection rule: {result['selection_rule']}")
    if mass_enabled and result["mass_optimal_fpa_deg"] is not None:
        optimal = next(row for row in result["rows"] if row["fpa_deg"] == result["mass_optimal_fpa_deg"])
        mass = optimal["mass"]
        print(
            "  entry-system mass at optimum: "
            f"{mass['entry_system_mass_kg']:.1f} kg "
            f"(TPS {mass['tps_mass_kg']:.1f} kg + structure {mass['structure_mass_kg']:.1f} kg)"
        )
        if result["shallowest_feasible_fpa_deg"] != result["mass_optimal_fpa_deg"]:
            print(f"  shallowest feasible EFPA: {result['shallowest_feasible_fpa_deg']:g} (max corridor margin)")
    return 0 if recommended is not None else 1


def _cmd_sweep_profiles(args) -> int:
    from .config import load_config
    from .outputs import write_json
    from .plots import profile_ensemble_figure
    from .profile_ensemble import run_profile_ensemble

    cfg = load_config(args.config)
    result = run_profile_ensemble(cfg)
    out = Path(args.out)
    write_json(out / "profile_ensemble.json", result)
    profile_ensemble_figure(result, out / "profile_ensemble.png")
    print(f"sweep-profiles complete -> {out} ({result['n_success']}/{result['n_selectors']} selectors)")
    print(json.dumps(result["metric_ranges"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
