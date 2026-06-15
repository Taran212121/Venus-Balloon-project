"""Pioneer Venus batch anchoring: run every config in a directory and emit a
per-probe comparison table for the flight-data anchoring exercise."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .integrator import integrate
from .models import to_plain
from .outputs import trajectory_rows, write_csv, write_json
from .validation import report_text, validate

SUMMARY_FIELDS = (
    "probe",
    "status",
    "failure_reason",
    "entry_fpa_deg",
    "m_entry_kg",
    "peak_g",
    "peak_q_pa",
    "peak_heat_w_m2",
    "heat_load_j_m2",
    "mach_handoff",
    "handoff_altitude_m",
    "validation_passed",
)


def run_batch(config_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    config_dir = Path(config_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    config_paths = sorted(config_dir.glob("*.yaml"))
    if not config_paths:
        raise FileNotFoundError(f"No *.yaml configs found in {config_dir}")

    rows: list[dict[str, Any]] = []
    for config_path in config_paths:
        probe = config_path.stem
        probe_dir = out_root / probe
        probe_dir.mkdir(parents=True, exist_ok=True)
        try:
            cfg = load_config(config_path)
            trajectory = integrate(cfg)
            report = validate(cfg, trajectory)
            write_csv(probe_dir / "trajectory.csv", *trajectory_rows(trajectory))
            write_json(probe_dir / "trajectory_events.json", {
                "success": trajectory.success,
                "failure_reason": trajectory.failure_reason,
                "event_times_s": trajectory.event_times_s,
            })
            write_json(probe_dir / "validation_report.json", to_plain(report))
            (probe_dir / "validation_report.txt").write_text(report_text(report), encoding="utf-8")
            row: dict[str, Any] = {
                "probe": probe,
                "status": "ok" if trajectory.success else "failed",
                "failure_reason": trajectory.failure_reason,
                "entry_fpa_deg": cfg.entry_fpa_deg,
                "m_entry_kg": cfg.m_entry_kg,
                "validation_passed": report.passed,
            }
            if trajectory.success:
                row.update(
                    peak_g=max(trajectory.deceleration_g),
                    peak_q_pa=max(trajectory.dynamic_pressure_pa),
                    peak_heat_w_m2=max(trajectory.heat_flux_total_w_m2),
                    heat_load_j_m2=trajectory.heat_load_total_j_m2[-1],
                    mach_handoff=trajectory.mach[-1],
                    handoff_altitude_m=trajectory.altitude_m[-1],
                )
        except Exception as exc:  # keep remaining probes running
            row = {"probe": probe, "status": "blocked", "failure_reason": str(exc)}
        rows.append(row)

    summary = {"config_dir": str(config_dir), "n_probes": len(rows), "rows": rows}
    write_json(out_root / "pioneer_venus_summary.json", summary)
    with (out_root / "pioneer_venus_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _comparison_plot(rows, out_root / "pioneer_venus_comparison.png")
    return summary


def _comparison_plot(rows: list[dict[str, Any]], path: Path) -> None:
    from .plots import comparison_bars

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    comparison_bars(
        labels=[row["probe"] for row in ok_rows],
        series={
            "peak deceleration [g]": [row["peak_g"] for row in ok_rows],
            "peak heat flux [MW/m$^2$]": [row["peak_heat_w_m2"] / 1e6 for row in ok_rows],
        },
        title="Pioneer Venus probes -- simulated entry metrics",
        path=path,
    )


def _unused_json_guard() -> None:  # pragma: no cover
    json.dumps({})
