# entry_traj

Phase-0 Venus entry trajectory tool (FA1, DSE Venus Aerobot Mission).

Ballistic entry from the 140 km interface to parachute-deploy hand-off:
nominal RK4 trajectory, Monte Carlo dispersions with first-order Sobol
attribution, Allen-Eggers and Pioneer Venus anchored validation, a TPS
sizing basis, EFPA feasibility sweeps, and a VIRA atmosphere-profile
ensemble. A small Streamlit control panel sits on top of the same API.

## Install

```sh
pip install -e .            # core: PyYAML + matplotlib
pip install -e ".[full]"    # + openpyxl, h5py, chaospy, streamlit
pip install -e ".[test]"    # + pytest
```

The core simulation runs on the standard library alone; PyYAML is needed to
load configs and matplotlib only for figures. Optional extras are degraded
gracefully: without openpyxl the workbook is skipped (with a note), without
chaospy a stdlib Latin-hypercube sampler is used, without h5py only the
synthetic GRAM dispersion mode is available.

## Run

```sh
python -m entry_traj run configs/config.example.yaml --out results
python -m entry_traj batch configs/pioneer_venus --out results/pioneer_venus
python -m entry_traj sweep-fpa configs/config.example.yaml --fpa-min -7 --fpa-max -11
python -m entry_traj sweep-profiles configs/config.example.yaml
streamlit run entry_traj/gui.py
```

Outputs of `run` (all in `--out`): `summary.json`, `validation_report.{json,txt}`,
`nominal_trajectory.csv`, `ensemble_metrics.csv`, `event_times.csv`,
`handoff_corridor.csv`, `mission_phase_table.{csv,md}`, `tps_sizing_basis.json`,
workbook + LaTeX macros, and PNG figures (trajectory, timeline, corridor,
Sobol tornado). `post_deploy_descent.{json,csv}` is added when the config
contains a `mission_timeline.post_deploy_model` block.

## Package layout

| Module | Purpose |
| --- | --- |
| `models.py` | Frozen dataclasses for config and results |
| `physics.py` | Gravity, Cd(M), flow/heating point metrics, interpolation |
| `atmosphere.py` | VIRA profile loaders, GRAM dispersions, atmosphere model |
| `config.py` | YAML defaults, deep-merge, construction, validation |
| `deploy.py` | Parachute deploy trigger resolution (`mach_q`, `q_window`) |
| `integrator.py` | RK4 entry integration, events, payload thermal gate |
| `monte_carlo.py` | LHS ensemble, percentiles, Sobol, hand-off corridor |
| `validation.py` | Allen-Eggers, caps, VCD envelope checks |
| `tps.py` | PV-calibrated TPS sizing basis |
| `post_deploy.py` | Staged-parachute descent to float hand-off (optional) |
| `sweep.py` | EFPA feasibility sweep over 3-sigma tails + entry-system mass trade |
| `profile_ensemble.py` | VIRA selector-grid ensemble |
| `batch.py` | Pioneer Venus anchoring batch |
| `outputs.py` | JSON/CSV/workbook/LaTeX writers, mission tables |
| `plots.py` | All matplotlib figures, one shared style |
| `api.py` / `cli.py` / `gui.py` | Orchestration, CLI, Streamlit panel |

## Config

The YAML schema is unchanged from v0.1 -- existing configs work as-is. See
`configs/config.example.yaml` (FA1 production baseline) and
`configs/pioneer_venus/` (flight-data anchoring set). `entry.fpa_deg` is
required and must be an exact published value; everything else has defaults.

## Notes

- v0.2 is a from-scratch rewrite of the v0.1 codebase with the same physics,
  config schema, CLI commands, and output file set.
- The standalone helper scripts from v0.1 (`scripts/`) targeted v0.1
  internals and were not ported; their useful functionality lives in the CLI
  subcommands.
- The mission power-profile table/plot from v0.1 was dropped; the
  event-annotated timeline figure and `mission_phase_table.*` replace it.
