"""entry_traj control panel -- a small Streamlit front end.

Run with:  streamlit run entry_traj/gui.py

Design notes: one accent colour, hairline rules, tabular numerals, no chrome.
The sidebar edits a working copy of the YAML config; runs are saved under
results/gui_runs/<timestamp> so the CLI and GUI share the same output format.
"""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path

import streamlit as st
import yaml

from entry_traj.config import config_from_mapping
from entry_traj.integrator import integrate
from entry_traj.models import to_plain
from entry_traj.monte_carlo import monte_carlo, summarize
from entry_traj.outputs import mission_phase_rows, trajectory_rows, write_csv, write_json
from entry_traj.sweep import sweep_fpa
from entry_traj.validation import report_text, validate

RUN_ROOT = Path("results/gui_runs")
DEFAULT_CONFIG = "configs/config.example.yaml"
DEFAULT_ENTRY_MASS_KG = 689.130

INK = "#1a1d21"
MUTED = "#6f6a60"
PAPER = "#f7f6f2"
CARD = "#fdfdfb"
ACCENT = "#b3552d"
TEAL = "#225b63"
HAIRLINE = "#e3e0d8"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', sans-serif;
    color: #1a1d21;
}
.stApp { background: #f7f6f2; }
[data-testid="stSidebar"] {
    background: #f1efe9;
    border-right: 1px solid #e3e0d8;
}
[data-testid="stSidebar"] .stButton button {
    width: 100%;
}
h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; }
.stButton button {
    background: #fdfdfb;
    color: #1a1d21;
    border: 1px solid #c9c4b8;
    border-radius: 2px;
    font-size: 0.85rem;
    padding: 0.35rem 0.9rem;
}
.stButton button:hover { border-color: #b3552d; color: #b3552d; }
.et-header {
    display: flex; align-items: baseline; gap: 1rem;
    border-bottom: 2px solid #1a1d21;
    padding-bottom: 0.4rem; margin-bottom: 0.2rem;
}
.et-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.25rem; font-weight: 500; letter-spacing: 0.18em;
}
.et-sub {
    font-size: 0.78rem; color: #6f6a60; letter-spacing: 0.06em;
}
.et-label {
    font-size: 0.68rem; color: #6f6a60;
    letter-spacing: 0.14em; text-transform: uppercase;
    margin-bottom: 0.1rem;
}
.et-strip {
    display: grid; grid-template-columns: repeat(6, 1fr);
    border: 1px solid #e3e0d8; background: #fdfdfb;
    margin: 0.8rem 0 0.4rem 0;
}
.et-cell {
    padding: 0.55rem 0.8rem;
    border-right: 1px solid #e3e0d8;
}
.et-cell:last-child { border-right: none; }
.et-value {
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-size: 1.05rem;
}
.et-unit { font-size: 0.72rem; color: #6f6a60; margin-left: 0.15rem; }
.et-note { font-size: 0.75rem; color: #6f6a60; }
.et-pass { color: #225b63; font-weight: 600; }
.et-fail { color: #b3552d; font-weight: 600; }
[data-testid="stMetric"] { background: transparent; }
.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; border-bottom: 1px solid #e3e0d8; }
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: #6f6a60; padding-bottom: 0.4rem;
}
.stTabs [aria-selected="true"] { color: #1a1d21; border-bottom: 2px solid #b3552d; }
</style>
"""


# --------------------------------------------------------------------------
# Config handling
# --------------------------------------------------------------------------


def _load_raw(path_text: str) -> dict:
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or dict()


def _set_in(raw: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = raw
    for key in keys[:-1]:
        node = node.setdefault(key, dict())
    node[keys[-1]] = value


def _get_in(raw: dict, dotted: str, default=None):
    node = raw
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _mass_basis_caption(raw: dict) -> str | None:
    no_margin = _get_in(raw, "mass_basis.entry_mass_without_system_margin_kg")
    with_margin = _get_in(raw, "mass_basis.entry_mass_with_system_margin_kg")
    if no_margin is None and with_margin is None:
        return None
    parts = []
    if no_margin is not None:
        parts.append(f"without system margin: {float(no_margin):.3f} kg")
    if with_margin is not None:
        parts.append(f"with system margin: {float(with_margin):.3f} kg")
    return "Mass basis -- " + "; ".join(parts)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def _sidebar(raw: dict, base_dir: Path) -> tuple[dict, dict]:
    """Render controls, return (edited raw config, actions)."""
    st.sidebar.markdown('<div class="et-label">Configuration</div>', unsafe_allow_html=True)
    edited = yaml.safe_load(yaml.safe_dump(raw)) or dict()  # deep copy

    with st.sidebar.expander("Entry state", expanded=True):
        interface_altitude = st.number_input(
            "Interface altitude [m]",
            value=float(_get_in(edited, "entry.interface_altitude_m", 140000.0)),
            step=1000.0,
            format="%.0f",
        )
        fpa = st.number_input(
            "EFPA [deg]",
            value=float(_get_in(edited, "entry.fpa_deg", -8.0) or -8.0),
            step=0.1,
            format="%.2f",
        )
        velocity = st.number_input(
            "Entry velocity [m/s]",
            value=float(_get_in(edited, "entry.velocity_m_s", 11000.0)),
            step=50.0,
            format="%.0f",
        )
        mass = st.number_input(
            "Entry mass [kg]",
            value=float(_get_in(edited, "entry.m_entry_kg", DEFAULT_ENTRY_MASS_KG)),
            step=0.1,
            format="%.3f",
        )
        mass_caption = _mass_basis_caption(edited)
        if mass_caption:
            st.caption(mass_caption)
        _set_in(edited, "entry.interface_altitude_m", interface_altitude)
        _set_in(edited, "entry.fpa_deg", fpa)
        _set_in(edited, "entry.velocity_m_s", velocity)
        _set_in(edited, "entry.m_entry_kg", mass)

    with st.sidebar.expander("Vehicle"):
        area = st.number_input(
            "Reference area [m2]",
            value=float(_get_in(edited, "vehicle.reference_area_m2", 2.72) or 2.72),
            step=0.01,
            format="%.3f",
        )
        nose = st.number_input(
            "Nose radius [m]",
            value=float(_get_in(edited, "vehicle.nose_radius_m", 0.93)),
            step=0.01,
            format="%.3f",
        )
        _set_in(edited, "vehicle.reference_area_m2", area)
        _set_in(edited, "vehicle.nose_radius_m", nose)
        forebody = _get_in(edited, "vehicle.forebody_wetted_area_m2")
        backshell = _get_in(edited, "vehicle.backshell_wetted_area_m2")
        if forebody is not None and backshell is not None:
            st.caption(
                f"TPS wetted areas: forebody {float(forebody):.1f} m2, "
                f"backshell {float(backshell):.1f} m2; reference area is projected aero area."
            )
        cd_value = _get_in(edited, "vehicle.cd")
        cd_table = _get_in(edited, "vehicle.cd_mach_table")
        if cd_table:
            st.caption(f"Cd(M) table with {len(cd_table)} rows (edit in YAML)")
            cd_for_beta = float(cd_table[0][1])
        else:
            cd_for_beta = st.number_input("Drag coefficient", value=float(cd_value or 1.0), step=0.05, format="%.2f")
            _set_in(edited, "vehicle.cd", cd_for_beta)
        if area > 0 and cd_for_beta > 0:
            beta = mass / (cd_for_beta * area)
            st.caption(f"ballistic coefficient ~ {beta:.1f} kg/m2")

    with st.sidebar.expander("Deploy window"):
        q_floor = st.number_input("q floor [Pa]", value=float(_get_in(edited, "deploy.q_floor_pa", 200.0)), step=25.0, format="%.0f")
        q_cap = st.number_input("q cap [Pa]", value=float(_get_in(edited, "deploy.q_cap_pa", 3000.0)), step=50.0, format="%.0f")
        mach_ceiling = st.number_input("Mach ceiling", value=float(_get_in(edited, "deploy.mach_ceiling", 2.2)), step=0.05, format="%.2f")
        min_alt = st.number_input("Altitude floor [m]", value=float(_get_in(edited, "deploy.min_altitude_m", 50000.0)), step=500.0, format="%.0f")
        for dotted, value in (("deploy.q_floor_pa", q_floor), ("deploy.q_cap_pa", q_cap), ("deploy.mach_ceiling", mach_ceiling), ("deploy.min_altitude_m", min_alt)):
            _set_in(edited, dotted, value)

    with st.sidebar.expander("Dispersions"):
        n_runs = st.number_input("Monte Carlo runs", value=int(_get_in(edited, "dispersions.n_runs", 1000)), min_value=8, max_value=20000, step=100)
        seed = st.number_input("Seed", value=int(_get_in(edited, "dispersions.seed", 1001)), step=1)
        mass_fraction = st.number_input("Mass dispersion [+/- fraction]", value=float(_get_in(edited, "dispersions.mass_fraction", 0.10)), step=0.01, format="%.3f")
        fpa_disp = st.number_input("EFPA dispersion [deg]", value=float(_get_in(edited, "dispersions.fpa_deg", 0.5)), step=0.05, format="%.2f")
        for dotted, value in (("dispersions.n_runs", int(n_runs)), ("dispersions.seed", int(seed)), ("dispersions.mass_fraction", mass_fraction), ("dispersions.fpa_deg", fpa_disp)):
            _set_in(edited, dotted, value)

    with st.sidebar.expander("Numerics"):
        dt = st.number_input("Time step [s]", value=float(_get_in(edited, "numerics.time_step_s", 0.25)), step=0.05, format="%.3f")
        model = st.selectbox(
            "Trajectory model",
            ("constant_fpa", "spherical_ballistic"),
            index=0 if _get_in(edited, "numerics.trajectory_model", "constant_fpa") == "constant_fpa" else 1,
        )
        _set_in(edited, "numerics.time_step_s", dt)
        _set_in(edited, "numerics.trajectory_model", model)

    st.sidebar.markdown('<div class="et-label">Execute</div>', unsafe_allow_html=True)
    actions = dict(
        run_nominal=st.sidebar.button("Run nominal"),
        run_mc=st.sidebar.button("Run Monte Carlo"),
        run_sweep=st.sidebar.button("Run EFPA sweep"),
    )
    with st.sidebar.expander("EFPA sweep range"):
        actions["sweep_lo"] = st.number_input("Shallow bound [deg]", value=-7.0, step=0.25, format="%.2f")
        actions["sweep_hi"] = st.number_input("Steep bound [deg]", value=-10.0, step=0.25, format="%.2f")
        actions["sweep_step"] = st.number_input("Step [deg]", value=0.25, min_value=0.05, step=0.05, format="%.2f")
    return edited, actions


# --------------------------------------------------------------------------
# Main panels
# --------------------------------------------------------------------------


def _metric_strip(nominal) -> None:
    deploy_t = nominal.event_times_s.get("handoff")
    cells = [
        ("peak decel", f"{max(nominal.deceleration_g):.1f}", "g"),
        ("peak q", f"{max(nominal.dynamic_pressure_pa) / 1000.0:.1f}", "kPa"),
        ("peak heat", f"{max(nominal.heat_flux_total_w_m2) / 1e6:.2f}", "MW/m2"),
        ("heat load", f"{nominal.heat_load_total_j_m2[-1] / 1e6:.1f}", "MJ/m2"),
        ("Mach @ hand-off", f"{nominal.mach[-1]:.2f}", ""),
        ("hand-off alt", f"{nominal.altitude_m[-1] / 1000.0:.1f}", "km"),
    ]
    html = ['<div class="et-strip">']
    for label, value, unit in cells:
        unit_html = f'<span class="et-unit">{unit}</span>' if unit else ""
        html.append(
            f'<div class="et-cell"><div class="et-label">{label}</div>'
            f'<div class="et-value">{value}{unit_html}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)
    if deploy_t is not None:
        st.markdown(
            f'<div class="et-note">hand-off at t = {deploy_t:.1f} s after interface</div>',
            unsafe_allow_html=True,
        )


def _trajectory_tab(nominal) -> None:
    from entry_traj.plots import timeline_figure, trajectory_figure

    run_dir = st.session_state["run_dir"]
    fig_path = run_dir / "nominal_trajectory.png"
    if not fig_path.exists():
        trajectory_figure(nominal, fig_path)
        timeline_figure(nominal, run_dir / "mission_timeline.png")
    st.image(str(fig_path), use_container_width=True)
    st.image(str(run_dir / "mission_timeline.png"), use_container_width=True)

    header, rows = mission_phase_rows(nominal)
    st.markdown('<div class="et-label">Mission phases</div>', unsafe_allow_html=True)
    st.dataframe(
        [dict(zip(header, row)) for row in rows],
        hide_index=True,
        use_container_width=True,
    )


def _monte_carlo_tab(summary) -> None:
    from entry_traj.plots import corridor_figure, tornado_figure

    if summary is None:
        st.markdown('<div class="et-note">No Monte Carlo results yet. Use "Run Monte Carlo" in the sidebar.</div>', unsafe_allow_html=True)
        return
    run_dir = st.session_state["run_dir"]
    corridor_path = run_dir / "handoff_corridor.png"
    tornado_path = run_dir / "sobol_tornado.png"
    if not corridor_path.exists():
        corridor_figure(summary, corridor_path)
        tornado_figure(summary, tornado_path, metric="peak_heat")

    st.markdown(
        f'<div class="et-note">{summary.n_success}/{summary.n_requested} runs reached hand-off</div>',
        unsafe_allow_html=True,
    )
    table = []
    for metric, stats in summary.metric_intervals.items():
        table.append(
            dict(
                metric=metric,
                p2_5=f"{stats['p2_5']:.4g}",
                p50=f"{stats['p50']:.4g}",
                p97_5=f"{stats['p97_5']:.4g}",
                units=stats["units"],
                driver=summary.dominant_driver.get(metric, ""),
            )
        )
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.image(str(corridor_path), use_container_width=True)
    st.image(str(tornado_path), use_container_width=True)


def _sweep_tab(sweep_result) -> None:
    from entry_traj.plots import fpa_sweep_figure

    if sweep_result is None:
        st.markdown('<div class="et-note">No sweep results yet. Use "Run EFPA sweep" in the sidebar.</div>', unsafe_allow_html=True)
        return
    run_dir = st.session_state["run_dir"]
    fig_path = run_dir / "fpa_sweep.png"
    if not fig_path.exists():
        fpa_sweep_figure(sweep_result, fig_path)
    recommended = sweep_result["recommended_fpa_deg"]
    mass_enabled = sweep_result.get("mass_model", {}).get("enabled", False)
    if recommended is None:
        text = "no feasible EFPA in range"
    elif mass_enabled and sweep_result.get("mass_optimal_fpa_deg") is not None:
        text = f"recommended EFPA {recommended:g} deg (mass-optimal: TPS vs structure trade)"
        shallowest = sweep_result.get("shallowest_feasible_fpa_deg")
        if shallowest is not None and shallowest != recommended:
            text += f" -- shallowest feasible {shallowest:g} deg (max corridor margin)"
    else:
        text = f"recommended EFPA {recommended:g} deg (shallowest feasible)"
    st.markdown(f'<div class="et-note">{text} -- {sweep_result["n_feasible"]} feasible candidates</div>', unsafe_allow_html=True)
    st.image(str(fig_path), use_container_width=True)

    def _mass_cols(row) -> dict:
        if not mass_enabled or "mass" not in row:
            return {}
        mass = row["mass"]
        return {
            "tps_kg": None if mass["tps_mass_kg"] is None else round(mass["tps_mass_kg"], 1),
            "struct_kg": None if mass["structure_mass_kg"] is None else round(mass["structure_mass_kg"], 1),
            "entry_sys_kg": None if mass["entry_system_mass_kg"] is None else round(mass["entry_system_mass_kg"], 1),
        }

    rows = [
        dict(
            fpa_deg=row["fpa_deg"],
            feasible=row["feasible"],
            **_mass_cols(row),
            **{name: ("pass" if ok else "fail") for name, ok in row["constraints"].items()},
        )
        for row in sweep_result["rows"]
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    if mass_enabled:
        st.caption(
            "Mass model: TPS from worst-tail design heat load (ablator mass-fraction correlation), "
            "structure from worst-tail peak g (configurable scaling). Ranking model only -- not a mass budget."
        )


def _validation_tab(report) -> None:
    if report is None:
        st.markdown('<div class="et-note">Run the nominal trajectory to generate the validation report.</div>', unsafe_allow_html=True)
        return
    status = '<span class="et-pass">PASS</span>' if report.passed else '<span class="et-fail">FAIL</span>'
    st.markdown(f'<div class="et-note">Overall: {status}</div>', unsafe_allow_html=True)
    for check in report.checks:
        ok = '<span class="et-pass">pass</span>' if check["passed"] else '<span class="et-fail">fail</span>'
        st.markdown(f"**{check['name']}** -- {ok}", unsafe_allow_html=True)
        details = {k: v for k, v in check.items() if k not in ("name", "passed", "violations", "note")}
        st.dataframe(
            [dict(field=k, value=_fmt(v)) for k, v in details.items()],
            hide_index=True,
            use_container_width=True,
        )
        if check.get("note"):
            st.caption(check["note"])


def _fmt(value) -> str:
    if isinstance(value, float):
        if value != value or math.isinf(value):
            return str(value)
        return f"{value:.6g}"
    return str(value)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="entry_traj", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    state = st.session_state
    state.setdefault("nominal", None)
    state.setdefault("report", None)
    state.setdefault("summary", None)
    state.setdefault("sweep", None)
    state.setdefault("run_dir", RUN_ROOT / "scratch")
    state.setdefault("history", [])

    config_path = st.sidebar.text_input("Config file", value=DEFAULT_CONFIG)
    base_dir = Path(config_path).parent
    try:
        raw = _load_raw(config_path)
    except (OSError, yaml.YAMLError) as exc:
        st.error(f"Could not read config: {exc}")
        return

    edited_raw, actions = _sidebar(raw, base_dir)

    tuple_id = _get_in(edited_raw, "architecture.tuple_id", "")
    st.markdown(
        '<div class="et-header"><span class="et-title">ENTRY-TRAJ</span>'
        f'<span class="et-sub">{Path(config_path).name} &middot; {tuple_id}</span></div>',
        unsafe_allow_html=True,
    )

    try:
        cfg = config_from_mapping(edited_raw, base_dir)
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"Config invalid: {exc}")
        return

    if actions["run_nominal"] or actions["run_mc"] or actions["run_sweep"]:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        state["run_dir"] = RUN_ROOT / stamp
        state["run_dir"].mkdir(parents=True, exist_ok=True)

    if actions["run_nominal"] or actions["run_mc"]:
        with st.spinner("Integrating nominal trajectory"):
            state["nominal"] = integrate(cfg)
            state["report"] = validate(cfg, state["nominal"])
        write_csv(state["run_dir"] / "nominal_trajectory.csv", *trajectory_rows(state["nominal"]))
        (state["run_dir"] / "validation_report.txt").write_text(report_text(state["report"]), encoding="utf-8")
        state["history"].append(f"{state['run_dir'].name}  nominal  EFPA {cfg.entry_fpa_deg:g}")

    if actions["run_mc"]:
        with st.spinner(f"Running {cfg.dispersions_n_runs} dispersed trajectories"):
            ensemble = monte_carlo(cfg, n=cfg.dispersions_n_runs, seed=cfg.dispersions_seed)
            state["summary"] = summarize(ensemble)
        write_json(state["run_dir"] / "summary.json", to_plain(state["summary"]))
        state["history"].append(f"{state['run_dir'].name}  monte-carlo  n={cfg.dispersions_n_runs}")

    if actions["run_sweep"]:
        lo, hi = sorted((actions["sweep_lo"], actions["sweep_hi"]))
        step = abs(actions["sweep_step"]) or 0.25
        values = []
        value = lo
        while value <= hi + 1e-9:
            values.append(round(value, 6))
            value += step
        with st.spinner(f"Sweeping {len(values)} EFPA candidates"):
            state["sweep"] = sweep_fpa(cfg, values)
        write_json(state["run_dir"] / "fpa_sweep.json", state["sweep"])
        state["history"].append(f"{state['run_dir'].name}  efpa-sweep  {lo:g}..{hi:g}")

    if state["history"]:
        with st.sidebar.expander("Run history"):
            for line in reversed(state["history"][-12:]):
                st.markdown(f'<div class="et-note">{line}</div>', unsafe_allow_html=True)

    if state["nominal"] is None:
        st.markdown(
            '<div class="et-note" style="margin-top:1.5rem">'
            "Set the entry state in the sidebar and run the nominal trajectory. "
            "Monte Carlo and the EFPA sweep build on the same configuration.</div>",
            unsafe_allow_html=True,
        )
        return

    if not state["nominal"].success:
        st.error(f"Nominal trajectory failed: {state['nominal'].failure_reason}")
        return

    _metric_strip(state["nominal"])
    tab_traj, tab_mc, tab_sweep, tab_val = st.tabs(["Trajectory", "Monte Carlo", "EFPA sweep", "Validation"])
    with tab_traj:
        _trajectory_tab(state["nominal"])
    with tab_mc:
        _monte_carlo_tab(state["summary"])
    with tab_sweep:
        _sweep_tab(state["sweep"])
    with tab_val:
        _validation_tab(state["report"])


main()
