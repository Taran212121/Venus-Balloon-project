"""Generate pgfplots LaTeX figures for the EDDI chapter from results/figrun2.

Emits four standalone snippets (figure environments) into
results/figrun2/figures/. Requires only tikz + pgfplots (compat 1.17,
groupplots library) and siunitx in the document preamble.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from entry_traj.atmosphere import AtmosphereModel
from entry_traj.config import load_config

G0 = 9.80665
RUN = ROOT / "results" / "figrun2"
OUT = RUN / "figures"
OUT.mkdir(exist_ok=True)

cfg = load_config(ROOT / "configs" / "config.example.yaml")
atm = AtmosphereModel(cfg)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


nom = read_csv(RUN / "nominal_trajectory.csv")
des = read_csv(RUN / "post_deploy_descent.csv")

nt = [float(r["time_s"]) for r in nom]
nh = [float(r["altitude_m"]) / 1000.0 for r in nom]
nv = [float(r["velocity_m_s"]) for r in nom]
nm = [float(r["mach"]) for r in nom]
ng = [float(r["deceleration_g"]) for r in nom]
nq = [float(r["dynamic_pressure_pa"]) / 1000.0 for r in nom]
nhf = [float(r["heat_flux_total_w_m2"]) for r in nom]

dt_ = [float(r["time_s"]) for r in des]
dh = [float(r["altitude_m"]) / 1000.0 for r in des]
dv = [float(r["velocity_m_s"]) for r in des]
dmass = [float(r["mass_kg"]) for r in des]

# descent mach / q from the same atmosphere model
dmach = []
dq = []
for h_km, v in zip(dh, dv):
    h = h_km * 1000.0
    a = atm.speed_of_sound_m_s(h)
    rho = atm.density_kg_m3(h)
    dmach.append(v / a)
    dq.append(0.5 * rho * v * v / 1000.0)

# descent deceleration (net |dv/dt| in g0), central difference
dgee = [0.0] * len(dt_)
for i in range(1, len(dt_) - 1):
    dgee[i] = abs((dv[i + 1] - dv[i - 1]) / (dt_[i + 1] - dt_[i - 1])) / G0

i_pg = max(range(len(ng)), key=lambda i: ng[i])
i_ph = max(range(len(nhf)), key=lambda i: nhf[i])
i_shock = max(range(len(dgee)), key=lambda i: dgee[i])
T_PH, T_PG = nt[i_ph], nt[i_pg]
T_DEP = dt_[0]
EV = dict(deploy=T_DEP, dgb=120.489, disreef=133.495, jett=137.489,
          infl0=140.489, infl1=170.489, float_=dt_[-1])
print("peak g %.2f at %.2f s ; peak heat %.3g W/m2 at %.2f s" % (ng[i_pg], T_PG, nhf[i_ph], T_PH))
print("opening shock %.2f g0 at t=%.2f s ; float %.1f s" % (dgee[i_shock], dt_[i_shock], dt_[-1]))


def coords(xs, ys, fmt="(%.5g,%.5g)"):
    return " ".join(fmt % (x, y) for x, y in zip(xs, ys))


def thin(idx_count, step, always=()):
    keep = sorted(set(list(range(0, idx_count, step)) + [idx_count - 1] + list(always)))
    return keep


def pick(xs, keep):
    return [xs[i] for i in keep]


# entry thinning: dense near the pulse (60-100 s), sparse elsewhere
keep_n = sorted(set(
    list(range(0, len(nt), 8)) + list(range(220, 340, 2)) + [i_pg, i_ph, len(nt) - 1]))
# descent thinning: dense over the first 80 s (shock + disreef), sparse after
keep_d = sorted(set(
    list(range(0, 400, 8)) + list(range(400, len(dt_), 60)) + [i_shock, len(dt_) - 1]))

L = []  # helper to assemble files


def emit(name, lines):
    text = "\n".join(lines) + "\n"
    assert "{" + "{" not in text, "double brace in " + name
    (OUT / name).write_text(text)
    print("wrote", name, len(text), "bytes")


# ---------------------------------------------------------------- fig 1
ax_common = "width=0.50\\textwidth, height=6.4cm, ylabel={Altitude [km]}, ymin=50, ymax=210, grid=major, grid style={black!12}, tick label style={font=\\footnotesize}, label style={font=\\footnotesize}, title style={font=\\footnotesize}"
f1 = []
f1.append("% Requires: \\usepackage{pgfplots} \\pgfplotsset{compat=1.17} \\usepgfplotslibrary{groupplots}")
f1.append("\\begin{figure}[ht]")
f1.append("\\centering")
f1.append("\\begin{tikzpicture}")
f1.append("\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=14mm}, " + ax_common + "]")
f1.append("\\nextgroupplot[xlabel={Velocity [km/s]}, xmin=0, xmax=11.5]")
f1.append("\\addplot[black, line width=0.9pt] coordinates {" + coords(pick([v / 1000.0 for v in nv], keep_n), pick(nh, keep_n)) + "};")
f1.append("\\addplot[only marks, mark=*, mark size=1.6pt] coordinates {(%.4g,%.4g) (%.4g,%.4g) (%.4g,%.4g)};" % (nv[i_ph] / 1000.0, nh[i_ph], nv[i_pg] / 1000.0, nh[i_pg], nv[-1] / 1000.0, nh[-1]))
f1.append("\\node[anchor=west, font=\\scriptsize] at (axis cs:%.4g,%.4g) {peak heating};" % (nv[i_ph] / 1000.0 + 0.15, nh[i_ph]))
f1.append("\\node[anchor=west, font=\\scriptsize] at (axis cs:%.4g,%.4g) {peak-g};" % (nv[i_pg] / 1000.0 + 0.15, nh[i_pg] - 2.0))
f1.append("\\node[anchor=south west, font=\\scriptsize] at (axis cs:%.4g,%.4g) {deploy};" % (nv[-1] / 1000.0 + 0.1, nh[-1] - 1.5))
f1.append("\\nextgroupplot[xlabel={Mach [--]}, xmode=log, xmin=1, xmax=70]")
f1.append("\\addplot[black, line width=0.9pt] coordinates {" + coords(pick(nm, keep_n), pick(nh, keep_n)) + "};")
f1.append("\\addplot[only marks, mark=*, mark size=1.6pt] coordinates {(%.4g,%.4g) (%.4g,%.4g) (%.4g,%.4g)};" % (nm[i_ph], nh[i_ph], nm[i_pg], nh[i_pg], nm[-1], nh[-1]))
f1.append("\\node[anchor=west, font=\\scriptsize] at (axis cs:%.4g,%.4g) {peak heating};" % (nm[i_ph] * 1.15, nh[i_ph]))
f1.append("\\node[anchor=west, font=\\scriptsize] at (axis cs:%.4g,%.4g) {peak-g};" % (nm[i_pg] * 1.15, nh[i_pg] - 2.0))
f1.append("\\node[anchor=south, font=\\scriptsize] at (axis cs:%.4g,%.4g) {deploy, M %.2f};" % (nm[-1] * 1.4, nh[-1] - 1.0, nm[-1]))
f1.append("\\draw[dashed, black!50] (axis cs:1.55,50) -- (axis cs:1.55,210);")
f1.append("\\end{groupplot}")
f1.append("\\end{tikzpicture}")
f1.append("\\caption{Altitude versus velocity (left) and Mach number (right) for the locked $\\gamma_E=-8.0^\\circ$ nominal entry. Markers: peak heating ($t\\approx\\SI{%.1f}{s}$), peak deceleration ($t\\approx\\SI{%.1f}{s}$), and parachute deploy ($q=\\SI{3000}{Pa}$, $M=%.2f$). Dashed line: deploy Mach gate 1.55.}" % (T_PH, T_PG, nm[-1]))
f1.append("\\label{fig:dep-alt-vel-mach}")
f1.append("\\end{figure}")
emit("fig_alt_vel_mach.tex", f1)

# ---------------------------------------------------------------- fig 2
keep_g = sorted(set(list(range(0, len(nt), 4)) + list(range(250, 340, 1)) + [i_pg, len(nt) - 1]))
f2 = []
f2.append("% Requires: \\usepackage{pgfplots} \\pgfplotsset{compat=1.17}")
f2.append("\\begin{figure}[ht]")
f2.append("\\centering")
f2.append("\\begin{tikzpicture}")
f2.append("\\begin{axis}[width=0.72\\textwidth, height=6.2cm, xlabel={Time from entry interface [s]}, ylabel={Deceleration [$g_0$]}, xmin=0, xmax=125, ymin=0, ymax=85, grid=major, grid style={black!12}, tick label style={font=\\footnotesize}, label style={font=\\footnotesize}]")
f2.append("\\addplot[black, line width=0.9pt] coordinates {" + coords(pick(nt, keep_g), pick(ng, keep_g)) + "};")
f2.append("\\addplot[only marks, mark=*, mark size=1.6pt] coordinates {(%.4g,%.4g)};" % (T_PG, ng[i_pg]))
f2.append("\\node[anchor=south east, font=\\scriptsize, align=right] at (axis cs:%.4g,%.4g) {%.1f $g_0$ (p50) at \\SI{%.1f}{s}\\\\ p97.5: 81.1 $g_0$};" % (T_PG - 2.0, ng[i_pg] + 2.0, ng[i_pg], T_PG))
f2.append("\\draw[dashed, black!50] (axis cs:%.4g,0) -- (axis cs:%.4g,85);" % (T_DEP, T_DEP))
f2.append("\\node[anchor=south east, rotate=90, font=\\scriptsize] at (axis cs:%.4g,2) {mortar fire};" % T_DEP)
f2.append("\\end{axis}")
f2.append("\\end{tikzpicture}")
f2.append("\\caption{Sensed deceleration versus time for the nominal $-8.0^\\circ$ entry. The peak of %.1f $g_0$ (p50; 81.1 $g_0$ at p97.5) sits a factor $\\approx$2.5 below the \\SI{200}{} $g_0$ structural cap.}" % ng[i_pg])
f2.append("\\label{fig:dep-decel-time}")
f2.append("\\end{figure}")
emit("fig_decel_time.tex", f2)

# ---------------------------------------------------------------- fig 3 (hero)
TMAX = 1060.0
events = [
    (T_PH, "peak heating"),
    (T_PG, "peak-g / peak-q"),
    (EV["deploy"], "deploy (M %.2f)" % nm[-1]),
    (EV["disreef"], "disreef"),
    (EV["jett"], "shell jettison"),
    (EV["infl0"], "inflation start"),
    (EV["infl1"], "inflation end"),
    (EV["float_"], "float hand-off"),
]

# stitched channels: entry + descent
ht_t = pick(nt, keep_n) + pick(dt_, keep_d)
ht_h = pick(nh, keep_n) + pick(dh, keep_d)
ma_t = ht_t
ma_v = pick(nm, keep_n) + pick(dmach, keep_d)
q_v = pick(nq, keep_n) + pick(dq, keep_d)
g_v = pick(ng, keep_n) + pick(dgee, keep_d)
mass_t = [0.0, EV["jett"], EV["jett"], dt_[-1]]
mass_v = [dmass[0], dmass[0], dmass[-1], dmass[-1]]
pow_t = [0.0, EV["infl0"], EV["infl0"], EV["infl1"], EV["infl1"], dt_[-1]]
pow_v = [9.3, 9.3, 18.6, 18.6, 9.0, 9.0]
spikes = [(EV["deploy"], 149.0), (EV["disreef"], 149.0), (EV["jett"], 149.0), (EV["deploy"] + 19.0, 149.0), (EV["infl0"], 12.6), (EV["deploy"] + 55.0, 149.0)]

q_floor = 0.004
q_v = [max(q, q_floor) for q in q_v]
ma_v = [max(m, 0.009) for m in ma_v]

axes = []
axes.append(("Altitude [km]", "ymin=50, ymax=210", coords(ht_t, ht_h), 50, 210, None))
axes.append(("Mach [--]", "ymode=log, ymin=0.009, ymax=80", coords(ma_t, ma_v), 0.009, 80, None))
axes.append(("$q$ [kPa]", "ymode=log, ymin=0.004, ymax=400", coords(ma_t, q_v), 0.004, 400, "\\addplot[black!45, dashed, domain=0:1060] {3.0}; \\node[anchor=south west, font=\\tiny, black!45] at (axis cs:640,3.0) {deploy gate \\SI{3}{kPa}};"))
axes.append(("Decel.\\ [$g_0$]", "ymin=0, ymax=85", coords(ma_t, g_v), 0, 85, None))
axes.append(("Mass [kg]", "ymin=480, ymax=660", coords(mass_t, mass_v), 480, 660, None))
axes.append(("Power [W]", "ymode=log, ymin=5, ymax=400", coords(pow_t, pow_v), 5, 400, "SPIKES"))

f3 = []
f3.append("% Requires: \\usepackage{pgfplots} \\pgfplotsset{compat=1.17} \\usepgfplotslibrary{groupplots}")
f3.append("% HERO composite: entry -> float on one shared t-axis; entry channels from the")
f3.append("% locked -8.0 deg nominal run, descent channels from the 1D post-deploy model.")
f3.append("\\begin{figure}[p]")
f3.append("\\centering")
f3.append("\\begin{tikzpicture}")
f3.append("\\begin{groupplot}[group style={group size=1 by 6, vertical sep=5pt, xlabels at=edge bottom, xticklabels at=edge bottom}, width=0.92\\textwidth, height=3.45cm, xmin=0, xmax=%.0f, grid=major, grid style={black!10}, tick label style={font=\\scriptsize}, label style={font=\\scriptsize}, ylabel style={font=\\scriptsize}, xlabel={Time from entry interface [s]}, every axis plot/.append style={line width=0.8pt}]" % TMAX)
for k, (ylab, yopt, data, ylo, yhi, extra) in enumerate(axes):
    f3.append("\\nextgroupplot[ylabel={" + ylab + "}, " + yopt + "]")
    if extra == "SPIKES":
        f3.append("\\addplot[black] coordinates {" + data + "};")
        f3.append("\\addplot[black, ycomb, mark=*, mark size=1.1pt] coordinates {" + " ".join("(%.4g,%.4g)" % s for s in spikes) + "};")
        f3.append("\\addplot[black!45, dashed, domain=0:%.0f] {289};" % TMAX)
        f3.append("\\node[anchor=north west, font=\\tiny, black!45] at (axis cs:640,289) {A/B-fire bound \\SI{289}{W} (\\SI{0.05}{s})};")
    else:
        f3.append("\\addplot[black] coordinates {" + data + "};")
        if extra:
            f3.append(extra)
    for (tev, name) in events:
        f3.append("\\draw[black!35, line width=0.5pt] (axis cs:%.4g,%.5g) -- (axis cs:%.4g,%.5g);" % (tev, ylo, tev, yhi))
    if k == 0:
        for (tev, name) in events:
            f3.append("\\node[anchor=west, rotate=55, font=\\tiny, inner sep=1pt] at (axis cs:%.4g,212) {%s};" % (tev, name))
f3.append("\\end{groupplot}")
f3.append("\\end{tikzpicture}")
f3.append("\\caption{EDDI composite timeline from entry interface to float hand-off: altitude, Mach, dynamic pressure, sensed deceleration, stack mass, and EDDI bus power on one shared time axis. Entry channels are the locked $\\gamma_E=-8.0^\\circ$ nominal Monte-Carlo run; descent channels are the 1D post-deploy model (deploy $t=\\SI{%.1f}{s}$, disreef $t=\\SI{%.1f}{s}$, jettison $t=\\SI{%.1f}{s}$, inflation \\SIrange{%.0f}{%.0f}{s}, float hand-off at \\SI{55}{km}, $t=\\SI{%.0f}{s}$). Mass steps follow the model basis (\\SI{630.6}{kg} entry stack, \\SI{502.75}{kg} post-jettison); power per Table~\\ref{tab:dep-power} with \\SI{0.05}{s} pyro transients drawn as stems.}" % (EV["deploy"], EV["disreef"], EV["jett"], EV["infl0"], EV["infl1"], EV["float_"]))
f3.append("\\label{fig:dep-hero-timeline}")
f3.append("\\end{figure}")
emit("fig_hero_timeline.tex", f3)

# ---------------------------------------------------------------- fig 4
f4 = []
f4.append("% Requires: \\usepackage{pgfplots} \\pgfplotsset{compat=1.17}")
f4.append("\\begin{figure}[ht]")
f4.append("\\centering")
f4.append("\\begin{tikzpicture}")
f4.append("\\begin{axis}[width=0.72\\textwidth, height=6.4cm, xmode=log, xmin=0.13, xmax=35, ymin=0.3, ymax=1.2, xlabel={Mach [--]}, ylabel={$C_d$ [--]}, grid=major, grid style={black!12}, tick label style={font=\\footnotesize}, label style={font=\\footnotesize}, legend style={font=\\scriptsize, at={(0.985,0.03)}, anchor=south east, draw=black!30}, legend cell align=left, xtick={0.2,0.5,1,2,5,10,25}, xticklabels={0.2,0.5,1,2,5,10,25}]")
f4.append("\\fill[black!8] (axis cs:1.55,0.3) rectangle (axis cs:2.0,1.2);")
f4.append("\\node[anchor=north, font=\\scriptsize, black!60] at (axis cs:1.76,1.19) {deploy window};")
f4.append("\\addplot[black, line width=0.9pt, mark=*, mark size=1.5pt] coordinates {(0.5,0.88) (0.8,0.92) (1.55,1.01) (5,1.02) (10,1.04) (25,1.05)};")
f4.append("\\addlegendentry{Sphere-cone, $S_\\mathrm{ref}$ (PV / Galileo / DAVINCI)}")
f4.append("\\addplot[black, line width=0.9pt, dashed, mark=square*, mark size=1.4pt, mark options={solid}] coordinates {(0.2,0.62) (1.0,0.55) (2.0,0.48)};")
f4.append("\\addlegendentry{DGB pilot, $S_0$ (Viking / MER / MSL)}")
f4.append("\\addplot[black, line width=0.9pt, dotted, mark=triangle*, mark size=1.7pt, mark options={solid}] coordinates {(0.2,0.80) (0.6,0.78)};")
f4.append("\\addlegendentry{Ringsail, $S_0$ (Apollo / Gemini, Knacke)}")
f4.append("\\draw[dashed, black!50] (axis cs:1.55,0.3) -- (axis cs:1.55,1.2);")
f4.append("\\end{axis}")
f4.append("\\end{tikzpicture}")
f4.append("\\caption{Heritage $C_d(M)$ database (Table~\\ref{tab:dep-cdm}): aeroshell on $S_\\mathrm{ref}$, parachutes on $S_0$. Shaded band: deployment window $M=$ 1.55--2.0; the nominal mortar fire is at $M=1.55$.}")
f4.append("\\label{fig:dep-cdm}")
f4.append("\\end{figure}")
emit("fig_cd_mach.tex", f4)

print("done")
