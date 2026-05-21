# Sensitivity Study

<aside>
🎯

**Purpose.** Keep the first sensitivity study simple. Use the combined pendulum + torsion model to understand how each design parameter influences the relevant outputs.

</aside>

## 1. Goal

The sensitivity study should answer:

```
How does each individual design parameter affect the dynamic and structural outputs?
```

The priority is **parameter sensitivity**, not optimization.

The first goal is to get engineering intuition:

```
If I increase cable length, what happens to period, amplitude, rate, tension, yaw, and IMU sampling need?
If I increase attachment radius, what happens to torsion?
If I increase damping, how much does settling improve?
If I change the number of cables, what happens to per-cable loads?
```

This is a **design-understanding step**, not a final design-selection or optimization step.

Do not add new physics yet:

- no active ADCS control;
- no aerodynamic damping;
- no added mass / internal gas participation;
- no fully coupled pendulum–torsion model;
- no detailed payload error budget.

Keep the current model as the motion generator.

---

## 2. Main approach

Use **one-at-a-time sensitivity sweeps** around a baseline case.

For each design parameter:

1. keep all other parameters fixed at the baseline value;
2. vary only one parameter;
3. run the model for each value;
4. save the relevant outputs in a table;
5. add a short trend summary explaining the global effect of that parameter.

The most important output for now is:

```
sensitivity_one_at_a_time.csv
```

This file should contain all one-at-a-time sweep results.

---

## 3. Baseline case

Use this as the first reference case:

```python
baseline_case = {
    "L_susp": 15.0,
    "N_cables": 4,
    "r": 1.0,
    "m_gondola": 400.0,
    "m_one_cable_total": 0.2,
    "gondola_shape": "solid_cylinder",
    "R_gondola": 2.8,
    "L_gondola": 0.6,
    "zeta_pendulum": 0.005,
    "zeta_torsion": 0.001,
    "disturbance_case": "medium",
}
```

Use this medium disturbance case as the baseline disturbance:

```python
medium_disturbance = {
    "pend_initial_angle_deg": 4.0,
    "pend_initial_rate_deg_s": 1.0,
    "torsion_initial_twist_deg": 5.0,
    "torsion_initial_yaw_rate_deg_s": 1.0,
    "pend_impulse_Ns": 0.0,
    "torsion_impulse_Nms": 0.0,
}
```

Before doing sweeps, confirm that:

- the script runs without errors;
- verification checks pass;
- pendulum visualization looks physically reasonable;
- baseline summary outputs are reasonable.

---

## 4. Parameters to vary

### 4.1 Cable length

```python
L_susp_values = [10.0, 12.5, 15.0, 20.0]
```

Design interpretation:

- shorter cable → lower feed-system mass, higher frequency, usually higher angular rates;
- longer cable → higher feed-system mass, lower frequency, usually lower angular rates.

The most important region is currently **10–15 m**, because Power gives a preliminary minimum balloon–gondola separation of about 10–15 m.

### 4.2 Number of cables

```python
N_cables_values = [3, 4, 6]
```

Design interpretation:

- fewer cables → lower mass/complexity, higher per-cable load;
- more cables → lower per-cable load, higher mass/complexity.

### 4.3 Cable attachment radius

```python
r_attach_values = [0.5, 1.0, 1.5, 2.0]
```

Design interpretation:

- larger attachment radius → higher torsional stiffness;
- higher torsional stiffness should reduce yaw/twist response;
- larger radius may require a larger gondola/suspension interface.

### 4.4 Damping ratio

```python
zeta_pendulum_values = [0.001, 0.003, 0.005, 0.01]
zeta_torsion_values  = [0.001, 0.003, 0.005, 0.01]
```

Interpretation:

- `0.001` → very lightly damped;
- `0.003` → low damping;
- `0.005` → moderate preliminary damping;
- `0.01` → optimistic / passive damping case.

For simplicity, run:

- one sweep where only `zeta_pendulum` changes;
- one sweep where only `zeta_torsion` changes.

### 4.5 Disturbance level

Keep three representative levels:

```python
disturbance_cases = {
    "low": {
        "pend_initial_angle_deg": 2.0,
        "pend_initial_rate_deg_s": 0.5,
        "torsion_initial_twist_deg": 2.0,
        "torsion_initial_yaw_rate_deg_s": 0.5,
        "pend_impulse_Ns": 0.0,
        "torsion_impulse_Nms": 0.0,
    },
    "medium": {
        "pend_initial_angle_deg": 4.0,
        "pend_initial_rate_deg_s": 1.0,
        "torsion_initial_twist_deg": 5.0,
        "torsion_initial_yaw_rate_deg_s": 1.0,
        "pend_impulse_Ns": 0.0,
        "torsion_impulse_Nms": 0.0,
    },
    "high": {
        "pend_initial_angle_deg": 8.0,
        "pend_initial_rate_deg_s": 2.0,
        "torsion_initial_twist_deg": 8.0,
        "torsion_initial_yaw_rate_deg_s": 2.0,
        "pend_impulse_Ns": 0.0,
        "torsion_impulse_Nms": 0.0,
    },
}
```

Keep all disturbance cases within the small-angle assumption.

### 4.6 Gondola mass

```python
m_gondola_values = [300.0, 400.0, 500.0]
```

Design interpretation:

- higher gondola mass → higher cable tension and structural loads;
- higher gondola mass changes the inertia of the suspended system;
- higher gondola mass also affects torsional stiffness because the suspended mass appears in the bifilar stiffness expression;
- this is important because the gondola mass is not fully fixed yet.

### 4.7 Cable mass

Vary the mass of one complete cable over the full suspension length:

```python
m_one_cable_total_values = [0.1, 0.2, 0.5, 1.0]
```

Design interpretation:

- higher cable mass → higher total suspended mass;
- higher cable mass → higher static and dynamic cable loads;
- higher cable mass changes the distributed mass in both pendulum and torsion models;
- cable mass also couples to cable number, because total cable mass is `N_cables * m_one_cable_total`.

For one-at-a-time sensitivity, vary only `m_one_cable_total` while keeping `N_cables` fixed at the baseline.

### 4.8 Gondola shape / inertia assumption

Vary the simplified gondola shape:

```python
gondola_shape_values = [
    "solid_cylinder",
    "cube",
    "solid_sphere",
    "rectangular_box",
]
```

The shape affects the model through the gondola inertias:

- `I_gondola_tilt` for pendulum / swing;
- `I_gondola_yaw` for torsion / yaw.

When the gondola shape changes, update the geometry parameters that define the inertia. Do not keep using cylinder radius/length for all shapes.

Use a helper function such as:

```python
def gondola_inertia(m, shape, R=None, h=None, side=None, a=None, b=None, c=None):
    """
    Return representative gondola inertias:
        I_tilt: roll/pitch inertia used in the pendulum model
        I_yaw:  yaw inertia used in the torsion model

    Coordinate convention:
        z = vertical / yaw axis
        x, y = horizontal axes
    """

    if shape == "solid_cylinder":
        # Vertical solid cylinder with radius R and height h
        I_tilt = (1.0 / 12.0) * m * (3.0 * R**2 + h**2)
        I_yaw  = 0.5 * m * R**2

    elif shape == "solid_sphere":
        # Solid sphere with radius R
        I_tilt = (2.0 / 5.0) * m * R**2
        I_yaw  = (2.0 / 5.0) * m * R**2

    elif shape == "cube":
        # Solid cube with side length side
        I_tilt = (1.0 / 6.0) * m * side**2
        I_yaw  = (1.0 / 6.0) * m * side**2

    elif shape == "rectangular_box":
        # Solid rectangular box with dimensions a, b, c
        # a = x length, b = y length, c = vertical height
        I_x = (1.0 / 12.0) * m * (b**2 + c**2)
        I_y = (1.0 / 12.0) * m * (a**2 + c**2)
        I_z = (1.0 / 12.0) * m * (a**2 + b**2)

        I_tilt = 0.5 * (I_x + I_y)
        I_yaw  = I_z

    else:
        raise ValueError(f"Unknown gondola shape: {shape}")

    return I_tilt, I_yaw
```

For the baseline shape:

```python
gondola_shape = "solid_cylinder"
R_gondola = 2.8
L_gondola = 0.6
```

For shape sensitivity, define representative dimensions for each shape. These values can be rough at this stage, but they should be explicit:

```python
gondola_geometry_cases = {
    "solid_cylinder": {
        "R": 2.8,
        "h": 0.6,
    },
    "cube": {
        "side": 2.0,
    },
    "solid_sphere": {
        "R": 1.5,
    },
    "rectangular_box": {
        "a": 3.0,
        "b": 2.0,
        "c": 0.8,
    },
}
```

In the model:

- use `I_gondola_tilt` in the pendulum inertia list;
- use `I_gondola_yaw` in the torsion mass matrix;
- use a representative vertical size for `L_gondola` when estimating cable tension / effective pendulum length.

For shape sensitivity, add the following to the output table:

```
gondola_shape
I_gondola_tilt
I_gondola_yaw
gondola_reference_size
```

The main purpose is to see whether gondola inertia assumptions noticeably affect pendulum and torsion outputs.

---

## 5. Outputs to save for each case

Keep the outputs high-level and useful for comparing trends.

### Case definition

```
case_id
sweep_name
varied_parameter
varied_value
L_susp
N_cables
r_attach
m_gondola
m_one_cable_total
zeta_pendulum
zeta_torsion
gondola_shape
I_gondola_tilt
I_gondola_yaw
disturbance_case
```

### Dynamic outputs

```
dominant_pendulum_period_s
dominant_torsion_period_s
dominant_pendulum_frequency_hz
dominant_torsion_frequency_hz
max_gondola_swing_angle_deg
max_gondola_swing_rate_deg_s
max_gondola_yaw_deg
max_gondola_yaw_rate_deg_s
settling_time_pendulum_s
settling_time_torsion_s
```

### Structural outputs

```
peak_total_cable_tension_N
peak_per_cable_tension_N
minimum_per_cable_tension_N
dynamic_static_tension_ratio
maximum_torsional_torque_Nm
maximum_torsion_element_twist_deg
```

### ADCS / IMU observability outputs

Keep the required IMU sampling frequency.

Use:

```python
f_max_relevant = max(np.max(freq_x), np.max(freq_z))
required_imu_sampling_10x_hz = 10.0 * f_max_relevant
required_imu_sampling_20x_hz = 20.0 * f_max_relevant
```

Save:

```
f_max_relevant_hz
required_imu_sampling_10x_hz
required_imu_sampling_20x_hz
```

Do not add a qualitative `imu_sampling_demand` label for now. The numerical sampling-frequency estimate is enough.

---

## 6. Required tables

For each one-at-a-time sweep, Copilot should generate a table.

Each table should show:

- the varied parameter value;
- the main dynamic outputs;
- the structural outputs;
- the required IMU sampling frequency;
- a final trend summary.

The tables can be printed in the terminal and saved to CSV.

---

## 7. Table 1 — Cable-length sensitivity

Vary:

```python
L_susp_values = [10.0, 12.5, 15.0, 20.0]
```

Keep all other parameters at baseline.

Required table columns:

```
L_susp
dominant_pendulum_period_s
dominant_torsion_period_s
max_gondola_swing_angle_deg
max_gondola_swing_rate_deg_s
max_gondola_yaw_deg
max_gondola_yaw_rate_deg_s
peak_per_cable_tension_N
maximum_torsional_torque_Nm
required_imu_sampling_10x_hz
```

At the end of the table, add a trend summary such as:

```
Global trend for cable length:
- Increasing cable length generally increases the pendulum period.
- Increasing cable length generally lowers pendulum frequency.
- Longer cable length may reduce angular rates, but can increase displacement/amplitude envelope.
- Longer cable length increases feed-system/cable length and mass.
- Shorter cable length is structurally/packaging attractive, but may produce faster dynamics.
```

The code should compute the actual numerical trends where possible, for example by comparing the first and last values.

---

## 8. Table 2 — Cable-number sensitivity

Vary:

```python
N_cables_values = [3, 4, 6]
```

Keep all other parameters at baseline.

Required table columns:

```
N_cables
dominant_pendulum_period_s
dominant_torsion_period_s
max_gondola_swing_angle_deg
max_gondola_swing_rate_deg_s
max_gondola_yaw_deg
max_gondola_yaw_rate_deg_s
peak_total_cable_tension_N
peak_per_cable_tension_N
dynamic_static_tension_ratio
maximum_torsional_torque_Nm
required_imu_sampling_10x_hz
```

At the end of the table, add a trend summary such as:

```
Global trend for number of cables:
- Increasing number of cables reduces peak load per cable.
- Total cable mass may increase if individual cable mass is kept constant.
- The effect on pendulum motion may be smaller than cable length or damping.
- The effect on torsion depends on how the cable geometry and stiffness scale with cable count.
```

---

## 9. Table 3 — Attachment-radius sensitivity

Vary:

```python
r_attach_values = [0.5, 1.0, 1.5, 2.0]
```

Keep all other parameters at baseline.

Required table columns:

```
r_attach
dominant_torsion_period_s
dominant_torsion_frequency_hz
max_gondola_yaw_deg
max_gondola_yaw_rate_deg_s
maximum_torsional_torque_Nm
maximum_torsion_element_twist_deg
dominant_pendulum_period_s
max_gondola_swing_angle_deg
required_imu_sampling_10x_hz
```

At the end of the table, add a trend summary such as:

```
Global trend for attachment radius:
- Increasing attachment radius increases torsional stiffness.
- Higher torsional stiffness generally lowers yaw/twist amplitudes.
- Higher torsional stiffness generally increases torsional natural frequency.
- Attachment radius should mainly affect torsion, not pendulum, unless geometry/mass changes are added later.
```

---

## 10. Table 4 — Pendulum damping sensitivity

Vary:

```python
zeta_pendulum_values = [0.001, 0.003, 0.005, 0.01]
```

Keep all other parameters at baseline.

Required table columns:

```
zeta_pendulum
dominant_pendulum_period_s
max_gondola_swing_angle_deg
max_gondola_swing_rate_deg_s
settling_time_pendulum_s
peak_per_cable_tension_N
required_imu_sampling_10x_hz
```

At the end of the table, add a trend summary such as:

```
Global trend for pendulum damping:
- Increasing damping should reduce settling time.
- Damping should not strongly change the natural period in this linear model.
- Damping may reduce peak response after the initial disturbance, depending on how the disturbance is applied.
- If damping strongly improves settling, passive damping is an important design lever.
```

---

## 11. Table 5 — Torsion damping sensitivity

Vary:

```python
zeta_torsion_values = [0.001, 0.003, 0.005, 0.01]
```

Keep all other parameters at baseline.

Required table columns:

```
zeta_torsion
dominant_torsion_period_s
max_gondola_yaw_deg
max_gondola_yaw_rate_deg_s
settling_time_torsion_s
maximum_torsional_torque_Nm
maximum_torsion_element_twist_deg
required_imu_sampling_10x_hz
```

At the end of the table, add a trend summary such as:

```
Global trend for torsion damping:
- Increasing torsion damping should reduce torsional settling time.
- Damping should not strongly change the torsional natural period in this linear model.
- Damping may reduce yaw/yaw-rate peaks after initial excitation.
- If yaw remains high despite damping, attachment radius or inertia may be stronger design levers.
```

---

## 12. Table 6 — Disturbance-level sensitivity

Vary:

```python
disturbance_case = ["low", "medium", "high"]
```

Keep the physical design parameters at baseline.

Required table columns:

```
disturbance_case
pend_initial_angle_deg
pend_initial_rate_deg_s
torsion_initial_twist_deg
torsion_initial_yaw_rate_deg_s
max_gondola_swing_angle_deg
max_gondola_swing_rate_deg_s
max_gondola_yaw_deg
max_gondola_yaw_rate_deg_s
peak_per_cable_tension_N
maximum_torsional_torque_Nm
required_imu_sampling_10x_hz
```

At the end of the table, add a trend summary such as:

```
Global trend for disturbance level:
- Increasing initial angle/rate should increase response amplitude and rates.
- Natural periods should not change in the linear model.
- Cable tension and torsional torque increase with more severe disturbances.
- This sweep shows how sensitive the design is to deployment/post-disturbance initial conditions.
```

---

## 13. Table 7 — Gondola-mass sensitivity

Vary:

```python
m_gondola_values = [300.0, 400.0, 500.0]
```

Keep all other parameters at baseline.

Required table columns:

```
m_gondola
dominant_pendulum_period_s
dominant_torsion_period_s
max_gondola_swing_angle_deg
max_gondola_swing_rate_deg_s
max_gondola_yaw_deg
max_gondola_yaw_rate_deg_s
peak_total_cable_tension_N
peak_per_cable_tension_N
dynamic_static_tension_ratio
maximum_torsional_torque_Nm
required_imu_sampling_10x_hz
```

At the end of the table, add a trend summary such as:

```
Global trend for gondola mass:
- Increasing gondola mass increases static and dynamic cable loads.
- Increasing gondola mass changes the inertia of the suspended system.
- Increasing gondola mass also changes torsional stiffness because the suspended mass appears in the bifilar stiffness expression.
- The effect on rates and periods should be read from the numerical trend table, because mass affects both inertia and stiffness terms.
```

---

## 14. Table 8 — Cable-mass sensitivity

Vary:

```python
m_one_cable_total_values = [0.1, 0.2, 0.5, 1.0]
```

Keep all other parameters at baseline, including `N_cables = 4`.

Required table columns:

```
m_one_cable_total
m_cables_total
dominant_pendulum_period_s
dominant_torsion_period_s
max_gondola_swing_angle_deg
max_gondola_swing_rate_deg_s
max_gondola_yaw_deg
max_gondola_yaw_rate_deg_s
peak_total_cable_tension_N
peak_per_cable_tension_N
dynamic_static_tension_ratio
maximum_torsional_torque_Nm
required_imu_sampling_10x_hz
```

At the end of the table, add a trend summary such as:

```
Global trend for cable mass:
- Increasing cable mass increases the total suspended mass.
- Increasing cable mass generally increases cable tension.
- Cable mass changes the distributed mass in both the pendulum and torsion discretizations.
- If the dynamic outputs are weakly sensitive to cable mass, cable mass can mainly be treated as a structural/mass-budget driver.
```

---

## 15. Table 9 — Gondola-shape / inertia sensitivity

Vary:

```python
gondola_shape_values = [
    "solid_cylinder",
    "cube",
    "solid_sphere",
    "rectangular_box",
]
```

Keep `m_gondola` fixed at the baseline value. For each shape, update the corresponding geometry parameters and recompute:

```
I_gondola_tilt
I_gondola_yaw
representative vertical size / L_gondola
```

Required table columns:

```
gondola_shape
gondola_reference_size
I_gondola_tilt
I_gondola_yaw
dominant_pendulum_period_s
dominant_torsion_period_s
max_gondola_swing_angle_deg
max_gondola_swing_rate_deg_s
max_gondola_yaw_deg
max_gondola_yaw_rate_deg_s
peak_per_cable_tension_N
maximum_torsional_torque_Nm
required_imu_sampling_10x_hz
```

At the end of the table, add a trend summary such as:

```
Global trend for gondola shape:
- Gondola shape affects the dynamics through the tilt and yaw inertias.
- Higher yaw inertia generally changes the torsional period and yaw-rate response.
- Higher tilt inertia can affect pendulum angular response.
- Shape sensitivity should be interpreted carefully because the representative dimensions are still preliminary.
```

Implementation note:

- do not compare shapes using the cylinder radius and length for every case;
- each shape must define its own dimensions;
- recompute `I_gondola_tilt`, `I_gondola_yaw`, and representative vertical size before running the model.

---

## 16. Trend-summary computation

For each table, add a simple automatic trend summary.

For every output column, compare the value at the lowest parameter value and the highest parameter value.

Use labels:

```
increases
decreases
approximately unchanged
non-monotonic
```

Simple implementation idea:

```python
def trend_label(x_values, y_values, rel_tol=0.05):
    y0 = y_values[0]
    y1 = y_values[-1]

    if abs(y1 - y0) <= rel_tol * max(abs(y0), 1e-12):
        return "approximately unchanged"

    diffs = np.diff(y_values)

    if np.all(diffs >= 0):
        return "increases"

    if np.all(diffs <= 0):
        return "decreases"

    return "non-monotonic"
```

Then print a compact trend table:

```
Trend summary for L_susp:
output variable                      trend
dominant_pendulum_period_s           increases
max_gondola_swing_rate_deg_s         decreases
required_imu_sampling_10x_hz         decreases
...
```

This trend summary is one of the most important outputs.

---

## 17. Output files

Save the complete one-at-a-time sweep to:

```
sensitivity_one_at_a_time.csv
```

Also save separate parameter tables if convenient:

```
sensitivity_length.csv
sensitivity_cable_number.csv
sensitivity_attachment_radius.csv
sensitivity_pendulum_damping.csv
sensitivity_torsion_damping.csv
sensitivity_disturbance.csv
sensitivity_gondola_mass.csv
sensitivity_cable_mass.csv
sensitivity_gondola_shape.csv
```

Save trend summaries to:

```
sensitivity_trend_summary.csv
```

A combined sweep can be added later, but it is not the priority.

---

## 18. Simple plots

Plots are useful, but the tables and trend summaries are the priority.

Suggested plots:

- `L_susp` vs dominant pendulum period;
- `L_susp` vs maximum swing rate;
- `L_susp` vs peak per-cable tension;
- `r_attach` vs dominant torsion period;
- `r_attach` vs maximum yaw rate;
- `zeta_pendulum` vs pendulum settling time;
- `zeta_torsion` vs torsion settling time;
- `N_cables` vs peak per-cable tension;
- `m_gondola` vs peak per-cable tension;
- `m_one_cable_total` vs peak per-cable tension;
- gondola shape vs dominant torsion period;
- disturbance case vs maximum swing/yaw response;
- parameter value vs required IMU sampling frequency.

---

## 19. Suggested implementation sequence for Copilot

Implement in this order:

```
1. Wrap the current single-run model into run_single_case(params).
2. Make run_single_case return a result dictionary.
3. Add required IMU sampling frequency output.
4. Add one-at-a-time sweeps.
5. For each parameter sweep, create and print a table.
6. For each parameter sweep, compute and print a trend summary.
7. Save all one-at-a-time results to sensitivity_one_at_a_time.csv.
8. Save the trend summaries to sensitivity_trend_summary.csv.
9. Add simple plots only after the tables work.
```

Do not add feasibility classification for now.

Do not add `overall_status`, `imu_sampling_demand`, or qualitative payload severity labels for now.

---

## 20. Final note for interpretation

The output of this study should mainly be a set of statements like:

```
Increasing cable length increases pendulum period and tends to reduce angular rate.
Increasing attachment radius reduces torsional yaw response and increases torsional frequency.
Increasing damping reduces settling time but does not strongly change natural period.
Increasing cable number reduces per-cable load.
Increasing gondola mass increases cable loads and changes inertia-driven responses.
Increasing cable mass increases total suspended mass and cable loads.
Changing gondola shape changes the response through tilt and yaw inertia.
Increasing disturbance level increases amplitudes, rates, and loads.
```

These statements should be based on the actual numerical trends in the generated tables.