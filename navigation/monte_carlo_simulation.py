# %% IMPORTS

import numpy as np

from config.config_loader import load_config

from environment.venus import VenusModel
from environment.atmosphere import AtmosphereModelFullVCD

from propagators.balloon_propagator import (
    WindsAndDensityBalloonPropagator,
)

from vehicles.balloon import BalloonState

from analysis.visualisation import (
    plot_balloon_altitude,
    plot_monte_carlo_groundtrack,
)

from paths import *


# %% CONFIG

SCENARIO = SCENARIOS_DIR / "base_scenario.yaml"

N_MONTE_CARLO_RUNS = 20

OUTPUT_FILE = (
    OUT_DIR / "mc_300day_20balloons.npz"
)


# %% MONTE CARLO

def run_monte_carlo(
    config,
    venus,
    n_runs
):
    """
    Run a Monte Carlo campaign using VCD EOF perturbations.

    Each run uses a different perturb_seed while
    keeping all other atmospheric parameters fixed.
    """

    dt = config.simulation.dt_s
    duration = config.simulation.duration_s

    n_steps = int(duration / dt)

    times = np.arange(n_steps) * dt

    altitude_history = np.zeros(
        (n_runs, n_steps),
        dtype=np.float32
    )

    latitude_history = np.zeros_like(
        altitude_history
    )

    longitude_history = np.zeros_like(
        altitude_history
    )

    for run_idx in range(n_runs):

        print(
            f"Monte Carlo run "
            f"{run_idx + 1}/{n_runs}"
        )

        # ----------------------------------
        # Atmosphere realization
        # ----------------------------------

        vcd_config = {
            "start_date": config.simulation.start_time,

            "hires_key": 0,

            "EUV_scena": 1,
            "albedo_scena": 1,

            # EOF perturbations
            "perturb_key": 2,
            "perturb_seed": run_idx + 1,

            # Only used for GW perturbations
            "perturb_gw_length": 0.0,
        }

        atmosphere = AtmosphereModelFullVCD(
            planet_radius=venus.radius,
            vcd_config=vcd_config
        )

        # ----------------------------------
        # Initial state
        # ----------------------------------

        balloon_state = BalloonState(
            latitude_deg=config.balloon.initial.latitude_deg,
            longitude_deg=config.balloon.initial.longitude_deg,
            altitude_m=config.balloon.initial.altitude_m
        )

        propagator = (
            WindsAndDensityBalloonPropagator(
                venus_model=venus,
                envelope_density=1.2
            )
        )

        # ----------------------------------
        # Propagation
        # ----------------------------------

        t = 0.0

        for step in range(n_steps):

            altitude_history[
                run_idx,
                step
            ] = balloon_state.altitude_m

            latitude_history[
                run_idx,
                step
            ] = balloon_state.latitude_deg

            longitude_history[
                run_idx,
                step
            ] = balloon_state.longitude_deg

            balloon_state = (
                propagator.step(
                    balloon_state,
                    dt,
                    atmosphere
                )
            )

            t += dt

        print(
            f"  Final altitude: "
            f"{balloon_state.altitude_m:.1f} m"
        )

    return {
        "times": times,
        "altitude": altitude_history,
        "latitude": latitude_history,
        "longitude": longitude_history,
    }


# %% ANALYSIS

def analyze_campaign(results):

    altitude = results["altitude"]

    median_altitude = np.median(
        altitude,
        axis=0
    )

    p05_altitude = np.percentile(
        altitude,
        5,
        axis=0
    )

    p95_altitude = np.percentile(
        altitude,
        95,
        axis=0
    )

    final_altitude = altitude[:, -1]

    print()
    print("========== Monte Carlo Summary ==========")
    print()

    print(
        f"Runs: {altitude.shape[0]}"
    )

    print()

    print(
        f"Final altitude mean: "
        f"{np.mean(final_altitude):.1f} m"
    )

    print(
        f"Final altitude std: "
        f"{np.std(final_altitude):.1f} m"
    )

    print(
        f"Final altitude min: "
        f"{np.min(final_altitude):.1f} m"
    )

    print(
        f"Final altitude max: "
        f"{np.max(final_altitude):.1f} m"
    )

    return (
        median_altitude,
        p05_altitude,
        p95_altitude
    )


# %% VISUALIZATION

def plot_altitude_envelope(
    times,
    altitude_history
):

    import matplotlib.pyplot as plt

    times_days = times / 86400.0

    median_altitude = np.median(
        altitude_history,
        axis=0
    )

    p05_altitude = np.percentile(
        altitude_history,
        5,
        axis=0
    )

    p95_altitude = np.percentile(
        altitude_history,
        95,
        axis=0
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.fill_between(
        times_days,
        p05_altitude,
        p95_altitude,
        alpha=0.3,
        label=r"5-95 % envelope"
    )

    plt.plot(
        times_days,
        median_altitude,
        linewidth=2,
        label="Median"
    )

    plt.xlabel(
        "Time [Earth days]"
    )

    plt.ylabel(
        "Altitude [m]"
    )

    plt.title(
        "Monte Carlo Balloon Altitude Envelope"
    )

    plt.grid(True)

    plt.legend()

    plt.show()


# %% MAIN

def main():

    config = load_config(
        SCENARIO
    )

    venus = VenusModel()

    results = run_monte_carlo(
        config=config,
        venus=venus,
        n_runs=N_MONTE_CARLO_RUNS
    )

    analyze_campaign(
        results
    )

    np.savez_compressed(
        OUTPUT_FILE,
        **results
    )

    print()
    print(
        f"Saved campaign to:\n"
        f"{OUTPUT_FILE}"
    )

    plot_altitude_envelope(
        results["times"],
        results["altitude"]
    )

    plot_monte_carlo_groundtrack(
        latitudes_deg=results['latitude'],
        longitudes_deg=results['longitude'],
        mode='trajectories',
        alpha=0.3
    )

def recover_data():
    RECOVER_PATH = OUT_DIR / "mc_300day_20balloons.npz"
    results = np.load(RECOVER_PATH)

    plot_altitude_envelope(
        results["times"],
        results["altitude"]
    )

    plot_monte_carlo_groundtrack(
        latitudes_deg=results['latitude'],
        longitudes_deg=results['longitude'],
        mode='density',
        alpha=0.3
    )

# %% RUN

if __name__ == "__main__":
    
    # main()
    recover_data()