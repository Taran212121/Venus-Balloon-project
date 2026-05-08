import numpy as np

from config.config_loader import load_config
from environment.venus import VenusModel
from environment.atmosphere import AtmosphereModel

from propagators.balloon_propagator import (
    FollowZonalWindBalloonPropagator
)

from propagators.orbiter_propagator import (
    TwoBodyNewtonPropagator
)

from vehicles.balloon import BalloonState
from analysis.visualisation import plot_orbit_3d, plot_simulation_3d, plot_link_analysis
from analysis.communications import compute_link_metrics

from paths import *

SCENARIO = SCENARIOS_DIR / "base_scenario.yaml"

ATMOSPHERE_FILE = (
    DATA_DIR / "venus_atmosphere.parquet"
)



def main():
    # Load config
    config = load_config(SCENARIO)

    # Environment model
    venus = VenusModel()
    atmosphere = AtmosphereModel(
        parquet_path=ATMOSPHERE_FILE,
        planet_radius=venus.radius
    )

    # Initial balloon state
    balloon_state = BalloonState(
        latitude_deg=config.balloon.initial.latitude_deg,
        longitude_deg=config.balloon.initial.longitude_deg,
        altitude_m=config.balloon.initial.altitude_m
    )

    # Propagators
    balloon_propagator = (
        FollowZonalWindBalloonPropagator(
            venus_model=venus
        )
    )

    orbiter_propagator = (
        TwoBodyNewtonPropagator(
            orbit_config=config.orbiter,
            venus_model=venus
        )
    )

    orbiter_state = (
        orbiter_propagator.initial_state
    )

    # Simulation timing
    dt = config.simulation.dt_s
    duration = config.simulation.duration_s
    n_steps = int(duration / dt)

    # Set up history
    time_history = []
    balloon_history = []
    balloon_history_vci = []
    orbiter_position_history = []
    orbiter_velocity_history = []

    # Simulation loop
    t = 0.0
    for step in range(n_steps):
        if step % 10_000 == 0:
            print(f"Step: {step}")

        # Store histories
        time_history.append(t)

        balloon_history.append(
            (
                balloon_state.latitude_deg,
                balloon_state.longitude_deg,
                balloon_state.altitude_m
            )
        )

        balloon_history_vci.append(
            balloon_state.to_vci(t=t, venus_model=venus)
        )

        orbiter_position_history.append(
            orbiter_state.position.copy()
        )

        orbiter_velocity_history.append(
            orbiter_state.velocity.copy()
        )

        # Propagate balloon
        balloon_state = (
            balloon_propagator.step(
                balloon_state,
                dt,
                atmosphere
            )
        )

        # Propagate orbiter
        orbiter_state = (
            orbiter_propagator.step(
                orbiter_state,
                dt,
                venus
            )
        )

        t += dt

    # Convert history to arrays
    time_history = np.array(time_history)
    balloon_history = np.array(balloon_history)
    balloon_history_vci = np.array(balloon_history_vci)
    orbiter_position_history = np.array(orbiter_position_history)
    orbiter_velocity_history = np.array(orbiter_velocity_history)

    # Diagnostic prints

    print()

    print("Simulation complete")

    print(f"Steps: {n_steps}")

    print()

    print("Final balloon state:")
    print(balloon_state)

    print()

    print("Final orbiter radius [km]:")
    print(
        np.linalg.norm(
            orbiter_state.position
        ) / 1000
    )

    # Plotting
    # plot_orbit_3d(
    #     orbiter_position_history,
    #     venus.radius
    # )

    plot_simulation_3d(
        orbiter_position_history,
        balloon_history_vci,
        venus.radius
    )

    # Link budget
    metrics = compute_link_metrics(
        time_history,
        orbiter_position_history,
        balloon_history_vci
    )

    plot_link_analysis(
        time_history,
        metrics['distances'],
        metrics['elevation_deg'],
        metrics['link_available']
    )


if __name__ == "__main__":
    main()