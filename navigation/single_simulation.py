# %% IMPORTS
import numpy as np

from config.config_loader import load_config
from environment.venus import VenusModel
from environment.atmosphere import (
    AtmosphereModel1D,
    AtmosphereModelSurrogateVCD,
    AtmosphereModelFullVCD
)

from propagators.balloon_propagator import (
    WindFollowingBalloonPropagator,
    ConstantAltitudeController,
    VerticalWindController,
    DensityTrackingController,
    ScriptedAltitudeController
)

from propagators.orbiter_propagator import (
    TwoBodyNewtonPropagator
)

from vehicles.balloon import BalloonState
from analysis.visualisation import (
    plot_link_track,
    plot_orbit_3d,
    plot_simulation_3d,
    compute_groundtrack,
    plot_groundtrack,
    plot_combined_groundtrack,
    plot_balloon_altitude
)

from tools.altitude_functions import

from analysis.communications import compute_link_metrics, generate_link_report

from paths import *

# %% CONFIG VARIABLES
SCENARIO = SCENARIOS_DIR / "single_scenario.yaml"

ATMOSPHERE_FILE = (
    DATA_DIR / "venus_atmosphere.parquet"
)

VCD_ARCHIVE_FILE = (
    DATA_DIR / "vcd_climatology_40km_70km.npz"
)

VISUALIZATIONS = {

    # 3D plots
    "orbit_3d": False,
    "simulation_3d": True,

    # Ground tracks
    "groundtrack_balloon": True,
    "groundtrack_orbiter": False,
    "groundtrack_combined": False,

    # Balloon tracking
    "balloon_altitude": True,

    # Communications
    "link_plot": False,
    "link_analysis": False,
}

# VCD config
vcd_config = {
    'hires_key': 0,
    'EUV_scena': 1,
    'albedo_scena': 1,
    'perturb_key': 2,
    'perturb_seed': 42
}
vcd_config['perturb_gw_length'] = 10000.0 if vcd_config['perturb_key'] in (1, 3) else 0 



# %% MAIN
def main():
    # Load config
    config = load_config(SCENARIO)

    # Environment model
    venus = VenusModel()
    # atmosphere = AtmosphereModel1D(
    #     parquet_path=ATMOSPHERE_FILE,
    #     planet_radius=venus.radius
    # )
    # atmosphere = AtmosphereModelSurrogateVCD(
    #     npz_path=VCD_ARCHIVE_FILE,
    #     planet_radius=venus.radius
    # )

    vcd_config['start_date'] = config.simulation.start_time
    atmosphere = AtmosphereModelFullVCD(
        vcd_config=vcd_config,
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
        WindFollowingBalloonPropagator(
            venus_model=venus,
            vertical_controller=DensityTrackingController(envelope_density=0.95)  # [kg/m^3]
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

    # Debug
    # print(atmosphere.sample_lat_lon_alt(0,0,50000,0))
    # print(atmosphere.data["rho"])

    # Simulation loop
    t = 0.0
    for step in range(n_steps):
        if step % 5_000 == 0:
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
    print("Orbital period [h]: ")
    period = np.sqrt(
        (4 * np.pi**2 * config.orbiter.initial_orbit.semi_major_axis_m**3) / venus.mu
    )
    print(period / 3600)

    # Analysis
    metrics = compute_link_metrics(
        time_history,
        orbiter_position_history,
        balloon_history_vci,
        min_elevation_deg=config.orbiter.min_elevation_deg,
        max_elevation_deg=config.orbiter.max_elevation_deg,
    )

    # Plotting
    if VISUALIZATIONS["orbit_3d"]:

        plot_orbit_3d(
            orbiter_position_history,
            venus.radius
        )

    if VISUALIZATIONS["simulation_3d"]:

        plot_simulation_3d(
            orbiter_position_history,
            balloon_history_vci,
            venus.radius
        )

    if VISUALIZATIONS["link_analysis"]:
        
        generate_link_report(
            time_history,
            orbiter_position_history,
            balloon_history_vci,
            scenario_name=config.scenario.name,
            min_elevation_deg=config.orbiter.min_elevation_deg,
            max_elevation_deg=config.orbiter.max_elevation_deg,
        )

    if VISUALIZATIONS["link_plot"]:

        plot_link_track(
            time_history,
            metrics["distances"],
            metrics["elevation_deg"],
            metrics["link_available"],
            min_elevation_deg=config.orbiter.min_elevation_deg,
            max_elevation_deg=config.orbiter.max_elevation_deg,
        )

    if VISUALIZATIONS["groundtrack_balloon"]:

        plot_groundtrack(
            latitudes_deg=balloon_history[:, 0],
            longitudes_deg=balloon_history[:, 1],
            label="Balloon",
            color="magenta"
        )

    if VISUALIZATIONS["groundtrack_orbiter"]:

        orbiter_lat_deg, orbiter_lon_deg = (
            compute_groundtrack(
                orbiter_position_history,
                time_history,
                venus
            )
        )

        plot_groundtrack(
            latitudes_deg=orbiter_lat_deg,
            longitudes_deg=orbiter_lon_deg,
            label="Orbiter",
            color="royalblue"
        )

    if VISUALIZATIONS["groundtrack_combined"]:

        orbiter_lat_deg, orbiter_lon_deg = (
            compute_groundtrack(
                orbiter_position_history,
                time_history,
                venus
            )
        )

        plot_combined_groundtrack(
            balloon_latitudes_deg=balloon_history[:, 0],
            balloon_longitudes_deg=balloon_history[:, 1],
            orbiter_latitudes_deg=orbiter_lat_deg,
            orbiter_longitudes_deg=orbiter_lon_deg
        )

    if VISUALIZATIONS["balloon_altitude"]:

        plot_balloon_altitude(
            balloon_positions=balloon_history,
            times=time_history,
            planet_radius=venus.radius,
            target_altitude=55_000,
            tolerance=15_000
        )

# %% RUN
if __name__ == "__main__":
    main()