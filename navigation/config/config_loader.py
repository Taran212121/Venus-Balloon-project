import yaml
from pathlib import Path

from config.config_models import (
    ScenarioConfig,
    ScenarioMetadata,
    PlanetConfig,
    SimulationConfig,
    BalloonConfig,
    BalloonInitialConfig,
    OrbiterConfig,
    OrbitalElements,
)


def load_config(path: Path) -> ScenarioConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return ScenarioConfig(
        scenario=ScenarioMetadata(
            name=data["scenario"]["name"]
        ),

        planet=PlanetConfig(
            body=data["planet"]["body"]
        ),

        simulation=SimulationConfig(
            start_time=data["simulation"]["start_time"],
            duration_s=data["simulation"]["duration_days"] * 24 * 3600,
            dt_s=data["simulation"]["dt_s"],
        ),

        balloon=BalloonConfig(
            initial=BalloonInitialConfig(
                latitude_deg=data["balloon"]["initial"]["latitude_deg"],
                longitude_deg=data["balloon"]["initial"]["longitude_deg"],
                altitude_m=data["balloon"]["initial"]["altitude_m"],
            )
        ),

        orbiter=OrbiterConfig(
            initial_orbit=OrbitalElements(
                semi_major_axis_m=data["orbiter"]["initial_orbit"]["semi_major_axis_m"],
                eccentricity=data["orbiter"]["initial_orbit"]["eccentricity"],
                inclination_deg=data["orbiter"]["initial_orbit"]["inclination_deg"],
                raan_deg=data["orbiter"]["initial_orbit"]["raan_deg"],
                argument_of_periapsis_deg=data["orbiter"]["initial_orbit"]["argument_of_periapsis_deg"],
                true_anomaly_deg=data["orbiter"]["initial_orbit"]["true_anomaly_deg"],
            )
        ),
    )

