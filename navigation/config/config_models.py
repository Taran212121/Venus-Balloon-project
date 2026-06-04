from dataclasses import dataclass



@dataclass
class SimulationConfig:
    start_time: str
    duration_s: float
    dt_s: float


@dataclass
class BalloonInitialConfig:
    latitude_deg: float
    longitude_deg: float
    altitude_m: float


@dataclass
class BalloonConfig:
    initial: BalloonInitialConfig


@dataclass
class OrbitalElements:
    semi_major_axis_m: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    argument_of_periapsis_deg: float
    true_anomaly_deg: float


@dataclass
class OrbiterConfig:
    initial_orbit: OrbitalElements
    min_elevation_deg: int
    max_elevation_deg: int


@dataclass
class PlanetConfig:
    body: str


@dataclass
class ScenarioMetadata:
    name: str


@dataclass
class ScenarioConfig:
    scenario: ScenarioMetadata
    planet: PlanetConfig
    simulation: SimulationConfig
    balloon: BalloonConfig
    orbiter: OrbiterConfig

    