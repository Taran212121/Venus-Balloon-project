from abc import ABC, abstractmethod
import numpy as np


from vehicles.balloon import BalloonState
from environment.atmosphere import AtmosphereModel


class BalloonPropagator(ABC):
    def __init__(self, *args, **kwargs):
        ...


    @abstractmethod
    def step(self, state, dt, env):
        ...

    
class ScriptedBalloonPropagator(BalloonPropagator):
    def __init__(self, scripted_path):
        # process scripted path
        ...

    def step(self, state, dt, env):
        # execute scripted path
        ...


class FollowZonalWindBalloonPropagator(BalloonPropagator):
    def __init__(self, venus_model):
        self.planet_radius = venus_model.radius

    def step(self, state, dt, env):
        """Balloon follows zonal wind along constant latitude."""

        latitude_rad = np.radians(state.latitude_deg)

        local_radius = (
            self.planet_radius
            + state.altitude_m
        ) * np.cos(latitude_rad)

        atmospheric_state = env.sample_altitude(state.altitude_m)

        zonal_velocity = atmospheric_state.wind_velocity[0]

        # [rad/s]
        longitude_rate = zonal_velocity / local_radius

        delta_longitude_deg = np.degrees(
            longitude_rate * dt
        )

        longitude_next = (
            state.longitude_deg
            + delta_longitude_deg
        )

        # Wrap longitude
        longitude_next = (
            (longitude_next + 180.0) % 360.0
        ) - 180.0

        return BalloonState(
            latitude_deg=state.latitude_deg,
            longitude_deg=longitude_next,
            altitude_m=state.altitude_m
        )
    

class FollowAllWindsBalloonPropagator(BalloonPropagator):
    def __init__(self, venus_model):
        self.planet_radius = venus_model.radius

    def step(self, state: BalloonState, dt: float, env: AtmosphereModel) -> BalloonState:
        """Balloon follows zonal, meridional and vertical winds"""

        latitude_rad = np.radians(state.latitude_deg)

        local_radius = (
            self.planet_radius
            + state.altitude_m
        ) * np.cos(latitude_rad)

        atmospheric_state = env.sample_lat_lon_alt(
            state.latitude_deg,
            state.longitude_deg,
            state.altitude_m,
            t=0
        )

        zonal_velocity = atmospheric_state.wind_velocity[0]
        meridional_velocity = atmospheric_state.wind_velocity[1]
        vertical_velocity = atmospheric_state.wind_velocity[2]

        vertical_velocity = 0  # NOTE: Velocity override as it seems to be diverging us to SPACE

        longitude_rate = zonal_velocity / local_radius # [rad/s]
        delta_longitude_deg = np.degrees(longitude_rate * dt)
        longitude_next = state.longitude_deg + delta_longitude_deg

        # Wrap longitude
        longitude_next = (
            (longitude_next + 180.0) % 360.0
        ) - 180.0

        latitude_rate = meridional_velocity / local_radius
        delta_latitude_deg = np.degrees(latitude_rate * dt)
        latitude_next = state.latitude_deg + delta_latitude_deg

        delta_altitude = vertical_velocity * dt
        altitude_next = state.altitude_m + delta_altitude

        return BalloonState(
            latitude_deg=latitude_next,
            longitude_deg=longitude_next,
            altitude_m=altitude_next
        )
    

class WindsAndDensityBalloonPropagator(BalloonPropagator):
    def __init__(self, venus_model, envelope_density):
        self.planet_radius = venus_model.radius
        self.rho = envelope_density

    def step(self, state: BalloonState, dt: float, env: AtmosphereModel) -> BalloonState:
        """Balloon follows zonal and meridional winds. Altitude varies to balance density"""

        latitude_rad = np.radians(state.latitude_deg)

        local_radius = (
            self.planet_radius
            + state.altitude_m
        ) * np.cos(latitude_rad)

        atmospheric_state = env.sample_lat_lon_alt(
            state.latitude_deg,
            state.longitude_deg,
            state.altitude_m,
            t=0
        )

        zonal_velocity = atmospheric_state.wind_velocity[0]
        meridional_velocity = atmospheric_state.wind_velocity[1]

        # Density adjustment contral law
        rho_env = atmospheric_state.density
        velocity_gain = 2 # [m/s]
        vertical_velocity = ((rho_env - self.rho) / rho_env) * velocity_gain

        longitude_rate = zonal_velocity / local_radius # [rad/s]
        delta_longitude_deg = np.degrees(longitude_rate * dt)
        longitude_next = state.longitude_deg + delta_longitude_deg

        # Wrap longitude
        longitude_next = (
            (longitude_next + 180.0) % 360.0
        ) - 180.0

        latitude_rate = meridional_velocity / local_radius
        delta_latitude_deg = np.degrees(latitude_rate * dt)
        latitude_next = state.latitude_deg + delta_latitude_deg

        delta_altitude = vertical_velocity * dt
        altitude_next = state.altitude_m + delta_altitude

        return BalloonState(
            latitude_deg=latitude_next,
            longitude_deg=longitude_next,
            altitude_m=altitude_next
        )