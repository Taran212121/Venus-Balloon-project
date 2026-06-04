from abc import ABC, abstractmethod
import numpy as np


from vehicles.balloon import BalloonState
from environment.atmosphere import AtmosphereModel, AtmosphericState


class BalloonPropagator(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, state: BalloonState, dt: float, env: AtmosphereModel, t: float = 0.0):
        pass


class VerticalController(ABC):
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def vertical_velocity(self, state: BalloonState, atmospheric_state: AtmosphericState, t: float) -> float:
        """Compute the commanded vertical velocity [m/s]"""
        pass


class WindFollowingBalloonPropagator(BalloonPropagator):
    def __init__(self, venus_model, vertical_controller):
        self.planet_radius = venus_model.radius
        self.vertical_controller = vertical_controller

    def step(self, state, dt, env, t=0):
        
        atmospheric_state = env.sample_lat_lon_alt(
            state.longitude_deg,
            state.latitude_deg,
            state.altitude_m,
            t
        )
        
        zonal_velocity = atmospheric_state.wind_velocity[0]

        meridional_velocity = atmospheric_state.wind_velocity[1]
        
        vertical_velocity = self.vertical_controller.vertical_velocity(
            state,
            atmospheric_state,
            t
        )

        latitude_rad = np.radians(state.latitude_deg)
        local_radius = (self.planet_radius + state.altitude_m) * np.cos(latitude_rad)

        # Latitude
        latitude_rate = meridional_velocity / local_radius
        delta_latitude_deg = np.degrees(latitude_rate * dt)
        latitude_next = state.latitude_deg + delta_latitude_deg

        latitude_next = np.clip(latitude_next, -90, 90)

        # Longitude
        longitude_rate = zonal_velocity / local_radius # [rad/s]
        delta_longitude_deg = np.degrees(longitude_rate * dt)
        longitude_next = state.longitude_deg + delta_longitude_deg

        longitude_next = ((longitude_next + 180.0) % 360.0) - 180.0

        # Altitude
        delta_altitude = vertical_velocity * dt
        altitude_next = state.altitude_m + delta_altitude

        return BalloonState(
            latitude_deg=latitude_next,
            longitude_deg=longitude_next,
            altitude_m=altitude_next
        )
    

class ConstantAltitudeController(VerticalController):

    def vertical_velocity(self, state: BalloonState, atmospheric_state: AtmosphericState, t: float) -> float:
        return 0.0
    

class VerticalWindController(VerticalController):

    def vertical_velocity(self, state: BalloonState, atmospheric_state: AtmosphericState, t: float) -> float:
        return atmospheric_state.wind_velocity[2]
    

class DensityTrackingController(VerticalController):
    def __init__(self, envelope_density, velocity_gain=2.0):
        self.envelope_density = envelope_density
        self.velocity_gain = velocity_gain

    def vertical_velocity(self, state: BalloonState, atmospheric_state: AtmosphericState, t: float) -> float:
        rho_env = atmospheric_state.density
        return ((rho_env - self.envelope_density) / rho_env) * self.velocity_gain


class ScriptedAltitudeController(VerticalController):
    def __init__(self, altitude_function, gain=0.01, max_vertical_velocity=None):
        self.altitude_function = altitude_function
        self.gain = gain
        self.max_vertical_velocity = max_vertical_velocity

    def vertical_velocity(self, state: BalloonState, atmospheric_state: AtmosphericState, t: float) -> float:
        
        target = self.altitude_function(t)
        error = target - state.altitude_m
        velocity = self.gain * error

        if self.max_vertical_velocity is not None:
            velocity = np.clip(velocity, -self.max_vertical_velocity, self.max_vertical_velocity)

        return velocity


# class FollowZonalWindBalloonPropagator(BalloonPropagator):
#     def __init__(self, venus_model):
#         self.planet_radius = venus_model.radius

#     def step(self, state, dt, env):
#         """Balloon follows zonal wind along constant latitude."""

#         latitude_rad = np.radians(state.latitude_deg)

#         local_radius = (
#             self.planet_radius
#             + state.altitude_m
#         ) * np.cos(latitude_rad)

#         atmospheric_state = env.sample_altitude(state.altitude_m)

#         zonal_velocity = atmospheric_state.wind_velocity[0]

#         # [rad/s]
#         longitude_rate = zonal_velocity / local_radius

#         delta_longitude_deg = np.degrees(
#             longitude_rate * dt
#         )

#         longitude_next = (
#             state.longitude_deg
#             + delta_longitude_deg
#         )

#         # Wrap longitude
#         longitude_next = (
#             (longitude_next + 180.0) % 360.0
#         ) - 180.0

#         return BalloonState(
#             latitude_deg=state.latitude_deg,
#             longitude_deg=longitude_next,
#             altitude_m=state.altitude_m
#         )
    

# class FollowAllWindsBalloonPropagator(BalloonPropagator):
#     def __init__(self, venus_model):
#         self.planet_radius = venus_model.radius

#     def step(self, state: BalloonState, dt: float, env: AtmosphereModel) -> BalloonState:
#         """Balloon follows zonal, meridional and vertical winds"""

#         latitude_rad = np.radians(state.latitude_deg)

#         local_radius = (
#             self.planet_radius
#             + state.altitude_m
#         ) * np.cos(latitude_rad)

#         atmospheric_state = env.sample_lat_lon_alt(
#             state.latitude_deg,
#             state.longitude_deg,
#             state.altitude_m,
#             t=0
#         )

#         zonal_velocity = atmospheric_state.wind_velocity[0]
#         meridional_velocity = atmospheric_state.wind_velocity[1]
#         vertical_velocity = atmospheric_state.wind_velocity[2]

#         longitude_rate = zonal_velocity / local_radius # [rad/s]
#         delta_longitude_deg = np.degrees(longitude_rate * dt)
#         longitude_next = state.longitude_deg + delta_longitude_deg

#         # Wrap longitude
#         longitude_next = (
#             (longitude_next + 180.0) % 360.0
#         ) - 180.0

#         latitude_rate = meridional_velocity / local_radius
#         delta_latitude_deg = np.degrees(latitude_rate * dt)
#         latitude_next = state.latitude_deg + delta_latitude_deg

#         delta_altitude = vertical_velocity * dt
#         altitude_next = state.altitude_m + delta_altitude

#         return BalloonState(
#             latitude_deg=latitude_next,
#             longitude_deg=longitude_next,
#             altitude_m=altitude_next
#         )
    

# class WindsAndDensityBalloonPropagator(BalloonPropagator):
#     def __init__(self, venus_model, envelope_density):
#         self.planet_radius = venus_model.radius
#         self.rho = envelope_density

#     def step(self, state: BalloonState, dt: float, env: AtmosphereModel) -> BalloonState:
#         """Balloon follows zonal and meridional winds. Altitude varies to balance density"""

#         latitude_rad = np.radians(state.latitude_deg)

#         local_radius = (
#             self.planet_radius
#             + state.altitude_m
#         ) * np.cos(latitude_rad)

#         atmospheric_state = env.sample_lat_lon_alt(
#             state.latitude_deg,
#             state.longitude_deg,
#             state.altitude_m,
#             t=0
#         )

#         zonal_velocity = atmospheric_state.wind_velocity[0]
#         meridional_velocity = atmospheric_state.wind_velocity[1]

#         # Density adjustment contral law
#         rho_env = atmospheric_state.density
#         velocity_gain = 2 # [m/s]
#         vertical_velocity = ((rho_env - self.rho) / rho_env) * velocity_gain

#         longitude_rate = zonal_velocity / local_radius # [rad/s]
#         delta_longitude_deg = np.degrees(longitude_rate * dt)
#         longitude_next = state.longitude_deg + delta_longitude_deg

#         # Wrap longitude
#         longitude_next = (
#             (longitude_next + 180.0) % 360.0
#         ) - 180.0

#         latitude_rate = meridional_velocity / local_radius
#         delta_latitude_deg = np.degrees(latitude_rate * dt)
#         latitude_next = state.latitude_deg + delta_latitude_deg

#         # Saveguard latitude
#         latitude_next = np.clip(latitude_next, -90, 90)

#         delta_altitude = vertical_velocity * dt
#         altitude_next = state.altitude_m + delta_altitude

#         return BalloonState(
#             latitude_deg=latitude_next,
#             longitude_deg=longitude_next,
#             altitude_m=altitude_next
#         )