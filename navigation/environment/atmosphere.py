import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.interpolate import interp1d


class AtmosphereProfile:
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)
        self.altitude = self.df["altitude"].to_numpy()  # [m]

    def get_column(self, column_name):
        return self.df[column_name].to_numpy()


@dataclass
class AtmosphericState:
    density: float             # [kg/m^3]
    pressure: float            # [Pa]
    temperature: float         # [K]
    wind_velocity: np.ndarray  # in the form (zonal, meridional, vertical), [m/s]


class AtmosphereModel1D:
    def __init__(self, parquet_path, planet_radius):

        self.planet_radius = planet_radius

        profile = AtmosphereProfile(parquet_path)

        altitude = profile.altitude

        self.temperature_interp = interp1d(
            altitude,
            profile.get_column("temperature"),
            bounds_error=False,
            fill_value="extrapolate"
        )

        self.pressure_interp = interp1d(
            altitude,
            profile.get_column("pressure"),
            bounds_error=False,
            fill_value="extrapolate"
        )

        self.density_interp = interp1d(
            altitude,
            profile.get_column("density"),
            bounds_error=False,
            fill_value="extrapolate"
        )

        self.zonal_wind_interp = interp1d(
            altitude,
            profile.get_column("zonal wind"),
            bounds_error=False,
            fill_value="extrapolate"
        )

        self.meridional_wind_interp = interp1d(
            altitude,
            profile.get_column("meridional wind"),
            bounds_error=False,
            fill_value="extrapolate"
        )

        self.vertical_wind_interp = interp1d(
            altitude,
            profile.get_column("vertical wind"),
            bounds_error=False,
            fill_value="extrapolate"
        )

    def altitude_from_position(self, position):

        radius = np.linalg.norm(position)

        return radius - self.planet_radius

    def sample(self, position, t):

        altitude = self.altitude_from_position(position)

        density = float(
            self.density_interp(altitude)
        )

        pressure = float(
            self.pressure_interp(altitude)
        )

        temperature = float(
            self.temperature_interp(altitude)
        )

        zonal = float(
            self.zonal_wind_interp(altitude)
        )

        meridional = float(
            self.meridional_wind_interp(altitude)
        )

        vertical = float(
            self.vertical_wind_interp(altitude)
        )

        return AtmosphericState(
            density=density,
            pressure=pressure,
            temperature=temperature,
            wind_velocity=np.array([
                zonal,
                meridional,
                vertical
            ])
        )
    
    def sample_altitude(self, altitude):
        position = np.array([self.planet_radius + altitude, 0, 0])
        return self.sample(position ,0)

    def density(self, position, t):
        return self.sample(position, t).density

    def wind_velocity(self, position, t):
        return self.sample(position, t).wind_velocity


    