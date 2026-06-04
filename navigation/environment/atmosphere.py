import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RegularGridInterpolator

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from coordinate_frames import cartesian_to_geodetic


@dataclass
class AtmosphericState:
    density: float             # [kg/m^3]
    pressure: float            # [Pa]
    temperature: float         # [K]
    wind_velocity: np.ndarray  # in the form (zonal, meridional, vertical), [m/s]


class AtmosphereModel(ABC):

    def __init__(self, planet_radius):
        self.planet_radius = planet_radius

    def altitude_from_position(self, position):
        return np.linalg.norm(position) - self.planet_radius

    @abstractmethod
    def sample(self, position: np.ndarray, t: float) -> AtmosphericState:
        pass
    
    @abstractmethod
    def sample_lat_lon_alt(self, lat, lon, alt, t) -> AtmosphericState:
        pass

    def density(self, position: np.ndarray, t: float) -> float:
        return self.sample(position, t).density

    def wind_velocity(self, position: np.ndarray, t: float) -> np.ndarray:
        return self.sample(position, t).wind_velocity
    

class AtmosphereProfile:
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)
        self.altitude = self.df["altitude"].to_numpy()  # [m]

    def get_column(self, column_name):
        return self.df[column_name].to_numpy()


class AtmosphereModel1D(AtmosphereModel):
    def __init__(self, parquet_path, planet_radius):
        super().__init__(planet_radius)

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
    
    def sample_lat_lon_alt(self, lat, lon, alt, t) -> AtmosphericState:
        return self.sample_altitude(alt)


class AtmosphereModelSurrogateVCD(AtmosphereModel):

    def __init__(self, npz_path, planet_radius):

        super().__init__(planet_radius)

        data = np.load(npz_path)
        self.data = data

        self.lat_grid = data["lat"]
        self.lon_grid = data["lon"]
        self.z_grid = data["z"]

        self.temperature_interp = RegularGridInterpolator(
            (self.lat_grid, self.lon_grid, self.z_grid),
            data["T"],
            bounds_error=False,
            fill_value=None
        )

        self.pressure_interp = RegularGridInterpolator(
            (self.lat_grid, self.lon_grid, self.z_grid),
            data["P"],
            bounds_error=False,
            fill_value=None
        )

        self.density_interp = RegularGridInterpolator(
            (self.lat_grid, self.lon_grid, self.z_grid),
            data["rho"],
            bounds_error=False,
            fill_value=None
        )

        self.zonal_wind_interp = RegularGridInterpolator(
            (self.lat_grid, self.lon_grid, self.z_grid),
            data["U"],
            bounds_error=False,
            fill_value=None
        )

        self.meridional_wind_interp = RegularGridInterpolator(
            (self.lat_grid, self.lon_grid, self.z_grid),
            data["V"],
            bounds_error=False,
            fill_value=None
        )

        self.vertical_wind_interp = RegularGridInterpolator(
            (self.lat_grid, self.lon_grid, self.z_grid),
            data["W"],
            bounds_error=False,
            fill_value=None
        )

    def sample(self, position, t):
        lat, lon, alt = cartesian_to_geodetic(position, self.planet_radius)
        return self.sample_lat_lon_alt(lat, lon, alt, t)
    
    def sample_lat_lon_alt(self, lat, lon, alt, t) -> AtmosphericState:
        point = np.array([  # Pure spherical coordinates
            lat,
            lon,
            alt / 1000  # To follow the interpolating grid in kilometers
        ])

        density     = float(self.density_interp(point))
        pressure    = float(self.pressure_interp(point))
        temperature = float(self.temperature_interp(point))
        zonal       = float(self.zonal_wind_interp(point))
        meridional  = float(self.meridional_wind_interp(point))
        vertical    = float(self.vertical_wind_interp(point))

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


class AtmosphereModelFullVCD(AtmosphereModel):
    def __init__(self, vcd_config, planet_radius):
        super().__init__(planet_radius)

        from tools.vcd_wrapper import sample_vcd, get_julian_date
        self.sample_vcd = sample_vcd
        self.get_julian_date = get_julian_date

        # Setup julian date
        start_date = vcd_config['start_date']
        self.start_datetime = datetime.fromisoformat(start_date.replace("Z", "+00:00"))

        ier, self.start_julian_date = self.get_julian_date(
            self.start_datetime.year,
            self.start_datetime.month,
            self.start_datetime.day,
            self.start_datetime.hour,
            self.start_datetime.minute,
            self.start_datetime.second
        )

        # Prepare vcd config
        self.z_key = 2  # Altitude above Venus reference sphere 6051848 [m]
        # z
        # lon
        # lat
        self.hires_key = vcd_config['hires_key']
        self.date_key = 0  # Use Julian date
        # juliandate
        self.localtime = 0.0       
        self.dset = r"C:\Users\juliu\OneDrive - Delft University of Technology\Bureaublad\BSc AE Y3\DESIGN SYNTHESIS EXERCISE\VCD2.3\VCD_DATA\\"
        self.EUV_scena = vcd_config['EUV_scena']
        self.albedo_scena = vcd_config['albedo_scena']
        self.varE107 = 0.0  # Unused by default (EUV_scena != 5)
        self.perturb_key = vcd_config['perturb_key']
        self.perturb_seed = vcd_config['perturb_seed']
        self.perturb_gw_length = vcd_config['perturb_gw_length']
        self.extvar_keys = np.zeros(100, dtype=np.int32)  # No extra variables requested by default

    def sample(self, position: np.ndarray, t: float) -> AtmosphericState:
        lat, lon, alt = cartesian_to_geodetic(position, self.planet_radius)
        return self.sample_lat_lon_alt(lat, lon, alt, t)
    
    def sample_lat_lon_alt(self, lat, lon, alt, t) -> AtmosphericState:

        julian_date = self.start_julian_date + t / 86400.0

        zon_wind, mer_wind, vert_wind, temp, pres, dens, ext, seed_out, ier = \
        self.sample_vcd(
            self.z_key,
            alt, lon, lat,
            self.hires_key,
            self.date_key,
            julian_date,
            self.localtime,
            self.dset,
            self.EUV_scena,
            self.albedo_scena,
            self.varE107,
            self.perturb_key,
            self.perturb_seed,
            self.perturb_gw_length,
            self.extvar_keys,
        )

        # NOTE: Very risky thing happening here, the document declares to keep the seed constant
        # self.perturb_seed = seed_out

        if ier != 0:
            raise RuntimeError(f"VCD failed at lat={lat}, lon={lon}, alt={alt}, ier={ier}")

        return AtmosphericState(
            density=dens,
            pressure=pres,
            temperature=temp,
            wind_velocity=np.array([zon_wind, mer_wind, vert_wind])
        )

