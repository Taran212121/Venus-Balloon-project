from numpy import pi

from dataclasses import dataclass


@dataclass
class VenusModel:
    radius: float = 6_051_800  # [m]
    mu: float = 3.2486E14  #  [N m^2 kg^-1]
    inertial_rot_period: float = -243.0226 # [days] retrograde rotation

    def __post_init__(self):
        self.inertial_rot_period_s = self.inertial_rot_period * 24 * 3600
        self.rotation_rate_rad_s = 2 * pi / self.inertial_rot_period_s

