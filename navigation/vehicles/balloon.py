from dataclasses import dataclass
import numpy as np

from coordinate_frames import geodetic_to_cartesian, vcf_to_vci

@dataclass
class BalloonState:

    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    time_s: float = 0.0 

    def to_vci(self, t, venus_model) -> np.ndarray:
        vcf = geodetic_to_cartesian(
            self.latitude_deg,
            self.longitude_deg,
            self.altitude_m,
            venus_model.radius
        )

        vci = vcf_to_vci(vcf, t, venus_model)
        
        return vci