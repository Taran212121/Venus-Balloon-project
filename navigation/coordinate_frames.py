from dataclasses import dataclass




@dataclass
class Position:
    long: float  # [deg]  (-180, 180]
    lat: float   # [deg]  [-90 , 90 ]
    alt: float   # [m]    [ 0  , inf)


