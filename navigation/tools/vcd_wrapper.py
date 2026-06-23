import sys
import os
from pathlib import Path
import numpy as np

# Crazy shenannigans for importing that god-forsaken VCD
sys.path.insert(
    0,
    r"C:\Users\juliu\OneDrive - Delft University of Technology\Bureaublad\BSc AE Y3\DESIGN SYNTHESIS EXERCISE\VCD2.3\vcd\python"
)
os.add_dll_directory(r"C:\Users\juliu\conda_temp\venus\Library\bin")
from fvcd import vcd, julian


def get_julian_date(year, month, day, hour, minute, second):
    return julian(month, day, year, hour, minute, second)


def sample_vcd(*args, **kwargs):
    return vcd.call_vcd(*args, **kwargs)


if __name__ == "__main__":
    dset = r"C:\Users\juliu\OneDrive - Delft University of Technology\Bureaublad\BSc AE Y3\DESIGN SYNTHESIS EXERCISE\VCD2.3\VCD_DATA\\"
    extvar_keys = np.ones(100, dtype=np.int32)
    # extvar_keys[39] = 1  # Net Solar Flux (SW) received at the top of the atmosphere (W/m2), positive downward
    # extvar_keys[40] = 1  # SW4 net flux at given altitude (W/m2), positive downward
    # extvar_keys[41] = 1  # LW5 net flux at given altitude (W/m2), positive upward

    zon_wind, mer_wind, vert_wind, temp, pres, dens, ext, seed_out, ier = \
    sample_vcd(
        2, 56000.0,
        0.0, 80.0,
        0,  # hires
        1,  # date key
        0.0,  # julian time
        12.0,  # local time
        dset,
        1,
        1,
        0.0,
        0,
        0,
        0,
        extvar_keys
    )
    
    print("Outputs: ")
    print(zon_wind)
    print(mer_wind)
    print(vert_wind)
    print(temp)
    print(pres)
    print(dens)

    print("Extra variables")
    print(ext[39], ext[40], ext[41])
    print(ext)

