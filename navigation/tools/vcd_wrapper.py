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
    extvar_keys = np.zeros(100, dtype=np.int32)

    zon_wind, mer_wind, vert_wind, temp, pres, dens, ext, seed_out, ier = \
    sample_vcd(
        2, 50000.0,
        0.0, 0.0,
        0,
        1,
        0.0,
        0.0,
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
