
import numpy as np


from tools.vcd_wrapper import get_julian_date, sample_vcd
from paths import DATA_DIR


def main():

    # Grid
    lat_grid = np.arange(-89, 90, 2)
    lon_grid = np.arange(0, 360, 4)
    z_grid   = np.arange(40, 71, 1)

    N_lat = len(lat_grid)
    N_lon = len(lon_grid)
    N_z   = len(z_grid)

    # Output arrays
    U = np.zeros((N_lat, N_lon, N_z), dtype=np.float32)
    V = np.zeros_like(U)
    W = np.zeros_like(U)
    T = np.zeros_like(U)
    P = np.zeros_like(U)
    rho = np.zeros_like(U)

    # Fixed inputs
    z_key = 2
    date_key = 1
    localtime = 12.0
    juliandate = 0.0

    dset = r"C:\Users\juliu\OneDrive - Delft University of Technology\Bureaublad\BSc AE Y3\DESIGN SYNTHESIS EXERCISE\VCD2.3\VCD_DATA\\"

    EUV_scena = 1
    albedo_scena = 1
    varE107 = 0.0

    perturb_key = 0
    perturb_seed = 0
    perturb_gw_length = 0.0

    extvar_keys = np.zeros(100, dtype=np.int32)

    # Main sampling loop
    for j, lon in enumerate(lon_grid):
        print(f"Computing longitude {4*j}")
        for i, lat in enumerate(lat_grid):
                for k, z in enumerate(z_grid):

                    zon_wind, mer_wind, vert_wind, temp, pres, dens, ext, seed_out, ier = \
                        sample_vcd(
                            z_key, float(z*1000),
                            float(lon), float(lat),
                            0,
                            date_key,
                            juliandate,
                            localtime,
                            dset,
                            EUV_scena,
                            albedo_scena,
                            varE107,
                            perturb_key,
                            perturb_seed,
                            perturb_gw_length,
                            extvar_keys
                        )
                    
                    if i == 0 and j == 0 and k == 0:
                        print("DEBUG SAMPLE:")
                        print(lat_grid[i], lon_grid[j], z_grid[k])
                        print(zon_wind, mer_wind, vert_wind)
                        print(temp, pres, dens)
                        print("ier:", ier)

                    if ier != 0:
                        raise RuntimeError(f"VCD failed at lat={lat}, lon={lon}, z={z}, ier={ier}")

                    U[i, j, k] = zon_wind
                    V[i, j, k] = mer_wind
                    W[i, j, k] = vert_wind
                    T[i, j, k] = temp
                    P[i, j, k] = pres
                    rho[i, j, k] = dens

    print("========== Sanity checks ==========")
    print("Shape of U: ", U.shape)
    print("isnan U: ", np.isnan(U).any())
    print("isinf U: ", np.isinf(U).any())
    print("U mean and std: ", U.mean(), U.std())
    print("T min and max: ", T.min(), T.max())

    np.savez_compressed(
    DATA_DIR / "vcd_climatology_40km_70km.npz",
    lat=lat_grid,
    lon=lon_grid,
    z=z_grid,
    U=U,
    V=V,
    W=W,
    T=T,
    P=P,
    rho=rho
    )

    print("Saved!")

def simplified_power_grid():
     
     # Grid
    lat_grid = np.array(
        [0, 15, 30, 45, 60, 75, 90],
        dtype=np.float32
    )

    z_grid = np.arange(
        40,
        71,
        1,
        dtype=np.float32
    )

    localtime_grid = np.arange(
        0.0,
        24.0,
        0.25,   # 15 venus-minute resolution
        dtype=np.float32
    )

    N_lat = len(lat_grid)
    N_z = len(z_grid)

    # Output
    sw_flux = np.zeros(
        (N_lat, N_z),
        dtype=np.float32
    )

    # Fixed inputs
    z_key = 2

    date_key = 1
    juliandate = 0.0

    dset = (r"C:\Users\juliu\OneDrive - Delft University of Technology\Bureaublad\BSc AE Y3\DESIGN SYNTHESIS EXERCISE\VCD2.3\VCD_DATA\\")

    EUV_scena = 1
    albedo_scena = 1
    varE107 = 0.0

    perturb_key = 0
    perturb_seed = 0
    perturb_gw_length = 0.0

    extvar_keys = np.zeros(100, dtype=np.int32)
    extvar_keys[40] = 1  # Request SW flux

    # Sampling loop
    for i, lat in enumerate(lat_grid):

        print(
            f"Computing latitude {lat:.0f} deg"
        )

        for k, z in enumerate(z_grid):

            flux_samples = []

            for localtime in localtime_grid:

                (
                    zon_wind,
                    mer_wind,
                    vert_wind,
                    temp,
                    pres,
                    dens,
                    extvar,
                    seed_out,
                    ier
                ) = sample_vcd(
                    z_key,
                    float(z * 1000.0),
                    0.0,                   # longitude
                    float(lat),
                    0,                     # hires
                    date_key,
                    juliandate,
                    float(localtime),
                    dset,
                    EUV_scena,
                    albedo_scena,
                    varE107,
                    perturb_key,
                    perturb_seed,
                    perturb_gw_length,
                    extvar_keys
                )

                if ier != 0:
                    raise RuntimeError(
                        f"VCD failed at "
                        f"lat={lat}, "
                        f"z={z}, "
                        f"lt={localtime}, "
                        f"ier={ier}"
                    )

                flux_samples.append(
                    extvar[40]
                )

            sw_flux[i, k] = np.mean(
                flux_samples
            )

            if i == 0 and k == 0:

                print("DEBUG SAMPLE")
                print(
                    "Mean SW flux:",
                    sw_flux[i, k]
                )

    # Diagnostics
    print()
    print("========== Sanity checks ==========")

    print("Shape:", sw_flux.shape)
    print("Min:", np.min(sw_flux))
    print("Max:", np.max(sw_flux))
    print("Mean:", np.mean(sw_flux))
    print("NaNs:", np.isnan(sw_flux).any())
    print("Infs:", np.isinf(sw_flux).any())

    # Saving
    np.savez_compressed(
        DATA_DIR / "vcd_mean_shortwave_flux.npz",
        lat=lat_grid,
        z=z_grid,
        sw_flux=sw_flux
    )

    print()
    print("Saved!")


if __name__ == "__main__":
    # main()
    simplified_power_grid()