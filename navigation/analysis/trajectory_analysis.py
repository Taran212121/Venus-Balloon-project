import numpy as np





def polar_fraction(latitudes, threshold_deg):
    polar_mask = np.abs(latitudes) > threshold_deg
    return np.mean(polar_mask, axis=1)


def check_vortex_captures(results, threshold_deg=60.0, time_fraction=0.7):
    """Study the outputs of MC simulation to identify trajectories that ended up in a polar vortex."""

    times = results['times']
    lat = results['latitude']
    lon = results['longitude']
    alt = results['altitude']
    dt = times[1] - times[0]


    polar_frac = polar_fraction(lat, threshold_deg)

    vortex_mask = polar_frac > time_fraction

    vortex_indices = np.where(vortex_mask)[0]

    print(
        f"Vortex-trapped runs: "
        f"{len(vortex_indices)} / {len(lat)}"
    )

    vortex_lat = lat[vortex_mask]
    vortex_lon = lon[vortex_mask]
    vortex_alt = alt[vortex_mask]

    vortex_results = {
        "times": times,
        'latitude': vortex_lat,
        'longitude': vortex_lon,
        'altitude': vortex_alt
    }

    return vortex_results