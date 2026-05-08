import numpy as np


def compute_link_metrics(
    times,
    orbiter_positions,
    balloon_positions,
    min_elevation_deg=20.0,
    max_elevation_deg=80.0
):
    """
    Compute communication link metrics between balloon and orbiter.

    Parameters
    ----------
    times : ndarray (N,)
        Simulation times.

    orbiter_positions : ndarray (N,3)
        Orbiter positions in inertial frame.

    balloon_positions : ndarray (N,3)
        Balloon positions in inertial frame.

    Returns
    -------
    results : dict
        Dictionary containing:
            distances
            elevation_deg
            link_available
            link_segments
    """

    # -------------------------------------------------
    # Relative vector balloon -> orbiter
    # -------------------------------------------------

    relative_vectors = (
        orbiter_positions
        - balloon_positions
    )

    # Distance
    distances = np.linalg.norm(
        relative_vectors,
        axis=1
    )

    # -------------------------------------------------
    # Balloon local zenith direction
    # -------------------------------------------------

    balloon_radii = np.linalg.norm(
        balloon_positions,
        axis=1
    )

    zenith_vectors = (
        balloon_positions
        / balloon_radii[:, None]
    )

    # -------------------------------------------------
    # Elevation angle
    # -------------------------------------------------

    los_unit_vectors = (
        relative_vectors
        / distances[:, None]
    )

    sin_elevation = np.sum(
        los_unit_vectors * zenith_vectors,
        axis=1
    )

    # Numerical safety
    sin_elevation = np.clip(
        sin_elevation,
        -1.0,
        1.0
    )

    elevation_rad = np.arcsin(
        sin_elevation
    )

    elevation_deg = np.degrees(
        elevation_rad
    )

    # -------------------------------------------------
    # Link availability
    # -------------------------------------------------

    link_available = (
        (elevation_deg >= min_elevation_deg)
        &
        (elevation_deg <= max_elevation_deg)
    )

    # -------------------------------------------------
    # Continuous link segments
    # -------------------------------------------------

    link_segments = []

    in_segment = False
    start_idx = None

    for i, available in enumerate(link_available):

        if available and not in_segment:

            start_idx = i
            in_segment = True

        elif not available and in_segment:

            end_idx = i - 1

            link_segments.append({
                "start_time": times[start_idx],
                "end_time": times[end_idx],
                "duration": (
                    times[end_idx]
                    - times[start_idx]
                ),
                "start_index": start_idx,
                "end_index": end_idx
            })

            in_segment = False

    # Handle case where final sample is linked
    if in_segment:

        end_idx = len(times) - 1

        link_segments.append({
            "start_time": times[start_idx],
            "end_time": times[end_idx],
            "duration": (
                times[end_idx]
                - times[start_idx]
            ),
            "start_index": start_idx,
            "end_index": end_idx
        })

    return {
        "distances": distances,
        "elevation_deg": elevation_deg,
        "link_available": link_available,
        "link_segments": link_segments
    }