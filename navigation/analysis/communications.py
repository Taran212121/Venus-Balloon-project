import numpy as np
from pathlib import Path

from paths import OUT_DIR


def seconds_to_hms(seconds):
    seconds = int(round(seconds))

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def find_segments(times, mask):
    """Find contiguous True segments in a boolean mask."""

    segments = []
    in_segment = False
    start_idx = None

    for i, value in enumerate(mask):

        if value and not in_segment:
            start_idx = i
            in_segment = True

        elif not value and in_segment:

            end_idx = i - 1

            segments.append({
                "start_index": start_idx,
                "end_index": end_idx,
                "start_time": times[start_idx],
                "end_time": times[end_idx],
                "duration": (times[end_idx] - times[start_idx])
            })

            in_segment = False

    if in_segment:

        end_idx = len(times) - 1

        segments.append({
            "start_index": start_idx,
            "end_index": end_idx,
            "start_time": times[start_idx],
            "end_time": times[end_idx],
            "duration": (times[end_idx] - times[start_idx])
        })

    return segments

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


def generate_link_report(
    times,
    orbiter_positions,
    balloon_positions,
    min_elevation_deg=20.0,
    max_elevation_deg=80.0,
    out_path=OUT_DIR,
    scenario_name="Unnamed Scenario",
):
    metrics = compute_link_metrics(
        times=times,
        orbiter_positions=orbiter_positions,
        balloon_positions=balloon_positions,
        min_elevation_deg=min_elevation_deg,
        max_elevation_deg=max_elevation_deg
    )

    distances = metrics["distances"]
    elevations = metrics["elevation_deg"]
    link_available = metrics["link_available"]

    link_segments = find_segments(
        times,
        link_available
    )

    blackout_segments = find_segments(
        times,
        ~link_available
    )

    # -------------------------------------------------
    # Overall statistics
    # -------------------------------------------------

    simulation_duration = (
        times[-1] - times[0]
    )

    dt = np.mean(np.diff(times))

    total_link_time = (
        np.sum(link_available) * dt
    )

    total_blackout_time = (
        simulation_duration
        - total_link_time
    )

    link_fraction = (
        total_link_time
        / simulation_duration
    )

    # -------------------------------------------------
    # Distance statistics
    # -------------------------------------------------

    overall_min_distance = np.min(distances)
    overall_mean_distance = np.mean(distances)
    overall_max_distance = np.max(distances)

    if np.any(link_available):

        link_distances = distances[
            link_available
        ]

        link_min_distance = np.min(
            link_distances
        )

        link_mean_distance = np.mean(
            link_distances
        )

        link_max_distance = np.max(
            link_distances
        )

    else:

        link_min_distance = np.nan
        link_mean_distance = np.nan
        link_max_distance = np.nan

    # -------------------------------------------------
    # Elevation statistics
    # -------------------------------------------------

    min_elevation = np.min(elevations)
    mean_elevation = np.mean(elevations)
    max_elevation = np.max(elevations)

    # -------------------------------------------------
    # Longest windows
    # -------------------------------------------------

    if len(link_segments) > 0:

        longest_link_window = max(
            link_segments,
            key=lambda s: s["duration"]
        )["duration"]

    else:

        longest_link_window = 0.0

    if len(blackout_segments) > 0:

        longest_blackout_window = max(
            blackout_segments,
            key=lambda s: s["duration"]
        )["duration"]

    else:

        longest_blackout_window = 0.0

    # -------------------------------------------------
    # Build report text
    # -------------------------------------------------

    report_lines = []

    report_lines.append(
        "=" * 60
    )

    report_lines.append(
        "VENUS COMMUNICATION LINK REPORT"
    )

    report_lines.append(
        "=" * 60
    )

    report_lines.append("")

    report_lines.append(
        f"Scenario: {scenario_name}"
    )

    report_lines.append("")

    report_lines.append(
        f"Simulation Duration: "
        f"{seconds_to_hms(simulation_duration)}"
    )

    report_lines.append("")

    report_lines.append(
        f"Minimum Elevation: "
        f"{min_elevation_deg:.1f} deg"
    )

    report_lines.append(
        f"Maximum Elevation: "
        f"{max_elevation_deg:.1f} deg"
    )

    report_lines.append("")

    report_lines.append(
        "-" * 60
    )

    report_lines.append(
        "OVERALL STATISTICS"
    )

    report_lines.append(
        "-" * 60
    )

    report_lines.append("")

    report_lines.append(
        f"Total Communication Time: "
        f"{seconds_to_hms(total_link_time)}"
    )

    report_lines.append(
        f"Total Blackout Time: "
        f"{seconds_to_hms(total_blackout_time)}"
    )

    report_lines.append(
        f"Link Availability: "
        f"{100 * link_fraction:.2f} %"
    )

    report_lines.append("")

    report_lines.append(
        f"Communication Windows: "
        f"{len(link_segments)}"
    )

    report_lines.append(
        f"Blackout Windows: "
        f"{len(blackout_segments)}"
    )

    report_lines.append("")

    report_lines.append(
        f"Longest Communication Window: "
        f"{seconds_to_hms(longest_link_window)}"
    )

    report_lines.append(
        f"Longest Blackout Window: "
        f"{seconds_to_hms(longest_blackout_window)}"
    )

    report_lines.append("")

    report_lines.append(
        "-" * 60
    )

    report_lines.append(
        "DISTANCE STATISTICS"
    )

    report_lines.append(
        "-" * 60
    )

    report_lines.append("")

    report_lines.append(
        f"Overall Min Distance: "
        f"{overall_min_distance/1000:.1f} km"
    )

    report_lines.append(
        f"Overall Mean Distance: "
        f"{overall_mean_distance/1000:.1f} km"
    )

    report_lines.append(
        f"Overall Max Distance: "
        f"{overall_max_distance/1000:.1f} km"
    )

    report_lines.append("")

    report_lines.append(
        f"Link Min Distance: "
        f"{link_min_distance/1000:.1f} km"
    )

    report_lines.append(
        f"Link Mean Distance: "
        f"{link_mean_distance/1000:.1f} km"
    )

    report_lines.append(
        f"Link Max Distance: "
        f"{link_max_distance/1000:.1f} km"
    )

    report_lines.append("")

    report_lines.append(
        "-" * 60
    )

    report_lines.append(
        "COMMUNICATION WINDOWS"
    )

    report_lines.append(
        "-" * 60
    )

    report_lines.append("")

    for i, segment in enumerate(
        link_segments,
        start=1
    ):

        start = segment["start_index"]
        end = segment["end_index"]

        segment_distances = (
            distances[start:end+1]
        )

        segment_elevations = (
            elevations[start:end+1]
        )

        report_lines.append(
            f"Window {i}"
        )

        report_lines.append(
            f"  Start: "
            f"{seconds_to_hms(segment['start_time'])}"
        )

        report_lines.append(
            f"  End: "
            f"{seconds_to_hms(segment['end_time'])}"
        )

        report_lines.append(
            f"  Duration: "
            f"{seconds_to_hms(segment['duration'])}"
        )

        report_lines.append(
            f"  Distance [km]: "
            f"{np.min(segment_distances)/1000:.1f}"
            f" / "
            f"{np.mean(segment_distances)/1000:.1f}"
            f" / "
            f"{np.max(segment_distances)/1000:.1f}"
        )

        report_lines.append(
            f"  Elevation [deg]: "
            f"{np.min(segment_elevations):.1f}"
            f" / "
            f"{np.mean(segment_elevations):.1f}"
            f" / "
            f"{np.max(segment_elevations):.1f}"
        )

        report_lines.append("")

    report_lines.append(
        "-" * 60
    )

    report_lines.append(
        "BLACKOUT WINDOWS"
    )

    report_lines.append(
        "-" * 60
    )

    report_lines.append("")

    for i, segment in enumerate(
        blackout_segments,
        start=1
    ):

        report_lines.append(
            f"Blackout {i}"
        )

        report_lines.append(
            f"  Start: "
            f"{seconds_to_hms(segment['start_time'])}"
        )

        report_lines.append(
            f"  End: "
            f"{seconds_to_hms(segment['end_time'])}"
        )

        report_lines.append(
            f"  Duration: "
            f"{seconds_to_hms(segment['duration'])}"
        )

        report_lines.append("")

    report_text = "\n".join(
        report_lines
    )

    out_path = Path(out_path)

    out_path.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = (
        out_path
        / f"{scenario_name}_link_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(report_text)

    print(
        f"Link report written to:\n"
        f"{report_path}"
    )

    return {
        "metrics": metrics,
        "link_fraction": link_fraction,
        "total_link_time": total_link_time,
        "total_blackout_time": total_blackout_time,
        "longest_link_window": longest_link_window,
        "longest_blackout_window": longest_blackout_window,
        "report_path": report_path
    }