import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

from paths import DATA_DIR

VENUS_MAP = DATA_DIR / "Cylindrical_Map_of_Venus.jpg"

def setup_venus_map_axes(
    figsize=(14, 7)
):
    """
    Create a Venus map plot with cylindrical texture.
    """

    fig, ax = plt.subplots(
        figsize=figsize
    )

    venus_map = imread(VENUS_MAP)

    ax.imshow(
        venus_map,
        extent=(-180.0, 180.0, -90.0, 90.0),
        aspect="auto",
        origin="upper",
        alpha=0.5
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")

    ax.grid(
        True,
        color="white",
        alpha=0.25
    )

    return fig, ax


def draw_venus(
    ax,
    planet_radius,
    sphere_alpha=0.35,
    line_alpha=0.5
):
    """
    Draw Venus with latitude and longitude lines.
    """

    # Sphere mesh
    u = np.linspace(0, 2*np.pi, 120)
    v = np.linspace(0, np.pi, 60)

    x = (
        planet_radius
        * np.outer(np.cos(u), np.sin(v))
    )

    y = (
        planet_radius
        * np.outer(np.sin(u), np.sin(v))
    )

    z = (
        planet_radius
        * np.outer(np.ones_like(u), np.cos(v))
    )

    # Venus sphere (transparent yellow/orange)
    ax.plot_surface(
        x,
        y,
        z,
        color="#d9b44a",
        alpha=sphere_alpha,
        linewidth=0,
        shade=True
    )

    # -------------------------------------------------
    # Latitude lines
    # -------------------------------------------------
    latitudes = np.arange(-75, 90, 15)

    theta = np.linspace(0, 2*np.pi, 300)

    for lat_deg in latitudes:

        lat = np.radians(lat_deg)

        r = planet_radius * np.cos(lat)

        z_lat = planet_radius * np.sin(lat)

        x_lat = r * np.cos(theta)
        y_lat = r * np.sin(theta)

        ax.plot(
            x_lat,
            y_lat,
            z_lat,
            color="black",
            alpha=line_alpha,
            linewidth=0.6
        )

    # -------------------------------------------------
    # Longitude lines
    # -------------------------------------------------
    longitudes = np.arange(0, 360, 15)

    phi = np.linspace(-np.pi/2, np.pi/2, 300)

    for lon_deg in longitudes:

        lon = np.radians(lon_deg)

        x_lon = (
            planet_radius
            * np.cos(phi)
            * np.cos(lon)
        )

        y_lon = (
            planet_radius
            * np.cos(phi)
            * np.sin(lon)
        )

        z_lon = (
            planet_radius
            * np.sin(phi)
        )

        ax.plot(
            x_lon,
            y_lon,
            z_lon,
            color="black",
            alpha=line_alpha,
            linewidth=0.6
        )


def set_equal_axes(ax, positions_list, planet_radius):
    """
    Force equal scaling on all axes.
    """

    all_positions = np.vstack(positions_list)

    max_range = np.max(np.abs(all_positions))

    max_range = max(max_range, planet_radius)

    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_box_aspect([1, 1, 1])


def plot_orbit_3d(
    orbiter_positions,
    planet_radius
):

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Venus
    draw_venus(
        ax,
        planet_radius
    )

    # Orbiter trajectory (blue)
    ax.plot(
        orbiter_positions[:, 0],
        orbiter_positions[:, 1],
        orbiter_positions[:, 2],
        color="royalblue",
        linewidth=2,
        label="Orbiter"
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    set_equal_axes(
        ax,
        [orbiter_positions],
        planet_radius
    )

    ax.legend()

    plt.show()


def plot_simulation_3d(
    orbiter_positions,
    balloon_positions,
    planet_radius
):

    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # Venus
    draw_venus(
        ax,
        planet_radius
    )

    # Orbiter trajectory (blue)
    ax.plot(
        orbiter_positions[:, 0],
        orbiter_positions[:, 1],
        orbiter_positions[:, 2],
        color="royalblue",
        linewidth=2,
        label="Orbiter"
    )

    # Balloon trajectory (red)
    ax.plot(
        balloon_positions[:, 0],
        balloon_positions[:, 1],
        balloon_positions[:, 2],
        color="magenta",
        linewidth=2,
        label="Balloon"
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    set_equal_axes(
        ax,
        [orbiter_positions, balloon_positions],
        planet_radius
    )

    ax.legend()

    plt.show()


def plot_link_analysis(
    times,
    distances,
    elevation_deg,
    link_available
):

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True
    )


    times_days = times / (3600 * 24)
    # -----------------------------------------
    # Distance
    # -----------------------------------------

    axes[0].plot(
        times_days,
        distances / 1000
    )

    axes[0].set_ylabel(
        "Distance [km]"
    )

    axes[0].grid(True)

    # -----------------------------------------
    # Elevation
    # -----------------------------------------

    axes[1].plot(
        times_days,
        elevation_deg,
        color="darkorange"
    )

    axes[1].axhline(
        20,
        color="red",
        linestyle="--"
    )

    axes[1].axhline(
        80,
        color="red",
        linestyle="--"
    )

    # Shade communication windows
    axes[1].fill_between(
        times_days,
        20,
        80,
        where=link_available,
        alpha=0.3
    )

    axes[1].set_ylabel(
        "Elevation [deg]"
    )

    axes[1].set_xlabel(
        "Time [Earth days]"
    )

    axes[1].grid(True)

    plt.show()


def compute_groundtrack(
    positions_vci,
    times,
    venus_model
):
    """
    Convert inertial spacecraft positions
    into Venus-fixed latitude/longitude.
    """

    latitudes_deg = []
    longitudes_deg = []

    omega = venus_model.rotation_rate_rad_s

    for r_vci, t in zip(positions_vci, times):

        # -----------------------------------------
        # Rotate into Venus-fixed frame
        # -----------------------------------------

        theta = omega * t

        c = np.cos(theta)
        s = np.sin(theta)

        rotation = np.array([
            [ c,  s, 0],
            [-s,  c, 0],
            [ 0,  0, 1]
        ])

        r_fixed = rotation @ r_vci

        x, y, z = r_fixed

        r_norm = np.linalg.norm(r_fixed)

        # -----------------------------------------
        # Latitude / longitude
        # -----------------------------------------

        latitude = np.degrees(
            np.arcsin(z / r_norm)
        )

        longitude = np.degrees(
            np.arctan2(y, x)
        )

        # Wrap longitude
        longitude = (
            (longitude + 180) % 360
        ) - 180

        latitudes_deg.append(latitude)
        longitudes_deg.append(longitude)

    return (
        np.array(latitudes_deg),
        np.array(longitudes_deg)
    )


def split_longitude_discontinuities(
    longitudes_deg,
    latitudes_deg,
    threshold=180.0
):
    """
    Split a ground track whenever longitude wraps
    around the map boundary.

    Returns a list of trajectory segments.
    """

    longitudes_deg = np.asarray(longitudes_deg)
    latitudes_deg = np.asarray(latitudes_deg)

    segments = []

    start_idx = 0

    for i in range(1, len(longitudes_deg)):

        delta = abs(
            longitudes_deg[i]
            - longitudes_deg[i - 1]
        )

        # Detect map wrap
        if delta > threshold:

            segments.append((
                longitudes_deg[start_idx:i],
                latitudes_deg[start_idx:i]
            ))

            start_idx = i

    # Final segment
    segments.append((
        longitudes_deg[start_idx:],
        latitudes_deg[start_idx:]
    ))

    return segments


def plot_groundtrack(
    latitudes_deg,
    longitudes_deg,
    label,
    color
):

    fig, ax = setup_venus_map_axes()

    segments = split_longitude_discontinuities(
        longitudes_deg,
        latitudes_deg
    )

    for lon_seg, lat_seg in segments:

        ax.plot(
            lon_seg,
            lat_seg,
            color=color,
            linewidth=1.5,
            label=label
        )

        label = None

    ax.legend()

    ax.set_title(
        f"{label} Ground Track"
    )

    plt.show()


def plot_combined_groundtrack(
    balloon_latitudes_deg,
    balloon_longitudes_deg,
    orbiter_latitudes_deg,
    orbiter_longitudes_deg
):

    fig, ax = setup_venus_map_axes()

    # Orbiter
    orbiter_segments = split_longitude_discontinuities(
        orbiter_longitudes_deg,
        orbiter_latitudes_deg
    )

    for i, (lon_seg, lat_seg) in enumerate(orbiter_segments):

        ax.plot(
            lon_seg,
            lat_seg,
            color="royalblue",
            linewidth=1.2,
            label="Orbiter" if i == 0 else None
        )

    # Balloon
    balloon_segments = split_longitude_discontinuities(
    balloon_longitudes_deg,
    balloon_latitudes_deg
    )

    for i, (lon_seg, lat_seg) in enumerate(balloon_segments):

        ax.plot(
            lon_seg,
            lat_seg,
            color="magenta",
            linewidth=1.2,
            label="Balloon" if i == 0 else None
        )

    # Start markers
    ax.scatter(
        orbiter_longitudes_deg[0],
        orbiter_latitudes_deg[0],
        color="royalblue",
        s=40,
        label="Orbiter Start"
    )

    ax.scatter(
        balloon_longitudes_deg[0],
        balloon_latitudes_deg[0],
        color="magenta",
        s=40,
        label="Balloon Start"
    )

    ax.legend()

    ax.set_title(
        "Venus Ground Tracks"
    )

    plt.show()

