import numpy as np
import matplotlib.pyplot as plt


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
        color="crimson",
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

    # -----------------------------------------
    # Distance
    # -----------------------------------------

    axes[0].plot(
        times,
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
        times,
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
        times,
        20,
        80,
        where=link_available,
        alpha=0.3
    )

    axes[1].set_ylabel(
        "Elevation [deg]"
    )

    axes[1].set_xlabel(
        "Time [s]"
    )

    axes[1].grid(True)

    plt.show()