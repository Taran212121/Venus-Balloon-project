import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

from paths import DATA_DIR, FRAME_DIR, OUT_DIR

VENUS_MAP = DATA_DIR / "Cylindrical_Map_of_Venus.jpg"

def setup_venus_map_axes(
    figsize=(14, 7),
    alpha=0.5,
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
        alpha=alpha
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


def plot_link_track(
    times,
    distances,
    elevation_deg,
    link_available,
    min_elevation_deg=20,
    max_elevation_deg=80
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
        min_elevation_deg,
        color="red",
        linestyle="--"
    )

    axes[1].axhline(
        max_elevation_deg,
        color="red",
        linestyle="--"
    )

    # Shade communication windows
    axes[1].fill_between(
        times_days,
        min_elevation_deg,
        max_elevation_deg,
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

    ax.set_title(f"{label} Ground Track")

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


def plot_balloon_altitude(
    balloon_positions,
    times,
    planet_radius,
    target_altitude=None,
    tolerance=None
):
    altitudes = np.linalg.norm(balloon_positions, axis=1) / 1000.0

    times_days = times / (24 * 3600)

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        times_days,
        altitudes,
        color="magenta",
        linewidth=2
    )

    if target_altitude is not None:

        target_km = target_altitude / 1000

        ax.axhline(
            target_km,
            linestyle="--"
        )

        if tolerance is not None:

            tol_km = tolerance / 1000

            ax.fill_between(
                times_days,
                target_km - tol_km,
                target_km + tol_km,
                alpha=0.2
            )

    ax.set_xlabel(
        "Time [Earth days]"
    )

    ax.set_ylabel(
        "Altitude [km]"
    )

    ax.grid(True)

    plt.show()


def plot_monte_carlo_groundtrack(
    latitudes_deg,
    longitudes_deg,
    mode="trajectories",   # "trajectories" or "density"
    alpha=0.1,
    color="magenta",
    savefig=False,
):
    """
    Plot Monte Carlo balloon ground tracks.

    Parameters
    ----------
    latitudes_deg : (n_runs, n_steps)
    longitudes_deg : (n_runs, n_steps)
    mode :
        - "trajectories": overlay all runs
        - "density": 2D histogram (recommended for many runs)
    """

    fig, ax = setup_venus_map_axes(figsize=(14, 6), alpha=1)

    n_runs = latitudes_deg.shape[0]

    # =========================================================
    # MODE 1: raw trajectories (good for small N ~ < 30)
    # =========================================================
    if mode == "trajectories":

        for i in range(n_runs):

            segments = split_longitude_discontinuities(
                longitudes_deg[i],
                latitudes_deg[i]
            )

            for lon_seg, lat_seg in segments:

                ax.plot(
                    lon_seg,
                    lat_seg,
                    color=color,
                    alpha=alpha,
                    linewidth=1.0,
                    label="Monte Carlo" if i == 0 else None
                )

        figname = "mc_vortex_gt_trajectory.png"
        # ax.set_title("Monte Carlo Balloon Ground Tracks")

    # =========================================================
    # MODE 2: density map (recommended for large N)
    # =========================================================
    elif mode == "density":

        lon_all = longitudes_deg.flatten()
        lat_all = latitudes_deg.flatten()

        bins_lon = np.linspace(-180, 180, 200)
        bins_lat = np.linspace(-90, 90, 100)

        H, xedges, yedges = np.histogram2d(
            lon_all,
            lat_all,
            bins=[bins_lon, bins_lat]
        )

        H = H.T  # for imshow orientation

        im = ax.imshow(
            H,
            extent=[-180, 180, -90, 90],
            origin="lower",
            cmap="inferno",
            aspect="auto",
            alpha=0.85
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Visit density")

        figname = "mc_gt_density.svg"
        # ax.set_title("Monte Carlo Ground Track Density")

    else:
        raise ValueError("mode must be 'trajectories' or 'density'")

    # ax.legend()
    plt.tight_layout()
    if savefig: plt.savefig(OUT_DIR / figname)
    plt.show()


def generate_heatmap_snapshots(
    times,
    latitude,
    longitude,
    days_per_frame=5,
    mode="instant",  # "instant" or "cumulative"
    cmap="inferno",
    n_frames=np.inf,
):

    dt = np.mean(np.diff(times))
    print(f"dt: {dt}")
    frame_stride = int(days_per_frame* 86400 / dt)
    frame_indices = np.arange(0, len(times), frame_stride)
    
    if len(frame_indices) > n_frames:
        frame_indices = frame_indices[:n_frames]

    print(f"Generating {len(frame_indices)} frames...")

    for frame_number, idx in enumerate(frame_indices):
        print(f"{frame_number}...")
        if frame_number == 0:
            continue
        
        if mode == "instant":
            lat = latitude[:, idx]
            lon = longitude[:, idx]
        elif mode == "cumulative":
            lat = latitude[:, :idx].flatten()
            lon = longitude[:, :idx].flatten()
        else:
            raise ValueError(f"Invalid mode for heatmap frame generation")

        bins_lon = np.linspace(-180, 180, 200)
        bins_lat = np.linspace(-90, 90, 100)

        heatmap, lat_edges, lon_edges = (
            np.histogram2d(
                lon,
                lat,
                bins=(bins_lon, bins_lat),
            )
        )

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )
        if mode == "instant":
            im = ax.imshow(
                heatmap.T,
                origin="lower",
                extent=[-180, 180, -90, 90],
                aspect="auto",
                cmap=cmap,
                vmin=0,
                vmax=4,
            )
        elif mode == "cumulative":
            im = ax.imshow(
                heatmap.T,
                origin="lower",
                extent=[-180, 180, -90, 90],
                aspect="auto",
                cmap=cmap
            )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Number of balloons" if mode == "instant" else "Visit number")
        day = (times[idx] / 86400)

        ax.set_title(f"Balloon Distribution (Day {day:.1f})")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")


        fig.tight_layout()

        frame_path = FRAME_DIR / f"frame_{frame_number:04d}.png"
        fig.savefig(frame_path, dpi=200)
        plt.close(fig)

    print(f"Frames saved to:\n {FRAME_DIR}")

# ffmpeg -framerate 5 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p heatmap.mp4

def plot_monte_carlo_density_comparison(
    datasets,
    labels,
    savefig=False
):
    if len(datasets) != 4:
        raise ValueError("Expected exactly 4 datasets")

    # -------------------------------------------------
    # Common histogram settings
    # -------------------------------------------------

    bins_lon = np.linspace(-180, 180, 200)
    bins_lat = np.linspace(-90, 90, 100)

    histograms = []

    global_max = 0

    # -------------------------------------------------
    # Compute all histograms first
    # -------------------------------------------------

    for latitudes_deg, longitudes_deg in datasets:

        lon_all = longitudes_deg.flatten()
        lat_all = latitudes_deg.flatten()

        H, _, _ = np.histogram2d(
            lon_all,
            lat_all,
            bins=[bins_lon, bins_lat]
        )

        H = H.T

        histograms.append(H)

        global_max = max(
            global_max,
            np.max(H)
        )

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()
    venus_img = plt.imread(DATA_DIR / "Cylindrical_Map_of_Venus.jpg")
    im = None

    for ax, H, label in zip(
        axes,
        histograms,
        labels
    ):

        # Venus background



        ax.imshow(
            venus_img,
            extent=[-180, 180, -90, 90],
            aspect="auto",
            alpha=1.0
        )

        im = ax.imshow(
            H,
            extent=[-180, 180, -90, 90],
            origin="lower",
            cmap="inferno",
            aspect="auto",
            alpha=0.85,
            vmin=0,
            vmax=global_max
        )

        ax.set_title(label)

        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)


    axes[2].set_xlabel("Longitude [deg]")
    axes[3].set_xlabel("Longitude [deg]")
    axes[0].set_ylabel("Latitude [deg]")
    axes[2].set_ylabel("Latitude [deg]")

    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.85,
        pad=0.02
    )

    cbar.set_label("Visit density")

    # plt.tight_layout()
    if savefig: plt.savefig(OUT_DIR / "mc_density_comparison.png", dpi=200, bbox_inches="tight")
    plt.show()