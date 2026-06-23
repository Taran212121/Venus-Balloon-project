import numpy as np

from matplotlib import pyplot as plt


from paths import OUT_DIR
from tools.trajectory_functions import *


def imu_error_plot():

    t_grid = np.linspace(8, 36, 1000)
    
    # Forward propagation only
    e1_grid = np.ones_like(t_grid)
    mask = (t_grid > 10) & (t_grid <= 34)
    e1_grid[mask] = 1 + (t_grid[mask] - 10)**2 / 500

    # Forward/backward propagation
    e2_grid = np.ones_like(t_grid)
    mask1 = (t_grid > 10) & (t_grid <= 22)
    e2_grid[mask1] = 1 + (t_grid[mask1] - 10)**2 / 500
    mask2 = (t_grid > 22) & (t_grid <= 34)
    e2_grid[mask2] = 1 + (34 - t_grid[mask2])**2 / 500

    # Plotting
    plt.figure(figsize=(6, 3))
    plt.axvspan(10, 34, alpha=0.15, color='grey')
    plt.plot(t_grid, e1_grid, lw=2, ls='-', label="Forward propagation", color='crimson')
    plt.plot(t_grid, e2_grid, lw=2, ls='--', label="Forward/backward propagation", color='navy')

    # plt.title("IMU Drift Errors")
    plt.xlabel("Time [h]")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUT_DIR / "imu_error_plot.svg")
    plt.show()


def example_altitude_profile():

    points = [
        (-10, 56),
        (0  , 56),
        (26 , 56),
        (40 , 54),
        (43 , 54),
        (44 , 55),
        (48 , 55),
        (49 , 56),
        (180, 56),
        (190, 56)
    ]

    alt_func = linear_spline(points)

    t_grid = np.linspace(0, 180, 1000)
    alt_grid = alt_func(t_grid)

    # Plotting
    plt.figure(figsize=(9, 3))
    plt.axvspan(-10, 80, alpha=0.15, color='gold')
    plt.axvspan(80, 160, alpha=0.15, color='slateblue')
    plt.axvspan(160, 190, alpha=0.15, color='gold')
    plt.plot(t_grid, alt_grid, color='crimson')

    plt.xlim(0, 180)
    plt.xlabel("Time [h]")
    plt.ylabel("Altitude [km]")
    plt.tight_layout()

    plt.savefig(OUT_DIR / "altitude_profile.svg")
    plt.show()








if __name__ == "__main__":
    # imu_error_plot()
    example_altitude_profile()