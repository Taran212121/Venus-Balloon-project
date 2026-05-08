import numpy as np


def rotation_matrix_3(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [ c, s, 0],
        [-s, c, 0],
        [ 0, 0, 1]
    ])


def rotation_matrix_1(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1, 0, 0],
        [0, c, s],
        [0,-s, c]
    ])


def keplerian_to_cartesian(
    semi_major_axis,
    eccentricity,
    inclination,
    raan,
    argument_of_periapsis,
    true_anomaly,
    mu
):
    """
    All angles in radians.
    """

    # Semi-latus rectum
    p = semi_major_axis * (1 - eccentricity**2)

    # Position in perifocal frame
    r_pf = np.array([
        p * np.cos(true_anomaly) / (1 + eccentricity * np.cos(true_anomaly)),
        p * np.sin(true_anomaly) / (1 + eccentricity * np.cos(true_anomaly)),
        0
    ])

    # Velocity in perifocal frame
    v_pf = np.sqrt(mu / p) * np.array([
        -np.sin(true_anomaly),
        eccentricity + np.cos(true_anomaly),
        0
    ])

    # Rotation: perifocal -> inertial
    rotation = (
        rotation_matrix_3(-raan)
        @ rotation_matrix_1(-inclination)
        @ rotation_matrix_3(-argument_of_periapsis)
    )

    r = rotation @ r_pf
    v = rotation @ v_pf

    return r, v