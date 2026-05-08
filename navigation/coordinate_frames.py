import numpy as np



def rotation_matrix_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1, 0, 0],
        [0, c,-s],
        [0, s, c]
    ])

def rotation_matrix_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def rotation_matrix_z(angle):

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1]
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
        rotation_matrix_z(raan)
        @ rotation_matrix_x(inclination)
        @ rotation_matrix_z(argument_of_periapsis)
    )

    r = rotation @ r_pf
    v = rotation @ v_pf

    return r, v


def geodetic_to_cartesian(
    latitude_deg,
    longitude_deg,
    altitude_m,
    planet_radius
):
    """
    Convert spherical geodetic coordinates
    to Venus-fixed Cartesian coordinates.
    """

    lat = np.radians(latitude_deg)
    lon = np.radians(longitude_deg)

    r = planet_radius + altitude_m

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return np.array([x, y, z])


def cartesian_to_geodetic(
    position,
    planet_radius
):
    """
    Cartesian Venus-fixed -> spherical geodetic.
    """

    x, y, z = position

    r = np.linalg.norm(position)

    latitude = np.arcsin(z / r)

    longitude = np.arctan2(y, x)

    altitude = r - planet_radius

    return (
        np.degrees(latitude),
        np.degrees(longitude),
        altitude
    )


def venus_rotation_rate(venus_model):

    period_seconds = (
        venus_model.inertial_rot_period
        * 24 * 3600
    )

    return 2 * np.pi / period_seconds


def vcf_to_vci(
    position_vcf,
    t,
    venus_model
):
    """
    Rotating frame -> inertial frame.
    """

    omega = venus_rotation_rate(
        venus_model
    )

    theta = omega * t

    rotation = rotation_matrix_z(theta)

    return rotation @ position_vcf


def vci_to_vcf(
    position_vci,
    t,
    venus_model
):
    """
    Inertial frame -> rotating frame.
    """

    omega = venus_rotation_rate(
        venus_model
    )

    theta = omega * t

    rotation = rotation_matrix_z(-theta)

    return rotation @ position_vci