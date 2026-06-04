from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import solve_ivp


from vehicles.orbiter import OrbiterState, state_to_vector, vector_to_state
from coordinate_frames import keplerian_to_cartesian



class OrbiterPropagator(ABC):

    def __init__(self, *args, **kwargs):
        ...


    @abstractmethod
    def step(self, state, dt, env):
        ...





class TwoBodyNewtonPropagator(OrbiterPropagator):
    def __init__(self, orbit_config, venus_model):
        self.mu = venus_model.mu

        orbit = orbit_config.initial_orbit

        r0, v0 = keplerian_to_cartesian(
            semi_major_axis=orbit.semi_major_axis_m,
            eccentricity=orbit.eccentricity,
            inclination=np.radians(orbit.inclination_deg),
            raan=np.radians(orbit.raan_deg),
            argument_of_periapsis=np.radians(
                orbit.argument_of_periapsis_deg
            ),
            true_anomaly=np.radians(orbit.true_anomaly_deg),
            mu=self.mu
        )

        self.initial_state = OrbiterState(
            position=r0,
            velocity=v0
        )

    def acceleration(self, r):
        """Two-body gravitational acceleration."""
        r2 = np.dot(r, r)
        r3 = r2 * np.sqrt(r2)
        return -self.mu * r / r3

    def step(self, state, dt, env):
        """
        Perform one symplectic integration step.
        """

        # Current state
        r = state.position
        v = state.velocity

        # Initial acceleration
        a0 = self.acceleration(r)

        # Position update
        r_new = (
            r
            + v * dt
            + 0.5 * a0 * dt * dt
        )

        # Acceleration update
        a1 = self.acceleration(r_new)

        # Velocity update
        v_new = (
            v
            + 0.5 * (a0 + a1) * dt
        )

        return OrbiterState(
            position=r_new,
            velocity=v_new
        )
