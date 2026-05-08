from dataclasses import dataclass

import numpy as np


@dataclass
class OrbiterState:
    position: np.ndarray  # [m]
    velocity: np.ndarray  # [m/s]


def state_to_vector(state: OrbiterState) -> np.ndarray:

    return np.hstack([
        state.position,
        state.velocity
    ])


def vector_to_state(vector: np.ndarray) -> OrbiterState:

    return OrbiterState(
        position=vector[:3],
        velocity=vector[3:]
    )