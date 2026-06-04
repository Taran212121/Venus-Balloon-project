import numpy as np
from typing import Callable


TrajectoryFunction = Callable[[float], float]


# ==============================
# BASE FUNCTIONS
# ==============================

def constant(magnitude: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        return magnitude
    
    return func


def step(activation_time: float, magnitude: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        if t >= activation_time:
            return magnitude
        else:
            return 0.0
    
    return func


def ramp(activation_time: float, slope: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        if t >= activation_time:
            return slope * (t - activation_time)
        else:
            return 0.0
    
    return func


def pulse(start_time: float, end_time: float, magnitude: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        if start_time <= t <= end_time:
            return magnitude
        else:
            return 0.0
    
    return func


def sine(amplitude: float, period: float, phase: float = 0.0) -> TrajectoryFunction:

    def func(t: float) -> float:
        return amplitude * np.sin(2 * np.pi * t / period + phase)
    
    return func


def cosine(amplitude: float, period: float, phase: float = 0.0) -> TrajectoryFunction:

    def func(t: float) -> float:
        return amplitude * np.cos(2 * np.pi * t / period + phase)
    
    return func


def exponential(activation_time: float, magnitude: float, tau: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        if t < activation_time:
            return 0.0
        else:
            return magnitude * (1 - np.exp(-(t - activation_time) / tau))
    
    return func


def triangle(amplitude: float, period: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        phase = (t % period) / period
        return 4 * amplitude * abs(phase - 0.5) - amplitude
    
    return func


def sawtooth(amplitude: float, period: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        phase = (t % period) / period
        return amplitude * phase
    
    return func


def gaussian(center_time: float, width: float, amplitude: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        return amplitude * np.exp( -0.5 * ((t - center_time) / width) ** 2)
    
    return func


def linear_transition(start_time: float, end_time: float, start_value: float, end_value: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        if t <= start_time:
            return start_value
        if t >= end_time:
            return end_value
        else:
            alpha = (t - start_time) / (end_time - start_time)
            return start_value + alpha * (end_value - start_value)
    
    return func


# ==============================
# COMPOSITION & OPERATORS
# ==============================

def add(*trajectories: TrajectoryFunction) -> TrajectoryFunction:

    def func(t: float) -> float:
        return sum(trajectory(t) for trajectory in trajectories)
    
    return func


def multiply(a: TrajectoryFunction, b: TrajectoryFunction) -> TrajectoryFunction:

    def func(t: float) -> float:
        return a(t) * b(t)

    return func


def offset(trajectory: TrajectoryFunction, value: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        return trajectory(t) + value
    
    return func


def scale(trajectory: TrajectoryFunction, value: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        return value * trajectory(t)
    
    return func


def shift(trajectory: TrajectoryFunction, dt: float) -> TrajectoryFunction:

    def func(t: float) -> float:
        return trajectory(t - dt)
    
    return func


test_trajectory = add(
    constant(55_000),
    sine(
        amplitude=1_000,
        period=24 * 3600
    ),
    gaussian(
        center_time=10 * 24 * 3600,
        width=4 * 3600,
        amplitude=3_000
    )
)



# test_trajectory = constant(52_000)
test_trajectory = linear_transition(1 * 24 * 3600, 8 * 24 * 3600, 52_000, 56_000)