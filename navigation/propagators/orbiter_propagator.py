from abc import ABC, abstractmethod




class OrbiterPropagator(ABC):

    def __init__(self, *args, **kwargs):
        ...


    @abstractmethod
    def step(self, state, dt, env):
        ...





class TwoBodyNewtonPropagator(OrbiterPropagator):
    def __init__(self, *args, **kwargs):
        # save all initial orbit params
        ...

    def step(self, state, dt, env):
        # run simple propagator using whatever numerical solver is stable for orbits
        ...
