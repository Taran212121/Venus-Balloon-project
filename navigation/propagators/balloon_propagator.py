from abc import ABC, abstractmethod




class BalloonPropagator(ABC):

    def __init__(self, *args, **kwargs):
        ...


    @abstractmethod
    def step(self, state, dt, env):
        ...

    




class ScriptedBalloonPropagator(BalloonPropagator):
    def __init__(self, scripted_path):
        # process scripted path
        ...

    def step(self, state, dt, env):
        # execute scripted path
        ...


