class MissileGym(object):
    scenario_names = {'scenario_1'}

    @classmethod
    def make(cls, scenario_name):
        pass

    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        pass

    def close(self):
        pass

    def step(self, action):
        pass

    def get_state(self):
        pass

    def set_state(self, state):
        pass

    def render(self, **kwargs):
        pass

    @property
    def action_space(self):
        pass

    def actione_sample(self):
        pass

    @property
    def observation_space_high(self):
        pass

    @property
    def observation_space_low(self):
        pass

if __name__ == "__main__":
    import numpy as np

    arr = np.array([1,2,3])
    print(1)