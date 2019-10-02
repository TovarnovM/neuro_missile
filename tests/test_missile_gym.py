import sys
import pytest
import os
import sys
import numpy as np
from pytest import approx

wd = os.path.abspath(__file__) # os.path.dirname(os.path.dirname(os.getcwd()))
wd = os.path.dirname(os.path.dirname(wd))
sys.path.append(wd+"\\src\\")

from missile_gym import MissileGym

def test_make():
    for name in MissileGym.scenario_names:
        env = MissileGym.make(name)
        env.reset()
        assert isinstance(env, MissileGym)
        env.close()

def test_get_state1():
    for name in MissileGym.scenario_names:
        env = MissileGym.make(name)
        env.reset()
        state1 = env.get_state()
        for i in range(np.random.randint(3,7)):
            env.step(env.actione_sample())
        env.reset()
        state2 = env.get_state()
        assert state1 == approx(state2)
        env.close()

def test_get_state2():
    for name in MissileGym.scenario_names:
        env = MissileGym.make(name)
        env.reset()
        state1 = env.get_state()
        assert isinstance(state1, np.ndarray)
        env.close()

def test_getset_state():
    for name in MissileGym.scenario_names:
        env = MissileGym.make(name)
        env.reset()
        for i in range(np.random.randint(3,7)):
            env.step(env.actione_sample())
        state1 = env.get_state()
        action1 = env.actione_sample()
        env.step(action1)
        state2 = env.get_state()
        env.set_state(state1)
        env.step(action1)
        state3 = env.get_state()
        assert state2 == approx(state3)
        env.close()

def test_step_returns():
    for name in MissileGym.scenario_names:
        env = MissileGym.make(name)
        env.reset()
        for i in range(np.random.randint(3,7)):
            observation, reward, done, info = env.step(env.actione_sample())
            observation_high = env.observation_space_high
            observation_low = env.observation_space_low
            assert isinstance(observation, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            assert len(observation_high) == len(observation)
            assert len(observation_low) == len(observation)
            assert observation < observation_high
            assert observation > observation_low
        env.close()



