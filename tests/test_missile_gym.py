import sys
import pytest
import os
import sys
import numpy as np
from pytest import approx
from math import *

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
            env.step(env.action_sample())
        env.reset()
        state2 = env.get_state()
        print(state1 - state2)
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
            env.step(env.action_sample())
        state1 = env.get_state()
        action1 = env.action_sample()
        env.step(action1)
        state2 = env.get_state()
        env.set_state(state1)
        env.step(action1)
        state3 = env.get_state()
        print(state2 - state3)
        assert state2 == approx(state3, abs=1e-3)
        env.close()

def test_step_returns():
    for name in MissileGym.scenario_names:
        env = MissileGym.make(name)
        env.reset()
        for i in range(np.random.randint(3,7)):
            observation, reward, done, info = env.step(env.action_sample())

        env.close()

def test_reset_returns():
    for name in MissileGym.scenario_names:
        env = MissileGym.make(name)
        observation = env.reset()
        assert isinstance(observation, np.ndarray)
        env.close()

def test_get_etta():
    for name in MissileGym.scenario_names:
        env = MissileGym.make(name)
        # observation = env.reset()
        break
    class FakeMiss:
        def __init__(self):
            self.pos = None
            self.Q = None
        

    miss = FakeMiss()
    target = FakeMiss()

    miss.pos = np.array([1, 1])
    miss.Q = 0
    target.pos = np.array([2,2])
    assert env._get_etta(miss, target) == approx(45)

    miss.pos = np.array([1, 1])
    miss.Q = 0
    target.pos = np.array([6,4])
    assert env._get_etta(miss, target) == approx(30.9637565320735)

    miss.pos = np.array([1, 1])
    miss.Q = pi/2
    target.pos = np.array([2,2])
    assert env._get_etta(miss, target) == approx(-45)

    miss.pos = np.array([1, 1])
    miss.Q = pi
    target.pos = np.array([2,2])
    assert env._get_etta(miss, target) == approx(-135)

    miss.pos = np.array([1, 1])
    miss.Q = pi
    target.pos = np.array([0,2])
    assert env._get_etta(miss, target) == approx(-45)

    miss.pos = np.array([1, 1])
    miss.Q = pi
    target.pos = np.array([-1,-1])
    assert env._get_etta(miss, target) == approx(45)


