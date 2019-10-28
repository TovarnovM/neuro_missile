import sys
import pytest
import os
import sys
import numpy as np
from pytest import approx

wd = os.path.abspath(__file__) # os.path.dirname(os.path.dirname(os.getcwd()))
wd = os.path.dirname(os.path.dirname(wd))
sys.path.append(wd+"/src/")

from missile import Missile

def test_missile_constructor():
    m = Missile.get_needle()
    assert m is not None

def test_get_standart_parameters_of_missile():
    parametrs_of_missile = Missile.get_standart_parameters_of_missile()
    assert (parametrs_of_missile == np.array([25, 0, 0, np.radians(30), 0])).all()

def test_set_init_cond():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    assert (m.get_state() == m.get_state_0()).all()

def test_get_state():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    assert (m.get_state() == np.array([25, 0, 0, np.radians(30), 0])).all()

def test_get_state_0():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    assert (m.get_state_0() == np.array([25, 0, 0, np.radians(30), 0])).all()

def test_set_state():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    state = np.array([30, 1, 1, np.radians(45), 0])
    m.set_state(state)
    assert (m.get_state() == state).all() and not (m.get_state_0() == m.get_state()).all()

def test_reset():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    state = np.array([30, 1, 1, np.radians(45), 0])
    m.set_state(state)
    m.reset()
    assert (m.get_state() == parameters_of_missile).all()


def test_action_space():
    m = Missile.get_needle()
    assert (m.action_space == np.array([-1, 0, 1])).all()

def test_action_sample():
    m = Missile.get_needle()
    action = m.action_sample()
    assert (action in m.action_space)

def test_pos():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    m.set_state(np.array([1, 2, 3, 4, 5]))
    assert (m.get_state()[1:3] == m.pos).all()

def test_vel():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    m.set_state(np.array([4, 0, 0, np.radians(0), 0]))
    assert (m.vel == np.array([4.0, 0.0])).all()

def test_step(self, action, tau):
    # TODO: add test
    pass

def test_x_axis(self):
    # TODO: add test
    pass
def test_get_summary(self):
    # TODO: add test
    pass
