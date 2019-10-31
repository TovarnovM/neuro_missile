import sys
import pytest
import os
import sys
import numpy as np
from pytest import approx

wd = os.path.abspath(__file__)
wd = os.path.dirname(os.path.dirname(wd))
sys.path.append(wd+"/src/")

from missile import Missile

def test_missile_constructor():
    m = Missile.get_needle()
    assert m is not None

def test_get_standart_parameters_of_missile():
    parametrs_of_missile = Missile.get_standart_parameters_of_missile()
    assert parametrs_of_missile == approx(np.array([25, 0, 0, np.radians(30), 0]))

def test_set_init_cond():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    assert m.get_state() == approx(m.get_state_0())

def test_get_state():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    assert m.get_state() == approx(np.array([25, 0, 0, np.radians(30), 0]))

def test_get_state_0():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    assert m.get_state_0() == approx(np.array([25, 0, 0, np.radians(30), 0]))

def test_set_state():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    state = np.array([30, 1, 1, np.radians(45), 0])
    m.set_state(state)
    assert m.get_state() == approx(state) and not m.get_state_0() == approx(m.get_state())

def test_reset():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    state = np.array([30, 1, 1, np.radians(45), 0])
    m.set_state(state)
    m.reset()
    assert m.get_state() == approx(parameters_of_missile)


def test_action_space():
    m = Missile.get_needle()
    assert m.action_space == approx(np.array([-1, 0, 1]))

def test_action_sample():
    m = Missile.get_needle()
    action = m.action_sample()
    assert (action in m.action_space)

def test_pos():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    m.set_state(np.array([1, 2, 3, 4, 5]))
    assert m.get_state()[1:3] == approx(m.pos)

def test_vel():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    m.set_state(np.array([4, 0, 0, np.radians(0), 0]))
    assert m.vel == approx(np.array([4.0, 0.0]))

def test_step():
    m = Missile.get_needle()
    parameters_of_missile = Missile.get_standart_parameters_of_missile()
    m.set_init_cond(parameters_of_missile)
    m.step(1, 1)
    m.step(0, 1)
    m.step(-1, 1)
    state = m.get_state()
    assert state == approx(np.array([88.48271373, 148.07413746, 78.54580254, 0.30156225, 3.002]))

