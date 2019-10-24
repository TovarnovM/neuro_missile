import sys
import pytest
import os
import sys
import numpy as np
from pytest import approx

wd = os.path.abspath(__file__) # os.path.dirname(os.path.dirname(os.getcwd()))
wd = os.path.dirname(os.path.dirname(wd))
sys.path.append(wd+"\\src\\")

from missile import Missile

def test_missile_constructor():
    m = Missile.get_needle()
    assert m is not None


def test_missile_init_cond():
    m = Missile.get_needle()
    parametrs_of_missile = Missile.get_standart_parametrs_of_missile()
    m.set_init_cond(parametrs_of_missile)
    m.step(m.action_sample(), 0.1)

# TODO покрыть тестами все методы класса Missile

