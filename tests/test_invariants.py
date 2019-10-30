import sys
import pytest
import os
import sys
import numpy as np
from pytest import approx

wd = os.path.abspath(__file__) # os.path.dirname(os.path.dirname(os.getcwd()))
wd = os.path.dirname(os.path.dirname(wd))
sys.path.append(wd+"/src/")

from invariants import Interp1d, Interp2d

def test_missile_constructor():
    x = np.arange(0, 50, 1)
    y = x * x 
    f = Interp1d(x, y)
    assert f(3) == 9

test_missile_constructor()