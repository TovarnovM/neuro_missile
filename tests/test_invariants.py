import sys
import pytest
import os
import sys
import numpy as np
from pytest import approx

wd = os.path.abspath(__file__)
wd = os.path.dirname(os.path.dirname(wd))
sys.path.append(wd+"/src/")

from invariants import Interp1d, Interp2d, InterpVec

def test_Interp1d():
    x = np.arange(0, 50, 1)
    y = x * x 
    f = Interp1d(x, y)
    assert f(3) == 9

def test_Interp2d():
    xx = np.arange(-5, 5, 0.5)
    yy = np.arange(-5, 5, 0.5)
    def f(x, y):
        return x**2 + y**2
    zz = f(*np.meshgrid(xx, yy))
    FF = Interp2d(xx, yy, zz)
    assert 0 == approx(FF(0,0))


def test_constantInterp1d():
    f = Interp1d.simple_constant(2)
    assert f(10) == approx(2)

def test_constantInterp2d():
    f = Interp2d.simple_constant(2)
    assert f(10,100) == approx(2)

def test_constantInterpVec():
    data = [(-1,(2,3)), (0,[0,0]), (2, (3,4))]
    iv = InterpVec(data)
    assert iv(-0.5) == approx([1,1.5])
    assert iv(1) == approx([1.5,2])
    assert iv(-10) == approx([2,3])
    assert iv(10) == approx([3,4])