import sys
import pytest
import os
import sys
import numpy as np
from pytest import approx

wd = os.path.abspath(__file__) # os.path.dirname(os.path.dirname(os.getcwd()))
wd = os.path.dirname(os.path.dirname(wd))
sys.path.append(wd+"\\src\\")

from invariants import Interp1d, Interp2d

# TODO сделать необходимые тесты