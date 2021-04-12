import numpy as np
from cydrone.drone import Drone2d
from easyvec import Vec2

d = Drone2d(m=1, J=2, F_max=3, L=1, Cx=0)
print(d.to_dict())

class DroneGym:
    def __init__(self, drone: Drone2d, pos0: Vec2, pos_trg: Vec2, vel_len_trg: float):
        self.time_curr = 0.0
        self.drone = drone
        self.pos0 = pos0.copy()
        self.pos_trg = pos_trg.copy()
        self.vel_len_trg = vel_len_trg

    