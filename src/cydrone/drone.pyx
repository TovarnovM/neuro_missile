from easyvec.vectors cimport Vec2, real
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos



cpdef double get_rho(double h_km):
    return 3.33121912e-06 * h_km**4 - 1.67661954e-04 * h_km**3 + 5.15632148e-03* h_km**2 -1.19537867e-01* h_km + 1.22544656e+00

cpdef double get_T_T0(double h_km):
    return (get_rho(h_km) / 1.22544656e+00)**(1/3)


cdef class Drone2d:
    cdef public double m, J, F_max, L, Cx, t
    cdef public double[:] state
    # [0, 1, 2,  3,  4,     5]
    # [x, y, Vx, Vy, alpha, omega]
    
    def __init__(self, m, J, F_max, L, Cx):
        self.m = m
        self.J = J
        self.F_max = F_max
        self.L = L
        self.Cx = Cx
        self.state = np.zeros(6)
        self.t = 0

    cpdef double[:] fill_dy(self, double t, double[:] y, double F1, double F2, double[:] dy):
        cdef double rho = get_rho(y[1]/1000)
        cdef double F_coeff = get_T_T0(y[1]/1000)
        F1 *= F_coeff * self.F_max
        F2 *= F_coeff * self.F_max

        cdef double V = (y[2]**2 + y[3]**2)**0.5
        cdef double F_Cx = self.Cx * rho * V**2 / 2
        cdef double F_Cx_x = -F_Cx*y[2]/V 
        cdef double F_Cx_y = -F_Cx*y[3]/V 
        cdef double F_x = -(F1 + F2) * sin(y[4])
        cdef double F_y =  (F1 + F2) * cos(y[4])

        dy[0] = y[2]                    # dx = vx
        dy[1] = y[3]                    # dy = vy
        dy[2] = (F_Cx_x + F_x)/self.m        # dvx = ax
        dy[3] = (F_Cx_y + F_y)/self.m - 9.81 # dvy = ay
        dy[4] = y[5]                    # dalpha = omega
        dy[5] = (F2 - F1)*self.L/self.J      # domega

    cpdef void step(self, double F1, double F2, double tau=0.1, int n=10):
        pass



cpdef real test(int a):
    cdef Vec2 v = Vec2(1,2) 
    cdef real b = a
    return v.x + b