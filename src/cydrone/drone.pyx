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
    cdef public double[:] state, tmp1, tmp2, tmp3, tmp4
    # [0, 1, 2,  3,  4,     5]
    # [x, y, Vx, Vy, alpha, omega]
    
    def __init__(self, m, J, F_max, L, Cx):
        self.m = m
        self.J = J
        self.F_max = F_max
        self.L = L
        self.Cx = Cx
        n = 6
        self.state = np.zeros(n)
        self.tmp1 = np.zeros(n)
        self.tmp2 = np.zeros(n)
        self.tmp3 = np.zeros(n)
        self.tmp4 = np.zeros(n)
        self.t = 0

    cpdef void fill_dy(self, double t, double[:] y, double F1, double F2, double[:] dy):
        cdef double rho = get_rho(y[1]/1000)
        cdef double F_coeff = get_T_T0(y[1]/1000)
        F1 *= F_coeff * self.F_max
        F2 *= F_coeff * self.F_max

        cdef double V = (y[2]**2 + y[3]**2)**0.5
        cdef double F_Cx, F_Cx_x, F_Cx_y 
        if V < 1e-8:
            F_Cx = 0
            F_Cx_x = 0
            F_Cx_y = 0
        else:
            F_Cx = self.Cx * rho * V**2 / 2
            F_Cx_x = -F_Cx*y[2]/V 
            F_Cx_y = -F_Cx*y[3]/V 
        cdef double F_x = -(F1 + F2) * sin(y[4])
        cdef double F_y =  (F1 + F2) * cos(y[4])

        dy[0] = y[2]                    # dx = vx
        dy[1] = y[3]                    # dy = vy
        dy[2] = (F_Cx_x + F_x)/self.m        # dvx = ax
        dy[3] = (F_Cx_y + F_y)/self.m - 9.81 # dvy = ay
        dy[4] = y[5]                    # dalpha = omega
        dy[5] = (F2 - F1)*self.L/self.J      # domega
        # print(f'rho{rho} F_coeff{F_coeff} F1{F1} F2{F2} V{V} F_Cx{F_Cx} F_Cx_x{F_Cx_x} F_Cx_y{F_Cx_y} F_x{F_x} F_y{F_y}')

    cpdef void step(self, double F1, double F2, double tau=0.1, int n=10):
        """
        F1 [0, 1]
        F2 [0, 1]
        """
        cdef double dt = tau / n
        cdef size_t i, j
        for i in range(n):
            self.fill_dy(self.t, self.state, F1, F2, self.tmp1) # k1
            for j in range(self.tmp1.shape[0]):
                self.tmp1[j] = self.state[j] + 0.5 * dt * self.tmp1[j]
            self.fill_dy(self.t + 0.5*dt, self.tmp1, F1, F2, self.tmp2) # k2
            for j in range(self.state.shape[0]):
                self.state[j] = self.state[j] + dt * self.tmp2[j]
            self.t += dt

    cpdef dict to_dict(self):
        cdef dict res = {
            't': self.t,
            'pos': Vec2(self.state[0], self.state[1]),
            'vel': Vec2(self.state[2], self.state[3]),
            'alpha': self.state[4],
            'omega': self.state[5]
        }
        return res

    cpdef void from_dict(self, dict state_dict):
        self.t = state_dict['t']
        self.state[0] = state_dict['pos'][0]
        self.state[1] = state_dict['pos'][1]
        self.state[2] = state_dict['vel'][0]
        self.state[3] = state_dict['vel'][1]
        self.state[4] = state_dict['alpha']
        self.state[5] = state_dict['omega']

    def to_numpy(self):
        return np.array(self.state)

    def from_numpy(self, state):
        self.state[:] = state[:]


cpdef real test(int a):
    cdef Vec2 v = Vec2(1,2) 
    cdef real b = a
    return v.x + b