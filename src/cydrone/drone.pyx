from easyvec.vectors cimport Vec2, real
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, fmax, pi



cpdef double get_rho(double h_km):
    return 3.33121912e-06 * h_km**4 - 1.67661954e-04 * h_km**3 + 5.15632148e-03* h_km**2 -1.19537867e-01* h_km + 1.22544656e+00

cpdef double get_T_T0(double h_km):
    return (get_rho(h_km) / 1.22544656e+00)**(1/3)


cpdef Vec2 get_B(double delta_t, Vec2 A, Vec2 vA):
    return A.add_vec(vA.mul_num(delta_t / 3))

cpdef Vec2 get_C(double delta_t, Vec2 D, Vec2 vD): 
    return  D.sub_vec(vD.mul_num(delta_t / 3))

cpdef double get_max_a(double delta_t, Vec2 A, Vec2 B, Vec2 C, Vec2 D):
    cdef Vec2 a1 = (C.sub_vec(B)).mul_num(6).sub_vec((B.sub_vec(A)).mul_num(6))
    cdef Vec2 a2 = (D.sub_vec(C)).mul_num(6).sub_vec((C.sub_vec(B)).mul_num(6))
    return fmax(a1.len(), a2.len())/delta_t/delta_t

cpdef (double, double) get_min_max_v(double delta_t, Vec2 A, Vec2 B, Vec2 C, Vec2 D, int n=42):
    cdef Vec2 BA = B.sub_vec(A)
    cdef Vec2 CB = C.sub_vec(B)
    cdef Vec2 DC = D.sub_vec(C)
    cdef double min_v = 1e13
    cdef double max_v = 0
    cdef int i
    cdef double dt = 1.0 / (n-1)
    cdef double t, vel_len
    for i in range(n):
        t = i * dt
        vel_len = (BA.mul_num(3*(1-t)*(1-t)/delta_t).add_vec(CB.mul_num(6*t*(1-t)/delta_t)).add_vec(DC.mul_num(3*t*t/delta_t))).len() 
        # print(t, vel_len)
        if vel_len < min_v:
            min_v = vel_len
        if vel_len > max_v:
            max_v = vel_len
    return min_v, max_v


cpdef (double, double) get_max_v_a(double delta_t, Vec2 A, Vec2 B, Vec2 C, Vec2 D, int n=33, int rounds=3):
    cdef Vec2 BA = B.sub_vec(A)
    cdef Vec2 CB = C.sub_vec(B)
    cdef Vec2 DC = D.sub_vec(C)
    cdef double min_v = 1e13
    cdef double max_v = 0
    cdef int i, j, imax
    cdef double dt = 1.0 / (n-1)
    cdef double t, vel_len, t0, t1
    t0 = 0.0
    t1 = 1.0
    for j in range(rounds):
        dt = (t1-t0) / (n-1)
        for i in range(n):
            t = t0 + i * dt
            vel_len = (BA.mul_num(3*(1-t)*(1-t)/delta_t).add_vec(CB.mul_num(6*t*(1-t)/delta_t)).add_vec(DC.mul_num(3*t*t/delta_t))).len() 
            if vel_len > max_v:
                max_v = vel_len
                imax = i
        t = t0 + imax * dt
        t0 = t - dt
        t1 = t + dt
        if t0 < 0:
            t0 = 0.0
        if t1 > 1:
            t1 = 1.0
    cdef Vec2 a1 = (CB.mul_num(6)).sub_vec(BA.mul_num(6))
    cdef Vec2 a2 = (DC.mul_num(6)).sub_vec(CB.mul_num(6))
    return max_v, fmax(a1.len(), a2.len())/delta_t/delta_t


cdef class Drone2d:
    @classmethod
    def get_DJI(cls):
        m = 5.0
        L = 0.5
        J = m*L**2
        return cls(m=m, J=J, F_max=42, L=L, Cx=0.2, g=9.81, mu_omega=1.0)

    cdef public double m, J, F_max, L, Cx, t, g, mu_omega
    cdef public double[:] state, tmp1, tmp2, tmp3, tmp4
    # [0, 1, 2,  3,  4,     5,     6]
    # [x, y, Vx, Vy, alpha, omega, t]
    
    def __init__(self, m, J, F_max, L, Cx, g, mu_omega):
        self.m = m
        self.J = J
        self.F_max = F_max
        self.L = L
        self.Cx = Cx
        self.g = g
        n = 7
        self.state = np.zeros(n)
        self.tmp1 = np.zeros(n)
        self.tmp2 = np.zeros(n)
        self.tmp3 = np.zeros(n)
        self.tmp4 = np.zeros(n)
        self.t = 0
        self.mu_omega = mu_omega

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
        dy[3] = (F_Cx_y + F_y)/self.m - self.g # dvy = ay
        dy[4] = y[5]                    # dalpha = omega
        dy[5] = (F2 - F1)*self.L/self.J - dy[5] * self.mu_omega     # domega
        dy[6] = 1   # dt
        # print(f'rho={rho} F_coeff={F_coeff} F1={F1} F2={F2} V={V} F_Cx={F_Cx} F_Cx_x={F_Cx_x} F_Cx_y={F_Cx_y} F_x={F_x} F_y={F_y}')

    cpdef void step(self, double F1, double F2, double tau=0.1, int n=10):
        """
        F1 [-1, 1]
        F2 [-1, 1]
        """
        cdef double dt = tau / n
        cdef size_t i, j
        F1 = 0.5*F1 + 0.5
        F2 = 0.5*F2 + 0.5
        # print(f'before {np.array(self.state)}')
        for i in range(n):
            self.fill_dy(self.t, self.state, F1, F2, self.tmp1) # k1
            for j in range(self.tmp1.shape[0]):
                self.tmp1[j] = self.state[j] + 0.5 * dt * self.tmp1[j]
            self.fill_dy(self.t + 0.5*dt, self.tmp1, F1, F2, self.tmp2) # k2
            for j in range(self.state.shape[0]):
                self.state[j] = self.state[j] + dt * self.tmp2[j]
            self.t += dt
            self.state[6] = self.t
            self.flip_alpha() 
            # print(f'after {np.array(self.state)}')


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
        self.state[6] = self.t

    cpdef void flip_alpha(self):
        if self.state[4] > pi:
            self.state[4] -= 2*pi
        elif self.state[4] <= -pi:
            self.state[4] += 2*pi


    def to_numpy(self):
        return np.array(self.state)

    def from_numpy(self, state):
        cdef size_t i 
        for i in range(self.state.shape[0]):
            self.state[i] = state[i]
        self.t = self.state[6]

    cpdef (double, double, double) get_vmin_vmax_amax(self, double delta_t, Vec2 pos_trg, Vec2 vel_trg, int n_vel_search=25):
        cdef Vec2 D = pos_trg
        cdef Vec2 A = Vec2(self.state[0], self.state[1])
        cdef Vec2 velA = Vec2(self.state[2], self.state[3])
        cdef Vec2 B = get_B(delta_t, A, velA)
        cdef Vec2 C = get_C(delta_t, D, vel_trg)
        cdef double amax = get_max_a(delta_t, A, B, C, D)
        cdef double vmin, vmax
        vmin, vmax = get_min_max_v(delta_t, A, B, C, D, n_vel_search)
        return vmin, vmax, amax

    cpdef double get_amax(self, double delta_t, Vec2 pos_trg, Vec2 vel_trg):
        cdef Vec2 D = pos_trg
        cdef Vec2 A = Vec2(self.state[0], self.state[1])
        cdef Vec2 velA = Vec2(self.state[2], self.state[3])
        cdef Vec2 B = get_B(delta_t, A, velA)
        cdef Vec2 C = get_C(delta_t, D, vel_trg)
        cdef double amax = get_max_a(delta_t, A, B, C, D)
        return amax

    def get_traject(self, delta_t, pos_trg, vel_trg, n_points=100):
        ts = np.linspace(0,1, n_points)
        cdef Vec2 A = Vec2(self.state[0], self.state[1])
        cdef Vec2 velA = Vec2(self.state[2], self.state[3])
        cdef Vec2 D = Vec2(pos_trg[0], pos_trg[1])
        cdef Vec2 B = get_B(delta_t, A, velA)
        cdef Vec2 C = get_C(delta_t, D, Vec2(vel_trg[0], vel_trg[1]))
        
        return np.array([
            (1-t)**3 * A + 3*t*(1-t)**2 * B + 3*t*t*(1-t)*C + t**3 * D
            for t in ts
        ])

    def get_traject_vels(self, delta_t, pos_trg, vel_trg, n_points=100):
        ts = np.linspace(0,1, n_points)
        cdef Vec2 A = Vec2(self.state[0], self.state[1])
        cdef Vec2 velA = Vec2(self.state[2], self.state[3])
        cdef Vec2 D = Vec2(pos_trg[0], pos_trg[1])
        cdef Vec2 B = get_B(delta_t, A, velA)
        cdef Vec2 C = get_C(delta_t, D, Vec2(vel_trg[0], vel_trg[1]))
        
        return np.array([
            (3*(1-t)**2*(B-A) + 6*t*(1-t)*(C-B) + 3*t**2*(D-C))/delta_t
            for t in ts
        ])

    def get_delta_t_minimum(self, pos_trg, vel_trg, vmax, amax, t_tol, n=33, rounds=3):
        cdef double t1, t2, t3, vmax_fact, amax_fact, vmax_, amax_, t_tol_
        cdef Vec2 A, VelA, B, C, D, velD
        cdef int t1_flag, t2_flag, t3_flag
        cdef int n_ = n
        cdef int rounds_ = rounds
        vmax_ = vmax
        amax_ = amax
        t_tol_ = t_tol

        A = Vec2(self.state[0], self.state[1])
        velA = Vec2(self.state[2], self.state[3])
        D = Vec2(pos_trg[0], pos_trg[1])
        velD = Vec2(vel_trg[0], vel_trg[1])

        t1 = 0
        t3 = A.sub_vec(velD).len() / vmax_
        if velD.len() > vmax_:
            vmax_ = velD.len()*1.01
        if velA.len() > vmax_:
            vmax_ = velA.len()*1.01

        while True:
            B = get_B(t3, A, velA)
            C = get_C(t3, D, velD)
            vmax_fact, amax_fact = get_max_v_a(t3, A, B, C, D, n_, rounds_)
            if vmax_fact > vmax_ or amax_fact > amax_:
                t3 *= 1.618
            else:
                break
        
        while t3 - t1 > t_tol_:
            t2 = (t1 + t3)/2
            B = get_B(t2, A, velA)
            C = get_C(t2, D, velD)
            vmax_fact, amax_fact = get_max_v_a(t2, A, B, C, D, n_, rounds_)
            if vmax_fact > vmax_ or amax_fact > amax_:
                t1 = t2
            else:
                t3 = t2

        return t2
        
             
            

            
        



    @property
    def vel(self):
        return Vec2(self.state[2], self.state[3])
    
    @property
    def vel_np(self):
        return np.array([self.state[2], self.state[3]])


    @property
    def pos(self):
        return Vec2(self.state[0], self.state[1])

    @property
    def pos_np(self):
        return np.array([self.state[0], self.state[1]])

    @property
    def alpha(self):
        return self.state[4]

    @property
    def omega(self):
        return self.state[5]


cpdef real test(int a):
    cdef Vec2 v = Vec2(1,2) 
    cdef real b = a
    return v.x + b