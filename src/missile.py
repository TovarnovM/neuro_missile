import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

class Missile(object):
    def __init__(self, parametrs_of_missile=None):
        self.d = parametrs_of_missile.get('d', 0)
        self.v_sr = parametrs_of_missile.get('v_sr', 0)
        self.mu_st = parametrs_of_missile.get('mu_st', 0)
        self.mu_mr = parametrs_of_missile.get('mu_mr', 0)
        self.alphamax = parametrs_of_missile.get('alphamax', 0)
        self.speed_change_alpha = parametrs_of_missile.get('speed_change_alpha', 0)
        
        self.J = parametrs_of_missile.get('J', 0)
        self.xi = parametrs_of_missile.get('xi', 0)
        self.Cx0 = parametrs_of_missile.get('Cx0', 0)
        self.Cya = parametrs_of_missile.get('Cya', 0)   
        self.betta = parametrs_of_missile.get('betta', 0)
        self.K_eff = parametrs_of_missile.get('K_eff', 0)
        
        self.t_mr = parametrs_of_missile.get('t_mr', 0)
        self.t_st = parametrs_of_missile.get('t_st', 0)

        self.t0 = parametrs_of_missile.get('t0', 0)
        self.v0 = parametrs_of_missile.get('v0', 0)
        self.x0 = parametrs_of_missile.get('x0', 0)
        self.y0 = parametrs_of_missile.get('y0', 0)
        self.m0 = parametrs_of_missile.get('m0', 0)
        self.tetta0 = parametrs_of_missile.get('tetta0', 0)
        self.alpha0 = parametrs_of_missile.get('alpha0', 0)

        self.Sm = np.pi * self.d ** 2 / 4

        self.m_pn = self.m0 - ( self.mu_st * self.m0 + self.mu_mr * ( self.m0 - self.mu_st * self.m0 ) )

        self.w_st = self.mu_st * self.m0
        self.w_mr = self.mu_mr * (self.m0 - self.w_st)
        self.G_st = self.w_st / self.t_st
        self.G_mr = self.w_mr / self.t_mr
        
        self.signal = 0

        self.t = [self.t0]
        self.v = [self.v0]
        self.x = [self.x0]
        self.y = [self.y0]
        self.tetta = [self.tetta0]
        self.alpha = [self.alpha0]
        self.P = [self.__get_thrust(self.t0)]
        self.m = [self.__get_weight(self.t0)]
        self.q = [self.__get_thrust(self.t0)]
        self.Cx = [self.__get_thrust(self.t0)]


    def __get_weight(self, t):
        if t < self.t_st:
            return round(self.m0 - self.G_st * t, 4)
        elif self.t_st <= t < self.t_mr + self.t_st:
            return round(self.m0 - self.w_st - self.G_mr * (t - self.t_st), 4)
        else:
            return self.m0 - self.w_st - self.w_mr

    def __get_thrust(self, t):
        if t < self.t_st:
          return round(self.K_eff * self.G_st * self.J, 4)
        elif self.t_st <= t < self.t_mr + self.t_st:
          return round(self.K_eff * self.G_mr * self.J, 4)
        else:
          return 0
    
    def __get_speeding_pressure(self, v):
        return ro * v ** 2 / 2

    def __get_Cx(self, alpha):
        return round(self.Cx0 + self.Cya * alpha / 57.3, 4)

    def __get_alpha(self, signal):
        if signal == 1 and self.alpha[-1] <= self.alphamax :
            return round(self.alpha[-1] + self.speed_change_alpha, 4)
        elif signal == -1 and np.abs(self.alpha[-1]) <= self.alphamax:
            return round(self.alpha[-1] - self.speed_change_alpha, 4)
        else:
            return self.alpha[-1]

    def set_thrust(self, t):
        self.P.append(self.__get_thrust(t))

    def set_weigth(self, t):
        self.m.append(self.__get_weight(t))

    def set_speeding_pressure(self, v):
        self.q.append(self.__get_speeding_pressure(v))

    def set_Cx(self, alpha):
        self.Cx.append(self.__get_Cx(alpha))

    def set_speed(self, next_v):
        self.v.append(next_v)

    def set_coordinates(self, *args, **kwargs):
        self.x.append(kwargs['x'])
        self.y.append(kwargs['y'])

    def set_tetta(self, next_tetta):
        self.tetta.append(next_tetta)

    def set_time(self, next_time):
        self.t.append(next_time)

    def set_alpha(self, signal):
        print(self.__get_alpha(signal))
        self.alpha.append(self.__get_alpha(signal))

    def get_solve(self, signal=0, condition=True):
        def system_du(t, y):
            v, x, y, Q = y
            
            P = self.__get_thrust(t)
            m = self.__get_weight(t)
            q = self.__get_speeding_pressure(v)
            alpha = self.__get_alpha(self.signal)
            Cx = self.__get_Cx(alpha)
            return [
                round(( P * np.cos(np.radians(alpha)) - q * self.Sm * Cx - m * g * np.sin(Q) ) / m, 4),
                round(v * np.cos(Q), 4),
                round(v * np.sin(Q), 4),
                round(( alpha * ( self.Cya * q * self.Sm * (1 + self.xi) + P / 57.3) / ( m * g ) - np.cos(Q)) * g / v, 4)
            ]

        
        OD = ode(system_du)
        r = OD.set_integrator('dopri5')
        r = OD.set_initial_value( [self.v[-1], self.x[-1], self.y[-1], np.radians(self.tetta[-1])], self.t[-1] ) 

        
        while OD.successful() and self.P[-1] > 0 and OD.y[2] >= 0 and condition:
            self.set_speed(round(OD.y[0], 4))
            self.set_coordinates(x = round(OD.y[1], 4), y = round(OD.y[2], 4))
            self.set_tetta(round(OD.y[3], 4))
            self.set_time(OD.t)
            self.set_thrust(OD.t)
            self.set_weigth(OD.t)
            self.set_speeding_pressure(round(OD.y[0], 4))
            self.set_alpha(self.signal)
            self.set_Cx(self.alpha[-1])

            OD.integrate(round(OD.t + dt, 4))

            # TODO: Move signal to input function
            self.signal = random.randint(-1, 1)


    def get_plot(self):
        plt.figure(1)
        plt.subplot(611)
        plt.plot(self.x, self.y)
        plt.ylabel('Y(X)')
        plt.subplot(612)
        plt.plot(self.t, self.v)
        plt.ylabel('V(t)')
        plt.subplot(613)
        plt.plot(self.t, self.tetta)
        plt.ylabel('Tetta(t)')
        plt.subplot(614)
        plt.plot(self.t, self.m)
        plt.ylabel('m(t)')
        plt.subplot(615)
        plt.plot(self.t, self.P)
        plt.ylabel('P(t)')
        plt.subplot(616)
        plt.plot(self.t, self.alpha)
        plt.ylabel('alpha(t)')
        plt.show()

    def reset_state(self):
        self.t = [self.t0]
        self.v = [self.v0]
        self.x = [self.x0]
        self.y = [self.y0]
        self.tetta = [self.tetta0]
        self.P = [self.__get_thrust(self.t0)]
        self.m = [self.__get_weight(self.t0)]
        self.q = [self.__get_thrust(self.t0)]
        self.Cx = [self.__get_thrust(self.t0)]

    def get_current_state(self):
        return {
            't': self.t[-1],
            'v': self.v[-1],
            'x': self.x[-1],
            'y': self.y[-1],
            'tetta': self.tetta[-1],
            'P': self.P[-1],
            'm': self.m[-1],
            'q': self.q[-1],
            'Cx': self.Cx[-1]
        }

    def get_state(self):
        return {
            't': self.t,
            'v': self.v,
            'x': self.x,
            'y': self.y,
            'tetta': self.tetta,
            'P': self.P,
            'm': self.m,
            'q': self.q,
            'Cx': self.Cx
        }



ro = 1.202
dt = 0.01
g = 9.81

missile_parametrs = {
    'd': 0.127,
    'v_sr': 500,
    'mu_st': 0.27,
    'mu_mr': 0.39,
    't_st': 6.0,
    't_mr': 3.0,
    'alphamax': 15,
    'J': 1900,
    'xi': 2,
    'Cx0': 0.35,
    'Cya': 0.04,
    'betta': 1.9,
    'K_eff': 0.3,
    't0': 0,
    'v0': 25,
    'x0': 0,
    'y0': 0,
    'm0': 11.8,
    'tetta0': 30,
    'speed_change_alpha': 0.1
}


missile = Missile(missile_parametrs)
missile.get_solve()
missile.get_plot()