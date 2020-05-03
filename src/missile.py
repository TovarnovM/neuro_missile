import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import ode
from invariants import Interp1d, Interp2d, InterpVec 
from constants.scenarios import scenarios
from math import *
import os

class Missile(object):
    """Класс ракета, которая которая летает в двухмерном пространстве в поле силы тяжести и которой можно управлять) 
    
    Ракета имеет свои аэродинамические и массо- и тяговременные характеристики. Данные характеристики задаются 
    в виде простых констант и интерполированных таблиц (классов Interp1d и Interp2d в модуле invariants.py)
    
    Ракета имеет свое текущее положение и скорость ц.м. в пространстве, угол наклона оси ракеты к горизонту, угол атаки, 
    и угол атаки, к которому ракета стремится с определенной скоростью. Управление как раз и происходит путем выбора
    потребного угла атаки с определенной частотой. Множестов возможных потребыных углов атаки будем полагать дискретным.

    Класс предоставляет необходимые методы для моделироания динамики ракеты по времени

    Класс является мутабельным и представляет состояние ракеты на текущий момент времени. Класс не хранит историю 
    изменений сових параметров
    """

    @classmethod
    def get_needle(cls):
        """Классовый метод создания стандартной ракеты (иглы) со всеми необходимыми аэродинамическими, 
        массо- и тяговременными характеристиками

        returns Missile
        """
        m0 = 11.8
        d = 0.072
        t_st = 4.5
        t_mr = 8.2
        
        Sm = np.pi * d ** 2 / 4
        w_st = 3.24
        w_mr = 1.357
        G_st = w_st / t_st
        G_mr = w_mr / t_mr
        
        P_st = 1787
        P_mr = 456
        J = 2700

        @np.vectorize
        def _get_m(t):
            if t < t_st:
                return m0 - G_st * t
            elif t_st <= t < t_mr + t_st:
                return m0 - w_st - G_mr * (t - t_st)
            else:
                return m0 - w_st - w_mr

        @np.vectorize
        def _get_P(t):
            if t < t_st:
                return P_st
            elif t_st <= t < t_mr + t_st:
                return P_mr
            else:
                return 0

        ts = np.linspace(0, t_st + t_mr + 0.5, 100)
        m_itr = Interp1d(ts, _get_m(ts))
        P_itr = Interp1d(ts, _get_P(ts))

        wd = os.path.abspath(__file__)
        wd = os.path.dirname(wd)
        fp = os.path.join(wd, 'constants/aerodynamic.csv')
        df = pd.read_csv(fp)
        df = df[df['D'] == 0]

        alpha_from_csv = np.unique(df['A'].to_numpy())
        M_from_csv = np.array(np.unique(df['M'].to_numpy()))
        Cx_from_csv = np.array(np.split(df['Cx'].to_numpy(), M_from_csv.size)).T
        Cya_from_csv = np.array(np.split(df['Cya'].to_numpy(), M_from_csv.size)).T

        M_for_Cya = np.array([0.6,0.9,1.1,1.5,2.0])
        Cya_from_mathcad = np.array([0.306,0.341,0.246,0.246,0.218])
        Cya_itr = Interp1d(M_for_Cya, Cya_from_mathcad)
        
        Cx_itr = Interp2d(alpha_from_csv, M_from_csv, Cx_from_csv)

        ro_itr = Interp1d(
            [0,    50,   100,  200,  300,  500,  1000, 2000, 3000, 5000, 8000, 1000, 12000,15000,20000],
            [1.225,1.219,1.213,1.202,1.190,1.167,1.112,1.007,0.909,0.736,0.526,0.414,0.312,0.195,0.089]
        )
        a_itr = Interp1d(
            [0,     50,   100,   200,   300,   400,   500,   600,   700,   800,   900,   1000,  5000,  10000, 20000],
            [340.29,340.1,339.91,339.53,339.14,338.76,338.38,337.98,337.60,337.21,336.82,336.43,320.54,299.53,295.07]
        ) 

        missile = cls(
            m_itr=m_itr,   # масса [кг] ракеты от времени [с]
            P_itr=P_itr,   # тяга [Н] ракеты от времени [с]
            Sm=Sm,         # площадь миделя [м^2] (к оторой относятся аэро коэффициенты)
            alphamax=15,   # максимальный угол атаки [градусы]
            speed_change_alpha=360,  # скорость изменения угола атаки [градусы / с]
            xi=0.1,           # коэффициент, характеризующий структуру подъёмной силы аэродинамической схемы ракеты [] TODO уточнить его
            Cx_itr=Cx_itr,              # интерполятор определения коэффициента лобового сопротивления ракеты от угла атаки [градусы] и от числа маха
            Cya_itr=Cya_itr,
            ro_itr=ro_itr,   # интерполятор определения плотности атмосферы [кг/м^3] от высоты [м] 
            a_itr=a_itr,      # интерполятор определения скорости звука воздуха [м/с] от высоты [м] 
            r_kill=15
        )
        return missile

    @staticmethod
    def get_standart_parameters_of_missile():
        """Возвращает словарь (или что там принимает метод set_init_cond) для стандартного начального состояния ракеты
        Returns:
            [np.ndarray] -- [v, x, y, Q, alpha, t]
        """
        return np.array([25, 0, 0, np.radians(30), 0, 0])    

    @classmethod
    def get_parameters_of_missile_to_meeting_target(cls, trg_pos, trg_vel, missile_pos=None, missile_vel_abs=500.0):
        """Возвращает состоняие ракеты, которая нацелена на мгновенную точку встречи с целью
        
        Arguments:
            trg_vel {tuple/list/np.ndarray} -- вектор скорости цели
            trg_pos {tuple/list/np.ndarray} -- положение цели
        
        Keyword Arguments:
            my_pos {tuple/list/np.ndarray} -- начальное положение ракеты, если не указано, то (0,0) (default: {None})
            my_vel_abs {float} -- средняя скорость ракеты (default: {500})
        
        Returns:
            [np.ndarray] -- [v, x, y, Q, alpha, t]
        """
        trg_vel = np.array(trg_vel)
        trg_pos = np.array(trg_pos)
        missile_pos = np.array(missile_pos) if missile_pos else np.array([0, 0])
        suc, meeting_point = cls.get_instant_meeting_point(trg_pos, trg_vel, missile_vel_abs, missile_pos)
        vis = meeting_point - missile_pos
        Q = np.arctan2(vis[1], vis[0])
        return np.array([25, missile_pos[0], missile_pos[1], Q, 0, 0]) 
    
    def __init__(self, *args, **kwargs):
        """Конструктор 
        """
        
        self.dt = kwargs.get('dt', 0.001)
        self.g = kwargs.get('g', 9.81)

        self.Sm = kwargs["Sm"]
        self.alphamax = kwargs["alphamax"]
        self.speed_change_alpha = kwargs["speed_change_alpha"]
        self.xi = kwargs["xi"]


        self.P_itr = kwargs["P_itr"]
        self.m_itr = kwargs["m_itr"]

        self.Cx_itr = kwargs["Cx_itr"]
        self.Cya_itr = kwargs["Cya_itr"]
        self.ro_itr = kwargs['ro_itr']
        self.a_itr = kwargs['a_itr']
        self.r_kill = kwargs['r_kill']

        self.alpha_targeting = 0
        
    def set_init_cond(self, parameters_of_missile=None):
        """Задает начальные параметры (положение, скорость, углы ...) и запоминает их для того,
        чтобы потом в них можно было вернуться при помощи reset()
        
        Arguments:

            parameters_of_missile 
        """
        if parameters_of_missile is None:
            parameters_of_missile = self.get_standart_parameters_of_missile()
        self.state = np.array(parameters_of_missile)
        self.state_0 = np.array(parameters_of_missile)

    def reset(self):
        """Возвращает ракету в начальное состояние
        """
        self.set_state(self.state_0)

    def get_state(self):
        """Метод получения вектора со всеми параметрами системы 
        (схож с вектором 'y' при интегрировании ode, но в векторе state еще должно быть t)
        
        Returns:
            [np.ndarray] -- [v,   x, y, Q,       alpha,   t]
                            [м/с, м, м, радианы, градусы, с]
        """
        return self.state
    
    def get_state_0(self):
        """Метод получения вектора со всеми параметрами системы 
        (схож с вектором 'y' при интегрировании ode, но в векторе state еще должно быть t)
        
        Returns:
            [np.ndarray] -- [v,   x, y, Q,       alpha,   t]
                            [м/с, м, м, радианы, градусы, с]
        """
        return self.state_0

    def set_state(self, state):
        """Метод задания нового (может полностью отличающегося от текущего) состояния ракеты

        Arguments:
            state - np.ndarray  (схож с вектором 'y' при интегрировании ode, но в векторе state еще должно быть t)
            [np.ndarray] -- [v,   x, y, Q,       alpha,   t]
                            [м/с, м, м, радианы, градусы, с]
        """
        self.state = np.array(state)

    def _get_dydt(self, t, y):
        """Функция правых частей системы ОДУ динамики ракеты. 
        !!! Функция не должна зависеть от мутабельных параметров объекта !!!
        Т.е. значения всех дифференцируемых параметров должны браться из 'y'
        
        Arguments:
            t {float} -- время
            y {np.ndarray} -- вектор состояния системы 
                            [v,   x, y, Q,       alpha   ]
                            [м/с, м, м, радианы, градусы ]
        returns {np.ndarray} -- dy/dt
                            [dv,    dx,  dy,  dQ,        dalpha    ]
                            [м/с^2, м/c, м/c, радианы/c, градусы/c ]
        """
        v, x, y, Q, alpha = y
        P = self.P_itr(t)
        m = self.m_itr(t)
        ro = self.ro_itr(y)
        a = self.a_itr(y)
        M = v/a
        Cya = self.Cya_itr(M)
        Cx = self.Cx_itr(alpha, M)

        alpha_diff = self.alpha_targeting - alpha
        dalpha = 0                       if abs(alpha_diff) < 1e-6 else \
                 self.speed_change_alpha if alpha_diff > 0 else \
                -self.speed_change_alpha  

        return np.array([
            ( P * np.cos(np.radians(alpha)) - ro * v ** 2 / 2 * self.Sm * Cx - m * self.g * np.sin(Q) ) / m,
            v * np.cos(Q),
            v * np.sin(Q),
            ( alpha * ( Cya * ro * v ** 2 / 2 * self.Sm * (1 + self.xi) + P / 57.3) / ( m * self.g ) - np.cos(Q)) * self.g / v,
            dalpha
        ], copy=False) 

    def _validate_y(self, y):
        if y[4] > self.alphamax:
            y[4] = self.alphamax
        elif y[4] < -self.alphamax:
            y[4] = -self.alphamax
        elif abs(y[4] - self.alpha_targeting) < 1e-4:
            y[4] = self.alpha_targeting
        if y[3] < -180 or y[3] > 180: # Q
            y[3] = y[3] % (2 * pi)
            y[3] = (y[3] + 2 * pi) % (2 * pi)
            if y[3] > pi:
                y[3] -= 2*pi
        return y

    def get_action_parallel_guidance(self, target, am=10, dny=1 ):
        """Метод, возвращающий аналог action'a, соответствующий идельному методу параллельного сближения
        
        Arguments:
            target {object} -- ссылка на цель. Обязательно должен иметь два свойства: pos->np.ndarray и vel->np.ndarray. 
                               Эти свойства аналогичны свойствам этого класса. pos возвращает координату цели, vel -- скорость
            am -- Коэффициент быстродействия
            dny --  Запас по перегрузке
        
        returns {float} -- [-1; 1] аналог action'a, только не int, а float. Если умножить его на self.alphamax, то получится
                           потребный угол атаки для обеспечения метода параллельного сближения
        """

        xc, yc = target.pos
        Qc = target.Q
        vc = target.v

        v, x, y, Q, alpha, t = self.state
        P = self.P_itr(t)
        m = self.m_itr(t)
        ro = self.ro_itr(y)
        a = self.a_itr(y)
        M = v/a
        Cya = self.Cya_itr(M)

        vis = target.pos - self.pos
        # Угол между линией визирования и горизонтом [рад]
        fi = np.arctan2(vis[1], vis[0])
        # Линия визирования
        r = np.linalg.norm(vis)

        vel_c_otn = target.vel - self.vel
        vis1 = vis / r
        vel_c_otn_tau = vis1 * np.dot(vis1, vel_c_otn)
        vel_c_otn_n = vel_c_otn - vel_c_otn_tau

        dfi_dt = copysign(np.linalg.norm(vel_c_otn_n)/r, np.cross(vis1, vel_c_otn_n))

        dQ_dt = am * dfi_dt
        nya = v * dQ_dt / self.g + np.cos(Q) + dny
        alpha_req = (nya *  m * self.g) / (Cya * ro * v ** 2 / 2 * self.Sm * (1 + self.xi) + P / 57.3)

        return alpha_req / self.alphamax

    def get_action_chaise_guidance(self, target, t_corr=1/30, dny=1):
        """Метод, возвращающий аналог action'a, соответствующий идельному методу чистой погони
        
        Arguments:
            target {object} -- ссылка на цель. Обязательно должен иметь два свойства: pos->np.ndarray и vel->np.ndarray. 
                               Эти свойства аналогичны свойствам этого класса. pos возвращает координату цели, vel -- скорость
        
        returns {float} -- [-1; 1] аналог action'a, только не int, а float. Если умножить его на self.alphamax, то получится
                           потребный угол атаки для обеспечения метода параллельного сближения
        """

        xc, yc = target.pos
        Qc = target.Q
        vc = target.v

        v, x, y, Q, alpha, t = self.state
        P = self.P_itr(t)
        m = self.m_itr(t)
        ro = self.ro_itr(y)
        a = self.a_itr(y)
        M = v/a
        Cya = self.Cya_itr(M)

        vis = target.pos + vc*t_corr - self.pos
        # Угол между линией визирования и горизонтом [рад]
        fi2 = np.arctan2(vis[1], vis[0])
        fi1 = Q

        dQ_dt = (fi2-fi1)/t_corr
        nya = v * dQ_dt / self.g + np.cos(Q) + dny
        alpha_req = (nya *  m * self.g) / (Cya * ro * v ** 2 / 2 * self.Sm * (1 + self.xi) + P / 57.3)

        return alpha_req / self.alphamax

    def step(self, action, tau):
        """Моделирует динамику ракеты за шаг по времени tau. 
        На протяжении tau управляющее воздействие на ракету постоянно (action)
        Меняет внутреннее состояние ракеты на момент окончания шага
        
        Arguments:
            action {int} -- управляющее воздействие на протяжении шага
            tau {float} -- длина шага по времени (не путать с шагом интегрирования)
        """
        
        self.alpha_targeting = self.alphamax * action

        y = self._validate_y(self.state[:-1])
        t = self.state[-1]  
        t_end = t + tau

        flag = True
        while flag:
            if t_end - t > self.dt:
                dt = self.dt 
            else:
                dt = t_end - t
                flag = False
            k1 = self._get_dydt(t, y)
            k2 = self._get_dydt(t + 0.5 * dt, self._validate_y(y + 0.5 * dt * k1))
            k3 = self._get_dydt(t + 0.5 * dt, self._validate_y(y + 0.5 * dt * k2))
            k4 = self._get_dydt(t + dt, self._validate_y(y+dt*k3))
            t += dt
            y = self._validate_y(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        self.set_state(np.append(y,t))

    @property
    def action_space(self):
        """Возвращает int'овый numpy-массив, элементы которого являются возможными действиями агента
        """
        return np.array([-1, 0, 1])

    def action_sample(self):
        """Возвращает случайное возможное действие (int)
        """
        return random.randint(-1, 1)

    @property
    def pos(self):
        """Свойство, возвращающее текущее положение ц.м. ракеты в виде numpy массива из двух элементов 
        np.array([x,y])
        """
        return np.array([self.state[1], self.state[2]])

    @property
    def vel(self):
        """Свойство, возвращающее текущий вектор скорости ракеты в виде numpy массива из двух элементов 
        np.array([Vx, Vy])
        """
        v = self.state[0]
        Q = self.state[3]
        return np.array([v * np.cos(Q), v * np.sin(Q)])

    @property
    def x_axis(self):
        """Свойство, возвращающее текущий нормированный вектор центральной оси ракеты в виде numpy массива из двух элементов 
        np.array([Axi_x, Axi_y])
        """
        x = self.x
        y = self.y
        Q = self.Q
        alpha = np.radians(self.alpha)
        return np.array([x * np.cos(Q  + alpha), y * np.sin(Q + alpha)])

    @property
    def v(self):
        return self.state[0]

    @property
    def x(self):
        return self.state[1]

    @property
    def y(self):
        return self.state[2]

    @property
    def Q(self):
        return self.state[3]

    @property
    def alpha(self):
        return self.state[4]
    
    @property
    def t(self):
        return self.state[5]

    @property
    def M(self):
        return self.v / self.a_itr(self.y)

    @property
    def Cya(self):
        return self.Cya_itr(self.M)  

    @property
    def Cx(self):
        return self.Cx_itr(self.alpha, self.M)

    def get_summary(self):
        """Возвращает словарь с основными текущими параметрами и характеристиками ракеты в данный момент
        """
        return { 
            't': self.t,
            'v': self.v,
            'x': self.x,
            'y': self.y,
            'Q': np.degrees(self.Q),
            'm': self.m_itr(self.t),
            'P': self.P_itr(self.t),
            'alpha': self.alpha,
            'alpha_targeting': self. alpha_targeting,
            'Cx': self.Cx_itr(self.alpha, self.M),
            'Cya': self.Cya_itr(self.M)
        }
    
    @staticmethod
    def get_instant_meeting_point(trg_pos, trg_vel, my_vel_abs, my_pos):
        """Метод нахождения мгновенной точки встречи ракеты с целью (с координатой trg_pos и скоростью trg_vel)
        
        Arguments:
            trg_pos {tuple/np.ndarray} -- координата цели
            trg_vel {tuple/np.ndarray} -- скорость цели
        
        Keyword Arguments:
            my_vel_abs {float} -- скорость ракеты
            my_pos {tuple/np.ndarray} -- положение ракеты

        retuns (bool, np.ndarray) - (успех/неуспех, координата точки)
        """
        trg_pos = np.array(trg_pos)
        trg_vel = np.array(trg_vel)

        my_pos = np.array(my_pos) 

        vis = trg_pos - my_pos
        vis1 = vis / np.linalg.norm(vis)

        trg_vel_tau = np.dot(trg_vel, vis1) * vis1
        trg_vel_n = trg_vel - trg_vel_tau
        trg_vel_n1 = trg_vel_n / np.linalg.norm(trg_vel_n)

        if np.linalg.norm(trg_vel_n) > my_vel_abs:
            return False, trg_pos

        my_vel_n = trg_vel_n
        my_vel_tau = vis1 * sqrt(my_vel_abs**2 - np.linalg.norm(my_vel_n)**2)

        vel_close = my_vel_tau - trg_vel_tau
        if np.dot(vis1, vel_close) <= 0:
            return False, trg_pos

        t = np.linalg.norm(vis) / np.linalg.norm(vel_close)
        return True, trg_pos + trg_vel * t

    def rotate_to_point(self, point):
        point = np.array(point)
        vis = point - self.pos
        Q = np.arctan2(vis[1], vis[0])
        self.state[3] = Q


class Target(object):
    @classmethod
    def get_target(cls, scenario_name='SUCCESS', scenario_i=1):
        velocity_vectors = scenarios[scenario_name][scenario_i]

        vel_interp = InterpVec(velocity_vectors)
        target = cls(vel_interp = vel_interp)
        parameters_of_target = cls.get_standart_parameters_of_target()
        target.set_init_cond(parameters_of_target=parameters_of_target)
        return target

    @classmethod
    def get_simple_target(cls, pos, vel):
        velocity_vectors = [[0, np.array(vel)]]
        vel_interp = InterpVec(velocity_vectors)
        target = cls(vel_interp = vel_interp)
        parameters_of_target = np.array([pos[0], pos[1], 0])
        target.set_init_cond(parameters_of_target=parameters_of_target)
        return target

    @staticmethod
    def get_standart_parameters_of_target():
        """Возвращает словарь (или что там принимает метод set_init_cond) для стандартного начального состояния цели
        [np.ndarray] -- [x, y, t]
        """ 
        return np.array([20, 1000, 0]) 

    def __init__(self, *args, **kwargs):
        """Конструктор 
        """
        self.g = kwargs.get('g', 9.81)
        self.dt = kwargs.get('dt', 0.001)
        self.vel_interp = kwargs['vel_interp'] # type() == invariants.InterpVec

    def set_init_cond(self, parameters_of_target=None):
        """Задает начальные параметры (положение) и запоминает их для того,
        чтобы потом в них можно было вернуться при помощи reset()
    
        """
        if parameters_of_target is None:
            parameters_of_target = self.get_standart_parameters_of_target()
        self.state = np.array(parameters_of_target)
        self.state_0 = np.array(parameters_of_target)

    def reset(self):
        """Возвращает цель в начальное состояние
        """
        self.set_state(self.state_0)

    def set_state(self, state):
        """Метод задания нового (может полностью отличающегося от текущего) состояния цели

        Arguments:
            [np.ndarray] -- [x, y, t]
                            [м, м, с]
        """
        self.state = np.array(state)

    def get_state(self):
        """Метод получения вектора со всеми параметрами системы 
        (схож с вектором 'y' при интегрировании ode, но в векторе state еще должно быть t)
        
        Returns:
            [np.ndarray] -- [x, y, t]
                            [м, м, с]
        """
        return self.state
    
    def get_state_0(self):
        """Метод получения вектора со всеми параметрами системы 
        (схож с вектором 'y' при интегрировании ode, но в векторе state еще должно быть t)
        
        Returns:
            [np.ndarray] -- [x, y, t]
                            [м, м, с]
        """
        return self.state_0

    def step(self, tau):
        """Моделирует кинематику цели за шаг по времени tau. 
        Меняет внутреннее состояние цели на момент окончания шага
        
        Arguments:
            tau {float} -- длина шага по времени (не путать с шагом интегрирования)
        """
        x, y, t = self.state
        t_end = t + tau
        flag = True
        while flag:
            if t_end - t > self.dt:
                dt = self.dt 
            else:
                dt = t_end - t
                flag = False
            t += dt
            vx, vy = self.vel_interp(t)
            x += vx * dt
            y += vy * dt
        self.set_state([x, y, t])

    @property
    def pos(self):
        """Свойство, возвращающее текущее положение ц.м. цели в виде numpy массива из двух элементов 
        np.array([x,y])
        """
        return self.state[:2]

    @property
    def vel(self):
        """Свойство, возвращающее текущий вектор скорости цели в виде numpy массива из двух элементов 
        np.array([Vx, Vy])
        """
        return self.vel_interp(self.t)

    @property
    def t(self):
        return self.state[-1]
    
    @property
    def Q(self):
        """Свойство, возвращающее угол тангажа
        Returns:
            Q {float} -- [рад]
        """
        vx, vy = self.vel_interp(self.t)
        return np.arctan2(vy, vx)

    @property
    def v(self):
        vx, vy = self.vel_interp(self.t)
        return np.sqrt(vx ** 2 + vy ** 2)

    @property
    def x(self):
        return self.pos[0]

    
    @property
    def y(self):
        return self.pos[1]

    def get_summary(self):
        """Возвращает словарь с основными текущими параметрами и характеристиками ракеты в данный момент
        """
        return { 
            't': self.t,
            'v': self.v,
            'x': self.x,
            'y': self.y,
            'Q': np.degrees(self.Q)
        }


if __name__ == "__main__":
    t = Target.get_target()
    t.set_init_cond()

    m = Missile.get_needle()
    m.set_init_cond()

    
    summaries = [m.get_summary()]
    tau = 0.1
    for _ in range(10):
        k = m.get_action_parallel_guidance(t)
        act = m.action_sample() * k
        for i in range(40):    
            m.step(act, tau)
            t.step(tau)
            summaries.append(m.get_summary())

    ts = [s['t'] for s in summaries]
    for k in summaries[0]:
        if k == 't':
            continue
        data = [s[k] for s in summaries]
        plt.plot(ts, data, label=k)
    plt.grid()
    plt.legend()
    plt.show()

