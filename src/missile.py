import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import ode
from invariants import Interp1d, Interp2d, InterpVec 
from math import *

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
    # TODO сделать описание ММ ракеты для РПЗ (в ворде)

    @classmethod
    def get_needle(cls):
        """Классовый метод создания стандартной ракеты (иглы) со всеми необходимыми аэродинамическими, 
        массо- и тяговременными характеристиками

        returns Missile
        """
        m0 = 11.8
        d = 0.072
        t_st = 4.531
        t_mr = 8.125
        
        Sm = np.pi * d ** 2 / 4
        w_st = 3.24
        w_mr = 1.357
        G_st = w_st / t_st
        G_mr = w_mr / t_mr
        
        P_st = 1787
        P_mr = 454
        J = 2500

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

        ts = np.linspace(0, t_st+t_mr+0.5, 100)
        m_itr = Interp1d(ts, _get_m(ts))
        P_itr = Interp1d(ts, _get_P(ts))

        df = pd.read_csv('constants/aerodynamic.csv')
        df = df[df['D'] == 0]

        alpha_from_csv = np.unique(df['A'].to_numpy())
        M_from_csv = np.array(np.unique(df['M'].to_numpy()))
        Cx_from_csv = np.array(np.split(df['Cx'].to_numpy(), alpha_from_csv.size))
        Cya_from_csv = np.array(np.split(df['Cya'].to_numpy(), alpha_from_csv.size))

        Cya_itr = Interp2d(alpha_from_csv, M_from_csv, Cya_from_csv)
        Cx_itr = Interp2d(alpha_from_csv, M_from_csv, Cx_from_csv)

        # TODO сделать нормальную таблицу плотности воздуха и скорости звука из https://ru.wikipedia.org/wiki/%D0%A1%D1%82%D0%B0%D0%BD%D0%B4%D0%B0%D1%80%D1%82%D0%BD%D0%B0%D1%8F_%D0%B0%D1%82%D0%BC%D0%BE%D1%81%D1%84%D0%B5%D1%80%D0%B0
        ro_itr = Interp1d.simple_constant(1.204)   
        a_itr = Interp1d.simple_constant(340)   

        missile = cls(
            m_itr=m_itr,   # масса [кг] ракеты от времени [с]
            P_itr=P_itr,   # тяга [Н] ракеты от времени [с]
            Sm=Sm,         # площадь миделя [м^2] (к оторой относятся аэро коэффициенты)
            alphamax=15,   # максимальный угол атаки [градусы]
            speed_change_alpha=30,  # скорость изменения угола атаки [градусы / с]
            xi=0.1,           # коэффициент, характеризующий структуру подъёмной силы аэродинамической схемы ракеты [] TODO уточнить его
            Cx_itr=Cx_itr,              # интерполятор определения коэффициента лобового сопротивления ракеты от угла атаки [градусы] и от числа маха
            Cya_itr=Cya_itr,
            ro_itr=ro_itr,   # интерполятор определения плотности атмосферы [кг/м^3] от высоты [м] 
            a_itr=a_itr      # интерполятор определения скорости звука воздуха [м/с] от высоты [м] 
        )
        return missile

    @staticmethod
    def get_standart_parameters_of_missile():
        """Возвращает словарь (или что там принимает метод set_init_cond) для стандартного начального состояния ракеты
        Returns:
            [np.ndarray] -- [v, x, y, Q, alpha, t]
        """
        return np.array([25, 0, 0, np.radians(30), 0, 0])    
    
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
        Cya = self.Cya_itr(alpha, M)
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
        elif abs(y[4] - self.alpha_targeting) < 1e-6:
            y[4] = self.alpha_targeting
        if y[3] < -180 or y[3] > 180: # Q
            y[3] = y[3] % (2*pi)
            y[3] = (y[3] + 2*pi) % (2*pi)
            if y[3] > pi:
                y[3] -= 2*pi
        return y

    def get_action_parallel_guidance(self, target):
        """Метод, возвращающий аналог action'a, соответствующий идельному методу параллельного сближения
        
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

        # Угол между линией визирования и горизонтом [рад]
        fi = np.arctan((yc - y) / (xc - x))

        
        # Линия визирования
        r = np.sqrt((yc - y) ** 2 + (xc - x) ** 2)
        # Угол между линией визирования и вектором скорости [рад]
        nu_c = Qc - fi
        nu = Q - fi
        dfi_dt = (vc * np.sin(nu_c) - v * np.sin(nu)) / r 

        am = 1 # Коэффициент быстродействия
        dny = 1 # Запас по перегрузке

        P = self.P_itr(t)
        m = self.m_itr(t)
        ro = self.ro_itr(y)
        a = self.a_itr(y)
        M = v/a
        Cya = self.Cya_itr(alpha, M)

        alpha_req = ((v * am * dfi_dt / self.g + np.cos(Q) + dny) *  m * self.g) \
            / (Cya * ro * v ** 2 / 2 * self.Sm * (1 + self.xi) + P / 57.3)

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
            k2 = self._get_dydt(t+0.5*dt, self._validate_y(y+0.5*dt*k1))
            k3 = self._get_dydt(t+0.5*dt, self._validate_y(y+0.5*dt*k2))
            k4 = self._get_dydt(t+dt, self._validate_y(y+dt*k3))
            t += dt
            y = self._validate_y(y + dt/6*(k1+2*k2+2*k3+k4))
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
        return self.Cya_itr(self.alpha, self.M)  

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
            't': self.t,
            'm': self.m_itr(self.t),
            'P': self.P_itr(self.t),
            'alpha': self.alpha,
            'alpha_targeting': self. alpha_targeting,
            'Cx': self.Cx_itr(self.alpha, self.M),
            'Cya': self.Cya_itr(self.alpha, self.M)
        }

class Target(object):
    @classmethod
    def get_target(cls):

        velocity_vectors = scenarios['SUCCESS'][0]

        vel_interp = InterpVec(velocity_vectors)
        target = cls(vel_interp = vel_interp)
        parameters_of_target = cls.get_standart_parameters_of_target()
        target.set_init_cond(parameters_of_target=parameters_of_target)
        return target

    @staticmethod
    def get_standart_parameters_of_target():
        """Возвращает словарь (или что там принимает метод set_init_cond) для стандартного начального состояния цели
        [np.ndarray] -- [x, y, t]
        """ 
        return np.array([1700, 1000, 0]) 

    def __init__(self, *args, **kwargs):
        """Конструктор 
        """
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
        x, y, current_time = self.state
        t = current_time + tau
        # TODO можно и поточнее)
        vx, vy = self.vel_interp(t)
        self.set_state([x + vx, y + vy, t])

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
        return np.arctan(vy / vx)

    @property
    def v(self):
        vx, vy = self.vel_interp(self.t)
        return np.sqrt(vx ** 2 + vy ** 2)

scenarios = {
    'SUCCESS': [
        [
            (0,[20,12]),
            (2,[22,5]),
            (4,[21,0]),
            (6,[19,-1]),
            (8,[22,2]),
            (10,[20,10]),
            (12,[50,0]),
            (14,[25,11]),
            (16,[25,4]),
            (18,[30,6]),
            (20,[20,9]),
            (22,[35,5]),
            (24,[25,0]),
            (26,[40,-6]),
            (28,[55,-10]),
            (30,[60,-3]),
            (32,[40,-1])
        ],
        [
            (0,[20,12]),
            (2,[28,5]),
            (4,[24,0]),
            (6,[26,-1]),
            (8,[20,2]),
            (10,[20,10]),
            (12,[20,0]),
            (14,[25,11]),
            (16,[20,4]),
            (18,[25,6]),
            (20,[25,9]),
            (22,[25,5]),
            (24,[20,0]),
            (26,[25,-6]),
            (28,[20,-10]),
            (30,[20,-3]),
            (32,[20,-1])
        ],
        [
            (0,[30,12]),
            (2,[38,5]),
            (4,[34,10]),
            (6,[36,9]),
            (8,[30,2]),
            (10,[30,0]),
            (12,[30,0]),
            (14,[35,-2]),
            (16,[30,-4]),
            (18,[35,-6]),
            (20,[35,-10]),
            (22,[35,-15]),
            (24,[30,-10]),
            (26,[35,-6]),
            (28,[30,-10]),
            (30,[30,-3]),
            (32,[30,-1])
        ],
        [
            (0,[30,12]),
            (2,[38,-5]),
            (4,[34,10]),
            (6,[36,-9]),
            (8,[30,2]),
            (10,[30,0]),
            (12,[30,0]),
            (14,[35,-2]),
            (16,[30,4]),
            (18,[35,-6]),
            (20,[35,10]),
            (22,[35,-15]),
            (24,[30,10]),
            (26,[35,-6]),
            (28,[30,10]),
            (30,[30,-3]),
            (32,[30,1])
        ]
    ],
    'FAIL': [
        [
            (0,[30,12]),
            (2,[42,5]),
            (4,[51,0]),
            (6,[29,-6]),
            (8,[32,-5]),
            (10,[30,0]),
            (12,[50,0]),
            (14,[45,3]),
            (16,[45,5]),
            (18,[40,1]),
            (20,[40,5]),
            (22,[45,9]),
            (24,[45,4]),
            (26,[40,0]),
            (28,[55,-1]),
            (30,[60,-3]),
            (32,[40,-1])
        ],
        [
            (0,[10,12]),
            (2,[42,5]),
            (4,[21,0]),
            (6,[29,6]),
            (8,[32,15]),
            (10,[40,20]),
            (12,[-36,10]),
            (14,[-30,-3]),
            (16,[-25,-5]),
            (18,[-10,-8]),
            (20,[0,-10]),
            (22,[-3,-7]),
            (24,[-5,-4]),
            (26,[-10,-3]),
            (28,[-5,-1]),
            (30,[0,0]),
            (32,[2,5])
        ],
        [
            (0,[30,12]),
            (2,[38,5]),
            (4,[34,0]),
            (6,[36,-1]),
            (8,[30,2]),
            (10,[30,10]),
            (12,[30,0]),
            (14,[35,11]),
            (16,[30,4]),
            (18,[35,6]),
            (20,[35,9]),
            (22,[35,5]),
            (24,[30,0]),
            (26,[35,-6]),
            (28,[30,-10]),
            (30,[30,-3]),
            (32,[30,-1])
        ],
        [
            (0,[-10,-12]),
            (2,[-18,-5]),
            (4,[-14,-10]),
            (6,[-16,-9]),
            (8,[-10,-2]),
            (10,[-10,0]),
            (12,[-10,0]),
            (14,[-15,-2]),
            (16,[-10,-4]),
            (18,[-15,-6]),
            (20,[-15,-10]),
            (22,[-15,-15]),
            (24,[-10,-10]),
            (26,[-15,-6]),
            (28,[-10,-10]),
            (30,[-10,-3]),
            (32,[-10,-1])
        ],
        [
            (0,[10,12]),
            (2,[38,5]),
            (4,[14,10]),
            (6,[16,9]),
            (8,[30,2]),
            (10,[10,0]),
            (12,[30,0]),
            (14,[15,2]),
            (16,[30,4]),
            (18,[15,6]),
            (20,[35,10]),
            (22,[15,15]),
            (24,[30,10]),
            (26,[15,-6]),
            (28,[30,-10]),
            (30,[10,-3]),
            (32,[30,1])
        ]
    ]
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
        for i in range(20):    
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

