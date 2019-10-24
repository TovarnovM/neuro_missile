import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from invariants import Interp1d, Interp2d 

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
        d = 0.127
        mu_st = 0.27
        mu_mr = 0.39
        t_st = 6.0
        t_mr = 3.0
        
        Sm = np.pi * d ** 2 / 4
        w_st = mu_st * m0
        w_mr = mu_mr * (m0 - w_st)
        G_st = w_st / t_st
        G_mr = w_mr / t_mr
        
        constants_of_the_rocket = {
            "mu_st": mu_st,
            "mu_mr": mu_mr,
            "t_st": t_st,
            "t_mr": t_mr,
            "w_st": w_st,
            "w_mr": w_mr,
            "G_st": G_st,
            "G_mr": G_mr,
            "Sm": Sm,
            "m0": m0,
            "d": d,
            "alphamax": 15,
            "speed_change_alpha": 0.1,
            "J": 1900,
            "xi": 2,
            "Cx0": 0.35,
            "Cya": 0.04,
            "betta": 1.9,
            "K_eff": 0.3,
            "alpha_0": 7
        }
        missile = cls(**constants_of_the_rocket)
        return missile

    @staticmethod
    def get_standart_parameters_of_missile():
        """Возвращает словарь (или что там принимает метод set_init_cond) для стандартного начального состояния ракеты
        Returns:
            [np.ndarray] -- [v, x, y, Q, t]
        """
        return np.array([25, 0, 0, np.radians(30), 0])    
    
    def __init__(self, *args, **kwargs):
        """Конструктор 
        """
        self.mu_st = kwargs.get("mu_st", 0)
        self.mu_mr = kwargs.get("mu_mr", 0)
        self.t_st = kwargs.get("t_st", 0)
        self.t_mr = kwargs.get("t_mr", 0)
        self.w_st = kwargs.get("w_st", 0)
        self.w_mr = kwargs.get("w_mr", 0)
        self.G_st = kwargs.get("G_st", 0)
        self.G_mr = kwargs.get("G_mr", 0)
        self.Sm = kwargs.get("Sm", 0)
        self.m0 = kwargs.get("m0", 0)
        self.d = kwargs.get("d", 0)
        self.v_sr = kwargs.get("v_sr", 0)
        self.alphamax = kwargs.get("alphamax", 0)
        self.speed_change_alpha = kwargs.get("speed_change_alpha", 0)
        self.J = kwargs.get("J", 0)
        self.xi = kwargs.get("xi", 0)
        self.Cx0 = kwargs.get("Cx0", 0)
        self.Cya = kwargs.get("Cya", 0)
        self.betta = kwargs.get("betta", 0)
        self.K_eff = kwargs.get("K_eff", 0)
        self.alpha = kwargs.get("alpha_0", 0)

        self.t_Interp = np.arange(0.0, 14.0, 0.01)
        self.alpha_Interp = np.arange(-self.alphamax, self.alphamax, 0.01)

        self.P = Interp1d(self.t_Interp, np.vectorize(self._get_P)(self.t_Interp))
        self.m = Interp1d(self.t_Interp, np.vectorize(self._get_m)(self.t_Interp))
        self.Cx = Interp1d(self.alpha_Interp, np.vectorize(self._get_Cx)(self.alpha_Interp))

        self.od = ode(self._get_dydt)
        
    def _get_m(self, t):
        if t < self.t_st:
            return self.m0 - self.G_st * t
        elif self.t_st <= t < self.t_mr + self.t_st:
            return self.m0 - self.w_st - self.G_mr * (t - self.t_st)
        else:
            return self.m0 - self.w_st - self.w_mr

    def _get_P(self, t):
        if t < self.t_st:
            return self.K_eff * self.G_st * self.J
        elif self.t_st <= t < self.t_mr + self.t_st:
            return self.K_eff * self.G_mr * self.J
        else:
            return 0

    def _get_Cx(self, alpha):
        return self.Cx0 + self.Cya * alpha / 57.3

    def _get_alpha(self, signal):
        if signal == 1 and self.alpha <= self.alphamax:
            return self.alpha + self.speed_change_alpha
        elif signal == -1 and -self.alpha <= self.alphamax:
            return self.alpha - self.speed_change_alpha
        else:
            return self.alpha


    def __init__(self, **kwargs):
        """Конструктор 
        """
        # TODO В зависимости от выбранной ММ определиться с выбором параметров, передаваемых в конструктор
        # Это обязательно должны быть аэродинамические, массо- и тяговременные характеристики. Задаются в виде констант
        # и объектов классов Interp1d и Interp2d
        # Сюда НЕ НАДО передавать начальные параметры ракеты, начальная инициализация будет в методе set_init_cond()
        pass

    def set_init_cond(self, parametrs_of_missile):
        """Задает начальные параметры (положение, скорость, углы ...) и запоминает их для того,
        чтобы потом в них можно было вернуться при помощи reset()
        
        Arguments:

            parameters_of_missile 
        """
        self.state = parameters_of_missile
        self.state_0 = parameters_of_missile


    def reset(self):
        """Возвращает ракету в начальное состояние
        """

        self.set_state(self.state_0)

    def get_state(self):
        """Метод получения вектора со всеми параметрами системы 
        (схож с вектором 'y' при интегрировании ode, но в векторе state еще должно быть t)
        
        Returns:
            [np.ndarray] -- [v, x, y, Q, t]
        """
        return self.state
    
    def get_state_0(self):
        """Метод получения вектора со всеми параметрами системы 
        (схож с вектором 'y' при интегрировании ode, но в векторе state еще должно быть t)
        
        Returns:
            [np.ndarray] -- [v, x, y, Q, t]
        """
        return self.state_0


    def set_state(self, state):
        """Метод задания нового (может полностью отличающегося от текущего) состояния ракеты

        Arguments:
            state - np.ndarray  (схож с вектором 'y' при интегрировании ode, но в векторе state еще должно быть t)
        """
        self.state = np.array(state)
        self.state_0 = np.array(state)


    def _get_dydt(self, t, y):
        """Функция правых частей системы ОДУ динамики ракеты. 
        !!! Функция не должна зависеть от мутабельных параметров объекта !!!
        Т.е. значения всех дифференцируемых параметров должны браться из 'y'
        
        Arguments:
            t {float} -- время
            y {np.ndarray} -- вектор состояния системы 

        returns {np.ndarray} -- dy/dt
        """
        v, x, y, Q = y
        alpha = self._get_alpha(self.action)

        return np.array([
            ( self.P(t) * np.cos(np.radians(alpha)) - ro * v ** 2 / 2 * self.Sm * self.Cx(alpha) - self.m(t) * g * np.sin(Q) ) / self.m(t),
            v * np.cos(Q),
            v * np.sin(Q),
            ( alpha * ( self.Cya * ro * v ** 2 / 2 * self.Sm * (1 + self.xi) + self.P(t) / 57.3) / ( self.m(t) * g ) - np.cos(Q)) * g / v
        ]) 

    def step(self, action, tau):
        """Моделирует динамику ракеты за шаг по времени tau. 
        На протяжении tau управляющее воздействие на ракету постоянно (action)
        Меняет внутреннее состояние ракеты на момент окончания шага
        
        Arguments:
            action {int} -- управляющее воздействие на протяжении шага
            tau {float} -- длина шага по времени (не путать с шагом интегрирования)
        """
        r = self.od.set_integrator('dopri5') 
        r = self.od.set_initial_value( self.state[:-1], self.state[-1] )  

        self.action = action
        while self.od.successful() and self.od.y[2] >= 0 and\
            self.P(self.state[-1]) > 0 and self.state_0[-1] + tau > self.state[-1]:

            self.alpha = self._get_alpha(self.action)
            self.state = np.concatenate([self.od.y, [self.od.t]])
            self.od.integrate(self.od.t + dt)


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
        # TODO реализовать метод.
        pass   

    def get_summary(self):
        """Возвращает словарь с основными текущими параметрами и характеристиками ракеты в данный момент
        """
        return { 
            'v': self.state[0],
            'x': self.state[1],
            'y': self.state[2],
            'Q': self.state[3],
            't': self.state[4],
            'm': self.m(self.state[4]),
            'P': self.P(self.state[4]),
            'alpha': self.alpha,
            'Cx': self.Cx(self.alpha)
        }

dt = 0.001
g = 9.81
ro = 1.202


m = Missile.get_needle()
parameters_of_missile = Missile.get_standart_parameters_of_missile()
m.set_init_cond(parameters_of_missile)
m.step(1, 2)
prev_state = m.get_state()
print(m.get_summary())

m.set_state(prev_state)
m.step(1, 1)
prev_state = m.get_state()
print(m.get_summary())

m.set_state(prev_state)
m.step(-1, 2)
prev_state = m.get_state()
print(m.get_summary())

m.set_state(prev_state)
m.step(1, 2)
prev_state = m.get_state()
print(m.get_summary())

m.set_state(prev_state)
m.step(0, 1)
prev_state = m.get_state()
print(m.get_summary())

m.set_state(prev_state)
m.step(-1, 10)
prev_state = m.get_state()
print(m.get_summary())

