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
        
        J = 2300

        # constants_of_the_rocket = {  # было
        #     "mu_st": mu_st,
        #     "mu_mr": mu_mr,
        #     "t_st": t_st,
        #     "t_mr": t_mr,
        #     "w_st": w_st,
        #     "w_mr": w_mr,
        #     "G_st": G_st,
        #     "G_mr": G_mr,
        #     "Sm": Sm,
        #     "m0": m0,
        #     "d": d,
        #     "alphamax": 15,
        #     "speed_change_alpha": 0.1,
        #     "J": 1900,
        #     "xi": 2,
        #     "Cx0": 0.35,
        #     "Cya": 0.04,
        #     "betta": 1.9,
        #     "K_eff": 0.3,
        #     "alpha_0": 7
        # }

        K_eff = 1

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
                return K_eff * G_st * J
            elif t_st <= t < t_mr + t_st:
                return K_eff * G_mr * J
            else:
                return 0

        ts = np.linspace(0, t_st+t_mr+0.5, 100)
        m_itr = Interp1d(ts, _get_m(ts))
        P_itr = Interp1d(ts, _get_P(ts))

        Cx0_itr2 = Interp2d.simple_constant(0.35)
        Cya_itr = Interp1d.simple_constant(0.04)


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
            Cx0_itr2=Cx0_itr2, # интерполятор определения коэффициента лобового сопротивления ракеты от угла атаки [градусы] и от числа маха
            Cya_itr=Cya_itr, # интерполятор определения коэффициента подъемной силы ракеты в полускоростной СК от числа маха
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

        self.Cx0_itr2 = kwargs["Cx0_itr2"]
        self.Cya_itr = kwargs["Cya_itr"]

        self.P_itr = kwargs["P_itr"]
        self.m_itr = kwargs["m_itr"]

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
        Cya = self.Cya_itr(M)
        Cx = self.Cx0_itr2(alpha, M) + Cya * alpha / 57.3

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
        return y

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
        return self.Cya_itr(self.M)  

    @property
    def Cx(self):
        return self.Cx0_itr2(self.alpha, self.M) + self.Cya * self.alpha / 57.3

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
            'Cx': self.Cx,
            'Cy': self.Cya * self.alpha
        }

if __name__ == "__main__":
    m = Missile.get_needle()
    m.set_init_cond()

    summaries = [m.get_summary()]
    tau = 0.1
    for _ in range(10):
        act = m.action_sample()
        for i in range(20):    
            m.step(act, tau)
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

