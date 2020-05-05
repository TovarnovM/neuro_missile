import numpy as np
from missile import Missile, Target
from invariants import Interp1d


class MissileGym(object):
    # set доступных сценариев для моделирования (различные поведения цели, различные варианты запуска и т.д.)
    scenario_names = {'standart', 'sc_simple_1', 'sc_simple_2', 'sc_simple_3'}

    @classmethod
    def make(cls, scenario_name):
        """Метод, создающий экземпляры класса для различных сценариев
        
        Arguments:
            scenario_name {str} -- имя сценария (должен совпадать с одним из scenario_names)
        """
        if scenario_name not in cls.scenario_names:
            raise AttributeError(f'Ошибка ! нет такого сценария "{scenario_name}", есть только {cls.scenario_names}')
        if scenario_name == 'standart':
            target = Target.get_target()
            missile = Missile.get_needle()
            mparams = Missile.get_parameters_of_missile_to_meeting_target(target.pos, target.vel)
            missile.set_init_cond(parameters_of_missile=mparams)
            return cls(missile=missile, target=target)
        if scenario_name == 'sc_simple_1':
            return cls.make_simple_scenario((1000,1000), (-200,0), missile_vel_abs=1.0)
        if scenario_name == 'sc_simple_2':
            return cls.make_simple_scenario((1000,2000), (-300,0), missile_vel_abs=1.0)
        if scenario_name == 'sc_simple_3':
            return cls.make_simple_scenario((2000,1000), (-500,0), missile_vel_abs=1.0)

    @classmethod
    def make_simple_scenario(cls, trg_pos, trg_vel, missile_pos=None, missile_vel_abs=500.0):
        trg_pos = np.array(trg_pos)
        trg_vel = np.array(trg_vel)

        target = Target.get_simple_target(trg_pos, trg_vel)
        missile = Missile.get_needle()
        mparams = Missile.get_parameters_of_missile_to_meeting_target(target.pos, target.vel, missile_pos, missile_vel_abs)
        suc, meeting_point = Missile.get_instant_meeting_point(target.pos, target.vel, missile_vel_abs, missile_pos if missile_pos is not None else (0,0))
        # print(suc, meeting_point)
        # print(mparams)
        missile.set_init_cond(parameters_of_missile=mparams)
        return cls(missile=missile, target=target)

    def __init__(self, *args, **kwargs):
        self.point_solution = np.array([])
        self.missile = kwargs['missile']
        self.target = kwargs['target']
        self._tau = kwargs.get('tau', 1/30) 
        self.t_max = kwargs.get('t_max', 14) 
        self._miss_state_len = self.missile.get_state().shape[0]
        self._trg_state_len = self.target.get_state().shape[0]
        self.prev_observation=self.get_current_observation()
 
    def reset(self):
        """Возвращает наше окружение в начальное состояние.
        Метод возвращает начальное наблюдение (observation)

        returns np.ndarray
        """
        self.missile.reset()
        self.target.reset()
        self.prev_observation = self.get_current_observation()
        return self.get_observation()

    def get_observation(self):
        return np.concatenate([self.prev_observation, self.get_current_observation()])
         
    def close(self):
        """Завершаем работу с окружением (если были открыты какие-то ресурсы, то закрываем их здесь)
        """
        # TODO реализовать метод
        pass

    def step(self, action):
        """ Основной метод. Сделать шаг по времени. Изменить внутреннее состояние и вернуть необходимые для тренеровки данные,
        а именно кортеж:
            (observation, reward, done, info)
            observation - numpy массив с наблюдаемыми раектой данными
            reward - float. награда, даваемая агенту за данный шаг
            done - bool. Флаг, показывающий закончено ли моделирование. True - всё, данный шаг терминальный, дальше делать шаги нельзя.
            info - словарь (может быть пустым), с дополнительной инфой (если надо)
        
        Arguments:
            action {скорее всего int} -- действие агента на данном шаге
        """
        self.prev_observation = self.get_current_observation()
        mpos0, tpos0 = self.missile.pos, self.target.pos
        self.missile.step(action, self._tau)
        self.target.step(self._tau)

        obs = self.get_observation()
        mpos1, tpos1 = self.missile.pos, self.target.pos
        mvel1, tvel1 = self.missile.vel, self.target.vel
        reward, done, info = self.get_reward_done_info(mpos0, tpos0, mpos1, tpos1, tvel1, mvel1)
        return obs, reward, done, info

    def step_with_guidance(self):
        if self.missile.v > self.target.v:
            action_parallel_guidance = self.missile.get_action_parallel_guidance(self.target)
        else:
            action_parallel_guidance = self.missile.get_action_chaise_guidance(self.target)
        # if -0.5 <= action_parallel_guidance <= 0.5:
        #     action_parallel_guidance = 0
        # elif action_parallel_guidance < -0.5:
        #     action_parallel_guidance = -1
        # else:
        #     action_parallel_guidance = 1
        obs, reward, done, info = self.step(action_parallel_guidance)
        return obs, reward, done, info

    def get_normal_reward(self, mpos0, tpos0, mpos1, tpos1):
        r0 = np.linalg.norm(tpos0 - mpos0)
        r1 = np.linalg.norm(tpos1 - mpos1)
        return (r0-r1)/r0

    def get_new_reward(self, mpos0, tpos0, mpos1, tpos1, tvel1, mvel1):
        r_norm = 10*self.get_normal_reward(mpos0, tpos0, mpos1, tpos1)
        suc, p = Missile.get_instant_meeting_point(tpos1, tvel1, np.linalg.norm(mvel1), mpos1)
        
        p = p - mpos1
        p1 = p / np.linalg.norm(p)
        v1 = mvel1 / np.linalg.norm(mvel1)
        r2 = p1 @ v1
        return 5*(r2**7 +r_norm - 1)
        # Missile.get_instant_meeting_point(tpos1, )

    def get_reward_done_info(self, mpos0, tpos0, mpos1, tpos1, tvel1, mvel1):
        info = {}
        if mpos1[1] < 0: # мы упали
            info['done_reason'] = 'мы упали'
            info['t'] = self.missile.t
            info['distance_to_target'] = np.linalg.norm(mpos1 - tpos1)
            return -200, True, info
        if self.is_hit(mpos0, tpos0, mpos1, tpos1):
            info['done_reason'] = 'мы попали'
            info['t'] = self.missile.t
            return 999, True, info
        if self.is_wrong_way(mpos1, mvel1, tpos1):
            info['done_reason'] = 'потеряли из виду'
            info['t'] = self.missile.t
            info['distance_to_target'] = np.linalg.norm(mpos1 - tpos1)
            return -200, True, info
        if self.missile.t > self.t_max:
            info['done_reason'] = 'слишком долго'
            info['t'] = self.missile.t
            return -200, True, info
        if self.missile.t > 10 and self.missile.v < 340:
            info['done_reason'] = 'слишком долго и медленно'
            info['t'] = self.missile.t
            return -200, True, info
        return self.get_new_reward(mpos0, tpos0, mpos1, tpos1, tvel1, mvel1), False, info

    @staticmethod
    def _r(mpos0, tpos0, mpos1, tpos1, r_kill):
        xm0, ym0 = mpos0
        xm1, ym1 = mpos1
        xt0, yt0 = tpos0
        xt1, yt1 = tpos1

        flag = False
        times = np.linspace(0.0, 1.0, 20)
        for t in times:
            dx = (xt0 - xm0) * (1 - t) + (xt1 - xm1) * t
            dy = (yt0 - ym0) * (1 - t) + (yt1 - ym1) * t
            r = np.sqrt(dx ** 2 + dy ** 2)
            if r < r_kill:
                flag = True
        return flag

    @staticmethod
    def _r1(mpos0, tpos0, mpos1, tpos1, r_kill):
        xm0, ym0 = mpos0
        xm1, ym1 = mpos1
        xt0, yt0 = tpos0
        xt1, yt1 = tpos1

        X_1 = xm1-xm0-xt1+xt0
        Y_1 = ym1-ym0-yt1+yt0
        A = X_1**2 + Y_1**2
        B = 2*X_1*(xm0+xt0) + 2*Y_1*(ym0+yt0)
        C = (xm0+xt0)**2 + (ym0+yt0)**2

        r0 = C
        r1 = A + B + C

        r_0 = B
        r_1 = 2*A + B
        if r_0 * r_1 >= 0:
            return min(r0, r1) <= r_kill**2
        t_0 = -B / (2*A)
        r_t0 = A * t_0**2 + B * t_0 + C
        return min(r0, r1, r_t0) <= r_kill**2

    def is_hit(self, mpos0, tpos0, mpos1, tpos1):
        r0 = np.linalg.norm(mpos0 - tpos0)
        r1 = np.linalg.norm(mpos1 - tpos1)

        r_kill = self.missile.r_kill
        if min(r1, r0) < r_kill:
            return True
        return MissileGym._r1(mpos0, tpos0, mpos1, tpos1, r_kill)

    def is_wrong_way(self, mpos, mvel, tpos):
        vis_n = (tpos-mpos)
        d = np.linalg.norm(vis_n)
        if d < 200:
            return False # Если мы уже достаточно близко к цели, то не проверяем, куда мы смотрим (авось поподем)
        vis_n /= d
        mis_axis = self.missile.x_axis
        return mvel @ vis_n < 0.78 # угол ежду осью ракеты и линией визирования меньше 38 градусов 

    def get_state(self):
        """метод, возвращающий numpy-массив, в котором хранится вся необходимая информация для воссоздания этого состояния
        """
        mis_state = self.missile.get_state()
        trg_state = self.target.get_state()
        return np.concatenate([mis_state, trg_state, self.prev_observation])

    def set_state(self, state):
        """метод, задающий новое состояние (state) окружения.

        return observation в новом состоянии
        
        Arguments:
            state {np.ndarray} -- numpy-массив, в котором хранится вся необходимая информация для задания нового состояния
        """
        self.missile.set_state(state[:self._miss_state_len])
        self.target.set_state(state[self._miss_state_len:self._miss_state_len+self._trg_state_len])
        self.prev_observation[:] = state[self._miss_state_len+self._trg_state_len:]

    def render_all_trajectory(self, **kwargs):
        """Отрисовать окружение в текущем состоянии 
        """
        reward = kwargs['reward']
        fig = plt.gcf()
        fig.show()
        fig.canvas.draw()
        vm, xm, ym, Qm, alpha, t = self.get_state()[0:6]
        P = self.missile.get_summary()['P']
        xt, yt = self.get_state()[6:8]
        vx, vy = self.missile.vel[:]
        vxt, vyt = self.target.vel
        vt = self.target.v
        Qt = self.target.Q
        npt = self.target.np
        npm = self.missile.np

        r = np.sqrt((xt - xm) ** 2 + (yt - ym) ** 2)

        plt.subplot(5, 1, 1)

        plt.plot([xm, xt], [ym, yt], linestyle='-', color="#dddddd")
        plt.plot([xm, xt], [ym, ym], color="#e2e2e2")

        plt.plot([xm, vx + xm], [ym,ym],  color="#92a8d1", label='Векторы V, Vx и Vy ракеты')
        plt.plot([xm, xm], [ym,vy + ym],  color="#92a8d1")
        plt.plot([xm, vx + xm], [ym,vy + ym],  color="#92a8d1")

        plt.plot([xt, vxt + xt], [yt,yt],  color="#eea29a", label='Векторы V, Vx и Vy цели')
        plt.plot([xt, xt], [yt,vyt + yt],  color="#eea29a")
        plt.plot([xt, vxt + xt], [yt,vyt + yt],  color="#eea29a")

        plt.plot(xm, ym, 'b.', xt, yt, 'r.', markersize=1)
        plt.title('Полет ракеты и цели')

        plt.subplot(5, 2, 3)
        plt.plot(t, reward, 'r.', markersize=1)
        plt.xlabel('t, сек')
        plt.ylabel('reward')
        
        plt.subplot(5, 2, 4)
        plt.plot(t, r, 'r.', markersize=1)
        plt.xlabel('t, с')
        plt.ylabel(r'r, м')

        plt.subplot(5, 2, 5)
        plt.plot(t, vm, 'r.', markersize=1)
        plt.xlabel('t, сек')
        plt.ylabel('Vm, м/с')
        
        plt.subplot(5, 2, 6)
        plt.plot(t, vt, 'r.', markersize=1)
        plt.xlabel('t, с')
        plt.ylabel('Vt, м/с')

        plt.subplot(5, 2, 7)
        plt.plot(t, Qm * 180 / np.pi, 'r.', markersize=1)
        plt.xlabel('t, с')
        plt.ylabel('Qm, рад')

        plt.subplot(5, 2, 8)
        plt.plot(t, Qt * 180 / np.pi, 'r.', markersize=1)
        plt.xlabel('t, с')
        plt.ylabel('Qt, рад')

        plt.subplot(5, 2, 9)
        plt.plot(t, npm, 'r.', markersize=1)
        plt.xlabel('t, с')
        plt.ylabel('npm, рад')

        plt.subplot(5, 2, 10)
        plt.plot(t, npt, 'r.', markersize=1)
        plt.xlabel('t, с')
        plt.ylabel('npt, рад')

        plt.pause(0.001)
        fig.canvas.draw()
        # TODO: Добавить сохранение

    def render(self, **kwargs):
        """Отрисовать окружение в текущем состоянии 
        """
        fig = kwargs['fig']
        ax = kwargs['ax']

        vm, xm, ym = self.get_state()[0:3]
        xt, yt = self.get_state()[6:8]

        vx, vy = self.missile.vel[:]
        vxt, vyt = self.target.vel

        # ax.plot([xm, vx + xm], [ym,ym],  color="#92a8d1", label='Векторы V, Vx и Vy ракеты')
        # ax.plot([xm, xm], [ym,vy + ym],  color="#92a8d1")
        ax.plot([xm, vx + xm], [ym,vy + ym],  color="#92a8d1", label=r'$\vec{V}$ ракеты')

        # ax.plot([xt, vxt + xt], [yt,yt],  color="#eea29a", label='Векторы V, Vx и Vy цели')
        # ax.plot([xt, xt], [yt,vyt + yt],  color="#eea29a")
        ax.plot([xt, vxt + xt], [yt,vyt + yt],  color="#eea29a", label=r'$\vec{V}$ цели')

        ax.plot([xm, xt], [ym, yt], linestyle='--', color="#dddddd", label='Линия визирования')
        ax.plot(xm, ym, 'b.',  markersize=10, label='Ракета')
        ax.plot(xt, yt, 'r.',  markersize=10, label='Цель')

        legend = ax.legend(fontsize='medium')


    @property
    def action_space(self):
        """Возвращает int'овый numpy-массив, элементы которого являются возможными действиями агента
        """
        return self.missile.action_space

    def action_sample(self):
        """Возвращает случайное возможное действие (int)
        """
        return self.missile.action_sample()

    def _get_etta(self, miss=None, target=None):
        """
            угол, между осью ракеты и линией визирования, градусы, -180..+180
        """
        miss = self.missile if miss is None else miss
        target = self.target if target is None else target
        Q = miss.Q
        vis = target.pos - miss.pos
        vis = vis / np.linalg.norm(vis)
        Q_vis = np.arctan2(vis[1], vis[0])
        angle = np.degrees(Q_vis - Q) % 360
        angle = (angle + 360) % 360
        if angle > 180:
            angle -= 360
        return angle

    def get_current_observation_raw(self):
        """Метод возвращает numpy-массив с наблюдаемыми раектой данными в текущем состоянии окружения
        0 - t    - время, с 0..15
        1 - etta - угол, между осью ракеты и линией визирования, градусы, -180..+180 
        2 - v    - скорость ракеты, м/с, 0..1000
        3 - Q    - угол тангажа, градусы, -180..+180
        4 - alpha
        """        
        t = self.missile.t
        etta = self._get_etta()
        v = self.missile.v
        # alpha_targeting = self.missile.alpha_targeting  3 - alpha_targeting - угол атаки, к которому стремится ракета, градусы, -alphamax..+alphamax 
        Q = np.degrees(self.missile.Q)
        alpha = self.missile.alpha
        return np.array([t, etta, v, Q, alpha])

    def get_current_observation(self):
        """Метод возвращает numpy-массив с НОРМИРОВАННЫМИ [0..1] наблюдаемыми раектой данными в текущем состоянии окружения
        
        """
        h = self.observation_current_raw_space_high
        l = self.observation_current_raw_space_low
        return (self.get_current_observation_raw() - l)/(h-l)

    @property
    def observation_current_raw_space_high(self):
        """Возвращает numpy-массив размерности как и наблюдения агента с максимально возможными показаниями наблюдения агента 
        """
        return np.array([
            self.t_max,
            180.0,
            1000,
            # self.missile.alphamax,
            180,
            self.missile.alphamax
        ])

    @property
    def observation_current_raw_space_low(self):
        """Возвращает numpy-массив размерности как и наблюдения агента с минимальныо возможными показаниями наблюдения агента 
        """
        return np.array([
            0,
            -180.0,
            0,
            # -self.missile.alphamax,
            -180,
            -self.missile.alphamax
        ])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    MissileGym.make_simple_scenario((1000, 1000), (300, 0))
    gym = MissileGym.make('standart')
    done = False
    reward = 0
    while not done:
        if reward > 0.4: 
            fig, ax = plt.subplots()
            gym.render(fig = fig, ax = ax)
            plt.title('Полет ракеты и цели')
            plt.show()
        obs, reward, done, info = gym.step_with_guidance()
    print(info['done_reason'])
