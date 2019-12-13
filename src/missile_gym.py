import numpy as np



class MissileGym(object):
    # set доступных сценариев для моделирования (различные поведения цели, различные варианты запуска и т.д.)
    scenario_names = {'scenario_1'}

    @classmethod
    def make(cls, scenario_name):
        """Метод, создающий экземпляры класса для различных сценариев
        
        Arguments:
            scenario_name {str} -- имя сценария (должен совпадать с одним из scenario_names)
        """
        # TODO реализовать метод
        pass

    def __init__(self, *args, **kwargs):
        self.missile = kwargs['missile']
        self.target = kwargs['target']
        self._tau = kwargs.get('tau', 0.1) 
        self._miss_state_len = self.missile.get_state().shape[0]
        
    def reset(self):
        """Возвращает наше окружение в начальное состояние.
        Метод возвращает начальное наблюдение (observation)

        returns np.ndarray
        """
        self.missile.reset()
        self.target.reset()

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
        # TODO реализовать метод
        pass

    def get_state(self):
        """метод, возвращающий numpy-массив, в котором хранится вся необходимая информация для воссоздания этого состояния
        """
        mis_state = self.missile.get_state()
        trg_state = self.target.get_state()
        return np.concatenate([mis_state, trg_state])


    def set_state(self, state):
        """метод, задающий новое состояние (state) окружения.

        return observation в новом состоянии
        
        Arguments:
            state {np.ndarray} -- numpy-массив, в котором хранится вся необходимая информация для задания нового состояния
        """
        self.missile.set_state(state[:self._miss_state_len])
        self.target.set_state(state[self._miss_state_len:])

    def render(self, **kwargs):
        """Отрисовать (где угодно) окружение в текущем состоянии 
        """
        # TODO реализовать метод
        pass

    @property
    def action_space(self):
        """Возвращает int'овый numpy-массив, элементы которого являются возможными действиями агента
        """
        return self.missile.action_space

    def action_sample(self):
        """Возвращает случайное возможное действие (int)
        """
        return self.missile.action_sample()

    def get_observation(self):
        """Метод возвращает numpy-массив с наблюдаемыми раектой данными в текущем состоянии окружения
        [t, etta, d, thetta, v, ]
        [0, 1,    2, 3,      4, ]
        """        
        t = self.missile.t
        p_miss = self.missile.pos
        p_trg = self.target.pos
        # TODO дописать

    @property
    def observation_space_high(self):
        """Возвращает numpy-массив размерности как и наблюдения агента с максимально возможными показаниями наблюдения агента 
        """
        # TODO реализовать метод
        pass

    @property
    def observation_space_low(self):
        """Возвращает numpy-массив размерности как и наблюдения агента с минимальныо возможными показаниями наблюдения агента 
        """
        # TODO реализовать метод
        pass
