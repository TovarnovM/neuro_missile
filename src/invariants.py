import numpy as np
import scipy.interpolate as interp

# TODO реализовать 2do методы и оттестировать в test_invariants.py

class Interp1d(object):
    """Класс, превращающий набор точек (x,f) в непрерывную функцию f(x), путем 
    линейной интерполяции между этими точками. Будет использоваться для 
    аэродинамических и массо-тяговременных характеристик ркеты типа P(t), m(t), и т.д.
    """
    def __init__(self, xs, fs):
        """Конструктор класса 
        
        Arguments:
            xs {iterable} -- абсциссы интерполируемой функции
            fs {iterable} -- ординаты интерполируемой функции
        """
        self.xs = np.array(xs)
        self.fs = np.array(fs)
        if self.xs.shape != self.fs.shape:
            raise AttributeError(f'Данные разных размеростей! xs{self.xs.shape}  fs{self.fs.shape}')

    def plot(self, fig, ax, **kwargs):
        """Отрисовать что там хранится

        Arguments:
            fig, ax = plt.subplots()
        """
        ax.plot(self.xs, self.fs)

    def __call__(self, x):
        """Основной метод получения интерполированных данных
        
        Arguments:
            x {float} -- абсцисса точки

        returns  - ордината точки
        """
        return np.interp(x, self.xs, self.fs)


class Interp2d(object):
    """Класс, похожий на предыдущий, но интерполирует он не одномерные а двумерные точки
    Что-то типа f(x, y), использовать для 
    """
    def __init__(self, xs, ys, fs):
        """Конструктор класса 
        
        Arguments:
            xs {iterable} -- абсциссы интерполируемой функции, len=N
            ys {iterable} -- ординаты интерполируемой функции, len=M
            fs - матрица N x M со значениями функции в соответствующих точках 
        """
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        self.xss, self.yss = np.meshgrid(self.xs, self.ys)
        self.fs = fs(self.xss, self.yss)

        self.func_inter = interp.Rbf(self.xss, self.yss, self.fs, function='cubic', smooth=0)

        if self.xs.size * self.ys.size != self.fs.size:
            raise AttributeError(f'Данные разных размеростей! xs{self.xs.shape} ys{self.ys.shape} fs{self.fs.shape}')


    def plot(self, fig, ax, **kwargs):
        """Отрисовать что там хранится

        Arguments:
            fig, ax = plt.subplots()
        """
        # TODO подумаль, как лучше отрисовать на плоском графике
        pass

    def __call__(self, x, y):
        """Основной метод получения интерполированных данных
        
        Arguments:
            x {float} -- 1 абсцисса  точки
            y {float} -- 2 абсцисса  точки

        returns  - ордината точки
        """
        return self.func_inter(x, y)
