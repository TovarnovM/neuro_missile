import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D

class Interp1d(object):
    """Класс, превращающий набор точек (x,f) в непрерывную функцию f(x), путем 
    линейной интерполяции между этими точками. Будет использоваться для 
    аэродинамических и массо-тяговременных характеристик ркеты типа P(t), m(t), и т.д.
    """

    @classmethod
    def simple_constant(cls, value):
        """Создает простую заглушку, возвращающую одно и то же число
        
        Arguments:
            value {[type]} -- значения для возврата
        """
        x=[0,1]
        y=[value, value]
        return cls(x,y)

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
        fig = plt.figure()
        plt.plot(self.xs, self.fs, 'k')
        plt.show()

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
    @classmethod
    def simple_constant(cls, value):
        """Создает простую заглушку, возвращающую одно и то же число
        
        Arguments:
            value {[type]} -- значения для возврата
        """
        x=[0,1]
        y=[0,1]
        z=[[value, value], [value, value]]
        return cls(x,y,z)


    def __init__(self, xs, ys, fs):
        """Конструктор класса 
        
        Arguments:
            xs {iterable} -- абсциссы интерполируемой функции, len=N
            ys {iterable} -- ординаты интерполируемой функции, len=M
            fs - матрица N x M со значениями функции в соответствующих точках 
        """
        
        self.xs = np.array(xs)
        self.ys = np.array(ys)
        # self.xss, self.yss = np.meshgrid(self.xs, self.ys)
        # self.fs = fs(self.xss, self.yss)    
        self.fs = np.array(fs)
        self.func_inter = interp.RectBivariateSpline(self.xs, self.ys, self.fs, kx=1, ky=1) # обчыная линейная интерполяция
        if self.xs.size * self.ys.size != self.fs.size:
            raise AttributeError(f'Данные разных размеростей! xs{self.xs.shape} ys{self.ys.shape} fs{self.fs.shape}')


    def plot(self):
        """Отрисовать что там хранится

        Arguments:
            fig, ax = plt.subplots()
        """

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(self.xss, self.yss, self.fs, cmap=cm.RdYlBu_r,
                            linewidth=0, antialiased=False)

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


    def __call__(self, x, y):
        """Основной метод получения интерполированных данных
        
        Arguments:
            x {float} -- 1 абсцисса  точки
            y {float} -- 2 абсцисса  точки

        returns  - ордината точки
        """
        return self.func_inter(x, y)[0,0]

class InterpVec(object):
    """Класс служит для подномерной интерполяции векторов

    """
    def __init__(self, tups):
        """Конструктор
        
        Arguments:
            tups {list} -- список кортежей [(время, (x,y)), (время, (x,y)), ...]
            или [(время, [x,y]), (время, [x,y]), ...]
            или [(время, np.array([x,y])), (время, np.array([x,y])), ...]
        """
        # TODO сделать его
        pass

    def __call__(self, t):
        """Возвращает интерполированное значение вектора
        
        Arguments:
            t {float} -- время

        returns {np.ndarray} - вектор
        """
        pass


if __name__ == "__main__":
    x1 = np.array([1,2,3])
    x2 = np.array([2,3,3.5])
    y = np.array([[3,3,4],[3,2,3],[1,2,3]])
    i2 = Interp2d(x1, x2, y)

    print(i2(1.5,2.5))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.linspace(-1,4, 100)
    y = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = i2(X[i,j], Y[i,j])
    surf = ax.plot_surface(X, Y, Z, cmap=cm.RdYlBu_r,
                        linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
   
    
