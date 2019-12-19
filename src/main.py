from scipy import interpolate
import numpy as np

velocity_vectors = [
    (0,[100,12]),
    (2,[100,41]),
    (4,[96,60]),
    (6,[90,50]),
    (8,[89,33]),
    (10,[90,10]),
    (12,[95,0]),
    (14,[105,21]),
    (16,[125,37]),
    (18,[110,60]),
    (20,[90,50]),
    (22,[85,44]),
    (24,[75,31]),
    (26,[20,61])
]

x = list(map(lambda x: x[0], velocity_vectors))
y = list(map(lambda x: x[1], velocity_vectors))

f = interpolate.interp1d(x, y, axis=0)

import matplotlib.pyplot as plt
from scipy import interpolate

t = np.linspace(0, 20, 20)

aa= list(map(lambda x: np.sqrt(f(x)[0]**2 + f(x)[1]**2), t))

plt.plot(t, aa, 'o')
plt.show()