import numpy as np
import matplotlib.pyplot as plt

loss = np.array([
    6697.8528,
    1667.5057,
    946.3358,
    836.9239,
    723.6884,
    660.3177,
    557.2845,
    562.1578,
    499.4271
])

k = np.array([
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10
])

figure, axis = plt.subplots()

axis.plot(k, loss)

axis.set_xlabel('k values')
axis.set_ylabel('Reconstruction Error')

plt.show()

