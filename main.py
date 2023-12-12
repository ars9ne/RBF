import numpy as np

# Функция для вычисления радиально-базисной функции
def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)

# Функция для обучения РБФ-сети
def rbf_network(X, Y, t):
    # X - входные данные
    # Y - выходные данные
    # t - количество радиально-базисных функций

    # Инициализация центров и ширины
    centers = np.random.uniform(X.min(), X.max(), size=t)
    widths = np.full(t, (X.max() - X.min()) / t)

    # Матрица для РБФ
    RBF_matrix = np.array([rbf(X, c, s) for c, s in zip(centers, widths)]).T

    # Обучение с использованием наименьших квадратов
    W = np.linalg.pinv(RBF_matrix).dot(Y)

    return W, centers, widths

# Пример использования
X = np.linspace(0, 10, 100)  # Входные данные
Y = np.sin(X)  # Целевые значения

# Обучение сети
W, centers, widths = rbf_network(X, Y, t=10)

RBF_matrix = np.array([rbf(X, c, s) for c, s in zip(centers, widths)]).T
predictions = RBF_matrix.dot(W)

import matplotlib.pyplot as plt
plt.plot(X, Y, label='Original')
plt.plot(X, predictions, label='RBF-Net')
plt.legend()
plt.show()
