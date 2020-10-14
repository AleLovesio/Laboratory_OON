import numpy as np
import matplotlib.pyplot as plt


def my_sum(a, b):
    return a+b


def my_division(a, b):
    return a / b


def my_power(base, exponent):
    return base ** exponent


def my_sin(n_points, arr):
    x = np.array([])
    y = np.array([i ** 2 for i in range(n_points)])
    x1 = np.ones(y.shape)

    plt.figure()
    plt.plot(arr, x1, 'or', label = 'example')
    plt.grid()
    plt.legend()
    plt.xlabel('xlab')
    plt.ylabel('ylab')
    plt.show()
    return np.sin(arr)


