#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import animation as ani
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

import sys


def E(X, g=lambda x: x):
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])


def V(X, g=lambda x: x):
    mean = E(X, g)
    return E(X, g=lambda x: (g(x) - mean) ** 2)


def normalization(x, mean, var):
    return (x - mean) / np.sqrt(var)


def plot_prob(X):
    x_set, f = X
    prob = np.array([f(x_k) for x_k in x_set])

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.bar(x_set, prob, label='prob')
    ax1.vlines(E(X), 0, 1, label='mean')
    ax1.set_xticks(np.append(x_set, E(X)))
    ax1.set_ylim(0, prob.max() * 1.2)

    ax1.set_xlabel('取り得る値')
    ax1.set_ylabel('確率')

    plt.tight_layout()  # タイトルの被りを防ぐ
    plt.legend()
    plt.show()


def F():
    _x_set = np.array([1, 2, 3, 4, 5, 6])

    # _x_set = np.arange(1,7)

    def _f(x):
        if x in _x_set:
            return x / np.sum(_x_set)
        else:
            return 0

    return _x_set, _f


def Bin(n, p):
    _x_set = np.arange(n + 1)

    def _f(x):
        if x in _x_set:
            return comb(n, x) * p ** x * (1 - p) ** (n - x)
        else:
            return 0

    return _x_set, _f


def Bern(p):
    """
    ベルヌーイ分布
    :param p:
    :return:
    """
    x_set = np.array([0, 1])

    # def f(x):
    #     if x in x_set:
    #         return (p ** x) * (1 - p) ** (1 - x)
    #     else:
    #         return 0
    #
    # return x_set, f

    from scipy import stats
    rv = stats.bernoulli(p)
    return x_set, rv.pmf


def Ge(p):
    """
    幾何分布
    :param p:
    :return:
    """
    x_set = np.arange(1, 30)

    def f(x):
        if x in x_set:
            return p * (1 - p) ** (x - 1)
        else:
            return 0

    return x_set, f


def Poisson(lam):
    """
    ポアソン分布
    :param lam:
    :return:
    """
    x_set = np.arange(20)

    def f(x):
        if x in x_set:
            from scipy.special import factorial
            return np.power(lam,x) / factorial(x) * np.exp(-lam)
        else:
            return 0

    return x_set, f