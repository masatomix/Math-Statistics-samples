#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import animation as ani
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def main(args):
    # 取り得る値
    x_set = np.array([1, 2, 3, 4, 5, 6])

    def f(x):
        if x in x_set:
            return x / np.sum(x_set)
        else:
            return 0

    # 確率変数を定義
    X = [x_set, f]

    # 確率を表示
    prob = np.array([f(x_k) for x_k in x_set])
    # print(dict(zip(x_set,prob)))

    print(np.all(prob >= 0))  # すべてが正か
    print("{:.3f}".format(np.sum(prob)))  # 足したら1.0か

    # 期待値の定義
    mean = np.sum([x_k * f(x_k) for x_k in x_set])

    # 分散の定義
    var = np.sum([(x_k - mean) ** 2 * f(x_k) for x_k in x_set])

    # 無作為抽出の実施結果
    sample = np.random.choice(x_set, 10000000, p=prob)

    print("期待値の定義　　: {:.3f}".format(mean))
    print("関数での出力　　: {:.3f}".format(E(X)))
    print("無作為抽出の平均値: {:.3f}".format(np.mean(sample)))

    print("分散の定義　: {:.3f}".format(var))
    print("関数での出力: {:.3f}".format(V(X)))
    print("無作為抽出の分散値: {:.3f}".format(np.var(sample)))

    def s(x):
        return (x - mean) / np.sqrt(var)

    print("{:.3f}".format(E(X, g=lambda x: s(x))))
    print("{:.3f}".format(V(X, g=lambda x: s(x))))

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.bar(x_set, prob)

    ax1.set_xlabel('取り得る値')
    ax1.set_ylabel('確率')

    plt.tight_layout()  # タイトルの被りを防ぐ

    # グラフに情報を表示
    plt.show()


def E(X, g=lambda x: x):
    """
    離散型確率変数の、期待値の定義
    :param X:
    :param g:
    :return:
    """
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])


def V(X, g=lambda x: x):
    mean = E(X, g)
    return E(X, g=lambda x: (g(x) - mean) ** 2)


# def V(X, g=lambda x: x):
#     x_set, f = X
#     mean = E(X, g)
#     return np.sum([(g(x_k) - mean) ** 2 * f(x_k) for x_k in x_set])


if __name__ == "__main__":
    main(sys.argv)
