#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import animation as ani
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

import sys


def main(args):
    def Bin(n, p):
        _x_set = np.arange(n + 1)

        def _f(x):
            if x in _x_set:
                return comb(n, x) * p ** x * (1 - p) ** (n - x)
            else:
                return 0

        return _x_set, _f

    # 確率変数を定義
    n = 10
    p = 0.3
    X = Bin(n, p)
    x_set, f = X

    # 確率を表示
    prob = np.array([f(x_k) for x_k in x_set])
    # print(dict(zip(x_set,prob)))

    print(np.all(prob >= 0))  # すべてが正か
    print(f"{np.sum(prob):.3f}")  # 足したら1.0か

    # 期待値の定義
    mean = np.sum([x_k * f(x_k) for x_k in x_set])

    # 分散の定義
    var = np.sum([(x_k - mean) ** 2 * f(x_k) for x_k in x_set])

    # 無作為抽出の実施結果
    sample = np.random.choice(x_set, 10000000, p=prob)

    print(f"期待値の定義　　: {mean:.3f}")
    print(f"関数での出力　　: {E(X):.3f}")
    print(f"無作為抽出の平均値: {np.mean(sample):.3f}")

    print(f"分散の定義　: {var:.3f}")
    print(f"関数での出力: {V(X):.3f}")
    print(f"無作為抽出の分散値: {np.var(sample):.3f}")

    def s(x):
        return (x - mean) / np.sqrt(var)

    print(f"{E(X, g=lambda x: s(x)):.3f}")
    print(f"{V(X, g=lambda x: s(x)):.3f}")

    plot_prob(X)


def E(X, g=lambda x: x):
    x_set, f = X
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])


def V(X, g=lambda x: x):
    mean = E(X, g)
    return E(X, g=lambda x: (g(x) - mean) ** 2)


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


if __name__ == "__main__":
    main(sys.argv)
