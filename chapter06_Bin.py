#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import animation as ani
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from utils import *
import sys

def main(args):
    # 確率変数を定義
    n=10
    p = 0.3
    X = Bin(n,p)
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

    print(f"{E(X, g=lambda x: normalization(x,mean,var)):.3f}")
    print(f"{V(X, g=lambda x: normalization(x,mean,var)):.3f}")

    plot_prob(X)

if __name__ == "__main__":
    main(sys.argv)
