#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import animation as ani
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def main(args):
    dice = [1, 2, 3, 4, 5, 6]  # サイコロの目
    prob = [1 / 21, 2 / 21, 3 / 21, 4 / 21, 5 / 21, 6 / 21]  # それらが出る確率(割合)

    # num_trial = 100000
    num_trial = 1000
    sample = np.random.choice(dice, num_trial, p=prob)  # サイコロをふる試行を、num_trial 回やった結果を取得。
    # 分布は、確率分布に近くなるはず。

    print(sample)

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)

    freq, _, _ = ax1.hist(sample, bins=6, range=(1, 7), density=True, rwidth=0.8)

    ax1.hlines(prob, np.arange(1, 7), np.arange(2, 8), colors='gray')
    ax1.set_xticks(np.linspace(1.5, 6.5, 6))

    ax1.set_xticklabels(np.arange(1, 7))

    ax1.set_xlabel('出目', fontsize=10)
    ax1.set_ylabel('相対度数', fontsize=10)

    plt.tight_layout()  # タイトルの被りを防ぐ

    # グラフに情報を表示
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
