#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import animation as ani
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def main(args):
    df = pd.read_csv('../python_stat_sample/data/ch4_scores400.csv')

    scores = np.array(df['点数'])
    print(scores[:10])

    sample = np.random.choice(scores, 20)
    np.random.seed(0)
    np.random.choice(scores, 20)

    for i in range(5):
        sample = np.random.choice(scores, 20)
        print(sample.mean())

    # データ全部を、ヒストグラム表示
    fig = plt.figure(figsize=(6, 10))
    ax1 = fig.add_subplot(3, 1, 1)
    freq, _, _ = ax1.hist(scores, bins=100, range=(0, 100), density=True)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 0.042)
    ax1.set_xlabel('点数')
    ax1.set_ylabel('相対度数')

    # データを無作為抽出した場合のデータ群を、ヒストグラム表示する
    hyohon = np.random.choice(scores, 10000)
    ax2 = fig.add_subplot(3, 1, 2)
    freq, _, _ = ax2.hist(hyohon, bins=100, range=(0, 100), density=True)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 0.042)
    ax2.set_xlabel('点数')
    ax2.set_ylabel('相対度数')

    # 20コの標本平均、を10000回やったデータ値の、ヒストグラム
    ax3 = fig.add_subplot(3, 1, 3)
    sample_means = [np.random.choice(scores, 20).mean()  for _ in range(10000)]
    freq, _, _ = ax3.hist(sample_means, bins=100, range=(0, 100), density=True)
    ax3.set_xlim(50, 90)
    ax3.set_ylim(0, 0.15)
    ax3.set_xlabel('点数')
    ax3.set_ylabel('相対度数')
    ax3.vlines(np.mean(scores), 0, 1, 'gray')

    plt.tight_layout()  # タイトルの被りを防ぐ
    # グラフに情報を表示
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
