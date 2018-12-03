#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from matplotlib import animation as ani


# https://deepage.net/features/numpy/
# https://deepage.net/features/numpy-dot.html
# https://qiita.com/jyori112/items/a15658d1dd17c421e1e2
# https://www.yukisako.xyz/entry/correlation-coefficient
# https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9
# https://www.yukisako.xyz/entry/correlation-coefficient

def main(args):
    data_path = "data.txt"
    execute_myself(data_path)
    execute_np_pandas(data_path)
    execute_pd_df(data_path)

    plot_histogram(data_path)
    plot_scatter(data_path)


def execute_myself(data_path):
    # データ読み込み
    data = np.loadtxt(data_path, delimiter='\t', skiprows=1, usecols=(0, 1))
    # print(data)

    data1 = data[:, 0]
    data2 = data[:, 1]

    # 平均
    ave1 = np.mean(data1)  # mx
    ave2 = np.mean(data2)  # my

    # 偏差
    dev1 = data1 - ave1  # X - mx
    dev2 = data2 - ave2  # Y - my

    # 偏差の積
    dev1_2 = dev1 ** 2  # (X-mx)^2  # |X|^2 つまりベクトルの長さの二乗(要素をΣしたら)
    dev2_2 = dev2 ** 2  # (Y-my)^2  # |Y|^2 つまりベクトルの長さの二乗(要素をΣしたら)
    dev1_dev2 = dev1 * dev2  # (X-mx) * (Y-my) # X・Y つまりベクトルの内積(要素をΣしたら)

    # 分散と、共分散
    v1 = np.mean(dev1_2)  # 1/n * Σ (X-mx)^2
    v2 = np.mean(dev2_2)  # 1/n * Σ (Y-my)^2
    cov = np.mean(dev1_dev2)  # 1/n * Σ (X-mx)*(Y-my)

    # 標準偏差
    std_dev1 = np.sqrt(v1)
    std_dev2 = np.sqrt(v2)

    # 相関係数
    cor = cov / (std_dev1 * std_dev2)
    #  cor = cov / (std_dev1 * std_dev2)
    #      =  1/n * Σ (X-mx)*(Y-my) /  sqrt (1/n * Σ (X-mx)^2 ) * sqrt ( 1/n * Σ (Y-my)^2 )
    #      =   Σ (X-mx)*(Y-my) /  sqrt ( Σ (X-mx)^2 ) * sqrt ( Σ (Y-my)^2 )
    #      =  (X-mx)・(Y-my) / |X-mx|*|Y-my|
    # ようするに、Xベクトルと Yベクトルの なす角θのcos(θ)を計算している
    # だから -1 <= cor <= 1 で、1だと相関アリ(なす角が0に近い)、0だと相関ナシ(直交)、か

    print('--- execute_myself ----')
    print('　　　平均: {0:.3f},{1:.3f}'.format(ave1, ave2))
    print('　標準偏差: {0:.3f},{1:.3f}'.format(std_dev1, std_dev2))
    print('　　　分散: {0:.3f},{1:.3f}'.format(v1, v2))
    print('　　共分散: {0:.3f}'.format(cov))
    print('　相関係数: {0:.3f}'.format(cor))

    # 要素の積
    data1_2 = data1 ** 2  # X^2
    data2_2 = data2 ** 2  # Y^2
    data1_data2 = data1 * data2  # X*Y
    ave1_2 = np.mean(data1_2)  # E[X^2]
    ave2_2 = np.mean(data2_2)  # E[Y^2]
    ave1_ave2 = np.mean(data1_data2)  # E[X*Y]

    print('２乗の平均: {0:.3f},{1:.3f}'.format(ave1_2, ave2_2))
    print('　　　分散: {0:.3f},{1:.3f}'.format(ave1_2 - ave1 ** 2, ave2_2 - ave2 ** 2))  # V[x] = E[X^2] - E[X]^2
    print('　　共分散: {0:.3f}'.format(ave1_ave2 - ave1 * ave2))  # cov[X,Y] = E[XY] - E[X]E[Y]
    print('-------')


def execute_np_pandas(data_path):
    # データ読み込み
    df = pd.read_table(data_path)

    data1 = df['英語']
    data2 = df['数学']

    # 平均
    ave1 = np.mean(data1)  # mx
    ave2 = np.mean(data2)  # my

    # 分散と、共分散
    v1 = np.var(data1, ddof=0)  # 1/n * Σ (X-mx)^2
    v2 = np.var(data2, ddof=0)  # 1/n * Σ (Y-my)^2
    cov = np.cov(data1, data2, ddof=0)[1, 0]  # 1/n * Σ (X-mx)*(Y-my)
    # 共分散は共分散行列で返ってくるので、[1,0],[0,1] をとればいい

    # 標準偏差
    std_dev1 = np.sqrt(v1)
    std_dev2 = np.sqrt(v2)

    # 相関係数
    cor = np.corrcoef(data1, data2)[1, 0]

    print('--- execute_np_pandas ----')
    print('　　　平均: {0:.3f},{1:.3f}'.format(ave1, ave2))
    print('　標準偏差: {0:.3f},{1:.3f}'.format(std_dev1, std_dev2))
    print('　　　分散: {0:.3f},{1:.3f}'.format(v1, v2))
    print('　　共分散: {0:.3f}'.format(cov))
    print('　相関係数: {0:.3f}'.format(cor))

    # 要素の積 以下、、とくにnpの特殊な関数はないので、省略
    print('-------')


def execute_pd_df(data_path):
    # データ読み込み
    df = pd.read_table(data_path)
    # print(df.head())

    data1 = df['英語']
    data2 = df['数学']

    # 平均
    ave1 = np.mean(data1)  # mx
    ave2 = np.mean(data2)  # my

    # 偏差
    dev1 = data1 - ave1  # X - mx
    dev2 = data2 - ave2  # Y - my

    summary = df.copy()
    summary['偏差1'] = dev1
    summary['偏差2'] = dev2

    # 偏差の積
    dev1_2 = dev1 ** 2  # (X-mx)^2  # |X|^2 つまりベクトルの長さの二乗(要素をΣしたら)
    dev2_2 = dev2 ** 2  # (Y-my)^2  # |Y|^2 つまりベクトルの長さの二乗(要素をΣしたら)
    dev1_dev2 = dev1 * dev2  # (X-mx) * (Y-my) # X・Y つまりベクトルの内積(要素をΣしたら)

    summary['偏差1の二乗'] = dev1_2
    summary['偏差2の二乗'] = dev2_2
    summary['偏差の積'] = dev1_dev2

    # 分散と、共分散
    v1 = np.cov(data1, data2, ddof=0)[0, 0]  # 1/n * Σ (X-mx)^2
    v2 = np.cov(data1, data2, ddof=0)[1, 1]  # 1/n * Σ (Y-my)^2
    cov = np.cov(data1, data2, ddof=0)[1, 0]  # 1/n * Σ (X-mx)*(Y-my)
    # 共分散は共分散行列で返ってくるので、[1,0],[0,1] をとればいい

    # 標準偏差
    std_dev1 = np.sqrt(v1)
    std_dev2 = np.sqrt(v2)

    summary['標準化1'] = dev1 / std_dev1  # X-m/σ 平均0、標準偏差1
    summary['標準化2'] = dev2 / std_dev2  # Y-m/σ 平均0、標準偏差1

    # いわゆる偏差値
    summary['偏差値1'] = 50 + 10 * summary['標準化1']  # 平均=50,標準偏差が10になるように、一度正規化してから再調整した変数
    summary['偏差値2'] = 50 + 10 * summary['標準化2']  # 平均=50,標準偏差が10になるように、一度正規化してから再調整した変数

    # 相関係数
    cor = np.corrcoef(data1, data2)[1, 0]

    print('--- execute_pd_df ----')
    print('　　　平均: {0:.3f},{1:.3f}'.format(ave1, ave2))
    print('　　　偏差: {0:.3f},{1:.3f}'.format(std_dev1, std_dev2))
    print('　　　分散: {0:.3f},{1:.3f}'.format(v1, v2))
    print('　　共分散: {0:.3f}'.format(cov))
    print('　相関係数: {0:.3f}'.format(cor))

    # 要素の積
    data1_2 = data1 ** 2  # X^2
    data2_2 = data2 ** 2  # Y^2
    data1_data2 = data1 * data2  # XY
    ave1_2 = np.mean(data1_2)
    ave2_2 = np.mean(data2_2)
    ave1_ave2 = np.mean(data1_data2)

    print('２乗の平均: {0:.3f},{1:.3f}'.format(ave1_2, ave2_2))
    print('　　　分散: {0:.3f},{1:.3f}'.format(ave1_2 - ave1 ** 2, ave2_2 - ave2 ** 2))  # V[x] = E[X^2] - E[X]^2
    print('　　共分散: {0:.3f}'.format(ave1_ave2 - ave1 * ave2))  # cov[X,Y] = E[XY] - E[X]E[Y]
    print('-------')

    # print(summary['英語'].describe())
    # print(summary['数学'].describe())
    print(summary)


def plot_scatter(data_path):
    """
    実データと偏差データをそれぞれプロット。
    :param data:
    :return:
    """

    data = np.loadtxt(data_path, delimiter='\t', skiprows=1, usecols=(0, 1))

    xorg = data[:, 0]
    yorg = data[:, 1]

    mx = np.mean(xorg)
    my = np.mean(yorg)

    x = xorg - mx
    y = yorg - my

    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.scatter(xorg, yorg, label='点数')
    ax1.scatter(mx, my, label='平均')

    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.grid(True)  # グリッド線

    # y = ax + b のa ,b を配列で返す?
    poly_fit = np.polyfit(xorg, yorg, 1)

    # 関数を作成
    poly_1d = np.poly1d(poly_fit)

    xs = np.linspace(xorg.min(), xorg.max())
    ys = poly_1d(xs)

    ax1.plot(xs, ys, label=f'{poly_fit[1]:.2f}+{poly_fit[0]:.2f}x')

    ax1.legend(loc='upper left')

    ax2.scatter(x, y)

    stdX = np.sqrt(np.var(xorg, ddof=0))
    stdY = np.sqrt(np.var(yorg, ddof=0))
    stdDataX = [stdX, -stdX, -stdX, stdX]
    stdDataY = [stdY, stdY, -stdY, -stdY]

    ax2.scatter(stdDataX, stdDataY)
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    ax2.grid(True)  # グリッド線

    plt.tight_layout()  # タイトルの被りを防ぐ

    # グラフに情報を表示
    plt.show()


def plot_histogram(data_path):
    # データ読み込み
    df = pd.read_table(data_path)

    summary = df.copy()

    # ヒストグラムを作るサンプル
    freq, _ = np.histogram(summary['英語'], bins=10, range=(0, 100))  # 10コに分ける

    # print(freq)
    freq_class = [f'{i}〜{i+10}' for i in range(0, 100, 10)]  # format文字列
    freq_dist_df = pd.DataFrame({'度数': freq}, index=pd.Index(freq_class, name='階級'))

    print(freq_dist_df)

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    # ax2 = fig.add_subplot(2, 1, 2)

    freq, _, _ = ax1.hist(summary['英語'], bins=10, range=(0, 100))

    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)

    ax1.set_xticks(np.linspace(0, 100, 10 + 1))
    ax1.set_yticks(np.arange(0, freq.max() + 1))

    plt.tight_layout()  # タイトルの被りを防ぐ

    # グラフに情報を表示
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
