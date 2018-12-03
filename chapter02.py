#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import animation as ani
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys


def main(args):
    # データ読み込み
    df = pd.read_csv('../python_stat_sample/data/ch2_scores_em.csv', index_col='生徒番号')
    print(df.head())

    # データ列取り出し
    scores = np.array(df['英語'])[:10]
    print(scores)

    # 新たなデータDF作成
    scores_df = pd.DataFrame({'点数': scores},
                             index=pd.Index(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], name='生徒'))

    print(scores_df)
    print(np.mean(scores))
    # print(scores_df.mean())

    mean = np.mean(scores)
    deviation = scores - mean

    print(deviation)

    an_scores = [50, 60, 58, 54, 51, 56, 57, 53, 52, 59]
    an_mean = np.mean(an_scores)
    an_deviation = an_scores - an_mean

    print(an_deviation)

    # コピーして、列追加
    summary_df = scores_df.copy()
    summary_df['偏差'] = deviation

    print(summary_df)
    print(summary_df.mean())

    print(np.mean(deviation ** 2))
    print(np.var(scores))


if __name__ == "__main__":
    main(sys.argv)
