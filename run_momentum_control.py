# -*- coding: utf-8 -*-
"""对照组: 20 日动量因子 —— 用完全相同的管线跑一遍

目的
----
ROE 的 IC 接近 0。要区分两种可能:
  (a) 管线(数据/对齐/收益计算)有问题, 什么因子都测不出来
  (b) 管线正常, 原始 ROE 在 A 股短周期上确实没有预测力
方法是换一个**公认有短期预测力**的因子(20 日动量)跑同一套流程。
若动量 IC 显著非 0, 则管线有效, (b) 成立。

输出: outputs/momentum_metrics.json, outputs/charts/06_对照组IC对比.png
"""
import json
import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['FZLanTingHei-R-GBK', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from alphalens import performance as perf
from alphalens.utils import get_clean_factor_and_forward_returns, get_forward_returns_columns

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, 'cache'), os.path.join(HERE, 'outputs')
CHARTS = os.path.join(OUT, 'charts')

FACTOR_START, FACTOR_END = '2019-01-01', '2026-07-31'
PERIODS = (1, 5, 10, 20)
QUANTILES = 5


def main():
    px = pd.read_parquet(os.path.join(CACHE, 'px_daily.parquet'))
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    dates = px.index[(px.index >= FACTOR_START) & (px.index <= FACTOR_END)]

    # 20 日动量: px[t] / px[t-20] - 1, 在每个观测日做横截面
    mom = px.pct_change(20).reindex(dates)
    factor = mom.stack()
    factor.index = factor.index.set_names(['date', 'asset'])
    factor = factor[np.isfinite(factor)]
    print('动量因子观测 %d 条, 覆盖 %d 日 × %d 只'
          % (len(factor), factor.index.levels[0].size, factor.index.levels[1].size), flush=True)

    clean = get_clean_factor_and_forward_returns(
        factor, px, quantiles=QUANTILES, periods=PERIODS, max_loss=0.6,
        cumulative_returns=True, filter_zscore=None)
    ret_cols = get_forward_returns_columns(clean.columns)
    print('  factor_data %s, 收益列 %s' % (clean.shape, list(ret_cols)), flush=True)

    ic = perf.factor_information_coefficient(clean)
    mean_ret, _ = perf.mean_return_by_quantile(clean, by_date=True)
    mean_ret_mean = mean_ret.groupby(level=0).mean()
    ls = perf.factor_returns(clean, demeaned=True)
    autocol = perf.factor_rank_autocorrelation(clean, period=1)

    summary = pd.DataFrame({
        'IC均值': ic.mean(), 'IC标准差': ic.std(),
        'ICIR': ic.mean() / ic.std(),
        't值': ic.mean() / ic.std() * np.sqrt(ic.count()),
        'IC>0占比': (ic > 0).mean(),
    })
    metrics = {
        'factor': 'mom_20 (20 交易日动量)',
        'config': {'start': FACTOR_START, 'end': FACTOR_END, 'periods_days': list(PERIODS),
                   'quantiles': QUANTILES, 'universe_days': int(len(dates))},
        'ic': {c: {k: (None if pd.isna(v) else round(float(v), 6)) for k, v in row.items()}
               for c, row in summary.iterrows()},
        'quantile_mean_return': {c: [round(float(x), 6) for x in mean_ret_mean[c]] for c in ret_cols},
        'top_minus_bottom': {c: round(float(mean_ret_mean.loc[QUANTILES, c] - mean_ret_mean.loc[1, c]), 6) for c in ret_cols},
        'long_short_ir': {c: round(float(ls[c].mean() / ls[c].std()), 4) for c in ret_cols},
        'factor_rank_autocorr': round(float(autocol.mean()), 4),
    }
    with open(os.path.join(OUT, 'momentum_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    ic.to_csv(os.path.join(OUT, 'momentum_ic_series.csv'))

    # ---- 对照图 ----
    roe = json.load(open(os.path.join(OUT, 'roe_metrics.json'), encoding='utf-8'))
    labels = list(ret_cols)
    x = np.arange(len(labels))
    w = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in ((axes[0], 'IC均值', 'IC 均值 (Pearson)'),
                           (axes[1], 'ICIR', 'ICIR 信息比')):
        a = [roe['ic'][c].get(key) or 0 for c in labels]
        b = [metrics['ic'][c][key] for c in labels]
        ax.bar(x - w / 2, a, w, label='ROE (原始)', color='#4C72B0')
        ax.bar(x + w / 2, b, w, label='20日动量 (对照)', color='#DD8452')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.axhline(0, color='k', lw=0.8); ax.set_title(title); ax.legend()
        ax.set_xlabel('持有期')
    plt.suptitle('ROE vs 20日动量 —— 同一管线的对照 (2019-01 ~ 2026-07, 5004 只)')
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, '06_对照组IC对比.png'), dpi=130); plt.close()

    print(json.dumps(metrics['ic'], ensure_ascii=False, indent=2), flush=True)
    print('完成 ->', os.path.join(OUT, 'momentum_metrics.json'), flush=True)


if __name__ == '__main__':
    main()
