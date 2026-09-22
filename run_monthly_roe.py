# -*- coding: utf-8 -*-
"""ROE 因子 · 月度调仓版（用 get_clean_factor，绕开 alphalens 的非日频崩溃）

为什么用 get_clean_factor 而不是 get_clean_factor_and_forward_returns
--------------------------------------------------------------------
后者 = compute_forward_returns + get_clean_factor，而崩溃发生在
compute_forward_returns 末尾的 `df.index.levels[0].freq = freq`：
月度因子索引无法通过 pandas 2.x 的 freq 一致性校验。
get_clean_factor 接受**外部传入的 forward_returns**，于是月度调仓可以正常做。

三种口径
--------
  A 收盘→收盘   : r = P_close(T+h)/P_close(T) - 1      (alphalens 默认口径, 用于对拍)
  B 次日开盘→开盘: r = P_open(T+1+h)/P_open(T+1) - 1   (A股 T+1 实际可成交口径)
  C 隔夜跳空    : g = P_open(T+1)/P_close(T) - 1       (单独统计, **不计入**因子收益)

持有期: 21/63/126/252 交易日 (≈1/3/6/12 个月)

附加: 用 alphalens 的 factor_information_coefficient 与自行重算的 IC **对拍**
      (设计文档"防线#3 不变量对账"的最小演示)

输出: outputs/monthly_roe_*.json / .csv, outputs/charts/1x_*.png
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
import statsmodels.api as sm

plt.rcParams['font.sans-serif'] = ['FZLanTingHei-R-GBK', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from alphalens import performance as perf
from alphalens.utils import get_clean_factor, get_forward_returns_columns

# 私有数据根目录（可用环境变量 ALPHALENS_DATA_ROOT 覆盖）
DATA_ROOT = os.environ.get(
    'ALPHALENS_DATA_ROOT',
    os.path.expanduser('~/alphalens-data'))

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, 'cache'), os.path.join(HERE, 'outputs')
CHARTS = os.path.join(OUT, 'charts')
PIT_PATH = os.path.join(DATA_ROOT, 'data/financial_pit/financial_pit.parquet')

FACTOR_START, FACTOR_END = '2019-01-01', '2026-07-31'
HORIZONS = (21, 63, 126, 252)          # 交易日 ≈ 1/3/6/12 月
LABELS = {21: '1M(21D)', 63: '3M(63D)', 126: '6M(126D)', 252: '12M(252D)'}
QUANTILES = 5
STALENESS_DAYS = 400


def build_factor(dates, codes):
    pit = pd.read_parquet(PIT_PATH, columns=['code', 'report_date', 'roe', 'avail_314'])
    pit = pit[pit['avail_314'].notna()].copy()
    pit['avail_314'] = pd.to_datetime(pit['avail_314'])
    pit = pit.sort_values('avail_314', kind='mergesort')
    panel = pd.DataFrame({
        'code': np.repeat(codes, len(dates)),
        'date': np.tile(dates.values, len(codes)),
    }).sort_values('date', kind='mergesort')
    m = pd.merge_asof(panel, pit, left_on='date', right_on='avail_314',
                      by='code', direction='backward')
    m['staleness_days'] = (m['date'] - m['avail_314']).dt.days
    m.loc[m['staleness_days'] > STALENESS_DAYS, 'roe'] = np.nan
    return m.set_index(['date', 'code'])['roe'].dropna(), m


def fwd_returns(px, dates, h, mode):
    """在调仓日 dates 上构造持有 h 个交易日的前向收益 (MultiIndex date,asset)。"""
    sub = px.reindex(dates)
    if mode == 'close':
        r = px.shift(-h).reindex(dates) / sub - 1.0
    else:  # next_open: T+1 买入, T+1+h 卖出
        nxt = px.shift(-1).reindex(dates)
        r = px.shift(-(1 + h)).reindex(dates) / nxt - 1.0
    s = r.stack()
    s.index = s.index.set_names(['date', 'asset'])
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def ic_stats(clean, col):
    """IC: Pearson 与 Spearman 都算, 并给出 NW 修正 t。"""
    g = clean.groupby(level='date')
    pear = g.apply(lambda x: x['factor'].corr(x[col]))
    spear = g.apply(lambda x: x['factor'].corr(x[col], method='spearman'))
    n = int(spear.count())
    lag = int(col.replace('D', '')) // 21 + 1
    out = {}
    for name, s in (('IC_pearson', pear), ('RankIC', spear)):
        v = s.dropna()
        naive = float(v.mean() / v.std() * np.sqrt(len(v))) if len(v) > 2 else np.nan
        try:
            nw = float(sm.OLS(v.values, np.ones(len(v)))
                       .fit(cov_type='HAC', cov_kwds={'maxlags': lag}).tvalues[0])
        except Exception:
            nw = np.nan
        out[name] = {'mean': round(float(v.mean()), 6), 'std': round(float(v.std()), 6),
                     'icir': round(float(v.mean() / v.std()), 4) if v.std() else None,
                     't_naive': round(naive, 3), 't_nw': round(nw, 3),
                     'pos_ratio': round(float((v > 0).mean()), 4), 'n': int(len(v))}
    return out, spear


def main():
    os.makedirs(CHARTS, exist_ok=True)

    px = pd.read_parquet(os.path.join(CACHE, 'px_daily.parquet'))
    px.index = pd.to_datetime(px.index)
    op = pd.read_parquet(os.path.join(CACHE, 'px_daily_open.parquet'))
    op.index = pd.to_datetime(op.index)
    print('收盘面板 %s  开盘面板 %s' % (px.shape, op.shape), flush=True)

    cal = px.index
    me = pd.Series(cal, index=cal).resample('ME').last().dropna()
    rebal = pd.DatetimeIndex(me.values)
    rebal = rebal[(rebal >= FACTOR_START) & (rebal <= FACTOR_END)]
    print('月度调仓日 %d 个: %s ~ %s' % (len(rebal), rebal[0].date(), rebal[-1].date()), flush=True)

    codes = [c for c in px.columns if px[c].notna().sum() > 250 and c in op.columns]
    print('参与股票 %d 只' % len(codes), flush=True)

    factor, pit_panel = build_factor(rebal, codes)
    print('ROE 因子观测 %d 条 (%d 个调仓日 × %d 只)'
          % (len(factor), factor.index.levels[0].size, factor.index.levels[1].size), flush=True)

    result = {'config': {
        'factor': 'roe (净资产收益率, PIT as-of avail_314)',
        'rebalance': 'monthly (每月最后交易日)',
        'periods': list(HORIZONS), 'quantiles': QUANTILES,
        'universe': len(codes), 'rebalances': int(len(rebal)),
        'staleness_days': STALENESS_DAYS,
    }, 'conventions': {}}

    # ---------- 三种口径的 forward returns ----------
    fr = {}
    print('构造前向收益...', flush=True)
    fr['close'] = pd.DataFrame({f'{h}D': fwd_returns(px, rebal, h, 'close') for h in HORIZONS})
    fr['open'] = pd.DataFrame({f'{h}D': fwd_returns(op, rebal, h, 'open') for h in HORIZONS})
    # 隔夜跳空 (只有 1 天, 与持有期无关)
    gap = (op.shift(-1).reindex(rebal) / px.reindex(rebal) - 1.0).stack()
    gap.index = gap.index.set_names(['date', 'asset'])
    fr['gap'] = pd.DataFrame({'1D': gap.replace([np.inf, -np.inf], np.nan).dropna()})
    for k, v in fr.items():
        print('  %-6s %s' % (k, v.shape), flush=True)

    # ---------- 逐口径清洗 + 指标 ----------
    cleans = {}
    for name, frd in fr.items():
        try:
            clean = get_clean_factor(factor, frd, quantiles=QUANTILES, max_loss=0.7)
        except Exception as e:
            print('  [%s] 清洗失败: %s' % (name, e), flush=True)
            continue
        cleans[name] = clean
        cols = get_forward_returns_columns(clean.columns)
        print('  [%s] factor_data %s 列=%s' % (name, clean.shape, list(cols)), flush=True)

        conv = {'factor_data_shape': list(clean.shape), 'periods': {}}
        for c in cols:
            st, _ = ic_stats(clean, c)
            conv['periods'][c] = st

        if name in ('close', 'open'):
            mr, _ = perf.mean_return_by_quantile(clean, by_date=True)
            mrm = mr.groupby(level=0).mean()
            conv['quantile_mean_return'] = {c: [round(float(x), 6) for x in mrm[c]] for c in cols}
            conv['top_minus_bottom'] = {
                c: round(float(mrm.loc[QUANTILES, c] - mrm.loc[1, c]), 6) for c in cols}
            ls = perf.factor_returns(clean, demeaned=True)
            conv['long_short_ir'] = {c: round(float(ls[c].mean() / ls[c].std()), 4) for c in cols}
            conv['factor_rank_autocorr'] = round(float(perf.factor_rank_autocorrelation(clean, period=1).mean()), 4)
            conv['turnover_mean'] = round(float(np.mean([
                perf.quantile_turnover(pd.Series(clean['factor_quantile']), q, period=1).mean()
                for q in range(1, QUANTILES + 1)])), 4)
            cleans[name + '_mrm'] = mrm
            cleans[name + '_ls'] = ls
        else:
            g = clean['1D']
            conv['gap_mean'] = round(float(g.mean()), 6)
            conv['gap_std'] = round(float(g.std()), 6)
            conv['gap_pos_ratio'] = round(float((g > 0).mean()), 4)
        result['conventions'][name] = conv

    # ---------- 对拍: alphalens 的 IC vs 自行重算 ----------
    cross = {}
    for name in ('close', 'open'):
        if name not in cleans:
            continue
        clean = cleans[name]
        cols = get_forward_returns_columns(clean.columns)
        al = perf.factor_information_coefficient(clean)
        for c in cols:
            mine, _ = ic_stats(clean, c)
            cross['%s_%s' % (name, c)] = {
                'alphalens_IC_mean': round(float(al[c].mean()), 8),
                'recomputed_RankIC_mean': round(mine['RankIC']['mean'], 8),
                'match': bool(abs(float(al[c].mean()) - mine['RankIC']['mean']) < 1e-6
                              or abs(float(al[c].mean()) - mine['IC_pearson']['mean']) < 1e-6),
            }
    result['cross_check'] = cross

    # ---------- 图表 ----------
    print('出图...', flush=True)
    labels = [LABELS[h] for h in HORIZONS]

    # 1. IC 对比: 收盘 vs 次日开盘
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(HORIZONS)); w = 0.36
    for i, (name, color, lab) in enumerate((('close', '#4C72B0', 'A 收盘→收盘'),
                                            ('open', '#DD8452', 'B 次日开盘→开盘'))):
        if name not in result['conventions']:
            continue
        v = [result['conventions'][name]['periods'][f'{h}D']['RankIC']['mean'] for h in HORIZONS]
        ax.bar(x + (i - 0.5) * w, v, w, label=lab, color=color)
    ax.axhline(0, color='k', lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('RankIC 均值'); ax.set_xlabel('持有期')
    ax.set_title('ROE 月度调仓 · RankIC 对比（T+1 开盘成交是否吃掉信号）')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, '11_月度IC对比.png'), dpi=130); plt.close()

    # 2/3. 分层平均收益
    for name, fname, tit in (('close', '12_月度分层_收盘', 'A 口径 收盘→收盘'),
                             ('open', '13_月度分层_开盘', 'B 口径 次日开盘→开盘')):
        if name + '_mrm' not in cleans:
            continue
        mrm = cleans[name + '_mrm']
        fig, ax = plt.subplots(figsize=(10, 5))
        mrm.T.plot(kind='bar', ax=ax, width=0.8)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_title('ROE 分层组合平均持有期收益 · %s（月度调仓, 5 分位）' % tit)
        ax.set_xlabel('持有期'); ax.set_ylabel('平均收益')
        ax.set_xticklabels(labels, rotation=0)
        ax.legend(title='分位', ncol=5, fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(CHARTS, f'{fname}.png'), dpi=130); plt.close()

    # 4. 各分位累计净值 (开盘口径, 1M)
    if 'open_ls' in cleans:
        mr, _ = perf.mean_return_by_quantile(cleans['open'], by_date=True)
        fig, ax = plt.subplots(figsize=(11, 5))
        for q in range(1, QUANTILES + 1):
            s = mr.xs(q, level=0)['21D']
            ax.plot(s.index, (1 + s.fillna(0)).cumprod(), label='Q%d' % q, lw=1.3)
        ax.set_yscale('log')
        ax.set_title('ROE 月度调仓 · 各分位累计净值（B 口径 次日开盘, 1 月持有）')
        ax.legend(ncol=5); ax.set_xlabel('')
        plt.tight_layout(); plt.savefig(os.path.join(CHARTS, '14_月度分位累计.png'), dpi=130); plt.close()

    # 5. 隔夜跳空
    if 'gap' in cleans:
        g = cleans['gap']['1D']
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        axes[0].hist(g.clip(-0.12, 0.12), bins=60, color='#55A868', alpha=0.85)
        axes[0].axvline(0, color='k', lw=0.8); axes[0].set_title('隔夜跳空分布（已截断 ±12%%）')
        axes[0].set_xlabel('P_open(T+1)/P_close(T) − 1')
        cum = (1 + g.groupby(level='date').mean()).cumprod()
        axes[1].plot(cum.index, cum.values, color='#55A868', lw=1.5)
        axes[1].axhline(1, color='k', lw=0.8)
        axes[1].set_title('隔夜跳空累计（等权, 月度）')
        plt.tight_layout(); plt.savefig(os.path.join(CHARTS, '15_隔夜跳空.png'), dpi=130); plt.close()

    # ---------- 落盘 ----------
    with open(os.path.join(OUT, 'monthly_roe_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    pd.DataFrame(cross).T.to_csv(os.path.join(OUT, 'monthly_roe_crosscheck.csv'))

    print(json.dumps({'RankIC': {k: {c: v['periods'][c]['RankIC'] for c in v['periods']}
                                 for k, v in result['conventions'].items()}},
                     ensure_ascii=False, indent=2), flush=True)
    print('完成 ->', OUT, flush=True)


if __name__ == '__main__':
    main()
