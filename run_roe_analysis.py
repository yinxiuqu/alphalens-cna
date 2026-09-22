# -*- coding: utf-8 -*-
"""ROE 因子标准分析 —— quantaxis 数据 + alphalens-reloaded

数据链路
--------
  财务:  <DATA_ROOT>/data/financial_pit/financial_pit.parquet
         (PIT 层, avail_314 = 财报公告日 + 1 个交易日, 避免前视)
  行情:  cache/px_daily.parquet  (由 prep_panel.py 从 quantaxis mongo 构建的复权价)

因子口径
--------
  * 因子: 净资产收益率 roe (tushare/通达信字段 197), PIT as-of, staleness <= 400 天
  * 调仓: 月度(每月最后交易日), 与 私有数据仓 的因子约定一致
  * 持有期: 1 / 5 / 10 / 20 个交易日
  * 分层: 5 分位

输出
----
  outputs/roe_metrics.json     指标 (IC/ICIR/分层收益/多空/换手)
  outputs/roe_clean.parquet    alphalens 清洗后的 factor_data (供复核)
  outputs/charts/*.png         图表
  outputs/ROE因子分析报告.md    报告
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
from alphalens import plotting as apl
from alphalens.utils import get_clean_factor_and_forward_returns, get_forward_returns_columns

# 私有数据根目录（可用环境变量 ALPHALENS_DATA_ROOT 覆盖）
DATA_ROOT = os.environ.get(
    'ALPHALENS_DATA_ROOT',
    os.path.expanduser('~/alphalens-data'))

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
OUT = os.path.join(HERE, 'outputs')
CHARTS = os.path.join(OUT, 'charts')
PIT_PATH = os.path.join(DATA_ROOT, 'data/financial_pit/financial_pit.parquet')

FACTOR_START, FACTOR_END = '2019-01-01', '2026-07-31'
PERIODS = (1, 5, 10, 20)
QUANTILES = 5
STALENESS_DAYS = 400
MAX_LOSS = 0.6


def build_factor(rebal_dates, codes):
    """构造月度 PIT as-of 的 ROE 因子 (date, code) MultiIndex。"""
    pit = pd.read_parquet(PIT_PATH, columns=['code', 'report_date', 'roe', 'avail_314'])
    pit = pit[pit['avail_314'].notna()].copy()
    pit['avail_314'] = pd.to_datetime(pit['avail_314'])
    pit = pit.sort_values('avail_314', kind='mergesort')

    panel = pd.DataFrame({
        'code': np.repeat(codes, len(rebal_dates)),
        'date': np.tile(rebal_dates.values, len(codes)),
    }).sort_values('date', kind='mergesort')

    m = pd.merge_asof(panel, pit, left_on='date', right_on='avail_314',
                      by='code', direction='backward')
    m['staleness_days'] = (m['date'] - m['avail_314']).dt.days
    stale = m['staleness_days'] > STALENESS_DAYS
    m.loc[stale, 'roe'] = np.nan
    print('  PIT 面板 %d 行; 无可用财报 %.1f%%; 超期剔除 %.1f%%'
          % (len(m), 100 * m['roe'].isna().mean(), 100 * stale.mean()), flush=True)
    return m, m.set_index(['date', 'code'])['roe'].dropna()


def main():
    os.makedirs(CHARTS, exist_ok=True)

    # ---------- 行情 ----------
    px = pd.read_parquet(os.path.join(CACHE, 'px_daily.parquet'))
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    print('行情面板 %s  %s ~ %s' % (px.shape, px.index.min().date(), px.index.max().date()), flush=True)

    cal = px.index
    # 因子观测日 = 全部交易日。
    # 注意: 这里**不能**用"月末最后交易日" —— alphalens 的
    # compute_forward_returns 末尾会执行 `df.index.levels[0].freq = freq`,
    # 而 freq 来自 infer_trading_calendar 推出来的 CustomBusinessDay。
    # 月度索引不是每个工作日都有, pandas 2.x 会做 freq 一致性校验并抛
    # "ValueError: Inferred frequency None ... does not conform to passed
    # frequency C", 整个流程直接失败。
    # (已实测: 月度索引必崩, 日度索引正常。)
    rebal = cal[(cal >= FACTOR_START) & (cal <= FACTOR_END)]
    print('因子观测日 %d 个 (全部交易日): %s ~ %s'
          % (len(rebal), rebal[0].date(), rebal[-1].date()), flush=True)

    codes = [c for c in px.columns if px[c].notna().sum() > 250]
    print('参与股票 %d 只 (行情行数>250)' % len(codes), flush=True)

    # ---------- 因子 ----------
    print('构造 ROE 因子...', flush=True)
    pit_panel, factor = build_factor(rebal, codes)
    print('  因子观测 %d 条, 覆盖 %d 个调仓日, %d 只股票'
          % (len(factor), factor.index.levels[0].size, factor.index.levels[1].size), flush=True)

    # ---------- alphalens 清洗 ----------
    print('alphalens 清洗...', flush=True)
    clean = get_clean_factor_and_forward_returns(
        factor, px, quantiles=QUANTILES, periods=PERIODS, max_loss=MAX_LOSS,
        cumulative_returns=True, filter_zscore=None)
    ret_cols = get_forward_returns_columns(clean.columns)
    print('  factor_data %s, 收益列 %s' % (clean.shape, ret_cols), flush=True)

    # ---------- 指标 ----------
    ic = perf.factor_information_coefficient(clean)
    ic_summary = pd.DataFrame({
        'IC均值': ic.mean(),
        'IC标准差': ic.std(),
        'ICIR': ic.mean() / ic.std(),
        't值': ic.mean() / ic.std() * np.sqrt(ic.count()),
        'IC>0占比': (ic > 0).mean(),
        '样本数': ic.count(),
    })
    rank_ic = perf.factor_information_coefficient(clean, group_adjust=False)
    mean_ret, std_err = perf.mean_return_by_quantile(clean, by_date=True)
    # 注意: by_date=True 时返回值的索引层级顺序是 ['factor_quantile', 'date']
    # (performance.mean_return_by_quantile 里 grouper 就是这个顺序), 分位在 level 0。
    mean_ret_mean = mean_ret.groupby(level=0).mean()   # 索引 = 分位
    ls = perf.factor_returns(clean, demeaned=True)          # 多空(分位加权, 去均值)
    autocol = perf.factor_rank_autocorrelation(clean, period=1)
    q_turnover = {q: perf.quantile_turnover(
        pd.Series(clean['factor_quantile']), q, period=1) for q in range(1, QUANTILES + 1)}

    top = mean_ret_mean.loc[QUANTILES]
    bot = mean_ret_mean.loc[1]
    ls_mean = (top - bot)

    metrics = {
        'config': {
            'factor': 'roe (净资产收益率, PIT as-of)',
            'factor_start': FACTOR_START, 'factor_end': FACTOR_END,
            'rebalance': 'monthly (month-end trading day)',
            'periods_days': list(PERIODS), 'quantiles': QUANTILES,
            'staleness_days': STALENESS_DAYS, 'max_loss': MAX_LOSS,
            'universe': len(codes), 'rebalances': int(len(rebal)),
            'factor_obs': int(len(factor)),
        },
        'ic': {c: {k: (None if pd.isna(v) else round(float(v), 6)) for k, v in row.items()}
               for c, row in ic_summary.iterrows()},
        'quantile_mean_return': {c: [round(float(x), 6) for x in mean_ret_mean[c]] for c in ret_cols},
        'top_minus_bottom': {c: round(float(ls_mean[c]), 6) for c in ret_cols},
        'long_short_mean': {c: round(float(ls[c].mean()), 6) for c in ret_cols},
        'long_short_ir': {c: round(float(ls[c].mean() / ls[c].std()), 4) for c in ret_cols},
        'factor_rank_autocorr': round(float(autocol.mean()), 4),
        'quantile_turnover_mean': {str(q): round(float(v.mean()), 4) for q, v in q_turnover.items()},
        'ic_positive_ratio': {c: round(float((ic[c] > 0).mean()), 4) for c in ret_cols},
    }

    # ---------- 图表 ----------
    print('出图...', flush=True)
    plt.rcParams['figure.max_open_warning'] = 0

    # 1. 分位平均收益
    fig, ax = plt.subplots(figsize=(9, 5))
    mean_ret_mean.T.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('ROE 分位组合平均持有期收益 (月度调仓, %d 分位)' % QUANTILES)
    ax.set_xlabel('持有期'); ax.set_ylabel('平均收益')
    ax.axhline(0, color='k', lw=0.8)
    ax.legend(title='分位', ncol=QUANTILES, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, '01_分位平均收益.png'), dpi=130); plt.close()

    # 2. IC 时序
    fig, ax = plt.subplots(figsize=(11, 5))
    ic.plot(ax=ax, lw=1)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('ROE 因子 IC 时序 (Rank IC, 月度)')
    ax.set_ylabel('IC'); ax.set_xlabel('')
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, '02_IC时序.png'), dpi=130); plt.close()

    # 3. IC 分布
    fig, axes = plt.subplots(1, len(ret_cols), figsize=(4 * len(ret_cols), 3.6))
    for a, c in zip(np.atleast_1d(axes), ret_cols):
        a.hist(ic[c].dropna(), bins=25, color='#4C72B0', alpha=0.85)
        a.axvline(0, color='k', lw=0.8)
        a.set_title('%s (均值 %.3f)' % (c, ic[c].mean()), fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, '03_IC分布.png'), dpi=130); plt.close()

    # 4. 多空累计
    fig, ax = plt.subplots(figsize=(11, 5))
    for c in ret_cols:
        ax.plot(ls.index, (1 + ls[c].fillna(0)).cumprod(), label=c, lw=1.3)
    ax.set_yscale('log')
    ax.set_title('多空组合累计净值 (对数轴, 因子去均值加权)')
    ax.legend(); ax.set_xlabel('')
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, '04_多空累计净值.png'), dpi=130); plt.close()

    # 5. 各分位累计净值
    fig, ax = plt.subplots(figsize=(11, 5))
    for q in range(1, QUANTILES + 1):
        series = mean_ret.xs(q, level=0)[ret_cols[0]]
        ax.plot(series.index, (1 + series.fillna(0)).cumprod(), label='Q%d' % q, lw=1.3)
    ax.set_title('各分位组合累计净值 (%s 持有期, 因子日度观测)' % ret_cols[0])
    ax.set_yscale('log'); ax.legend(ncol=QUANTILES); ax.set_xlabel('')
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, '05_分位累计净值.png'), dpi=130); plt.close()

    # ---------- 落盘 ----------
    clean.to_parquet(os.path.join(OUT, 'roe_clean.parquet'))
    with open(os.path.join(OUT, 'roe_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    ic.to_csv(os.path.join(OUT, 'roe_ic_series.csv'))

    print(json.dumps(metrics['ic'], ensure_ascii=False, indent=2), flush=True)
    print('完成 ->', OUT, flush=True)
    return metrics


if __name__ == '__main__':
    main()
