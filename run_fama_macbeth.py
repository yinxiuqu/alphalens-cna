# -*- coding: utf-8 -*-
"""Fama-MacBeth 截面回归 —— 石川《因子投资：方法与实践》的核心方法

alphalens 只有分组排序(portfolio sort), **完全没有截面回归**。
本脚本用现有数据把它做出来, 作为 alphalens-cna 该有的能力演示。

方法
----
每个调仓日 t 做一次横截面 OLS:
    r_{i, t→t+1} = γ_0t + γ_1t·ROE_i + γ_2t·PB_i + γ_3t·lnMV_i + γ_4t·TURN_i + ε_i
然后对 γ 的时间序列做统计推断:
    γ̄ = mean(γ_t)        朴素 t = γ̄ / (std(γ_t)/√T)
    Newey-West 修正 t (月度重叠 → 自相关)

按石川书的标准流程:
  1. 去极值  MAD 法: 中位数 ± 3 × 1.4826 × MAD
  2. 标准化  横截面 z-score
  3. 正交化  ROE 对 (PB, lnMV, TURN) 回归取残差 → 看增量信息

三种设定
  A 单因子:   只用 ROE
  B 多因子:   ROE + PB + lnMV + TURN
  C 正交化:   用 ROE 对控制变量回归后的残差

输出: outputs/fm_*.json/.csv, outputs/charts/2x_*.png
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
from pymongo import MongoClient

# 私有数据根目录（可用环境变量 ALPHALENS_DATA_ROOT 覆盖）
DATA_ROOT = os.environ.get(
    'ALPHALENS_DATA_ROOT',
    os.path.expanduser('~/alphalens-data'))

plt.rcParams['font.sans-serif'] = ['FZLanTingHei-R-GBK', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, 'cache'), os.path.join(HERE, 'outputs')
CHARTS = os.path.join(OUT, 'charts')
PIT_PATH = os.path.join(DATA_ROOT, 'data/financial_pit/financial_pit.parquet')

START, END = '2019-01-01', '2026-07-31'
H_FWD = 21                    # 21 交易日 ≈ 1 个月
STALENESS = 400
MIN_STOCKS = 200              # 单次横截面最少股票数


# --------------------------------------------------------------------------- #
def get_rebal_dates():
    px = pd.read_parquet(os.path.join(CACHE, 'px_daily.parquet'))
    px.index = pd.to_datetime(px.index)
    cal = px.index.sort_values()
    me = pd.Series(cal, index=cal).resample('ME').last().dropna()
    r = pd.DatetimeIndex(me.values)
    return px, r[(r >= START) & (r <= END)]


def fetch_exposures(dates):
    """按日取 stock_daily_basic (有 date 索引, 按日查很快)"""
    db = MongoClient('127.0.0.1', 27017, serverSelectionTimeoutMS=15000)['quantaxis']
    rows = []
    for i, d in enumerate(dates, 1):
        ds = d.strftime('%Y-%m-%d')
        cur = db.stock_daily_basic.find(
            {'date': ds}, {'code': 1, 'pb': 1, 'total_mv': 1, 'turnover_rate': 1, 'pe_ttm': 1, '_id': 0})
        for r in cur:
            r['date'] = d
            rows.append(r)
        if i % 20 == 0 or i == len(dates):
            print('   取数 %d/%d  累计 %d 行' % (i, len(dates), len(rows)), flush=True)
    df = pd.DataFrame(rows)
    for c in ('pb', 'total_mv', 'turnover_rate', 'pe_ttm'):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def roe_asof(dates, codes):
    pit = pd.read_parquet(PIT_PATH, columns=['code', 'roe', 'avail_314'])
    pit = pit[pit['avail_314'].notna()].copy()
    pit['avail_314'] = pd.to_datetime(pit['avail_314'])
    pit = pit.sort_values('avail_314', kind='mergesort')
    panel = pd.DataFrame({'code': np.repeat(codes, len(dates)),
                          'date': np.tile(dates.values, len(codes))}).sort_values('date', kind='mergesort')
    m = pd.merge_asof(panel, pit, left_on='date', right_on='avail_314', by='code', direction='backward')
    m['staleness'] = (m['date'] - m['avail_314']).dt.days
    m.loc[m['staleness'] > STALENESS, 'roe'] = np.nan
    return m[['code', 'date', 'roe']]


def fwd_ret_1m(px, dates, mode='close'):
    if mode == 'close':
        r = px.shift(-H_FWD).reindex(dates) / px.reindex(dates) - 1.0
    else:
        op = pd.read_parquet(os.path.join(CACHE, 'px_daily_open.parquet'))
        op.index = pd.to_datetime(op.index)
        r = op.shift(-(1 + H_FWD)).reindex(dates) / op.shift(-1).reindex(dates) - 1.0
    s = r.stack()
    s.index = s.index.set_names(['date', 'code'])
    return s.rename('fwd_ret')


# --------------------------------------------------------------------------- #
def mad_winsorize(s, n=3):
    med = s.median()
    mad = (s - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return s
    hi, lo = med + n * 1.4826 * mad, med - n * 1.4826 * mad
    return s.clip(lo, hi)


def zscore(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) else s * np.nan


def prep_panel(expo, roe, fwd):
    df = expo.merge(roe, on=['code', 'date'], how='left').merge(
        fwd.reset_index(), on=['date', 'code'], how='left')
    df = df[df['total_mv'].notna() & (df['total_mv'] > 0)]
    df['ln_mv'] = np.log(df['total_mv'])
    df = df.replace([np.inf, -np.inf], np.nan)

    def norm(g):
        out = {}
        for c in ('roe', 'pb', 'ln_mv', 'turnover_rate'):
            out[c + '_z'] = zscore(mad_winsorize(g[c]))
        return pd.DataFrame(out, index=g.index)

    parts = [g.join(norm(g)) for _, g in df.groupby('date')]
    return pd.concat(parts).dropna(subset=['fwd_ret'])


def fama_macbeth(panel, xs, min_n=MIN_STOCKS):
    """逐期横截面 OLS, 返回 γ 的 DataFrame (index=date, columns=因子)。"""
    gammas = []
    for d, g in panel.groupby('date'):
        sub = g.dropna(subset=xs + ['fwd_ret'])
        if len(sub) < min_n:
            continue
        X = sm.add_constant(sub[xs].values, has_constant='add')
        try:
            res = sm.OLS(sub['fwd_ret'].values, X).fit()
        except Exception:
            continue
        rec = {'date': d, 'n': len(sub), 'r2': res.rsquared}
        rec.update(dict(zip(['const'] + xs, res.params)))
        gammas.append(rec)
    return pd.DataFrame(gammas).set_index('date')


def fm_stats(g):
    """γ 序列的均值 / 朴素 t / Newey-West t。"""
    out = {}
    T = len(g)
    for c in g.columns:
        if c in ('n', 'r2'):
            continue
        v = g[c].dropna()
        mean = float(v.mean())
        t_naive = float(mean / (v.std() / np.sqrt(len(v)))) if len(v) > 2 else np.nan
        try:
            t_nw = float(sm.OLS(v.values, np.ones(len(v)))
                         .fit(cov_type='HAC', cov_kwds={'maxlags': 3}).tvalues[0])
        except Exception:
            t_nw = np.nan
        out[c] = {'gamma_mean': round(mean, 6), 'std': round(float(v.std()), 6),
                  't_naive': round(t_naive, 3), 't_nw': round(t_nw, 3),
                  'pos_ratio': round(float((v > 0).mean()), 4), 'T': len(v)}
    return out


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(CHARTS, exist_ok=True)
    px, rebal = get_rebal_dates()
    print('月度调仓日 %d 个' % len(rebal), flush=True)

    print('取横截面暴露 (stock_daily_basic, 按日索引)...', flush=True)
    expo = fetch_exposures(rebal)
    print('   暴露面板 %s' % (expo.shape,), flush=True)

    codes = sorted(expo['code'].unique())
    print('构造 ROE PIT as-of...', flush=True)
    roe = roe_asof(rebal, codes)

    print('构造前向收益...', flush=True)
    fwd_c = fwd_ret_1m(px, rebal, 'close').reset_index()
    fwd_o = fwd_ret_1m(px, rebal, 'open').reset_index()

    result = {'config': {'rebalance': 'monthly', 'forward_days': H_FWD,
                         'winsorize': 'MAD(3)', 'standardize': 'cross-sectional z-score',
                         'min_stocks_per_period': MIN_STOCKS,
                         'range': [str(rebal[0].date()), str(rebal[-1].date())], 'periods': len(rebal)}}

    for tag, fwd in (('close', fwd_c), ('open', fwd_o)):
        panel = prep_panel(expo, roe, fwd)
        print('[%s] 面板 %s, %d 期' % (tag, panel.shape, panel['date'].nunique()), flush=True)

        # 相关矩阵
        corr = panel[['roe_z', 'pb_z', 'ln_mv_z', 'turnover_rate_z']].corr()
        result.setdefault('factor_corr', {})[tag] = corr.round(4).to_dict()

        # A 单因子
        gA = fama_macbeth(panel, ['roe_z'])
        # B 多因子
        gB = fama_macbeth(panel, ['roe_z', 'pb_z', 'ln_mv_z', 'turnover_rate_z'])
        # C 正交化: ROE 对控制变量回归取残差
        orth = []
        for d, g in panel.groupby('date'):
            sub = g.dropna(subset=['roe_z', 'pb_z', 'ln_mv_z', 'turnover_rate_z'])
            if len(sub) < MIN_STOCKS:
                continue
            Xo = sm.add_constant(sub[['pb_z', 'ln_mv_z', 'turnover_rate_z']].values, has_constant='add')
            b = np.linalg.lstsq(Xo, sub['roe_z'].values, rcond=None)[0]
            r = sub['roe_z'].values - Xo @ b
            orth.append(pd.DataFrame({'date': d, 'code': sub['code'].values, 'roe_orth': r}))
        panel_o = panel.merge(pd.concat(orth), on=['date', 'code'], how='inner')
        gC = fama_macbeth(panel_o, ['roe_orth'])

        result.setdefault('specs', {})[tag] = {
            'A_univariate': fm_stats(gA),
            'B_multivariate': fm_stats(gB),
            'C_orthogonalized': fm_stats(gC),
            'mean_r2': round(float(gB['r2'].mean()), 4),
            'mean_n': int(gB['n'].mean()),
        }
        result.setdefault('gamma_series', {})[tag] = {
            'A': gA['roe_z'].round(6).rename(lambda d: str(d.date())).to_dict(),
            'B_roe': gB['roe_z'].round(6).rename(lambda d: str(d.date())).to_dict(),
        }
        pd.DataFrame({'A_univariate': gA['roe_z'], 'B_multivariate': gB['roe_z']}).to_csv(
            os.path.join(OUT, 'fm_gamma_%s.csv' % tag))
        if tag == 'open':
            gB.to_csv(os.path.join(OUT, 'fm_gamma_full_open.csv'))

    # ---------- 图 ----------
    S = result['specs']['open']
    specs = [('A 单因子 ROE', 'A_univariate'), ('B 多因子(控 PB/规模/换手)', 'B_multivariate'),
             ('C 正交化 ROE', 'C_orthogonalized')]
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [S[k]['roe_z' if k != 'C_orthogonalized' else 'roe_orth']['gamma_mean'] for _, k in specs]
    ts = [S[k]['roe_z' if k != 'C_orthogonalized' else 'roe_orth']['t_nw'] for _, k in specs]
    x = np.arange(len(specs))
    b = ax.bar(x, vals, 0.5, color=['#4C72B0', '#DD8452', '#55A868'])
    for xi, (v, t) in enumerate(zip(vals, ts)):
        ax.text(xi, v, '  γ=%.4f\nt(NW)=%.2f' % (v, t), ha='center',
                va='bottom' if v >= 0 else 'top', fontsize=9)
    ax.axhline(0, color='k', lw=0.8); ax.set_xticks(x); ax.set_xticklabels([n for n, _ in specs])
    ax.set_ylabel('因子收益率 γ 均值 (月)')
    ax.set_title('Fama-MacBeth 截面回归 · ROE 的因子收益率（月度, 次日开盘口径）')
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, '21_FM因子收益率.png'), dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(11, 5))
    g = pd.read_csv(os.path.join(OUT, 'fm_gamma_full_open.csv'), index_col=0, parse_dates=True)
    ax.plot(g.index, g['roe_z'], lw=1.2, color='#4C72B0')
    ax.axhline(g['roe_z'].mean(), color='r', ls='--', lw=1,
               label='均值 %.4f' % g['roe_z'].mean())
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('ROE 因子收益率 γ 时序（多因子设定, 月度）')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(CHARTS, '22_FM_gamma时序.png'), dpi=130); plt.close()

    with open(os.path.join(OUT, 'fama_macbeth.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result['specs']['open'], ensure_ascii=False, indent=2), flush=True)
    print('完成 ->', OUT, flush=True)


if __name__ == '__main__':
    main()
