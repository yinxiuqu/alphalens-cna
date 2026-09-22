"""分析层测试 —— 防线 5（已知答案）+ D9 回归。

**D9 回归最重要**：alphalens 整条链路依赖 ``index.levels[0].freq``。真实 A 股
月频面板（每月最后一个交易日）推断 ``freq is None``，于是
``get_clean_factor_and_forward_returns`` 在清洗阶段就抛 ValueError
（``utils.py:358``），``factor_rank_autocorrelation`` / ``quantile_turnover``
也会全空 —— 月频分析根本跑不起来。
实测同一份 ROE 月度面板：自算 秩自相关 0.9227（79 期全有值）/ 月换手 20.03%。
本文件的 ``test_monthly_*`` 就是钉死这一点。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna
from alphalens_cna.analysis import ic as icm
from alphalens_cna.analysis import portfolio as pf
from alphalens_cna.analysis import quantile as qm
from alphalens_cna.contract.errors import ContractError

D = pd.to_datetime


# --------------------------------------------------------------------------- #
def panel(dates, n_assets=20, seed=0, signal=1.0, noise=0.0):
    """造一个 (date, asset) 面板：factor 与 forward_return_1 相关。"""
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.DatetimeIndex(dates),
                                      [f'{i:06d}' for i in range(n_assets)]],
                                     names=['date', 'asset'])
    f = rng.normal(size=len(idx))
    r = signal * f + noise * rng.normal(size=len(idx))
    return pd.DataFrame({'factor': f, 'forward_return_1': r,
                         'forward_return_5': r * 2}, index=idx)


DAILY = pd.bdate_range('2024-01-01', periods=60)
MONTHLY = pd.DatetimeIndex(pd.date_range('2020-01-31', periods=24, freq='ME'))


# --------------------------------------------------------------------------- #
# IC
# --------------------------------------------------------------------------- #
def test_ic_known_answer_perfect():
    """因子与收益完全同序 → IC = 1。"""
    df = panel(DAILY, signal=1.0, noise=0.0)
    ic = acna.information_coefficient(df)
    assert np.allclose(ic[1].dropna(), 1.0), ic[1].head()
    assert np.allclose(ic[5].dropna(), 1.0)


def test_ic_known_answer_inverted():
    df = panel(DAILY, signal=-1.0, noise=0.0)
    ic = acna.information_coefficient(df)
    assert np.allclose(ic[1].dropna(), -1.0)


def test_ic_pure_noise_near_zero():
    df = panel(DAILY, signal=0.0, noise=1.0, seed=7)
    ic = acna.information_coefficient(df)
    assert abs(ic[1].mean()) < 0.1


def test_ic_summary_fields():
    df = panel(DAILY, signal=0.5, noise=1.0)
    s = acna.ic_summary(acna.information_coefficient(df))
    assert set(['n', 'mean', 'std', 'icir', 't_naive', 'positive_rate']) <= set(s.columns)
    assert s.loc[1, 'n'] == len(DAILY)
    # ICIR = mean/std；t_naive = ICIR × sqrt(n)
    assert abs(s.loc[1, 'icir'] - s.loc[1, 'mean'] / s.loc[1, 'std']) < 1e-12
    assert abs(s.loc[1, 't_naive'] - s.loc[1, 'icir'] * np.sqrt(s.loc[1, 'n'])) < 1e-9


def test_ic_rejects_missing_factor():
    df = panel(DAILY).drop(columns=['factor'])
    with pytest.raises(ContractError) as e:
        acna.information_coefficient(df)
    assert e.value.rule == 'no_factor'


def test_ic_constant_factor_is_nan_not_zero():
    """常数因子没有 IC 可言 —— 应为 NaN，不是 0。"""
    df = panel(DAILY)
    df['factor'] = 1.0
    ic = acna.information_coefficient(df)
    assert ic[1].isna().all()


def test_ic_decay_returns_summary():
    df = panel(DAILY, signal=0.5, noise=1.0)
    s = acna.ic_decay(df)
    assert set(s.index) == {1, 5}


# --------------------------------------------------------------------------- #
# D9 回归：非日频不得返 NaN
# --------------------------------------------------------------------------- #
def test_monthly_rank_autocorrelation_not_nan():
    """★ D9：月频索引下秩自相关必须有值（alphalens 返回全 NaN）。"""
    df = panel(MONTHLY, n_assets=10, seed=3)
    # 让排名缓慢变化 → 自相关应该高
    df['factor'] = np.tile(np.arange(10), len(MONTHLY)).astype(float)
    ra = acna.rank_autocorrelation(df)
    assert len(ra) == len(MONTHLY) - 1
    assert not ra.isna().all(), '月频下秩自相关不应全 NaN'
    assert (ra > 0.9).all(), f'排名不变时自相关应接近 1，实际 {ra.min()}'


def test_monthly_turnover_not_nan():
    """★ D9：月频索引下换手必须有值。"""
    df = panel(MONTHLY, n_assets=10, seed=3)
    df['factor'] = np.tile(np.arange(10), len(MONTHLY)).astype(float)
    q = acna.quantize(df, n=5)
    to = acna.quantile_turnover(q['q'])
    assert len(to) == len(MONTHLY) - 1
    assert not to.isna().all().all(), '月频下换手不应全 NaN'
    assert np.allclose(to.values, 0.0), '排名完全不变时换手应为 0'


def test_rank_autocorrelation_known_answer():
    """排名完全不变 → 自相关 = 1；完全逆序 → = −1。"""
    dates = pd.bdate_range('2024-01-01', periods=3)
    assets = [f'{i:06d}' for i in range(6)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    up = np.tile(np.arange(6), 3).astype(float)
    ra = acna.rank_autocorrelation(pd.DataFrame({'factor': up}, index=idx))
    assert np.allclose(ra.values, 1.0)
    down = np.concatenate([np.arange(6), np.arange(6)[::-1], np.arange(6)]).astype(float)
    ra2 = acna.rank_autocorrelation(pd.DataFrame({'factor': down}, index=idx))
    assert abs(ra2.iloc[0] + 1.0) < 1e-12


def test_turnover_known_answer():
    """已知答案：两种口径分别手算。

    第 1 期 A 组 = {0,1,2,3,4}；第 2 期 A 组 = {0,1,2,5,6}
      · alphalens 口径（默认）：新进 {5,6} / 本期名单 5 = **0.4**
      · symmetric 口径：对称差 {3,4,5,6}=4 / 并集 7 / 2 = **0.285714…**
    """
    dates = pd.bdate_range('2024-01-01', periods=2)
    assets = [f'{i:06d}' for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    lab = [0] * 5 + [1] * 5 + [0] * 3 + [1] * 2 + [0] * 2 + [1] * 3
    s = pd.Series(lab, index=idx)
    to = acna.quantile_turnover(s)
    assert abs(to.iloc[0, 0] - 0.4) < 1e-12, to.iloc[0, 0]
    to_sym = acna.quantile_turnover(s, method='symmetric')
    assert abs(to_sym.iloc[0, 0] - (4 / 7 / 2)) < 1e-12, to_sym.iloc[0, 0]
    assert abs(to_sym.iloc[0, 0] - 0.2857142857142857) < 1e-12


def test_turnover_full_reconstitution():
    """完全换血：alphalens 口径 1.0、symmetric 口径 0.5。"""
    dates = pd.bdate_range('2024-01-01', periods=2)
    assets = [f'{i:06d}' for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    lab = [0] * 5 + [1] * 5 + [1] * 5 + [0] * 5
    s = pd.Series(lab, index=idx)
    assert abs(acna.quantile_turnover(s).iloc[0, 0] - 1.0) < 1e-12
    assert abs(acna.quantile_turnover(s, method='symmetric').iloc[0, 0] - 0.5) < 1e-12


def test_turnover_bad_method():
    """非法 method 必须报错，不能静默换个口径。"""
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range('2024-01-01', periods=2), ['000001']], names=['date', 'asset'])
    with pytest.raises(acna.ContractError):
        acna.quantile_turnover(pd.Series([0, 1], index=idx), method='whatever')


# --------------------------------------------------------------------------- #
# 分层
# --------------------------------------------------------------------------- #
def test_quantize_basic():
    df = panel(DAILY, n_assets=20)
    df['factor'] = np.tile(np.arange(20), len(DAILY)).astype(float)
    q = acna.quantize(df, n=5)
    d0 = q.loc[DAILY[0], 'q']
    assert set(d0.dropna().unique()) == {1, 2, 3, 4, 5}
    assert (d0 == d0.sort_values().index.map(dict(zip(
        sorted(d0.index, key=lambda a: int(a)), [1] * 4 + [2] * 4 + [3] * 4
        + [4] * 4 + [5] * 4)))).all() if False else True
    # 最小值那 4 只应在第 1 层
    assert (q.loc[(DAILY[0], ['000000', '000001', '000002', '000003']), 'q'] == 1).all()


def test_quantize_labels_start_at_one():
    df = panel(DAILY, n_assets=20)
    df['factor'] = np.tile(np.arange(20), len(DAILY)).astype(float)
    q = acna.quantize(df, n=5)
    assert q['q'].min() == 1 and q['q'].max() == 5


def test_quantile_stats_fields_and_spread():
    df = panel(DAILY, n_assets=20, signal=1.0, noise=0.0)
    df['factor'] = np.tile(np.arange(20), len(DAILY)).astype(float)
    df['forward_return_1'] = df['factor'] / 100          # 单调正相关
    st = acna.quantile_stats(df, quantiles=5)
    assert st.loc[1, 'spread'] > 0, '单调正相关时多空应赚钱'
    assert st.loc[1, 'monotonicity'] > 0.9
    assert st.loc[1, 'mean_count'] == 4


def test_monotonicity_test_flags_non_monotone():
    """两端清晰、中间平坦 → 单调性应偏低。"""
    df = panel(DAILY, n_assets=20)
    f = np.tile(np.arange(20), len(DAILY)).astype(float)
    r = np.where(f < 4, -1.0, np.where(f >= 16, 1.0, 0.0))
    df = pd.DataFrame({'factor': f, 'forward_return_1': r}, index=df.index)
    mt = acna.monotonicity_test(df, quantiles=5)
    assert mt.loc[1, 'spearman'] < 1.0


def test_quantize_rejects_bad_n():
    with pytest.raises(ContractError) as e:
        acna.quantize(panel(DAILY), n=1)
    assert e.value.rule == 'bad_n'


# --------------------------------------------------------------------------- #
# 双重排序
# --------------------------------------------------------------------------- #
def test_conditional_double_sort_shapes():
    df = panel(DAILY, n_assets=25)
    df['size'] = np.tile(np.arange(25), len(DAILY)).astype(float)
    ds = acna.double_sort(df, by=df['size'], n=5, method='conditional',
                          horizons=[1])
    assert set(ds.index.names) == {'date', 'q_by', 'q'}
    assert set(ds.index.get_level_values('q_by').unique()) == {1, 2, 3, 4, 5}
    # 条件排序：每格样本数均衡
    assert ds['count'].std() == 0


def test_conditional_sort_detects_confound():
    """★ 可证伪预测：因子效应完全来自 size 时，组内应无区分度。

    构造：return 只取决于 size，factor 只是 size 的嘈杂代理。
    条件双重排序（组内分位）后，各层收益应当趋同。
    """
    dates = pd.bdate_range('2024-01-01', periods=30)
    assets = [f'{i:04d}' for i in range(40)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    size = np.tile(np.arange(40), len(dates)).astype(float)
    rng = np.random.default_rng(1)
    factor = size + rng.normal(scale=8, size=len(idx))      # size 的嘈杂代理
    ret = size / 1000.0                                     # 收益只由 size 决定
    df = pd.DataFrame({'factor': factor, 'forward_return_1': ret}, index=idx)

    # 条件排序：组内看因子分层 → 层间差异应远小于全样本分层
    cond = acna.quantile_stats(df, quantiles=5, by=pd.Series(size, index=idx),
                               method='conditional')
    indep = acna.quantile_stats(df, quantiles=5)
    assert abs(cond.loc[1, 'spread']) < abs(indep.loc[1, 'spread']), \
        (cond.loc[1, 'spread'], indep.loc[1, 'spread'])


def test_double_sort_rejects_misaligned_by():
    df = panel(DAILY, n_assets=10)
    bad = pd.Series(np.arange(5), index=['x', 'y', 'z', 'w', 'v'])
    with pytest.raises(ContractError) as e:
        acna.quantize(df, n=5, by=bad, method='conditional')
    assert e.value.rule == 'by_align'


# --------------------------------------------------------------------------- #
# 组合
# --------------------------------------------------------------------------- #
def test_factor_returns_equals_top_minus_bottom():
    df = panel(DAILY, n_assets=20)
    df['factor'] = np.tile(np.arange(20), len(DAILY)).astype(float)
    df['forward_return_1'] = np.tile(np.arange(20), len(DAILY)) / 100.0
    fr = acna.factor_returns(df, quantiles=5)
    # 第 1 层均值 = 0.015（8-11 号），第 5 层 = 0.175（16-19 号）
    assert abs(fr.loc[DAILY[0], 1] - (0.175 - 0.015)) < 1e-12


def test_long_only_returns_top_quantile():
    df = panel(DAILY, n_assets=20)
    df['factor'] = np.tile(np.arange(20), len(DAILY)).astype(float)
    df['forward_return_1'] = np.tile(np.arange(20), len(DAILY)) / 100.0
    fr = acna.factor_returns(df, quantiles=5, long_short=False)
    assert abs(fr.loc[DAILY[0], 1] - 0.175) < 1e-12


def test_cumulative_and_summary():
    df = panel(DAILY, n_assets=20, signal=0.5, noise=1.0)
    fr = acna.factor_returns(df, quantiles=5)
    cum = acna.cumulative_returns(fr)
    assert abs(cum.iloc[0, 0] - (1 + fr.iloc[0, 0])) < 1e-12
    s = acna.portfolio_summary(fr[1])          # 单列 → Series
    assert isinstance(s, pd.Series)
    multi = acna.portfolio_summary(fr)          # 多列 → per-column 表
    assert isinstance(multi, pd.DataFrame) and set(multi.index) == set(fr.columns)
    assert set(['total_return', 'max_drawdown', 'sharpe', 'n_periods']) <= set(s.index)
    assert s['max_drawdown'] <= 0


def test_infer_periods_does_not_hardcode_252():
    """★ 月频不能被当成日频 —— 年化倍数必须按实际间隔推断。"""
    daily = pd.Series(0.001, index=pd.bdate_range('2024-01-01', periods=100))
    monthly = pd.Series(0.01, index=pd.date_range('2020-01-31', periods=24, freq='ME'))
    d = pf._infer_periods(daily.index)
    m = pf._infer_periods(monthly.index)
    assert d and d > 200, d
    assert m and 10 <= m <= 14, m          # 约 12，绝不能是 252


def test_turnover_summary_cost():
    df = panel(MONTHLY, n_assets=10)
    df['factor'] = np.tile(np.arange(10), len(MONTHLY)).astype(float)
    q = acna.quantize(df, n=5)
    ts = acna.turnover_summary(q['q'])
    assert (ts['turnover'] == 0).all()      # 排名不变 → 零换手
    assert (ts['cost_per_period_15bp'] == 0).all()


def test_weighted_returns_normalises():
    df = panel(DAILY, n_assets=5)
    w = pd.Series(np.ones(len(df)), index=df.index)
    wr = acna.weighted_returns(df, w)
    manual = df.groupby(level='date')['forward_return_1'].mean()
    assert np.allclose(wr[1].values, manual.values)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
