"""截面回归 / Fama-MacBeth 测试 —— 防线 5（已知答案）。

回归最容易"看起来对"：给出一堆系数，谁也不知道对不对。
所以这里全部用**能手算**的数据：真系数已知、OLS 与 numpy 对拍、
不足样本的日子必须是 NaN 而不是 0。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna  # noqa: E402
from alphalens_cna.inference.newey_west import (  # noqa: E402
    auto_lags, nw_tstat, nw_variance, variance_inflation,
)


def panel(dates, assets, seed=0, n_x=2):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    X = pd.DataFrame({f'f{i + 1}': rng.normal(size=len(idx)) for i in range(n_x)},
                     index=idx)
    return X, rng, idx


# --------------------------------------------------------------------------- #
# 截面回归
# --------------------------------------------------------------------------- #
def test_ols_recovers_exact_line():
    """★ 完全共线的三点：y = 0 + 2x，截距 0、斜率 2，必须精确复原。"""
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp('2024-01-01'), c) for c in ('a', 'b', 'c')],
        names=['date', 'asset'])
    X = pd.DataFrame({'x': [1.0, 2.0, 3.0]}, index=idx)
    y = pd.Series([2.0, 4.0, 6.0], index=idx)
    coef = acna.cross_sectional_regression(y, X, min_obs=3)
    assert coef.shape == (1, 2)
    assert abs(coef.iloc[0]['const'] - 0.0) < 1e-12
    assert abs(coef.iloc[0]['x'] - 2.0) < 1e-12


def test_ols_matches_numpy_lstsq():
    """逐期系数与 numpy.linalg.lstsq 逐位一致。"""
    dates = pd.bdate_range('2024-01-01', periods=5)
    assets = [f'{i:06d}' for i in range(30)]
    X, rng, idx = panel(dates, assets, seed=1)
    y = pd.Series(0.5 * X['f1'].values - 2.0 * X['f2'].values
                  + rng.normal(0, 0.1, len(idx)), index=idx)
    ours = acna.cross_sectional_regression(y, X)
    A = np.column_stack([np.ones(30), X.iloc[:30].values])
    beta, *_ = np.linalg.lstsq(A, y.iloc[:30].values, rcond=None)
    assert np.abs(ours.iloc[0].values - beta).max() == 0.0


def test_ols_thin_cross_section_is_nan_not_zero():
    """★ 样本不足的日期必须是 NaN —— 造一个 0 出来就是伪造数据。"""
    dates = pd.bdate_range('2024-01-01', periods=3)
    assets = [f'{i:06d}' for i in range(20)]
    X, rng, idx = panel(dates, assets, seed=2)
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    # 第 2 天只留 2 个观测
    d = idx.get_level_values('date')
    keep = ~((d == dates[1]) & (idx.get_level_values('asset') > '000001'))
    X2 = X[keep]
    y2 = y[keep]
    coef = acna.cross_sectional_regression(y2, X2, min_obs=10)
    assert len(coef) == 3
    assert coef.iloc[1].isna().all(), '样本不足应为 NaN'
    assert coef.iloc[0].notna().all() and coef.iloc[2].notna().all()


def test_ols_no_constant():
    dates = pd.bdate_range('2024-01-01', periods=2)
    assets = [f'{i:06d}' for i in range(20)]
    X, rng, idx = panel(dates, assets, seed=3, n_x=1)
    y = pd.Series(3.0 * X['f1'].values, index=idx)
    coef = acna.cross_sectional_regression(y, X, add_constant=False)
    assert list(coef.columns) == ['f1']
    assert np.allclose(coef['f1'].values, 3.0)


def test_ols_index_validation():
    """索引不对要报错，不能猜。"""
    X = pd.DataFrame({'x': [1.0, 2.0]},
                     index=pd.Index([1, 2], name='date'))
    y = pd.Series([1.0, 2.0], index=X.index)
    with pytest.raises(acna.ContractError):
        acna.cross_sectional_regression(y, X)


def test_ols_index_mismatch_raises():
    dates = pd.bdate_range('2024-01-01', periods=2)
    assets = [f'{i:06d}' for i in range(5)]
    X, rng, idx = panel(dates, assets, seed=4, n_x=1)
    y = pd.Series(rng.normal(size=len(idx)), index=idx)[:-1]
    with pytest.raises(acna.ContractError):
        acna.cross_sectional_regression(y, X)


# --------------------------------------------------------------------------- #
# Fama-MacBeth
# --------------------------------------------------------------------------- #
def test_fama_macbeth_recovers_true_premium():
    """★ 真 λ = (0.02, −0.01)，FM 两步法必须复原出来。"""
    dates = pd.bdate_range('2024-01-01', periods=40)
    assets = [f'{i:06d}' for i in range(50)]
    X, rng, idx = panel(dates, assets, seed=5)
    y = pd.Series(0.02 * X['f1'].values - 0.01 * X['f2'].values
                  + rng.normal(0, 0.01, len(idx)), index=idx)
    res = acna.fama_macbeth(y, X, horizon=1)
    assert abs(res.summary.loc['f1', 'mean'] - 0.02) < 2e-3
    assert abs(res.summary.loc['f2', 'mean'] + 0.01) < 2e-3
    assert int(res.summary.loc['f1', 'n_periods']) == 40
    assert res.coef.shape == (40, 3)


def test_fama_macbeth_reports_both_t_stats():
    """★ summary 必须同时给朴素 t 与 NW t —— 让人看见重叠观测吹大了多少。"""
    dates = pd.bdate_range('2024-01-01', periods=60)
    assets = [f'{i:06d}' for i in range(50)]
    X, rng, idx = panel(dates, assets, seed=6, n_x=1)
    y = pd.Series(0.02 * X['f1'].values, index=idx)
    res = acna.fama_macbeth(y, X, horizon=1)
    for c in ('t_naive', 't_nw', 'n_eff', 'vif', 'lags'):
        assert c in res.summary.columns
    # 完美拟合 → 残差 0 → λ 无波动 → 两个 t 都应该发散或 NaN，但不许是错的数
    assert np.isfinite(res.summary.loc['f1', 'mean'])


def test_fama_macbeth_nw_shrinks_t_under_overlap():
    """★ 重叠观测（正自相关）下 NW 必须把 t 压下来，不是抬上去。"""
    rng = np.random.default_rng(7)
    T = 200
    # 造一个强正自相关的 λ 序列（模拟 h=21 重叠）
    e = rng.normal(size=T)
    lam = np.zeros(T)
    for t in range(1, T):
        lam[t] = 0.9 * lam[t - 1] + e[t]
    lam = 0.01 + 0.001 * lam
    st = nw_tstat(lam, lags=20, horizon=21)
    assert abs(st['t_nw']) < abs(st['t_naive']), (st['t_naive'], st['t_nw'])
    assert st['vif'] > 1


def test_fm_str_warns_about_naive_t():
    dates = pd.bdate_range('2024-01-01', periods=30)
    assets = [f'{i:06d}' for i in range(30)]
    X, rng, idx = panel(dates, assets, seed=8, n_x=1)
    y = pd.Series(0.01 * X['f1'].values + rng.normal(0, 0.05, len(idx)),
                  index=idx)
    txt = str(acna.fama_macbeth(y, X, horizon=1))
    assert 'NW t' in txt and 't_naive' in txt
    assert '只用 `t_nw` 下结论' in txt


def test_fm_bad_horizon():
    dates = pd.bdate_range('2024-01-01', periods=3)
    assets = [f'{i:06d}' for i in range(30)]
    X, rng, idx = panel(dates, assets, seed=9, n_x=1)
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    with pytest.raises(acna.ContractError):
        acna.fama_macbeth(y, X, horizon=0)


def test_fm_empty_when_no_valid_section():
    """一期都凑不齐 → 不许崩，也不许编数。"""
    dates = pd.bdate_range('2024-01-01', periods=3)
    assets = [f'{i:06d}' for i in range(3)]
    X, rng, idx = panel(dates, assets, seed=10, n_x=2)
    y = pd.Series(rng.normal(size=len(idx)), index=idx)
    res = acna.fama_macbeth(y, X, min_obs=10)
    assert res.summary['n_periods'].max() == 0
    assert '无有效截面' in str(res)


# --------------------------------------------------------------------------- #
# Shanken 修正
# --------------------------------------------------------------------------- #
def test_shanken_inflation_zero_premium_is_one():
    """λ ≈ 0 时膨胀因子 ≈ 1（无修正）—— 这是它唯一能精确验证的性质。"""
    S = np.eye(2)
    assert abs(acna.shanken_inflation([0.0, 0.0], S) - 1.0) < 1e-12
    assert acna.shanken_inflation([1.0, 1.0], S) > 1.0


def test_shanken_inflation_shape_check():
    with pytest.raises(acna.ContractError):
        acna.shanken_inflation([1.0, 1.0, 1.0], np.eye(2))
    with pytest.raises(acna.ContractError):
        acna.shanken_inflation([1.0], np.ones((2, 3)))


# --------------------------------------------------------------------------- #
# NW 的保守下限（本库的立场）
# --------------------------------------------------------------------------- #
def test_vif_never_below_one():
    """★ NW **永不**比朴素方差更小。

    原因：减样本均值使 E[γ̂_j] = −σ²/T，短样本上 NW 会系统性**低估**方差
    （实测 T=30 iid：均值 vif≈0.90，即少校正 10%），而低估方差 = 高估显著性。
    本库直接把这个方向封死：vif ≥ 1。
    """
    for seed in range(50):
        x = np.random.default_rng(seed).normal(size=30)
        assert variance_inflation(x, lags=auto_lags(30, 1)) >= 1.0 - 1e-12
    for T in (20, 60, 120):
        for seed in range(20):
            x = np.random.default_rng(seed + 500).normal(size=T)
            assert variance_inflation(x, lags=auto_lags(T, 1)) >= 1.0 - 1e-12


def test_lags_zero_is_exactly_naive():
    """lags=0 → t_nw 与 t_naive 精确相等（回归测试，防分母再走样）。"""
    x = np.random.default_rng(1).normal(size=50)
    st = nw_tstat(x, lags=0)
    assert st['vif'] == 1.0
    assert st['t_nw'] == st['t_naive']


def test_nw_variance_positive_autocorr_inflates():
    """正自相关（重叠观测）→ vif > 1，方向不能反。"""
    rng = np.random.default_rng(2)
    e = rng.normal(size=500)
    x = np.zeros(500)
    for t in range(1, 500):
        x[t] = 0.8 * x[t - 1] + e[t]
    assert variance_inflation(x, lags=10) > 1.5
    assert nw_variance(x, lags=10) > nw_variance(x, lags=0)


def test_reported_lags_are_clamped():
    """★ 报告的 lags 必须与实际计算一致 —— 不许出现"32 期报 125 阶滞后"。"""
    x = np.random.default_rng(0).normal(size=32)
    st = nw_tstat(x, lags=125, horizon=126)
    assert st['lags'] <= 30, st['lags']
    assert st['lags'] == 30
    # 不指定时，持有期再长也不能超过 T−2
    st2 = nw_tstat(x, horizon=126)
    assert st2['lags'] <= 30
