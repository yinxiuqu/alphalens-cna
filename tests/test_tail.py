"""尾部统计测试 —— 防线 5（已知答案）+ 模块的存在理由。

**最重要的一个测试是 `test_crash_spread_sees_what_mean_cannot`**：
构造一份"两组均值相同、但低分组左尾厚得多"的数据 ——
均值差 ≈ 0、IC ≈ 0，可崩盘命中率差必须显著为负。
如果这个测试不过，这个模块就没有存在价值。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna  # noqa: E402

R = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 1.0])


# --------------------------------------------------------------------------- #
# 已知答案
# --------------------------------------------------------------------------- #
def test_var_is_order_statistic_not_interpolated():
    """★ 不插值：10 个样本 level=5% 与 10% 都取第 1 小 = −0.5。

    若用 np.quantile 线性插值会得到 −0.455（落在两个样本中间），
    导致"最差 5%"和"最差 10%"选出同样多的样本 —— 尾部比例与 level 脱钩。
    """
    assert acna.value_at_risk(R, 0.05) == -0.5
    assert acna.value_at_risk(R, 0.10) == -0.5
    assert acna.value_at_risk(R, 0.20) == -0.4


def test_cvar_is_tail_mean():
    assert abs(acna.cvar(R, 0.20) - (-0.45)) < 1e-12
    assert abs(acna.cvar(R, 0.50) - (-0.30)) < 1e-12
    assert abs(acna.cvar(R, 0.10) - (-0.5)) < 1e-12


def test_expected_shortfall_alias():
    assert acna.expected_shortfall(R, 0.2) == acna.cvar(R, 0.2)


def test_cvar_never_above_var():
    """CVaR 一定 ≤ VaR（尾部均值不会比门槛更靠右）。"""
    for lv in (0.05, 0.1, 0.2, 0.4):
        assert acna.cvar(R, lv) <= acna.value_at_risk(R, lv) + 1e-12


def test_tail_fraction_matches_level():
    """尾部比例 = ⌈level·n⌉/n ≥ level —— 与 level 一致，不脱钩。"""
    for n in (7, 10, 37, 100):
        x = np.random.default_rng(n).normal(size=n)
        for lv in (0.01, 0.05, 0.1, 0.25):
            st = acna.crash_stats(x, level=lv)
            k = max(1, int(np.ceil(lv * n)))
            assert st['n_crash'] == k
            assert st['hit_rate'] >= lv - 1e-12


# --------------------------------------------------------------------------- #
# 崩盘统计
# --------------------------------------------------------------------------- #
def test_crash_stats_quantile_mode_self_consistent():
    st = acna.crash_stats(R, level=0.2)
    assert st['n'] == 10 and st['n_crash'] == 2
    assert st['hit_rate'] == 0.2
    assert abs(st['mean_crash'] - acna.cvar(R, 0.2)) < 1e-12   # 同一个尾部
    assert st['threshold_mode'] == 'quantile'


def test_crash_stats_absolute_threshold():
    """绝对门槛：一年腰斩（−50%）。"""
    st = acna.crash_stats(R, threshold=-0.5)
    assert st['threshold_mode'] == 'absolute'
    assert st['n_crash'] == 1 and abs(st['mean_crash'] + 0.5) < 1e-12


def test_crash_stats_no_crash():
    st = acna.crash_stats([0.1, 0.2, 0.3], threshold=-0.5)
    assert st['n_crash'] == 0 and np.isnan(st['mean_crash'])
    assert st['hit_rate'] == 0.0


def test_crash_stats_empty():
    st = acna.crash_stats([np.nan, np.nan])
    assert st['n'] == 0 and np.isnan(st['hit_rate'])


def test_crash_stats_ignores_nan():
    st = acna.crash_stats([-0.5, -0.4, np.nan, 0.1], level=0.5)
    assert st['n'] == 3


# --------------------------------------------------------------------------- #
# 其它尾部件
# --------------------------------------------------------------------------- #
def test_downside_deviation_only_counts_losses():
    """★ 右尾极端值不该影响下行波动 —— 这是它和总波动率的区别。"""
    base = [-0.1, -0.05, 0.02, 0.03, 0.05]
    a = acna.downside_deviation(base)
    b = acna.downside_deviation(base + [10.0])       # 加一个 +1000%
    assert abs(a - b) < 1e-12
    # 但总波动会被它拉爆
    assert np.std(base + [10.0]) > 10 * np.std(base)


def test_tail_ratio_sign():
    good = [0.5, 0.1, 0.0, -0.01, -0.02]
    bad = [0.02, 0.01, 0.0, -0.1, -0.5]
    assert acna.tail_ratio(good, 0.2) > 1
    assert acna.tail_ratio(bad, 0.2) < 1


def test_bad_level_raises():
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(acna.ContractError):
            acna.cvar(R, bad)
        with pytest.raises(acna.ContractError):
            acna.value_at_risk(R, bad)


# --------------------------------------------------------------------------- #
# 分位 × 持有期
# --------------------------------------------------------------------------- #
def mk_clean(factor, ret, q):
    idx = factor.index
    return pd.DataFrame({'factor': factor.values, 'factor_quantile': q,
                         'forward_return_21': ret}, index=idx)


def test_tail_by_quantile_shape():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2024-01-01', periods=10)
    assets = [f'{i:06d}' for i in range(50)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    fv = pd.Series(rng.normal(size=len(idx)), index=idx)
    ret = rng.normal(0, 0.2, len(idx))
    q = pd.Series(pd.qcut(fv.groupby(level='date').rank(method='first'), 5,
                          labels=[1, 2, 3, 4, 5]).astype(int).values, index=idx)
    df = pd.DataFrame({'factor': fv.values, 'factor_quantile': q.values,
                       'forward_return_21': ret}, index=idx)
    out = acna.tail_by_quantile(df, level=0.2)
    assert out.index.names == ['h', 'q']
    assert set(out['q'] if 'q' in out else out.index.get_level_values('q')) == {1, 2, 3, 4, 5}
    for c in ('mean', 'cvar', 'hit_rate', 'n_crash', 'n', 'threshold'):
        assert c in out.columns
    assert (out['hit_rate'] <= 0.25).all()


def test_tail_by_quantile_missing_quantile_raises():
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range('2024-01-01', periods=2), ['000001']],
        names=['date', 'asset'])
    df = pd.DataFrame({'forward_return_21': [0.1, 0.2]}, index=idx)
    with pytest.raises(acna.ContractError):
        acna.tail_by_quantile(df)


# --------------------------------------------------------------------------- #
# ★ 模块的存在理由
# --------------------------------------------------------------------------- #
def test_crash_spread_sees_what_mean_cannot():
    """★★ 本模块的存在理由：**两组均值完全相同，但低分组崩得频繁得多**。

    构造（每期 100 只）：
      · Q1（20 只）：40% 概率 −60%，60% 概率 +40%  → 均值 = 0
      · Q5（20 只）： 5% 概率 −15%，95% 概率 +0.79% → 均值 = 0
      · 中间 60 只：N(0, 0.01)                   → 均值 ≈ 0
    **三组均值都是 0 → 均值差 ≈ 0、IC ≈ 0**，但 Q1 的崩盘命中率是 Q5 的 8 倍。
    """
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2024-01-01', periods=60)
    assets = [f'{i:06d}' for i in range(100)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    n = len(idx)
    pos = idx.get_level_values('asset').str[-3:].astype(int).to_numpy()
    is_lo, is_hi = pos < 20, pos >= 80
    fv = np.where(is_lo, -1.0, np.where(is_hi, 1.0, 0.0))

    ret = rng.normal(0, 0.01, n)                       # 中间组
    draw = rng.random(n)
    ret[is_lo] = np.where(draw[is_lo] < 0.40, -0.60, +0.40)
    ret[is_hi] = np.where(draw[is_hi] < 0.05, -0.15, +0.05 * 0.15 / 0.95)
    df = pd.DataFrame({'factor': fv,
                       'factor_quantile': np.where(is_lo, 1,
                                                   np.where(is_hi, 5, 3)),
                       'forward_return_21': ret}, index=idx)

    # ── 绝对门槛：语义最清楚 ────────────────────────────────────────
    sp = acna.crash_spread(df, threshold=-0.40)
    row = sp.loc[21]
    assert abs(row['mean_spread']) < 0.05, f"均值差应≈0，实际 {row['mean_spread']}"
    assert row['hit_lo'] > 0.30, row['hit_lo']       # Q1 崩盘率 ≈40%
    assert row['hit_hi'] < 0.05, row['hit_hi']       # Q5 崩盘率 ≈0%
    assert row['hit_spread'] < -0.30, row['hit_spread']
    assert row['t_hit'] < -10, row['t_hit']          # 极其显著

    # ── 分位门槛（默认）：当日截面最差 10% ──────────────────────────
    sp2 = acna.crash_spread(df, level=0.10)
    row2 = sp2.loc[21]
    assert row2['hit_lo'] > row2['hit_hi'] + 0.15, (row2['hit_lo'], row2['hit_hi'])
    assert row2['hit_spread'] < -0.15, row2['hit_spread']

    # ── 尾部均值同向可见 ────────────────────────────────────────────
    tb = acna.tail_by_quantile(df, level=0.10)
    assert tb.loc[(21, 1), 'cvar'] < tb.loc[(21, 5), 'cvar'] - 0.2


def test_crash_spread_symbol_convention():
    """符号直觉：低分位更容易崩 → hit_spread（QN−Q1）为**负**。"""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range('2024-01-01', periods=20)
    assets = [f'{i:06d}' for i in range(40)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    pos = idx.get_level_values('asset').str[-3:].astype(int).to_numpy()
    lo = pos < 10
    ret = rng.normal(0, 0.01, len(idx))
    ret[lo] = np.where(rng.random(lo.sum()) < 0.5, -0.5, 0.5)
    df = pd.DataFrame({'factor': np.where(lo, -1.0, 1.0),
                       'factor_quantile': np.where(lo, 1, 5),
                       'forward_return_21': ret}, index=idx)
    row = acna.crash_spread(df, threshold=-0.3).loc[21]
    assert row['hit_spread'] < 0 and row['t_hit'] < 0
