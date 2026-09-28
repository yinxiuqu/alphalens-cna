"""因子相关性 / 冗余度测试 —— 对拍 + 已知答案 + 记账。

最容易被写错的四处，本文件各钉一条：

1. **池化 vs 逐期** —— 池化会把"两因子共同的时间漂移"算成相关。
   实测（本文件 ``test_no_pooling_drift_trap`` 的构造）：
   **池化 0.9685，逐期均值 0.0214**。若实现退化成池化，这条测试必红。
2. **逐位对拍** —— 手工 ``groupby(level='date').corr()`` 与
   ``factor_correlation`` 的结果对到 1e-12（两条独立路径）。
3. **重叠不足 = NaN + 原因**，且**不销毁原始逐期值**（不静默、不硬凑）。
4. **冗余判定用绝对值** —— ρ = −1 同样冗余。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphalens_cna.analysis.correlation import (  # noqa: E402
    FactorCorrResult,
    RedundancyResult,
    factor_correlation,
    redundancy_check,
)
from alphalens_cna.contract.errors import ContractError  # noqa: E402
from alphalens_cna.contract.panels import FactorPanel  # noqa: E402

DATES = pd.bdate_range('2024-01-01', periods=30)
ASSETS = [f'{i:06d}' for i in range(40)]


# --------------------------------------------------------------------------- #
def make_factors(dates=DATES, n_assets=40, seed=0):
    """三个固定的合成因子（40 只 × 30 期）。

    * ``ROE``   —— 基准
    * ``PB``    —— 与 ROE 高度相关（同一份基础信号 + 小噪声）
    * ``MOM``   —— 与两者都无关（独立噪声）
    """
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(dates), [f'{i:06d}' for i in range(n_assets)]],
        names=['date', 'asset'])
    base = rng.normal(size=len(idx))
    return {
        'ROE': pd.Series(base + rng.normal(size=len(idx)), index=idx),
        'PB': pd.Series(base + 0.1 * rng.normal(size=len(idx)), index=idx),
        'MOM': pd.Series(rng.normal(size=len(idx)), index=idx),
    }


def wide_of(factors):
    return pd.DataFrame({k: v for k, v in factors.items()})


# --------------------------------------------------------------------------- #
# 1. 逐位对拍（手工 pandas）
# --------------------------------------------------------------------------- #
def test_matches_manual_pandas_bitwise():
    """★ 手工 ``groupby(level='date').corr('spearman')`` 与我们的结果 1e-12 对上。"""
    f = make_factors()
    df = wide_of(f)
    res = factor_correlation(f, min_overlap=20)

    man = df.groupby(level='date').corr(method='spearman')
    for a, b in (('ROE', 'PB'), ('ROE', 'MOM'), ('PB', 'MOM')):
        s = man.loc[(slice(None), a), b].dropna()
        assert abs(float(res.matrix.loc[a, b]) - float(s.mean())) < 1e-12, (a, b)
        assert abs(float(res.median.loc[a, b]) - float(s.median())) < 1e-12
        assert abs(float(res.positive_rate.loc[a, b])
                   - float((s > 0).mean())) < 1e-12
        assert int(res.n_periods.loc[a, b]) == len(s) == 30
        # 对称性：矩阵必须两面同值
        assert res.matrix.loc[a, b] == res.matrix.loc[b, a]


def test_matches_manual_pairwise_loop_bitwise():
    """★ 第二条独立路径：逐对取共同样本、逐期 ``Series.corr``，同样对到 1e-12。

    这条与上一条查的**不是同一件事**：上一条是"聚合对不对"，
    这条是"逐期截面的样本选择和秩相关算法对不对"。
    """
    f = make_factors(seed=5)
    df = wide_of(f)
    res = factor_correlation(f)
    by_date = df.groupby(level='date')
    for a, b in (('ROE', 'PB'), ('MOM', 'PB')):
        manual = {}
        for d, sub in by_date:
            m = sub[a].notna() & sub[b].notna()
            manual[d] = sub.loc[m, a].corr(sub.loc[m, b], method='spearman')
        s = pd.Series(manual).dropna()
        assert abs(float(res.matrix.loc[a, b]) - float(s.mean())) < 1e-12
        assert int(res.n_periods.loc[a, b]) == len(s)


def test_identical_factors_give_exact_one():
    """完全相同的两个因子 → 每期 ρ = 1，矩阵逐位等于 1。"""
    f = make_factors()
    res = factor_correlation({'A': f['ROE'], 'B': f['ROE'], 'C': f['MOM']})
    assert res.matrix.loc['A', 'B'] == 1.0
    assert (res.per_period[('A', 'B')] == 1.0).all()
    assert res.positive_rate.loc['A', 'B'] == 1.0


# --------------------------------------------------------------------------- #
# 2. 绝不池化
# --------------------------------------------------------------------------- #
def test_no_pooling_drift_trap():
    """★ 共同时间漂移的陷阱：池化 0.97，逐期均值 ≈ 0 —— 实现必须是后者。

    构造：两个因子都由**同一个逐期漂移**驱动（漂移压倒一切），
    但每期截面内两者毫无关系。池化会把漂移读成"高度相关"。
    """
    rng = np.random.default_rng(7)
    n_d, n_a = 60, 40
    idx = pd.MultiIndex.from_product([pd.bdate_range('2024-01-01', periods=n_d),
                                      [f'{i:03d}' for i in range(n_a)]],
                                     names=['date', 'asset'])
    drift = np.repeat(rng.normal(scale=3.0, size=n_d), n_a)
    x = pd.Series(drift + rng.normal(scale=0.5, size=len(idx)), index=idx)
    y = pd.Series(drift * 0.95 + rng.normal(scale=0.5, size=len(idx)), index=idx)

    pooled = float(pd.DataFrame({'x': x, 'y': y})
                   .corr(method='spearman').loc['x', 'y'])
    res = factor_correlation({'X': x, 'Y': y})

    assert pooled > 0.9, f'构造失败：池化相关应很高，实际 {pooled}'
    assert abs(float(res.matrix.loc['X', 'Y'])) < 0.15, (
        f'逐期均值应接近 0（漂移不该被读成相关），实际 {res.matrix.loc["X", "Y"]}')
    # 逐期为正的比例也应当接近"抛硬币"
    assert 0.3 < float(res.positive_rate.loc['X', 'Y']) < 0.7


# --------------------------------------------------------------------------- #
# 3. 冗余判定：已知答案
# --------------------------------------------------------------------------- #
def test_identical_candidate_is_redundant():
    """已知答案：候选 = 库里已有的因子 → ρ = 1 → 判为冗余。"""
    f = make_factors()
    cand = f['ROE'].rename('ROE_copy')
    res = redundancy_check(cand, {'ROE': f['ROE'], 'MOM': f['MOM']},
                           threshold=0.7)
    assert isinstance(res, RedundancyResult)
    assert res.redundant is True
    assert res.candidate == 'ROE_copy'
    top = res.worst.iloc[0]
    assert top['factor'] == 'ROE'
    assert float(top['abs_rho']) > 0.999
    assert int(top['n_periods']) == 30
    assert res.reason.startswith('非常冗余')
    assert '|ρ|=1.00' in res.reason and '阈值 0.7' in res.reason
    assert '最相关：ROE' in res.reason and '重叠 30 期' in res.reason
    assert '❌ 冗余' in str(res)


def test_orthogonal_candidate_is_not_redundant():
    """已知答案：正交/无关因子（固定种子，余量充足）→ 非冗余。"""
    f = make_factors(seed=11)
    res = redundancy_check(f['MOM'], {'ROE': f['ROE'], 'PB': f['PB']},
                           threshold=0.7, method='spearman')
    assert res.redundant is False
    assert float(res.worst.iloc[0]['abs_rho']) < 0.3        # 离阈值 0.7 有大余量
    assert res.reason.startswith('不冗余')
    assert '< 阈值 0.7' in res.reason
    assert '✅ 不冗余' in str(res)


def test_threshold_uses_absolute_value():
    """★ 判定用**绝对值**：把因子取反（ρ = −1）依旧是冗余。"""
    f = make_factors()
    res = redundancy_check(-f['ROE'], {'ROE': f['ROE'], 'MOM': f['MOM']},
                           threshold=0.7)
    assert res.redundant is True
    top = res.worst.iloc[0]
    assert float(top['rho']) < -0.999          # 负相关
    assert float(top['abs_rho']) > 0.999       # 但绝对值过阈值
    assert '非常冗余' in res.reason


def test_redundancy_worst_ordering_and_cap():
    """``worst`` 按 |ρ| 降序、最多 5 个（库里有 6 个候选时只给前 5）。"""
    f = make_factors()
    lib = {'PB': f['PB'], 'MOM': f['MOM'], 'N1': f['MOM'] * 3,
           'N2': -f['MOM'], 'N3': f['MOM'] / 2, 'N4': f['PB'] * 2}
    res = redundancy_check(f['ROE'], lib, threshold=0.7)
    assert len(res.worst) == 5                       # 6 个库里因子 → 截到 5
    assert list(res.worst['abs_rho']) == sorted(res.worst['abs_rho'],
                                                reverse=True)
    assert res.worst.iloc[0]['factor'] == 'PB'       # 最相关的是 PB
    assert res.n_library == len(lib) == 6
    # 未被截断的完整明细还在 stats 里（6 行，不丢证据）
    assert len(res.stats) == 6
    assert res.stats['n_periods'].min() == 30


def test_redundancy_candidate_from_factor_panel():
    """契约对象也能当候选（取 ``value`` 列），且名字取不到时记为 candidate。"""
    f = make_factors()
    d = f['ROE']
    panel = FactorPanel(pd.DataFrame(
        {'value': d.to_numpy(), 'available_at': d.index.get_level_values('date')},
        index=d.index))
    res = redundancy_check(panel, {'ROE': f['ROE']}, threshold=0.7)
    assert res.candidate == 'candidate'
    assert res.redundant is True
    assert float(res.worst.iloc[0]['abs_rho']) > 0.999


def test_redundancy_relation_to_factor_correlation():
    """冗余判定与 ``factor_correlation`` 的矩阵**同一个数**（不另算一套）。"""
    f = make_factors()
    res = redundancy_check(f['ROE'], {'PB': f['PB'], 'MOM': f['MOM']})
    r = factor_correlation({'ROE': f['ROE'], 'PB': f['PB'], 'MOM': f['MOM']})
    got = dict(zip(res.worst['factor'], res.worst['rho']))
    assert abs(got['PB'] - float(r.matrix.loc['ROE', 'PB'])) < 1e-15
    assert abs(got['MOM'] - float(r.matrix.loc['ROE', 'MOM'])) < 1e-15


# --------------------------------------------------------------------------- #
# 4. 重叠不足：NaN + 原因（不静默、不硬凑、不销毁证据）
# --------------------------------------------------------------------------- #
def test_min_overlap_insufficient_nan_and_reason():
    """★ 只重叠 3 期、阈值 20 期 → NaN + ``insufficient`` 写明还差几期。"""
    rng = np.random.default_rng(3)
    f = make_factors()
    d2 = DATES[:3]
    idx2 = pd.MultiIndex.from_product([d2, ASSETS], names=['date', 'asset'])
    short = pd.Series(rng.normal(size=len(idx2)), index=idx2).rename('SHORT')

    res = factor_correlation({'A': f['ROE'], 'B': short}, min_overlap=20)
    assert np.isnan(res.matrix.loc['A', 'B'])
    assert np.isnan(res.median.loc['A', 'B'])
    assert np.isnan(res.positive_rate.loc['A', 'B'])
    assert int(res.n_periods.loc['A', 'B']) == 3

    assert len(res.insufficient) == 1
    row = res.insufficient.iloc[0]
    assert int(row['n_periods']) == 3
    assert int(row['need']) == 20
    assert int(row['short']) == 17
    assert 'A' in row['reason'] and 'B' in row['reason']
    assert '3' in row['reason'] and '20' in row['reason'] and '17' in row['reason']
    assert res.insufficient.index.names == ['factor_a', 'factor_b']


def test_insufficient_pair_keeps_raw_per_period_values():
    """被掩码的对**原始逐期值仍在**（不销毁证据），只是矩阵里不给数。"""
    rng = np.random.default_rng(4)
    f = make_factors()
    idx2 = pd.MultiIndex.from_product([DATES[:3], ASSETS], names=['date', 'asset'])
    short = pd.Series(rng.normal(size=len(idx2)), index=idx2).rename('SHORT')
    res = factor_correlation({'A': f['ROE'], 'B': short}, min_overlap=20)

    raw = res.per_period[('A', 'B')]
    assert raw.notna().sum() == 3
    assert np.isfinite(res.stats.loc[('A', 'B'), 'mean'])      # 原值还在
    assert res.stats.loc[('A', 'B'), 'comparable'] == False
    assert int(res.stats.loc[('A', 'B'), 'short']) == 17


def test_min_overlap_boundary_is_inclusive():
    """边界：重叠期数 == min_overlap → 给数（``>=``，不是 ``>``）。"""
    f = make_factors()
    res = factor_correlation(f, min_overlap=30)
    assert not np.isnan(res.matrix.loc['ROE', 'PB'])
    assert len(res.insufficient) == 0
    res2 = factor_correlation(f, min_overlap=31)
    assert np.isnan(res2.matrix.loc['ROE', 'PB'])
    assert len(res2.insufficient) == 3
    # 空表也要**列齐全**（下游会直接按列取用）
    assert list(res.insufficient.columns) == ['n_periods', 'n_overlap', 'need',
                                              'short', 'reason']
    assert res.insufficient.index.names == ['factor_a', 'factor_b']
    assert list(res2.insufficient.columns) == list(res.insufficient.columns)


def test_positive_rate_denominator_and_n_overlap():
    """常数截面：该期算不出相关 → 不计入 ``n_periods``，但计入 ``n_overlap``。

    ⚠️ 口径：``positive_rate`` 的分母是**有值的期数**，不是总期数。
    """
    f = make_factors()
    a = f['ROE'].copy()
    d0 = DATES[0]
    a.loc[d0] = 1.0                       # 第一天是常数截面 → 无相关可言
    res = factor_correlation({'A': a, 'B': f['MOM']})
    assert int(res.n_periods.loc['A', 'B']) == 29
    assert int(res.n_overlap.loc['A', 'B']) == 30
    assert res.n_dates == 30
    s = res.per_period[('A', 'B')].dropna()
    assert abs(float(res.positive_rate.loc['A', 'B'])
               - float((s > 0).mean())) < 1e-15
    # 对角线口径 = 该因子"有值（≥ 3 只）"的期数 → 常数截面那期**仍然算**
    assert int(res.n_periods.loc['A', 'A']) == 30
    # 而非对角线 = 能算出相关的期数 → 常数那期不算
    assert int(res.n_periods.loc['B', 'B']) == 30


# --------------------------------------------------------------------------- #
# 5. 入参形态
# --------------------------------------------------------------------------- #
def test_multi_column_dataframe_equals_mapping():
    """多列 DataFrame 与 Mapping 两种写法结果逐位相同，且保序。"""
    f = make_factors()
    r_map = factor_correlation(f)
    r_df = factor_correlation(wide_of(f))
    assert r_map.factors == r_df.factors == ('ROE', 'PB', 'MOM')
    for a, b in (('ROE', 'PB'), ('ROE', 'MOM'), ('PB', 'MOM')):
        assert r_map.matrix.loc[a, b] == r_df.matrix.loc[a, b]


def test_factor_panel_input_equals_dataframe():
    """契约对象（FactorPanel）与裸 DataFrame 结果逐位相同。"""
    f = make_factors()
    panels = {}
    for k, s in f.items():
        panels[k] = FactorPanel(pd.DataFrame(
            {'value': s.to_numpy(),
             'available_at': s.index.get_level_values('date')}, index=s.index))
    r_panel = factor_correlation(panels)
    r_series = factor_correlation(f)
    assert r_panel.matrix.equals(r_series.matrix) or np.allclose(
        r_panel.matrix.to_numpy(), r_series.matrix.to_numpy(), atol=0)
    assert r_panel.n_periods.equals(r_series.n_periods)


def test_alignment_uses_union_of_dates():
    """两因子期数不同 → 按**并集**对齐，重叠期数如实写进 n_overlap。"""
    rng = np.random.default_rng(9)
    f = make_factors()
    part = pd.Series(rng.normal(size=len(ASSETS) * 10),
                     index=pd.MultiIndex.from_product([DATES[:10], ASSETS],
                                                      names=['date', 'asset']))
    res = factor_correlation({'A': f['ROE'], 'B': part}, min_overlap=10)
    assert res.n_dates == 30
    assert int(res.n_overlap.loc['A', 'B']) == 10
    assert int(res.n_periods.loc['A', 'B']) == 10
    assert not np.isnan(res.matrix.loc['A', 'B'])


# --------------------------------------------------------------------------- #
# 6. 非法输入（一律 ContractError，且能定位）
# --------------------------------------------------------------------------- #
def test_single_factor_is_rejected():
    f = make_factors()
    with pytest.raises(ContractError) as e:
        factor_correlation({'ROE': f['ROE']})
    assert e.value.contract == 'correlation'
    assert e.value.rule == 'too_few'
    assert 'ROE' in str(e.value)


def test_ambiguous_factor_value_is_rejected():
    f = make_factors()
    with pytest.raises(ContractError) as e:
        factor_correlation({'A': f['ROE'], 'B': wide_of(f).iloc[:, :3]})
    assert e.value.rule == 'factor_ambiguous'
    assert 'B' in str(e.value)


def test_non_numeric_factor_is_rejected():
    f = make_factors()
    bad = f['ROE'].copy()
    bad[:] = np.nan
    bad = bad.astype(object)
    bad.iloc[:] = 'x'
    with pytest.raises(ContractError) as e:
        factor_correlation({'A': f['ROE'], 'B': bad})
    assert e.value.rule == 'not_numeric'
    assert 'B' in str(e.value)          # 必须指名道姓


def test_all_nan_factor_is_rejected():
    f = make_factors()
    bad = pd.Series(np.nan, index=f['ROE'].index)
    with pytest.raises(ContractError) as e:
        factor_correlation({'A': f['ROE'], 'B': bad})
    assert e.value.rule == 'all_nan'


def test_duplicate_index_is_rejected():
    f = make_factors()
    bad = pd.concat([f['ROE'], f['ROE'].iloc[:5]])
    with pytest.raises(ContractError) as e:
        factor_correlation({'A': f['ROE'], 'B': bad})
    assert e.value.rule == 'index_unique'


def test_bad_index_is_rejected():
    f = make_factors()
    flat = pd.Series(np.arange(30.0), index=DATES)
    with pytest.raises(ContractError) as e:
        factor_correlation({'A': f['ROE'], 'B': flat})
    assert e.value.rule == 'index'


def test_panel_dataframe_passed_instead_of_mapping():
    """把 FactorPanel.df（value + available_at）当多因子表传 → 明确报错。"""
    d = make_factors()['ROE']
    panel_df = pd.DataFrame({'value': d.to_numpy(),
                             'available_at': d.index.get_level_values('date')},
                            index=d.index)
    with pytest.raises(ContractError) as e:
        factor_correlation(panel_df)
    assert e.value.rule == 'looks_like_panel'
    assert 'value' in str(e.value)


def test_bad_method_and_min_overlap_are_rejected():
    f = make_factors()
    with pytest.raises(ContractError) as e:
        factor_correlation(f, method='kendall')
    assert e.value.rule == 'bad_method'
    with pytest.raises(ContractError) as e:
        factor_correlation(f, min_overlap=0)
    assert e.value.rule == 'bad_min_overlap'


def test_redundancy_bad_inputs():
    f = make_factors()
    with pytest.raises(ContractError) as e:
        redundancy_check(f['ROE'], {})
    assert e.value.rule == 'empty_library'
    with pytest.raises(ContractError) as e:
        redundancy_check(f['ROE'].rename('ROE'), {'ROE': f['ROE']})
    assert e.value.rule == 'name_collision'
    assert 'ROE' in str(e.value)
    with pytest.raises(ContractError) as e:
        redundancy_check(f['ROE'], {'MOM': f['MOM']}, threshold=1.5)
    assert e.value.rule == 'bad_threshold'
    with pytest.raises(ContractError) as e:
        redundancy_check(f['ROE'], {'MOM': f['MOM']}, threshold=np.nan)
    assert e.value.rule == 'bad_threshold'


def test_empty_library_result_when_all_overlap_poor():
    """库里全是对不上的期 → ``redundant=False`` 但 reason 明说"无法判定"。"""
    rng = np.random.default_rng(13)
    f = make_factors()
    idx2 = pd.MultiIndex.from_product([DATES[:3], ASSETS], names=['date', 'asset'])
    short = pd.Series(rng.normal(size=len(idx2)), index=idx2)
    res = redundancy_check(f['ROE'], {'SHORT': short}, min_overlap=20)
    assert res.redundant is False
    assert res.reason.startswith('无法判定')
    assert '重叠不足' in res.reason and '还差' in res.reason
    assert len(res.worst) == 0
    assert len(res.insufficient) == 1
    # 明细表仍然完整（不因为"没结论"就把证据丢掉）
    assert 'n_overlap' in res.stats.columns


# --------------------------------------------------------------------------- #
# 7. 直接导入路径 + 结果对象
# --------------------------------------------------------------------------- #
def test_direct_import_and_result_objects():
    """公开函数能直接按模块路径导入，返回的是带文档的 dataclass。"""
    from alphalens_cna.analysis import correlation as mod
    for nm in ('factor_correlation', 'redundancy_check'):
        assert callable(getattr(mod, nm))
        assert nm in mod.__all__
    f = make_factors()
    res = factor_correlation(f)
    assert isinstance(res, FactorCorrResult)
    assert isinstance(res.matrix, pd.DataFrame)
    assert res.matrix.shape == (3, 3)
    assert res.mean is res.matrix                     # mean 是别名，不是另一套数
    assert set(res.stats.index.get_level_values(0)) == {'ROE', 'PB'}
    # summary() 按 |mean| 降序，且与矩阵一致
    s = res.summary()
    assert list(s['mean'].abs()) == sorted(s['mean'].abs(), reverse=True)
    assert abs(float(s.iloc[0]['mean']) - float(res.matrix.loc['PB', 'ROE'])) < 1e-15
    assert 'FactorCorr' in str(res)
    assert FactorCorrResult.__doc__ is not None
    assert RedundancyResult.__doc__ is not None
