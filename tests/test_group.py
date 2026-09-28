"""分组 IC / 组内一致性 / 双重排序测试 —— 对拍 + 不变量 + 记账。

三条最关键的：

1. **★ 正确性锚（不变量）** —— 样本量加权的组内 IC 之和
   ``Σ_g (n_g/n)·IC_g`` 与"组内秩池化相关"**逐位**相等（1e-12）。
   ⚠️ 规格里写的是"= 朴素全局 IC"，**实测不成立**（本文件两条测试分别证明：
   差额 = 组间均值差，全协方差分解逐位成立）。等量分组、无并列值是前提。
2. **记账** —— 组内样本不足 / 分组键缺失 / 宫格缺格：一律 NaN + 台账，
   不填 0、不静默丢样本（``n_obs + dropped == len(data)``）。
3. **口径** —— 组内 IC 与 ``information_coefficient`` 同一把尺子；
   双重排序的池化单调性与 ``quantile_stats.monotonicity`` 逐位相同。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alphalens_cna.analysis.group import (  # noqa: E402
    DoubleSortResult,
    GroupedIC,
    double_sort,
    group_consistency,
    grouped_ic,
)
from alphalens_cna.analysis.ic import information_coefficient  # noqa: E402
from alphalens_cna.analysis.quantile import (  # noqa: E402
    quantile_stats,
    quantize,
)
from alphalens_cna.contract.errors import ContractError  # noqa: E402

ASSETS = [f'{i:06d}' for i in range(50)]


# --------------------------------------------------------------------------- #
def make_panel(dates=None, n_assets=50, n_dates=30, seed=0, n_groups=5,
               signal=0.6):
    """合成面板：``factor`` / ``forward_return_1`` / ``forward_return_5`` /
    ``ln_mv`` / ``grp``。

    ``grp`` = 每期按 ``ln_mv`` 分的 **n_groups 个等量组**（等量是加权恒等式的前提）。
    """
    rng = np.random.default_rng(seed)
    dates = (pd.bdate_range('2024-01-01', periods=n_dates) if dates is None
             else pd.DatetimeIndex(dates))
    idx = pd.MultiIndex.from_product(
        [dates, [f'{i:06d}' for i in range(n_assets)]], names=['date', 'asset'])
    mv = rng.normal(size=len(idx))
    grp = quantize(pd.DataFrame({'factor': mv}, index=idx), n=n_groups)['q']
    fac = 0.6 * mv + rng.normal(size=len(idx))
    ret = signal * fac + rng.normal(size=len(idx))
    return pd.DataFrame({
        'factor': fac, 'forward_return_1': ret, 'forward_return_5': ret * 2,
        'ln_mv': mv, 'grp': grp.to_numpy()}, index=idx)


# --------------------------------------------------------------------------- #
# 1. ★ 正确性锚：样本量加权的组内 IC 之和
# --------------------------------------------------------------------------- #
def test_weighted_group_ic_equals_within_rank_pooled():
    """★ 不变量：``Σ_g (n_g/n)·IC_g`` == 组内秩池化相关（1e-12，逐期）。

    右边**手工独立复算**：把每个观测换成组内秩，再逐期算 Pearson。

    ⚠️ 等式成立的前提（本测试显式断言）：**各组样本量相等**、无并列值。
    """
    df = make_panel(seed=3, n_assets=40, n_dates=24)
    sizes = df.groupby([df.index.get_level_values('date'), 'grp']).size()
    assert sizes.nunique() == 1, '前提失败：分组不是等量的'
    assert sizes.iloc[0] == 8

    g = grouped_ic(df, by='grp')
    assert isinstance(g, GroupedIC)
    w = g.weighted_ic()

    dates = df.index.get_level_values('date')
    rank = (pd.DataFrame({'f': df['factor'], 'r': df['forward_return_1'],
                          'r5': df['forward_return_5']})
            .groupby([dates, df['grp'].to_numpy()]).rank())
    manual = pd.Series({d: sub['f'].corr(sub['r'])
                        for d, sub in rank.groupby(level='date')})

    assert np.allclose(w[1].sort_index().to_numpy(),
                       manual.sort_index().to_numpy(), atol=1e-12), '锚不成立'
    assert np.allclose(w[1].to_numpy(), g.within_rank_ic[1].to_numpy(),
                       atol=1e-12)
    assert np.allclose(w[5].to_numpy(), g.within_rank_ic[5].to_numpy(),
                       atol=1e-12)


def test_weighted_group_ic_is_not_naive_global_ic_and_why():
    """★ 规格勘误：加权和 **≠** 朴素全局 IC；差额就是**组间均值差**。

    逐位成立的分解（本测试独立复算到 1e-12）：

    .. code-block:: text

        Cov(x, y) = Σ_g w_g · Cov_g(x, y) + Σ_g w_g (x̄_g − x̄)(ȳ_g − ȳ)
                    └── 组内部分 ──┘     └──── 组间部分（被相关系数丢掉）────┘
    """
    df = make_panel(seed=3, n_assets=40, n_dates=24)
    g = grouped_ic(df, by='grp')

    gap = float(g.weighted_ic()[1].mean() - g.global_ic[1].mean())
    assert abs(gap) > 0.02, f'本构造下两口径应当明显不同，实际差 {gap}'

    # 独立复算全协方差分解
    tot, within, betw = [], [], []
    for _, sub in df.groupby(level='date'):
        x = sub['factor'].to_numpy()
        y = sub['forward_return_1'].to_numpy()
        gg = sub['grp'].to_numpy()
        xb, yb = x.mean(), y.mean()
        tot.append(float(((x - xb) * (y - yb)).mean()))
        w_sum, b_sum = 0.0, 0.0
        for k in np.unique(gg):
            m = gg == k
            w_sum += m.mean() * float(((x[m] - x[m].mean())
                                       * (y[m] - y[m].mean())).mean())
            b_sum += m.mean() * float((x[m].mean() - xb) * (y[m].mean() - yb))
        within.append(w_sum)
        betw.append(b_sum)
    tot, within, betw = map(np.array, (tot, within, betw))
    assert np.allclose(tot, within + betw, atol=1e-12)   # 恒等式成立
    # 组间项不为 0 —— 这就是"加权和 ≠ 全局 IC"的全部原因
    assert float(np.abs(betw).mean()) > 1e-4


def test_weighted_identity_needs_equal_group_sizes():
    """⚠️ 组量不等时该恒等式**不成立**（本测试把边界钉住，避免被误当成恒真）。"""
    df = make_panel(seed=3, n_assets=40, n_dates=24)
    # 人为把第 5 组的一部分挪到第 4 组 → 组量 10/10/10/15/5
    grp = df['grp'].to_numpy().copy()
    aid = np.tile(np.arange(40), 24)
    move = (grp == 5) & (aid < 5)
    grp[move] = 4
    d2 = df.copy()
    d2['grp'] = grp
    sizes = d2.groupby([d2.index.get_level_values('date'), 'grp']).size()
    assert sizes.nunique() > 1, '前提失败：应当出现不等量分组'

    g = grouped_ic(d2, by='grp')
    diff = float(np.abs(g.weighted_ic()[1] - g.within_rank_ic[1]).max())
    assert diff > 1e-6, f'组量不等时不该逐位相等（实测差 {diff}）'


# --------------------------------------------------------------------------- #
# 2. 分组 IC 的正确性与口径
# --------------------------------------------------------------------------- #
def test_grouped_ic_matches_manual_pandas():
    """逐 (期, 组) 的组内 Spearman 与手工 pandas 对到 1e-12。"""
    df = make_panel(seed=1, n_assets=40, n_dates=20)
    g = grouped_ic(df, by='grp')
    dates = df.index.get_level_values('date')

    manual = {}
    for (d, k), sub in df.groupby([dates, df['grp'].to_numpy()]):
        manual[(d, k)] = sub['factor'].corr(sub['forward_return_1'],
                                            method='spearman')
    man = pd.Series(manual)
    man.index = pd.MultiIndex.from_tuples(man.index, names=['date', 'grp'])
    man = man.sort_index()
    ours = g.ic[1].sort_index()
    assert len(man) == len(ours) == 20 * 5
    assert np.allclose(man.to_numpy(), ours.to_numpy(), atol=1e-12)
    # counts = 组内有效样本数（等量组 → 每格 10）
    assert (g.counts[1] == 8).all()
    assert g.ic.shape == (100, 2) and g.counts.shape == (100, 2)


def test_global_ic_same_ruler_as_information_coefficient():
    """``global_ic`` 与库里的 ``information_coefficient`` **逐位**相同（同一把尺子）。"""
    df = make_panel(seed=2, n_assets=40, n_dates=20)
    g = grouped_ic(df, by='grp')
    ref = information_coefficient(df, method='spearman')
    assert np.allclose(g.global_ic[1].to_numpy(), ref[1].to_numpy(), atol=0)
    assert np.allclose(g.global_ic[5].to_numpy(), ref[5].to_numpy(), atol=0)
    # 组内 IC 的均值 ≠ 全局 IC 的均值（两回事，别混用）
    assert abs(float(g.summary['mean'].mean() - g.global_ic.mean().mean())) > 0.01


def test_summary_time_aggregation_fields():
    """summary 的时间汇总口径：mean/median/std/icir/positive_rate/n_periods。"""
    df = make_panel(seed=4, n_assets=40, n_dates=20)
    g = grouped_ic(df, by='grp')
    row = g.summary.loc[(1, 1.0)]
    s = g.ic[1].xs(1.0, level='grp')
    assert int(row['n_periods']) == 20
    assert abs(float(row['mean']) - float(s.mean())) < 1e-15
    assert abs(float(row['median']) - float(s.median())) < 1e-15
    assert abs(float(row['std']) - float(s.std(ddof=1))) < 1e-15
    assert abs(float(row['icir']) - float(s.mean() / s.std(ddof=1))) < 1e-12
    assert abs(float(row['positive_rate']) - float((s > 0).mean())) < 1e-15
    assert abs(float(row['mean_count']) - float(g.counts[1].xs(1.0, level='grp').mean())) < 1e-15


# --------------------------------------------------------------------------- #
# 3. 记账：样本不足 / 分组键缺失
# --------------------------------------------------------------------------- #
def make_small_group_panel(seed=5, small=3, n_assets=40, n_dates=20):
    """第 1 组每期只有 ``small`` 只票，其余分成 4 个 ~12 只的组。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-01', periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f'{i:06d}' for i in range(n_assets)]],
                                     names=['date', 'asset'])
    aid = np.tile(np.arange(n_assets), n_dates)
    grp = np.where(aid < small, 1.0, 2.0 + (aid % 4))
    fac = rng.normal(size=len(idx))
    ret = 0.6 * fac + rng.normal(size=len(idx))
    return pd.DataFrame({'factor': fac, 'forward_return_1': ret,
                         'forward_return_5': ret * 2, 'grp': grp}, index=idx)


def test_min_group_size_skips_and_accounts():
    """★ 组内样本不足 → 该期该组 NaN + 记账（跳了多少期、为什么）。"""
    df = make_small_group_panel()
    g = grouped_ic(df, by='grp', min_group_size=5)

    # 第 1 组每期都只有 3 只 → 全部跳过（NaN，不是 0）
    assert g.ic[1].xs(1.0, level='grp').isna().all()
    assert g.ic[5].xs(1.0, level='grp').isna().all()
    assert g.counts[1].xs(1.0, level='grp').isna().all()
    assert int(g.summary.loc[(1, 1.0), 'n_periods']) == 0
    # 其它组正常
    assert int(g.summary.loc[(1, 2.0), 'n_periods']) == 20

    # 台账：30 期 × 2 个持有期
    assert len(g.skipped) == 20 * 2
    assert set(g.skipped['horizon']) == {1, 5}
    assert (g.skipped['n'] == 3).all()
    assert g.skipped['reason'].str.contains('min_size 5').all()
    assert g.skipped['reason'].str.contains('3').all()
    # skip_summary 与台账自洽，且回答"跳过了多少期"
    assert int(g.skip_summary['n_cells'].sum()) == 40
    assert int(g.skip_summary['n_periods'].iloc[0]) == 20
    assert '跳过 40 格' in str(g)

    # 把下限降到 3，同一份数据就出数了（证明是"门槛"在起作用，不是算不出来）
    g3 = grouped_ic(df, by='grp', min_group_size=3)
    assert not g3.ic[1].xs(1.0, level='grp').isna().any()
    assert len(g3.skipped) == 0


def test_missing_group_labels_are_booked():
    """分组键是 NaN 的行**不属于任何组**，但要记账（不静默丢样本）。"""
    df = make_panel(seed=6, n_assets=40, n_dates=20)
    d0 = df.index.get_level_values('date')[0]
    mask = (df.index.get_level_values('date') == d0) & \
        (df.index.get_level_values('asset') < '000005')
    df.loc[mask, 'grp'] = np.nan

    g = grouped_ic(df, by='grp')
    led = g.skipped[g.skipped['reason'].str.contains('分组键缺失')]
    assert len(led) == 2                       # 2 个持有期各一条
    assert (led['n'] == 5).all()
    assert led['date'].nunique() == 1
    # 组标签里不该冒出 NaN 这个"隐形分组"
    assert not g.ic.index.get_level_values('grp').isna().any()
    assert g.ic.shape[0] == 20 * 5


# --------------------------------------------------------------------------- #
# 4. 组内一致性
# --------------------------------------------------------------------------- #
def make_one_group_only_panel(seed=21, n_assets=45, n_dates=30, target=5.0):
    """因子**只**在 ``grp == target`` 那一组里有效。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2024-01-01', periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f'{i:06d}' for i in range(n_assets)]],
                                     names=['date', 'asset'])
    ctrl = rng.normal(size=len(idx))
    grp = quantize(pd.DataFrame({'factor': ctrl}, index=idx), n=5)['q'].to_numpy()
    fac = rng.normal(size=len(idx))
    ret = 0.9 * fac * (grp == target) + rng.normal(size=len(idx))
    return pd.DataFrame({'factor': fac, 'forward_return_1': ret,
                         'grp': grp}, index=idx)


def test_group_consistency_finds_the_only_working_group():
    """★ "因子是不是只在某一组里有效" —— 信号只在第 5 组时，读数必须指出来。"""
    df = make_one_group_only_panel()
    c = group_consistency(grouped_ic(df, by='grp'))

    assert int(c.loc[(1, 5.0), 'n_periods']) == 30
    assert float(c.loc[(1, 5.0), 'mean']) > 0.4          # 第 5 组真有效
    assert float(c.loc[(1, 5.0), 'same_sign_rate']) == 1.0
    assert int(c.loc[(1, 5.0), 'sign']) == 1

    for k in (1.0, 2.0, 3.0, 4.0):                        # 其余组是噪声
        assert abs(float(c.loc[(1, k), 'mean'])) < 0.2
        assert float(c.loc[(1, k), 'same_sign_rate']) < 0.7

    across = c.attrs['across_groups']
    assert int(across.loc[1, 'n_groups']) == 5
    assert float(across.loc[1, 'max_group_mean']) > 0.4
    assert float(across.loc[1, 'min_group_mean']) < 0.1
    assert float(across.loc[1, 'std_of_means']) > 0.15   # 组间离散度
    assert float(across.loc[1, 'sign_agreement_rate']) <= 0.8   # 不是全组同号
    assert c.attrs['by_name'] == 'grp'


def test_group_consistency_horizons_and_bad_input():
    df = make_panel(seed=8, n_assets=40, n_dates=20)
    g = grouped_ic(df, by='grp')
    c = group_consistency(g, horizons=[5])
    assert set(c.index.get_level_values('horizon')) == {5}
    with pytest.raises(ContractError) as e:
        group_consistency(g, horizons=[99])
    assert e.value.rule == 'no_horizon'
    with pytest.raises(ContractError) as e:
        group_consistency(df)                      # 传 DataFrame 而不是结果对象
    assert e.value.rule == 'bad_grouped'
    assert 'grouped_ic' in str(e.value)


# --------------------------------------------------------------------------- #
# 5. 双重排序
# --------------------------------------------------------------------------- #
def test_double_sort_grid_is_n_by_times_n():
    """★ 宫格数 = ``n_by × n``，且索引恰好是完整的笛卡尔积。"""
    df = make_panel(seed=11, n_assets=50, n_dates=20)
    res = double_sort(df, by='ln_mv', n_by=5, n=5)
    assert isinstance(res, DoubleSortResult)
    assert res.mean.shape == (25, 2)
    assert list(res.mean.columns) == [1, 5]
    assert list(res.mean.index) == [(b, q) for b in range(1, 6)
                                    for q in range(1, 6)]
    assert res.count.shape == (25,) and res.n_periods.shape == (25,)
    assert int((res.mean.isna()).sum().sum()) == 0        # 每格至少出现过一次
    assert int(res.n_periods.max()) == 20
    assert bool((res.n_periods > 0).all())
    assert int(res.count.min()) >= 1
    # 60 只票摊到 25 格 → 有的格子某些期会空，必须记成"部分缺"（不是填 0）
    partial = res.missing[res.missing['kind'] == '部分缺']
    assert len(partial) == int((res.n_periods < 20).sum())
    # 宽表 = 5×5
    wide = res.to_frame(1)
    assert wide.shape == (5, 5)
    assert float(wide.loc[5, 5]) == float(res.mean.loc[(5, 5), 1])
    assert res.n_obs + int(res.dropped['n_rows'].sum()) == len(df)
    assert 'DoubleSort' in str(res)


def test_double_sort_missing_cells_are_nan_not_zero():
    """★ 缺格**不填 0**：控制变量只能分出 4 层时，第 5 层整行缺格并记账。"""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range('2024-01-01', periods=20)
    n_assets = 50
    idx = pd.MultiIndex.from_product([dates, [f'{i:06d}' for i in range(n_assets)]],
                                     names=['date', 'asset'])
    aid = np.tile(np.arange(n_assets), 20)
    # 离散控制变量：只有 4 个不同取值 → qcut 只会分出 4 层
    mv = np.where(aid < 3, 0.0, 1.0 + (aid % 4))
    fac = 0.6 * mv + rng.normal(size=len(idx))
    ret = 0.5 * fac + rng.normal(size=len(idx))
    df = pd.DataFrame({'factor': fac, 'forward_return_1': ret, 'ln_mv': mv},
                      index=idx)

    res = double_sort(df, by='ln_mv', n_by=5, n=5)
    nan_cells = {(b, q) for b, q in res.mean.index
                 if bool(res.mean.loc[(b, q)].isna().all())}
    assert nan_cells == {(5, q) for q in range(1, 6)}
    assert int((res.mean == 0).sum().sum()) == 0          # 不填 0
    miss = res.missing[res.missing['kind'] == '全缺']
    assert len(miss) == 5
    assert set(miss['q_by']) == {5}
    assert miss['reason'].str.contains('第 5 层').all()
    assert miss['reason'].str.contains('合并').all()
    assert res.missing['kind'].iloc[0] == '全缺'          # 全缺排在前面
    assert bool(np.isnan(res.count.loc[(5, 1)]))          # 计数也是 NaN


def test_double_sort_sample_accounting_is_exact():
    """``n_obs + dropped.n_rows.sum() == len(data)`` —— 每条样本都有去向。"""
    df = make_panel(seed=12, n_assets=50, n_dates=16)
    rng = np.random.default_rng(0)
    df = df.copy()
    df.loc[df.index[:15], 'factor'] = np.nan          # 因子缺失
    df.loc[df.index[15:25], 'ln_mv'] = np.nan         # 控制变量缺失
    df.loc[df.index[25:30], 'forward_return_1'] = np.nan
    res = double_sort(df, by='ln_mv', n_by=5, n=5)
    assert res.n_obs + int(res.dropped['n_rows'].sum()) == len(df)
    assert int(res.dropped['n_rows'].sum()) > 0
    txt = ' | '.join(res.dropped['reason'])
    assert 'factor' in txt and 'by=' in txt
    assert res.n_dates == 16


def test_double_sort_pooled_monotonicity_same_ruler_as_quantile_stats():
    """口径锚：``pooled_monotonicity`` 与 ``quantile_stats.monotonicity`` 逐位相同。"""
    df = make_panel(seed=13, n_assets=50, n_dates=18)
    for method in ('independent', 'conditional'):
        res = double_sort(df, by='ln_mv', n_by=5, n=5, method=method)
        ref = quantile_stats(df, quantiles=5, by=df['ln_mv'], method=method,
                             horizons=[1, 5])
        assert np.allclose(res.pooled_monotonicity.to_numpy(),
                           ref['monotonicity'].to_numpy(), atol=1e-12), method


def test_double_sort_within_group_monotonicity_is_per_group():
    """★ 组内单调性必须**逐组**算：因子只在一个 by 组里有效时，读数要分开。"""
    rng = np.random.default_rng(31)
    n_dates, n_assets = 30, 120
    dates = pd.bdate_range('2024-01-01', periods=n_dates)
    idx = pd.MultiIndex.from_product([dates, [f'{i:06d}' for i in range(n_assets)]],
                                     names=['date', 'asset'])
    mv = rng.normal(size=len(idx))
    q_by = quantize(pd.DataFrame({'factor': mv}, index=idx), n=5)['q'].to_numpy()
    fac = rng.normal(size=len(idx))
    ret = 1.2 * fac * (q_by == 5) + rng.normal(size=len(idx))
    df = pd.DataFrame({'factor': fac, 'forward_return_1': ret, 'ln_mv': mv},
                      index=idx)

    res = double_sort(df, by='ln_mv', n_by=5, n=5, method='conditional')
    mono = res.monotonicity['spearman']
    assert float(mono.loc[(5.0, 1)]) > 0.9              # 最大规模组里单调
    others = mono.drop(index=5.0, level='q_by')
    assert float(others.abs().max()) < 0.6              # 其余组不单调
    assert float(res.pooled_monotonicity[1]) > 0.9      # 池化后看着"很单调"
    assert set(mono.index.get_level_values('q_by')) == {1.0, 2.0, 3.0, 4.0, 5.0}


def test_double_sort_conditional_matches_quantize_labels():
    """``conditional`` 的层序与 ``quantize``（同一套分位口径）一致。"""
    df = make_panel(seed=14, n_assets=40, n_dates=10)
    lab = quantize(df, n=5, by=df['ln_mv'], method='conditional')
    res = double_sort(df, by='ln_mv', n_by=5, n=5, method='conditional')
    # 用同一套标签自己算一遍逐期均值 → 与 grid 对上
    tmp = df[['forward_return_1']].join(lab).dropna()
    man = (tmp.groupby([tmp.index.get_level_values('date'), 'q_by', 'q'])
           ['forward_return_1'].mean())
    got = res.grid['forward_return_1'].sort_index()
    assert np.allclose(man.sort_index().to_numpy(), got.to_numpy(), atol=1e-15)


# --------------------------------------------------------------------------- #
# 6. 非法输入（一律 ContractError，且能定位）
# --------------------------------------------------------------------------- #
def test_by_column_missing_is_rejected():
    df = make_panel(seed=9, n_assets=30, n_dates=12)
    with pytest.raises(ContractError) as e:
        grouped_ic(df, by='行业')
    assert e.value.contract == 'group' and e.value.rule == 'by_missing'
    assert '行业' in str(e.value)          # 指明是哪个名字
    assert 'grp' in str(e.value)           # 并列出可用列
    with pytest.raises(ContractError) as e:
        double_sort(df, by='行业')
    assert e.value.rule == 'by_missing'


def test_missing_factor_column_is_rejected():
    df = make_panel(seed=9, n_assets=30, n_dates=12).drop(columns=['factor'])
    with pytest.raises(ContractError) as e:
        grouped_ic(df, by='grp')
    assert e.value.rule == 'no_factor'
    assert 'forward_return_1' in str(e.value)
    with pytest.raises(ContractError) as e:
        double_sort(df, by='ln_mv')
    assert e.value.rule == 'no_factor'


def test_bad_layer_counts_are_rejected():
    df = make_panel(seed=9, n_assets=30, n_dates=12)
    with pytest.raises(ContractError) as e:
        double_sort(df, by='ln_mv', n_by=1)
    assert e.value.rule == 'bad_n_by'
    assert 'n_by' in str(e.value)
    with pytest.raises(ContractError) as e:
        double_sort(df, by='ln_mv', n=1)
    assert e.value.rule == 'bad_n'
    with pytest.raises(ContractError) as e:
        double_sort(df, by='ln_mv', n_by=0)
    assert e.value.rule == 'bad_n_by'


def test_bad_method_and_min_group_size_are_rejected():
    df = make_panel(seed=9, n_assets=30, n_dates=12)
    with pytest.raises(ContractError) as e:
        grouped_ic(df, by='grp', method='kendall')
    assert e.value.rule == 'bad_method'
    with pytest.raises(ContractError) as e:
        grouped_ic(df, by='grp', min_group_size=2)
    assert e.value.rule == 'bad_min_group_size'
    assert '3' in str(e.value)
    with pytest.raises(ContractError) as e:
        double_sort(df, by='ln_mv', method='both')
    assert e.value.rule == 'bad_method'


def test_by_must_be_numeric_for_double_sort():
    """分类变量（行业名）不能做双重排序的 ``by`` —— 要能明确指出怎么改。"""
    df = make_panel(seed=9, n_assets=30, n_dates=12).copy()
    df['industry'] = np.where(np.arange(len(df)) % 3 == 0, '银行', '医药')
    with pytest.raises(ContractError) as e:
        double_sort(df, by='industry')
    assert e.value.rule == 'by_not_numeric'
    assert 'grouped_ic' in str(e.value)          # 给了替代方案
    # 同样这份数据，分组 IC 是能用的（分类变量本来就该走那条路）
    g = grouped_ic(df, by='industry')
    assert set(g.ic.index.get_level_values('industry')) == {'银行', '医药'}


def test_two_by_columns_are_rejected_for_double_sort():
    df = make_panel(seed=9, n_assets=30, n_dates=12)
    two = pd.DataFrame({'half': (df['ln_mv'] > 0).astype(int), 'grp': df['grp']})
    with pytest.raises(ContractError) as e:
        double_sort(df, by=two)
    assert e.value.rule == 'by_ambiguous'
    # 但分组 IC 支持多键分组
    g = grouped_ic(df, by=two, min_group_size=3)
    assert g.ic.index.names == ['date', 'half', 'grp']
    assert g.ic[1].notna().any()
    # 格子数 = 每期实际出现的 (half, grp) 组合数（独立数一遍，不靠函数自己说）
    n_cells = (df.assign(half=(df['ln_mv'] > 0).astype(int))
               .groupby([df.index.get_level_values('date'), 'half', 'grp'])
               .ngroups)
    assert set(g.ic.index.get_level_values('half')) == {0, 1}
    assert g.ic.shape[0] == n_cells
    assert g.n_groups == 8                     # 出现过 8 种 (half, grp) 组合


def test_misaligned_by_is_rejected():
    df = make_panel(seed=9, n_assets=30, n_dates=12)
    bad = pd.Series(np.arange(10.0), index=pd.RangeIndex(10), name='bad')
    with pytest.raises(ContractError) as e:
        grouped_ic(df, by=bad)
    assert e.value.rule == 'by_align'


def test_missing_returns_are_rejected():
    df = make_panel(seed=9, n_assets=30, n_dates=12).drop(columns=['forward_return_1',
                                          'forward_return_5'])
    with pytest.raises(ContractError) as e:
        grouped_ic(df, by='grp')
    assert 'forward_return' in str(e.value)
    with pytest.raises(ContractError) as e:
        double_sort(df, by='ln_mv')
    assert 'forward_return' in str(e.value)


def test_bad_index_is_rejected():
    df = make_panel(seed=9, n_assets=30, n_dates=12)
    flat = df.copy()
    flat.index = pd.RangeIndex(len(flat))
    with pytest.raises(ContractError) as e:
        grouped_ic(flat, by='grp')
    assert e.value.rule in ('index', 'index_names')


def test_oracle_and_result_objects():
    """直接导入路径可用；结果是带文档的 dataclass。"""
    from alphalens_cna.analysis import group as mod
    for nm in ('grouped_ic', 'group_consistency', 'double_sort'):
        assert callable(getattr(mod, nm))
        assert nm in mod.__all__
    assert GroupedIC.__doc__ and DoubleSortResult.__doc__
    df = make_panel(seed=15, n_assets=40, n_dates=20)
    g = grouped_ic(df, by='grp')
    assert list(g.horizons) == [1, 5]
    assert g.by_name == 'grp' and g.n_groups == 5 and g.n_dates == 20
    assert g.group_names == ['grp']
    assert list(g.to_frame().columns) == [1, 5]
    assert np.allclose(g.to_frame().to_numpy(),
                       g.summary['mean'].unstack('horizon').to_numpy())
