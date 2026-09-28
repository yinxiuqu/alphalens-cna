"""契约层补口（0.4.0 的 F1–F6）：来自用户反馈，逐条实测后修掉的三个缺口。

| # | 缺口 | 修法 |
|---|---|---|
| F1 | `_check_finite` 只查 `numeric` 声明的列 → `Exposures`/`Universe`/`Grouping`/`Events`（`numeric=()`）**一个列都不查** | 改为「`numeric` ∪ **所有数值 dtype 列**」 |
| F2 | `build_report` 不把 `exposures` 送进边界 | 送进 `ensure_contract`，并用包装后的对象 |
| F3 | `groupby` 没进边界 → 形状不对时漏**裸 `AttributeError`/`IndexError`**，且指向下游错层 | 送进边界；`Grouping` 放宽为「恰好一列」（不再强制叫 `group`），`neutralize` 同口径 |
| F4 | `build_report` 没有 `tradability` 入参（只能隐式往 prices 塞列） | 新增 `tradability=None`，给了就用且先校验 |
| F5 | `align_event_windows` 只做 `_as_df(events)`，不校验 | 包装成 `Events`（构造即校验） |
| F6 | `events` 是否该进 `build_report` | **不加**（它不做事件研究），在 docstring 写明边界 |

实测背景：把 `inf` 塞进 `exposures`，0.3.0 会一路跑通，且结果与把 `inf` 换成 `NaN`
**逐位相同** —— 非法值被静默当成缺失，用户拿不到任何信号。
"""

from __future__ import annotations

import inspect
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna                                          # noqa: E402
from alphalens_cna.contract.errors import ContractError               # noqa: E402


def mk(n_date=40, n_assets=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2023-01-02', periods=n_date)
    assets = [f'{600000 + i:06d}' for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    close = pd.Series(
        np.tile(rng.uniform(8, 40, n_assets), n_date)
        * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    px = pd.DataFrame({'raw_close': close}, index=idx)
    px['raw_open'] = close
    px['raw_high'] = close * 1.004
    px['raw_low'] = close * 0.996
    px['prev_close'] = px.groupby(level='asset')['raw_close'].shift(1).fillna(close)
    px['adj_factor'] = 1.0
    for c in ('open', 'high', 'low', 'close'):
        px[f'adj_{c}'] = px[f'raw_{c}']
    px['volume'] = 1e5
    f = pd.DataFrame({'value': rng.normal(size=len(idx))}, index=idx)
    f['available_at'] = f.index.get_level_values('date')
    return dates, idx, px, f, rng


# ── F1：±inf 在**每个**契约上都要拦（NaN 仍合法）──────────────────────────
def test_inf_is_refused_on_every_contract():
    """★ 曾经：`numeric = ()` 的四个契约对 inf **完全无感**。"""
    dates, idx, px, f, rng = mk()
    bad = idx[7]

    exp = pd.DataFrame({'size': rng.normal(size=len(idx))}, index=idx)
    exp.loc[bad, 'size'] = np.inf
    uni = pd.DataFrame({'in_universe': True, 'mktcap': 1.0}, index=idx)
    uni.loc[bad, 'mktcap'] = np.inf
    ev = pd.DataFrame({'event_type': ['e'] * len(idx), 'surprise': 1.0}, index=idx)
    ev.loc[bad, 'surprise'] = np.inf
    grp = pd.DataFrame({'g': [1.0] * len(idx)}, index=idx)        # 单列数值分组
    grp.loc[bad, 'g'] = np.inf
    px_bad = px.copy()
    px_bad.loc[bad, 'adj_close'] = np.inf

    for label, ctor, data in [
        ('Exposures', acna.Exposures, exp),
        ('Universe', acna.Universe, uni),
        ('Events', acna.Events, ev),
        ('Grouping', acna.Grouping, grp),
        ('PricePanel', acna.PricePanel, px_bad),
    ]:
        with pytest.raises(ContractError) as e:
            ctor(data)
        assert e.value.rule == 'non_finite', f'{label}: {e.value.rule}'


def test_nan_is_still_a_legal_missing_value():
    """★ 别把 NaN 也拦了 —— 停牌 / 未上市就是 NaN。"""
    dates, idx, px, f, rng = mk()
    exp = pd.DataFrame({'size': rng.normal(size=len(idx))}, index=idx)
    exp.loc[idx[7], 'size'] = np.nan
    assert isinstance(acna.Exposures(exp), acna.Exposures)


# ── F3a：Grouping 只要求「恰好一列」，不强制列名叫 group ────────────────────
def test_grouping_accepts_any_single_column():
    dates, idx, px, f, rng = mk()
    ind = pd.DataFrame({'industry': rng.choice(['a', 'b'], len(idx))}, index=idx)
    g = acna.Grouping(ind)
    assert g.group_col == 'industry', 'group_col 要告诉调用方用的是哪一列'

    g2 = acna.Grouping(ind.rename(columns={'industry': 'group'}))
    assert g2.group_col == 'group'

    both = ind.copy()
    both['group'] = 'x'
    assert acna.Grouping(both).group_col == 'group', '有 group 时优先用它'


def test_grouping_rejects_ambiguous_multi_column():
    dates, idx, px, f, rng = mk()
    two = pd.DataFrame({'a': ['x'] * len(idx), 'b': ['y'] * len(idx)}, index=idx)
    with pytest.raises(ContractError) as e:
        acna.Grouping(two)
    assert e.value.rule == 'ambiguous_group'
    assert 'group' in str(e.value)          # 报错要给出修法


# ── F2/F3/F4：可选契约在**入口**就被拦住（不再漏到下游）────────────────────
def test_exposures_with_inf_is_refused_at_the_entry():
    dates, idx, px, f, rng = mk()
    exp = pd.DataFrame({'size': rng.normal(size=len(idx))}, index=idx)
    exp.loc[idx[7], 'size'] = np.inf
    with pytest.raises(ContractError) as e:
        acna.build_report(f, px, acna.Calendar(dates), horizons=(1,), quantiles=3,
                          exposures=exp)
    assert (e.value.contract, e.value.rule) == ('Exposures', 'non_finite')


@pytest.mark.parametrize('groupby,contract,rule', [
    ('industry', 'validate', 'type'),
    (pd.DataFrame(index=pd.MultiIndex.from_product(
        [pd.bdate_range('2023-01-02', periods=2), ['600000']],
        names=['date', 'asset'])), 'Grouping', 'empty'),
    (pd.DataFrame({'a': ['x'], 'b': ['y']},
                  index=pd.MultiIndex.from_tuples(
                      [(pd.Timestamp('2023-01-02'), '600000')],
                      names=['date', 'asset'])), 'Grouping', 'ambiguous_group'),
])
def test_bad_groupby_is_refused_at_the_entry_not_downstream(
        groupby, contract, rule):
    """★ 曾经是裸 `AttributeError: 'str' object has no attribute 'iloc'`
    与裸 `IndexError` —— 报错既不指向入口、也不告诉人怎么修。"""
    dates, idx, px, f, rng = mk()
    with pytest.raises(ContractError) as e:
        acna.build_report(f, px, acna.Calendar(dates), horizons=(1,), quantiles=3,
                          groupby=groupby)
    assert (e.value.contract, e.value.rule) == (contract, rule)


def test_legal_groupby_column_name_is_not_penalised():
    """★ 关键的反向测试：列名叫 `industry`（不叫 group）+ 中性化，必须照跑。"""
    dates, idx, px, f, rng = mk()
    ind = pd.DataFrame({'industry': rng.choice(['a', 'b', 'c'], len(idx))}, index=idx)
    rep = acna.build_report(f, px, acna.Calendar(dates), horizons=(1, 5), quantiles=3,
                            groupby=ind, preprocess=('neutralize',))
    assert rep.verdict is not None
    assert 'neutralize' in rep.preprocess


def test_neutralize_group_column_name_does_not_change_the_numbers():
    """`neutralize(groups=…)` 与 `Grouping.group_col` 同口径 —— 改名不该改数字。"""
    dates, idx, px, f, rng = mk()
    ind = pd.DataFrame({'industry': rng.choice(['a', 'b'], len(idx))}, index=idx)
    # ⚠️ 传 Series（`neutralize` 与输入同型；传 DataFrame 回来就是 DataFrame）
    factor = f['value']
    a = acna.neutralize(factor, groups=ind)
    b = acna.neutralize(factor, groups=ind.rename(columns={'industry': 'group'}))
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_explicit_tradability_is_used_and_validated():
    dates, idx, px, f, rng = mk()
    cal = acna.Calendar(dates)
    tr = acna.compute_tradability(px, calendar=cal)
    rep = acna.build_report(f, px, cal, horizons=(1,), quantiles=3, tradability=tr)
    assert 0.0 <= rep.verdict.tradable_ratio <= 1.0

    with pytest.raises(ContractError) as e:                 # 只有无关列 → 入口拦下
        acna.build_report(f, px, cal, horizons=(1,), quantiles=3,
                          tradability=pd.DataFrame({'x': [1.0]}, index=idx[:1]))
    assert (e.value.contract, e.value.rule) == ('Tradability', 'missing_signal')


# ── F5：events 的真入口也要校验 ────────────────────────────────────────────
def test_align_event_windows_validates_events():
    dates, idx, px, f, rng = mk()
    # ⚠️ 窗口 (-5, +20) 在 40 天面板上很容易越界，而**越界也记进 `not_on_calendar`**
    #    （键名把"日期不在历"和"窗口越界"混在一起 —— 见 §0.5 的备注）。
    #    这里显式给小窗口 + 事件放在第 10 天，避开越界，专测事件表契约。
    mid = idx[idx.get_level_values('date') == dates[10]][:10]
    bad = pd.DataFrame({'x': [1.0] * 10}, index=mid)
    with pytest.raises(ContractError) as e:
        acna.align_event_windows(px, bad, acna.Calendar(dates), window=(-2, 4))
    assert (e.value.contract, e.value.rule) == ('Events', 'missing_columns')

    good = pd.DataFrame({'event_type': ['earnings'] * 10}, index=mid)
    w = acna.align_event_windows(px, good, acna.Calendar(dates), window=(-2, 4))
    assert len(w.df) > 0, '合法事件表必须真的对齐出窗口'


# ── F6：events 不进 build_report（边界要显式）──────────────────────────────
def test_build_report_has_no_events_parameter():
    """它不做事件研究 —— 加个 `events` 入参只会让人以为报告里有对应的一节。"""
    params = inspect.signature(acna.build_report).parameters
    assert 'events' not in params
    doc = inspect.getdoc(acna.build_report) or ''
    assert 'align_event_windows' in doc, '边界要写在文档里，不能只靠"没这个参数"'
