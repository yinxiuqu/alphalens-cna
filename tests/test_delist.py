"""退市收益约定测试 —— 防线 5（已知答案）。

核心命题：**"持有期内退市"和"窗口内停牌"是两件事，不能混。**
* 退市 → 股票再也不会回来 → 可按约定清算（默认按最后成交价）
* 停牌 → 期末还没复牌，但之后还在交易 → 仍按缺失处理

混掉的后果：把停牌股按停牌前的旧价"清算"，等于凭空冻结一个过时价格。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna  # noqa: E402
from alphalens_cna.engine.returns import (  # noqa: E402
    DELIST_POLICIES, ReturnModel, forward_returns,
)

DATES = pd.bdate_range('2024-01-01', periods=6)
ASSETS = ['A', 'B', 'C']


def mk(price_b=None, b_gap_days=(), flat=10.0):
    """3 只票 6 天。A/C 恒为 10 元；B 的价格用 ``price_b`` 指定（None = 缺失）。

    ``b_gap_days`` 里的天 B 缺价，但**之后还回来**（= 停牌，不是退市）。
    """
    idx = pd.MultiIndex.from_product([DATES, ASSETS], names=['date', 'asset'])
    px = pd.DataFrame({c: flat for c in
                       ('adj_open', 'adj_close', 'raw_open', 'raw_close',
                        'raw_high', 'raw_low', 'adj_high', 'adj_low',
                        'prev_close', 'adj_factor')}, index=idx)
    if price_b is not None:
        d = idx.get_level_values('date')
        a = idx.get_level_values('asset')
        for i, v in enumerate(price_b):
            # None = 该日**没有价格**（退市或停牌），必须显式置 NaN
            px.loc[(a == 'B') & (d == DATES[i]),
                   ['adj_open', 'adj_close']] = (np.nan if v is None else v)
    for i in b_gap_days:
        d = idx.get_level_values('date')
        a = idx.get_level_values('asset')
        px.loc[(a == 'B') & (d == DATES[i]), ['adj_open', 'adj_close']] = np.nan
    return px


def ret(df, policy='nan', delist_return=-0.3, h=3, asset='B', day=0):
    m = ReturnModel(entry='same_open', exit_price='open',
                    delist_policy=policy, delist_return=delist_return)
    r = forward_returns(df, acna.Calendar(DATES), [h], model=m)
    row = r.df.xs(asset, level='asset').iloc[day]
    return row[f'forward_return_{h}'], row[f'delisted_{h}'], r


# --------------------------------------------------------------------------- #
# 已知答案
# --------------------------------------------------------------------------- #
def test_delist_last_price_known_answer():
    """★ B 第 1/2 天 10→9→8，第 3 天起消失；h=3（第 0 天入场，期末第 3 天）。

    按最后成交价清算：8 / 10 − 1 = **−0.20**
    """
    df = mk(price_b=[10.0, 9.0, 8.0, None, None, None])
    v, delisted, r = ret(df, policy='last_price')
    assert delisted
    assert abs(v - (8.0 / 10.0 - 1)) < 1e-12, v
    assert abs(v + 0.20) < 1e-12
    assert r.ledger[3]['delist_filled'] == 3, 'A/C 也在第 0 天入场，B 是 1 条'


def test_delist_haircut_known_answer():
    """再打 −30% 折扣：8 × 0.7 / 10 − 1 = **−0.44**"""
    df = mk(price_b=[10.0, 9.0, 8.0, None, None, None])
    v, _, _ = ret(df, policy='haircut', delist_return=-0.3)
    assert abs(v - (8.0 * 0.7 / 10.0 - 1)) < 1e-12, v
    assert abs(v + 0.44) < 1e-12


def test_delist_nan_keeps_old_behaviour():
    """★ 默认仍是旧行为：剔除，且台账记 ``delisted_dropped``。"""
    df = mk(price_b=[10.0, 9.0, 8.0, None, None, None])
    v, delisted, r = ret(df, policy='nan')
    assert np.isnan(v)
    assert delisted is True or delisted == True                    # noqa: E712
    assert r.ledger[3].get('delisted_dropped', 0) >= 1
    assert 'delist_filled' not in r.ledger[3]


# --------------------------------------------------------------------------- #
# 关键区分：停牌 ≠ 退市
# --------------------------------------------------------------------------- #
def test_suspension_inside_window_is_not_delisting():
    """★ B 第 2 天缺价但第 3 天**回来了** → 是停牌，不是退市 → 不许清算。

    期末（第 3 天）有价，所以本来就取得到；这里测的是第 3 天也缺、
    但第 4/5 天有价的情形 —— 那属于"期末还没复牌"，仍按缺失处理。
    """
    df = mk(price_b=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0], b_gap_days=(2, 3))
    v, delisted, r = ret(df, policy='last_price', h=3)
    assert not delisted, '期末缺价但之后还交易 → 不是退市'
    assert np.isnan(v), '停牌期末未复牌 → 仍按缺失处理'
    assert 'delist_filled' not in r.ledger[3]


def test_asset_that_returns_after_window_never_filled():
    """B 只在期末缺价、其后仍有价 → 无论哪种约定都不填。"""
    for pol in DELIST_POLICIES:
        df = mk(price_b=[10.0, 10.0, 10.0, None, 10.0, 10.0])
        v, delisted, _ = ret(df, policy=pol, h=3)
        assert not delisted, pol
        assert np.isnan(v), pol


# --------------------------------------------------------------------------- #
# 不变量：不许动没退市的票
# --------------------------------------------------------------------------- #
def test_policy_does_not_touch_survivors():
    """★ 换 delist_policy **不得**改变任何未退市样本的收益（逐位相同）。"""
    df = mk(price_b=[10.0, 9.0, 8.0, None, None, None])
    base = None
    for pol in DELIST_POLICIES:
        r = forward_returns(df, acna.Calendar(DATES), [1, 3],
                            model=ReturnModel(entry='same_open',
                                              exit_price='open',
                                              delist_policy=pol))
        sub = r.df[r.df.index.get_level_values('asset').isin(['A', 'C'])][
            ['forward_return_1', 'forward_return_3']]
        if base is None:
            base = sub
        else:
            assert np.array_equal(np.asarray(base.values, dtype=float),
                                  np.asarray(sub.values, dtype=float),
                                  equal_nan=True), pol


def test_delisted_flag_column():
    """``delisted_h`` 列必须存在且只对退市样本为真。"""
    df = mk(price_b=[10.0, 9.0, 8.0, None, None, None])
    r = forward_returns(df, acna.Calendar(DATES), [3],
                        model=ReturnModel(entry='same_open', exit_price='open',
                                          delist_policy='last_price'))
    flag = r.df['delisted_3']
    assert flag.dtype == bool
    b = r.df.xs('B', level='asset')['delisted_3']
    a = r.df.xs('A', level='asset')['delisted_3']
    assert b.iloc[0] and not a.any(), '只有 B 会退市'


# --------------------------------------------------------------------------- #
# 参数校验
# --------------------------------------------------------------------------- #
def test_bad_policy_raises():
    with pytest.raises(acna.ContractError):
        ReturnModel(delist_policy='whatever')


@pytest.mark.parametrize('bad', (0.1, -1.0, -2.0))
def test_bad_delist_return_raises(bad):
    """折扣必须落在 (−1, 0]：正数是"退市后还涨"，−1 以下是"倒欠钱"。"""
    with pytest.raises(acna.ContractError):
        ReturnModel(delist_policy='haircut', delist_return=bad)


def test_haircut_zero_equals_last_price():
    """折扣 0 应当与 ``last_price`` 完全等价。"""
    df = mk(price_b=[10.0, 9.0, 8.0, None, None, None])
    a, _, _ = ret(df, policy='last_price')
    b, _, _ = ret(df, policy='haircut', delist_return=0.0)
    assert abs(a - b) < 1e-15
