"""契约层测试 —— 防线 1（契约校验）+ 防线 5（已知答案）。

每条"应当被拒绝"的用例，既验证**会拒绝**，也验证**错误信息能定位**。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna
from alphalens_cna.contract.errors import ContractError

DATES = pd.bdate_range('2024-01-02', periods=5)
ASSETS = ['600000', '000001', '300750']
IDX = pd.MultiIndex.from_product([DATES, ASSETS], names=['date', 'asset'])


# --------------------------------------------------------------------------- #
# 造数据的小工具
# --------------------------------------------------------------------------- #
def make_prices(n=5, assets=None, factor=1.0):
    assets = assets or ASSETS
    idx = pd.MultiIndex.from_product([DATES[:n], assets], names=['date', 'asset'])
    k = len(idx)
    df = pd.DataFrame({
        'raw_open':   np.tile([10.0, 12.0, 30.0], n)[:k],
        'raw_high':   np.tile([10.3, 12.2, 30.8], n)[:k],
        'raw_low':    np.tile([9.9, 11.9, 29.8], n)[:k],
        'raw_close':  np.tile([10.2, 12.1, 30.5], n)[:k],
        'prev_close': np.tile([10.0, 12.0, 30.0], n)[:k],
        'adj_factor': factor,
        'volume':     1e5,
        'amount':     1e7,
    }, index=idx)
    for c in ('open', 'high', 'low', 'close'):
        df[f'adj_{c}'] = df[f'raw_{c}'] * df['adj_factor']
    return df


def make_factor(n=5, assets=None):
    assets = assets or ASSETS
    idx = pd.MultiIndex.from_product([DATES[:n], assets], names=['date', 'asset'])
    k = len(idx)
    return pd.DataFrame({
        'value': np.tile([0.5, 0.3, 0.8], n)[:k],
        'available_at': np.repeat(DATES[:n], len(assets)),
    }, index=idx)


def make_universe(n=5, assets=None):
    assets = assets or ASSETS
    idx = pd.MultiIndex.from_product([DATES[:n], assets], names=['date', 'asset'])
    return pd.DataFrame({'in_universe': True}, index=idx)


# --------------------------------------------------------------------------- #
# 防线 5：已知答案 —— 正常路径必须能过
# --------------------------------------------------------------------------- #
def test_happy_path():
    px = acna.PricePanel(make_prices())
    f = acna.FactorPanel(make_factor())
    cal = acna.Calendar(DATES)
    u = acna.Universe(make_universe())

    assert acna.validate_inputs(f, px, cal, u) is True
    assert len(px) == 15
    assert len(f) == 15
    assert len(cal) == 5


def test_repr_is_readable():
    px = acna.PricePanel(make_prices())
    s = repr(px)
    assert 'PricePanel' in s and '15 行' in s and '3 只' in s


# --------------------------------------------------------------------------- #
# 防线 1：该拒绝的必须拒绝
# --------------------------------------------------------------------------- #
def test_reject_lookahead():
    """前视：available_at 晚于 date。"""
    f = make_factor()
    f.loc[(DATES[0], '600000'), 'available_at'] = DATES[2]   # 未来信息
    with pytest.raises(ContractError) as e:
        acna.FactorPanel(f)
    assert e.value.rule == 'lookahead'
    assert 'available_at' in str(e.value)


def test_reject_duplicate_index():
    f = make_factor()
    f = pd.concat([f, f.iloc[[0]]])
    with pytest.raises(ContractError) as e:
        acna.FactorPanel(f)
    assert e.value.rule == 'index_unique'


def test_reject_non_positive_price():
    px = make_prices()
    px.loc[(DATES[0], '600000'), 'raw_close'] = -1.0
    with pytest.raises(ContractError) as e:
        acna.PricePanel(px)
    assert e.value.rule == 'positive'
    assert 'raw_close' in str(e.value)


def test_reject_missing_adj_factor():
    """只给复权价、不给因子 —— 换不了口径。"""
    px = make_prices().drop(columns=['adj_factor'])
    with pytest.raises(ContractError) as e:
        acna.PricePanel(px)
    assert e.value.rule == 'missing_columns'


def test_reject_adj_without_raw():
    """只给复权价、不给原始价 —— 判不了涨跌停。"""
    px = make_prices().drop(columns=[c for c in make_prices().columns
                                     if c.startswith('raw_') and c != 'raw_close'])
    px = px.rename(columns={'raw_close': 'close'})
    with pytest.raises(ContractError) as e:
        acna.PricePanel(px)
    assert e.value.rule in ('missing_columns', 'adjust_incomplete')


def test_reject_adj_not_self_consistent():
    """adj_close ≠ raw_close × adj_factor。"""
    px = make_prices()
    px['adj_close'] = px['adj_close'] * 1.01      # 人为破坏自洽
    with pytest.raises(ContractError) as e:
        acna.PricePanel(px)
    assert e.value.rule == 'adjust_self_consistent'
    assert 'adj_close' in str(e.value)


def test_reject_off_calendar():
    """数据里出现非交易日。"""
    f = make_factor()
    px = make_prices()
    cal = acna.Calendar(DATES[:3])                # 只给前 3 天
    with pytest.raises(ContractError) as e:
        acna.validate_inputs(f, px, cal)
    assert e.value.rule == 'off_calendar'


def test_reject_factor_not_in_prices():
    """因子里有行情里没有的股票。"""
    f = make_factor()
    px = make_prices(assets=['600000', '000001'])     # 少一只
    with pytest.raises(ContractError) as e:
        acna.validate_inputs(f, px, acna.Calendar(DATES))
    assert e.value.rule == 'factor_not_in_prices'


def test_reject_survivorship():
    """生存者偏差：因子样本不在 as-of 股票池内。"""
    u = make_universe()
    u.loc[(DATES[0], '600000'), 'in_universe'] = False   # 当天不该在池里
    with pytest.raises(ContractError) as e:
        acna.validate_inputs(acna.FactorPanel(make_factor()),
                             acna.PricePanel(make_prices()),
                             acna.Calendar(DATES), u)
    assert e.value.rule == 'factor_outside_universe'
    assert '生存者偏差' in str(e.value)


def test_reject_bad_index_names():
    f = make_factor()
    f.index = f.index.set_names(['dt', 'code'])
    with pytest.raises(ContractError) as e:
        acna.FactorPanel(f)
    assert e.value.rule == 'index_names'


def test_reject_tz_aware():
    f = make_factor()
    d = f.index.get_level_values('date').tz_localize('Asia/Shanghai')
    f.index = pd.MultiIndex.from_arrays(
        [d, f.index.get_level_values('asset')], names=['date', 'asset'])
    with pytest.raises(ContractError) as e:
        acna.FactorPanel(f)
    assert e.value.rule == 'timezone'


# --------------------------------------------------------------------------- #
# Calendar：自持日历，不依赖 pandas freq
# --------------------------------------------------------------------------- #
def test_calendar_shift():
    cal = acna.Calendar(DATES)
    assert cal.shift(DATES[0], 2) == DATES[2]
    assert cal.shift(DATES[4], -1) == DATES[3]
    assert cal.shift(DATES[4], 1) is None            # 越界
    assert cal.next(DATES[0]) == DATES[1]
    assert cal.prev(DATES[2]) == DATES[1]


def test_calendar_works_on_non_daily():
    """★ 关键回归：非日频索引不得崩（alphalens 的 D1）。"""
    monthly = pd.DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-29'])
    cal = acna.Calendar(monthly)
    assert len(cal) == 3
    assert cal.shift(monthly[0], 1) == monthly[1]
    # 用月频因子 + 月频日历跑通契约（alphalens 在这里抛 freq 错）
    idx = pd.MultiIndex.from_product([monthly, ASSETS], names=['date', 'asset'])
    f = pd.DataFrame({'value': 1.0, 'available_at': np.repeat(monthly, 3)}, index=idx)
    assert len(acna.FactorPanel(f)) == 9


# --------------------------------------------------------------------------- #
# 防线 3：两口径收益必须一致（D5 的副产品）
# --------------------------------------------------------------------------- #
def test_adjust_agreement_passes():
    """同一份数据两种常数缩放 → 收益完全相同。"""
    idx = pd.MultiIndex.from_product([DATES, ['600000']], names=['date', 'asset'])
    a = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=idx)
    b = a * 7.3                                   # 任意常数缩放
    r = acna.check_adjust_agreement(a, b)
    assert r['ok'] and r['n_bad'] == 0
    assert r['max_diff'] < 1e-12


def test_adjust_agreement_catches_bad_data():
    """两口径收益不一致 → 抛错。"""
    idx = pd.MultiIndex.from_product([DATES, ['600000']], names=['date', 'asset'])
    a = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4], index=idx)
    b = pd.Series([1.0, 1.1, 1.5, 1.3, 1.4], index=idx)   # 中间被人为改动
    with pytest.raises(ContractError) as e:
        acna.check_adjust_agreement(a, b)
    assert e.value.rule == 'two_sources_differ'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))


# ── 2026-09-25 审计：自洽校验的 NaN 洞 ─────────────────────────────
def test_reject_adj_missing_where_raw_exists():
    """★ 原始价在、复权价缺 —— 必须拦下。

    判据是 `diff > tol`，而 `NaN > tol` 恒为 False，于是缺行此前被"自洽"放行。
    后果不只是漏检：体检层算复权收益时**前值填充**，缺失行算出 0% 收益，
    再被读成"复权后正常（复权价连续）"→ 假 all-clear。
    """
    px = make_prices()
    px.loc[(DATES[2], '600000'), 'adj_close'] = np.nan
    with pytest.raises(ContractError) as e:
        acna.PricePanel(px)
    assert e.value.rule == 'adjust_incomplete'
    assert 'adj_close' in str(e.value)


def test_reject_factor_missing_where_raw_exists():
    """同一个洞的另一半：`adj_factor` 缺行时 `expect` 也是 NaN，照样放行过。"""
    px = make_prices()
    px.loc[(DATES[2], '600000'), 'adj_factor'] = np.nan
    with pytest.raises(ContractError) as e:
        acna.PricePanel(px)
    assert e.value.rule == 'adjust_incomplete'
    assert 'adj_factor' in str(e.value)


def test_suspension_missing_on_both_sides_is_allowed():
    """反向：停牌日 raw 与 adj **两边都缺** —— 规格里的正常表示，不许拦。"""
    px = make_prices()
    row = (DATES[2], '600000')
    for c in ('raw_open', 'raw_high', 'raw_low', 'raw_close',
              'adj_open', 'adj_high', 'adj_low', 'adj_close'):
        px.loc[row, c] = np.nan
    acna.PricePanel(px)                      # 不抛异常即通过


# ── 2026-09-25 审计：±inf 必须拦下，NaN 必须放行 ───────────────────
def _mk_price_edge():
    dates = pd.bdate_range('2024-01-02', periods=3)
    idx = pd.MultiIndex.from_product([dates, ['600000']], names=['date', 'asset'])
    df = pd.DataFrame({c: [10.0, 10.3, 9.9] for c in
                       ['raw_open', 'raw_high', 'raw_low', 'raw_close']}, index=idx)
    df['prev_close'] = [10.0, 10.0, 10.0]
    df['adj_factor'] = [1.0] * 3
    df['volume'] = 1e5
    for c in ('open', 'high', 'low', 'close'):
        df[f'adj_{c}'] = df[f'raw_{c}']
    return df


def test_reject_positive_inf_price():
    """★ ±inf 不是合法数值（`_check_positive` 判的是 `<= 0`，+inf 会穿过去）。"""
    df = _mk_price_edge()
    df.loc[df.index[1], 'raw_close'] = np.inf
    with pytest.raises(ContractError) as e:
        acna.PricePanel(df)
    assert e.value.rule == 'non_finite'
    assert 'raw_close' in str(e.value)


def test_reject_negative_inf_factor():
    df = _mk_price_edge()
    df.loc[df.index[1], 'adj_factor'] = -np.inf
    with pytest.raises(ContractError) as e:
        acna.PricePanel(df)
    assert e.value.rule == 'non_finite'


def test_nan_is_still_a_legal_missing_value():
    """反向：NaN 是规格里的缺失表示 —— 停牌两边都缺、或单边缺，都不许因为
    「非有限」被拦（拦的是 inf，不是 NaN）。"""
    df = _mk_price_edge()
    row = df.index[1]
    for c in ('raw_close', 'adj_close'):
        df.loc[row, c] = np.nan
    acna.PricePanel(df)                      # 停牌：两边都缺 → 放行

    df2 = _mk_price_edge()
    df2.loc[df2.index[1], 'adj_close'] = np.nan      # 单边缺 → 由 adjust_incomplete 管
    with pytest.raises(ContractError) as e:
        acna.PricePanel(df2)
    assert e.value.rule == 'adjust_incomplete'
