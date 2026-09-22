"""可成交性测试 —— 防线 5（已知答案）+ 对 execution.py 缺口的回归。

重点锁住：
* **北交所 30%** —— ``execution.py`` 的 ``board_pct`` 不认北交所，会返回 10%
* **必须用原始价** —— 用复权价算涨停价会差 23%
* **eps 用 1e-6 而非 0.01** —— 后者会把"低于涨停价 1 分"误判为买不进
* **limit_status 3/6 才是一字板** —— 2/5 是盘中封板，开盘完全能成交
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
from alphalens_cna.engine.tradability import (
    board_pct, compute_tradability, is_st_name, limit_prices, limit_ratio_of,
)

D = pd.to_datetime


# --------------------------------------------------------------------------- #
# 防线 5：板块比例 —— 手算得出来
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('code,expect', [
    ('600000', 0.10), ('000001', 0.10), ('002415', 0.10),      # 主板
    ('300750', 0.20), ('301029', 0.20), ('302132', 0.20),      # 创业板
    ('688981', 0.20), ('689009', 0.20),                        # 科创板
    # ★ execution.py 缺的：北交所应为 30%，它返回 10%
    ('430047', 0.30), ('832317', 0.30), ('833874', 0.30),
    ('871981', 0.30), ('889999', 0.30), ('920305', 0.30),
])
def test_board_pct(code, expect):
    assert board_pct(code) == expect, f'{code} 应为 {expect}'


def test_northbound_exchange_is_30pct():
    """★ 回归：北交所必须是 30%，不是 10%。"""
    up, dn = limit_prices(10.00, '832317')
    assert (up, dn) == (13.00, 7.00), (up, dn)
    # 对照：主板同样前收
    up, dn = limit_prices(10.00, '600000')
    assert (up, dn) == (11.00, 9.00)


# --------------------------------------------------------------------------- #
# ST
# --------------------------------------------------------------------------- #
def test_is_st_name():
    assert is_st_name('ST中天') is True
    assert is_st_name('*ST艾格') is True
    assert is_st_name('SST前锋') is True
    assert is_st_name('S*ST龙昌') is True
    assert is_st_name('平安银行') is False
    assert is_st_name('华ST科技') is False      # 仅前缀判定
    assert is_st_name(None) is False
    assert is_st_name(float('nan')) is False


def test_st_limit_by_board():
    """主板 ST 5%；创业/科创 ST **仍 20%**；北交所 ST 仍 30%。"""
    assert limit_ratio_of('600000', 'ST中天') == 0.05
    assert limit_ratio_of('000001', '*ST某') == 0.05
    assert limit_ratio_of('300750', '*ST某') == 0.20      # 创业板不变
    assert limit_ratio_of('688981', 'ST某') == 0.20       # 科创板不变
    assert limit_ratio_of('832317', 'ST某') == 0.30       # 北交所不变

    up, dn = limit_prices(10.00, '600000', name='ST中天')
    assert (up, dn) == (10.50, 9.50)


def test_limit_price_rounding():
    """四舍五入到分。"""
    assert limit_prices(3.33, '600000') == (3.66, 3.00)
    assert limit_prices(0, '600000') == (None, None)
    assert limit_prices(None, '600000') == (None, None)
    assert limit_prices(10.0, '600000', unlimited=True) == (None, None)


# --------------------------------------------------------------------------- #
# 面板级判定
# --------------------------------------------------------------------------- #
def make_panel(rows, asset='600000'):
    """rows: [(date, prev_close, open)]"""
    idx = pd.MultiIndex.from_arrays(
        [D([r[0] for r in rows]), [asset] * len(rows)], names=['date', 'asset'])
    return pd.DataFrame({
        'raw_open':  [r[2] for r in rows],
        'raw_close': [r[2] for r in rows],
        'prev_close': [r[1] for r in rows],
    }, index=idx)


def test_limit_up_blocks_buy():
    """一字涨停：开盘 = 涨停价 → 买不进；跌停 → 卖不出。"""
    p = make_panel([
        ('2024-01-02', 10.00, 10.00),
        ('2024-01-03', 10.00, 11.00),      # 涨停
        ('2024-01-04', 11.00, 11.00),
    ])
    t = compute_tradability(p, names=pd.Series('正常股', index=p.index), new_stock_days=0)
    assert t.loc[(D('2024-01-03'), '600000'), 'open_at_limit_up']
    assert not t.loc[(D('2024-01-03'), '600000'), 'can_buy_open']
    assert t.loc[(D('2024-01-03'), '600000'), 'can_sell_open']
    assert t.loc[(D('2024-01-03'), '600000'), 'reason_buy'] == 'limit_up'


def test_eps_is_tiny_not_one_cent():
    """★ 回归：低于涨停价 1 分**可以**买 —— 用 0.01 容差会误判。"""
    p = make_panel([
        ('2024-01-02', 10.00, 10.00),
        ('2024-01-03', 10.00, 10.99),      # 涨停价 11.00，差 1 分
    ])
    t = compute_tradability(p, names=pd.Series('正常股', index=p.index), new_stock_days=0)
    row = t.loc[(D('2024-01-03'), '600000')]
    assert not row['open_at_limit_up'], '低于涨停价 1 分不应算封板'
    assert row['can_buy_open']


def test_suspended():
    p = make_panel([
        ('2024-01-02', 10.00, 10.00),
        ('2024-01-03', 10.00, np.nan),     # 停牌
    ])
    t = compute_tradability(p, names=pd.Series('正常股', index=p.index), new_stock_days=0)
    row = t.loc[(D('2024-01-03'), '600000')]
    assert row['suspended'] and not row['can_buy_open'] and not row['can_sell_open']
    assert row['reason_buy'] == 'suspended'


def test_st_uses_pit_name():
    """同样前收 10 元，ST 那天涨停价是 10.50 而非 11.00。"""
    p = make_panel([
        ('2024-01-02', 10.00, 10.00),
        ('2024-01-03', 10.00, 10.50),
        ('2024-01-04', 10.00, 10.50),
    ])
    nm = pd.Series('正常股', index=p.index)
    nm.loc[(D('2024-01-03'), '600000')] = 'ST某某'      # 仅那天是 ST
    t = compute_tradability(p, names=nm, new_stock_days=0)
    assert t.loc[(D('2024-01-03'), '600000'), 'is_st']
    assert t.loc[(D('2024-01-03'), '600000'), 'limit_up_price'] == 10.50
    assert t.loc[(D('2024-01-03'), '600000'), 'open_at_limit_up']
    # 前一天不是 ST，涨停价仍是 11.00
    assert t.loc[(D('2024-01-02'), '600000'), 'limit_up_price'] == 11.00


def _newstock_panel(n=70, start='2024-01-02', with_calendar=True):
    dates = pd.bdate_range(start, periods=n)
    idx = pd.MultiIndex.from_arrays([dates, ['600000'] * n], names=['date', 'asset'])
    p = pd.DataFrame({'raw_open': 10.0, 'raw_close': 10.0, 'prev_close': 10.0}, index=idx)
    cal = acna.Calendar(dates) if with_calendar else None
    return p, dates, idx, cal


def test_new_stock_window():
    """给了上市日：前 60 个交易日算新股 → 不可买。"""
    p, dates, idx, cal = _newstock_panel()
    t = compute_tradability(p, names=pd.Series('新股', index=idx), calendar=cal,
                            list_dates={'600000': dates[0]}, new_stock_days=60)
    assert t['listed_days'].iloc[0] == 1
    assert t['is_new_stock'].iloc[59], '第 60 日仍是新股'
    assert not t['is_new_stock'].iloc[60], '第 61 日起不是'
    assert t['reason_buy'].iloc[0] == 'new_stock'
    assert not t['can_buy_open'].iloc[0]
    assert t['can_sell_open'].iloc[0], '新股只是不能买，持有仍可卖'


def test_new_stock_days_default_is_60():
    p, dates, idx, cal = _newstock_panel()
    t = compute_tradability(p, names=pd.Series('新股', index=idx), calendar=cal,
                            list_dates={'600000': dates[0]})    # 不传阈值
    assert t['is_new_stock'].iloc[59] and not t['is_new_stock'].iloc[60]


def test_new_stock_filter_skipped_without_list_dates():
    """★ 安全默认：上市日未知时**不做**新股过滤。

    否则截断面板（如只取一天）会把所有股票判成"第 1 天"，
    进而**静默排除全部数据** —— 比"不排除"危险得多。
    """
    p, dates, idx, cal = _newstock_panel(n=1)
    t = compute_tradability(p, names=pd.Series('新股', index=idx))
    assert not t['is_new_stock'].any(), '上市日未知时不应判新股'
    assert t['can_buy_open'].all()
    assert not t['listed_days_known'].any()
    assert 'new_stock_filter' in t.attrs


def test_listed_days_known_flag():
    p, dates, idx, cal = _newstock_panel(n=5)
    t1 = compute_tradability(p, names=pd.Series('x', index=idx))
    assert not t1['listed_days_known'].any()
    t2 = compute_tradability(p, names=pd.Series('x', index=idx),
                             list_dates={'600000': dates[0]},
                             calendar=acna.Calendar(dates))
    assert t2['listed_days_known'].all()
    # 给日历 → 按交易日精确数；不给 → 自然日折算（仍是"已知"）
    t3 = compute_tradability(p, names=pd.Series('x', index=idx),
                             list_dates={'600000': dates[0]})
    assert t3['listed_days_known'].all()
    # 无日历是自然日折算，天数会有偏差 —— 但只要"上市日已知"过滤就启用
    assert t3['listed_days'].iloc[0] == 1


# --------------------------------------------------------------------------- #
# limit_status 降级路径
# --------------------------------------------------------------------------- #
def test_limit_status_3_blocks_buy_2_does_not():
    """★ 实测语义：3 = 一字涨停（买不进），2 = 盘中封板（**开盘能买**）。"""
    p = make_panel([
        ('2024-01-02', 10.00, 10.00),
        ('2024-01-03', 10.00, 10.00),      # 收盘涨停但开盘没封
        ('2024-01-04', 10.00, 10.00),
    ])
    ls = pd.Series([1.0, 2.0, 3.0], index=p.index)
    t = compute_tradability(p, limit_status=ls, names=None, new_stock_days=0)
    assert not t['open_at_limit_up'].iloc[1], 'ls=2（盘中封板）开盘应可买'
    assert t['open_at_limit_up'].iloc[2], 'ls=3（一字板）开盘买不进'
    # 只有 ls ∈ {3,6} 才触发降级查表；ls=2 无可降级，仍标 price
    assert t['calc_source'].iloc[2] == 'limit_status'
    assert t['calc_source'].iloc[1] == 'price'


def test_limit_status_6_blocks_sell():
    p = make_panel([('2024-01-02', 10.00, 10.00), ('2024-01-03', 10.00, 10.00)])
    ls = pd.Series([1.0, 6.0], index=p.index)
    t = compute_tradability(p, limit_status=ls, names=None, new_stock_days=0)
    assert not t['can_sell_open'].iloc[1]
    assert t['reason_sell'].iloc[1] == 'limit_down'


def test_limit_status_0_is_suspended():
    p = make_panel([('2024-01-02', 10.00, 10.00), ('2024-01-03', 10.00, 10.00)])
    ls = pd.Series([1.0, 0.0], index=p.index)
    t = compute_tradability(p, limit_status=ls, names=None, new_stock_days=0)
    assert t['suspended'].iloc[1]


def test_self_computed_when_names_available():
    """有 PIT 名称时走自算（零前视），不查表。"""
    p = make_panel([('2024-01-02', 10.00, 10.00)])
    ls = pd.Series([6.0], index=p.index)      # 故意给一个会误判的值
    t = compute_tradability(p, limit_status=ls, names=pd.Series('正常股', index=p.index), new_stock_days=0)
    assert t['calc_source'].iloc[0] == 'price'


# --------------------------------------------------------------------------- #
# 必须用原始价
# --------------------------------------------------------------------------- #
def test_rejects_missing_raw_open():
    idx = pd.MultiIndex.from_arrays([[D('2024-01-02')], ['600000']], names=['date', 'asset'])
    p = pd.DataFrame({'adj_open': 10.0, 'raw_close': 10.0}, index=idx)
    with pytest.raises(ContractError) as e:
        compute_tradability(p)
    assert e.value.rule == 'missing_column'
    assert '原始不复权' in str(e.value)


def test_raw_and_adj_give_different_limit_prices():
    """★ 回归：用复权价算涨停价会严重偏离。

    600519 实盘：2016-08-26 涨停价 333.31（用原始前收 303.01 算）。
    若误用当时的前复权前收 245.46，会算成 270.01 —— **差 23%**。
    """
    raw_prev, adj_prev = 303.01, 245.4612
    up_raw, _ = limit_prices(raw_prev, '600519')
    up_adj, _ = limit_prices(adj_prev, '600519')
    assert up_raw == 333.31
    assert up_adj == 270.01
    assert abs(up_raw - up_adj) / up_raw > 0.18, '差距应有 ~19%'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
