"""成交模型与前向收益测试 —— 防线 5（已知答案）+ 口径回归。

重点：
* **三列并列**（可获取 / 跳空 / 合计）算得对，且 ``total = (1+fwd)(1+gap) − 1``
* ``entry='next_open'`` 用的是 **T+1 开盘**，不是 T 收盘
* 一字涨停买不进 → 该样本被剔除并计入原因账
* **退化配置**（收盘成交）复刻 alphalens 口径 —— P0 对拍基线
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
from alphalens_cna.engine.returns import ReturnModel, forward_returns
from alphalens_cna.engine.tradability import compute_tradability

D = pd.to_datetime
ASSET = '600000'


# --------------------------------------------------------------------------- #
def make_prices(rows, adj_factor=1.0):
    """rows: [(date, open, close)]"""
    idx = pd.MultiIndex.from_arrays(
        [D([r[0] for r in rows]), [ASSET] * len(rows)], names=['date', 'asset'])
    raw_open = np.array([r[1] for r in rows], dtype=float)
    raw_close = np.array([r[2] for r in rows], dtype=float)
    df = pd.DataFrame({'raw_open': raw_open, 'raw_close': raw_close,
                       'raw_high': np.maximum(raw_open, raw_close),
                       'raw_low': np.minimum(raw_open, raw_close),
                       'prev_close': np.r_[np.nan, raw_close[:-1]],
                       'adj_factor': adj_factor}, index=idx)
    for c in ('open', 'close', 'high', 'low'):
        df[f'adj_{c}'] = df[f'raw_{c}'] * df['adj_factor']
    return df


DATES = D(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08'])
CAL = acna.Calendar(DATES)


# --------------------------------------------------------------------------- #
# 防线 5：已知答案
# --------------------------------------------------------------------------- #
def test_known_answer_next_open():
    """T 日信号 → T+1 开盘买 → 持有 1 日 → T+2 开盘卖。

    open:  10, 11, 13, 15, 17
    close: 10, 12, 14, 16, 18
    """
    px = make_prices([(d, o, o + 0) for d, o in zip(DATES, [10, 11, 13, 15, 17])])
    px['raw_close'] = [10, 12, 14, 16, 18]
    px['adj_close'] = px['raw_close'] * px['adj_factor']
    r = forward_returns(px, CAL, [1], model=ReturnModel(entry='next_open'))

    # 信号日 1/2：T+1 = 1/3 开盘 11 买入，持有 1 日 → 1/4 开盘 13 卖出
    row = r.df.loc[(DATES[0], ASSET)]
    assert row['entry_date'] == DATES[1]
    assert row['exit_date_1'] == DATES[2]
    assert abs(row['forward_return_1'] - (13 / 11 - 1)) < 1e-12
    # 跳空 = T+1 开盘 / T 收盘 − 1 = 11/10 − 1
    assert abs(row['overnight_gap_1'] - (11 / 10 - 1)) < 1e-12
    # 合计 = (1+fwd)(1+gap) − 1 = 13/10 − 1
    assert abs(row['total_return_1'] - (13 / 10 - 1)) < 1e-12


def test_total_equals_composition():
    px = make_prices([(d, o, o * 1.1) for d, o in zip(DATES, [10, 11, 13, 15, 17])])
    r = forward_returns(px, CAL, [1, 2])
    for h in (1, 2):
        f = r.df[f'forward_return_{h}']
        g = r.df[f'overnight_gap_{h}']
        t = r.df[f'total_return_{h}']
        ok = f.notna() & g.notna()
        assert np.allclose(t[ok], (1 + f[ok]) * (1 + g[ok]) - 1), h


def test_entry_is_t_plus_1_not_t():
    """★ 回归：入场必须是 T+1，不是 T —— 这正是 alphalens 的口径问题。"""
    px = make_prices([(d, o, o) for d, o in zip(DATES, [10, 11, 13, 15, 17])])
    r = forward_returns(px, CAL, [1], model=ReturnModel(entry='next_open'))
    row = r.df.loc[(DATES[0], ASSET)]
    # 若错误地用 T 日开盘，forward 会是 11/10−1；正确应是 13/11−1
    assert abs(row['forward_return_1'] - (13 / 11 - 1)) < 1e-12
    assert abs(row['forward_return_1'] - (11 / 10 - 1)) > 1e-6


def test_degenerate_close_mode_matches_alphalens():
    """退化配置：收盘买、收盘卖 —— 与 alphalens 的 pct_change(h).shift(-h) 同口径。

    ⚠️ 这里 open 与 close **必须不同** —— 否则测不出"入场价取错列"的 bug
    （早先实现无论哪种 entry 都取 adj_open，用 open==close 的数据会被掩盖）。
    """
    closes = [10.0, 12.0, 14.0, 16.0, 18.0]
    px = make_prices([(d, c * 0.9, c) for d, c in zip(DATES, closes)])
    r = forward_returns(px, CAL, [1, 2],
                        model=ReturnModel(entry='close', exit_price='close'))
    s = pd.Series(closes, index=DATES)
    manual = s.pct_change(1).shift(-1)          # alphalens 口径
    got = r.df['forward_return_1'].droplevel('asset')
    assert np.allclose(got.values, manual.values, equal_nan=True), \
        (got.values, manual.values)
    # 收盘成交没有跳空暴露
    assert (r.df['overnight_gap_1'].dropna() == 0).all()
    # 入场价必须是收盘价：若误用开盘价，forward 会变成 close/0.9open 那一套
    assert abs(r.df['forward_return_1'].iloc[0] - (12.0 / 10.0 - 1)) < 1e-12


def test_weekend_is_skipped():
    px = make_prices([(d, o, o) for d, o in zip(DATES, [10, 11, 13, 15, 17])])
    r = forward_returns(px, CAL, [1], model=ReturnModel(entry='next_open'))
    d = r.df.reset_index()
    d = d[d['asset'] == ASSET].set_index('date')
    # 1/5(周五) 的入场日是 1/8(周一)，不是 1/6(周六)
    assert pd.Timestamp(d.loc[DATES[3], 'entry_date']) == DATES[4]


# --------------------------------------------------------------------------- #
# 可成交性联动
# --------------------------------------------------------------------------- #
def test_limit_up_entry_is_skipped():
    """入场日一字涨停 → 买不进 → 该样本剔除，并计入原因账。"""
    rows = [(DATES[0], 10.0, 10.0), (DATES[1], 11.0, 11.0),   # 1/3 一字涨停
            (DATES[2], 12.0, 12.0), (DATES[3], 12.0, 12.0)]
    px = make_prices(rows)
    px['prev_close'] = [np.nan, 10.0, 11.0, 12.0]
    t = compute_tradability(px, names=pd.Series('正常股', index=px.index),
                            new_stock_days=0)
    assert t.loc[(DATES[1], ASSET), 'open_at_limit_up']

    r = forward_returns(px, CAL, [1], tradability=t,
                        model=ReturnModel(entry='next_open', policy='skip'))
    row = r.df.loc[(DATES[0], ASSET)]          # 入场日正是 1/3
    assert not row['entry_tradable']
    assert row['entry_reason'] == 'limit_up'
    assert np.isnan(row['forward_return_1'])
    assert r.ledger[1]['limit_up'] == 1
    assert r.ledger[1]['dropped'] >= 1


def test_mark_only_keeps_the_sample():
    rows = [(DATES[0], 10.0, 10.0), (DATES[1], 11.0, 11.0),
            (DATES[2], 12.0, 12.0), (DATES[3], 12.0, 12.0)]
    px = make_prices(rows)
    px['prev_close'] = [np.nan, 10.0, 11.0, 12.0]
    t = compute_tradability(px, names=pd.Series('正常股', index=px.index),
                            new_stock_days=0)
    r = forward_returns(px, CAL, [1], tradability=t,
                        model=ReturnModel(entry='next_open', policy='mark_only'))
    row = r.df.loc[(DATES[0], ASSET)]
    assert row['entry_reason'] == 'limit_up'      # 标记保留
    assert np.isfinite(row['forward_return_1'])   # 收益照算


def test_no_tradability_warns_by_being_permissive():
    """不给可成交性时只按"有价格"判 —— 能被调用，但会保留一字板样本。"""
    px = make_prices([(d, o, o) for d, o in zip(DATES, [10, 11, 13, 15, 17])])
    r = forward_returns(px, CAL, [1])
    assert r.df['entry_tradable'].sum() >= 3


# --------------------------------------------------------------------------- #
# 复权 / 边界
# --------------------------------------------------------------------------- #
def test_uses_adjusted_prices_not_raw():
    """★ 回归：收益必须用复权价 —— 原始价在除权日会假跳变。"""
    # 10 送 10：1/4 起价格减半
    rows = [(DATES[0], 10.0, 10.0), (DATES[1], 10.0, 10.0),
            (DATES[2], 5.0, 5.0), (DATES[3], 5.0, 5.0), (DATES[4], 5.0, 5.0)]
    px = make_prices(rows)
    # 除权前因子 0.5，除权后 1.0（前复权）
    px['adj_factor'] = [0.5, 0.5, 1.0, 1.0, 1.0]
    for c in ('open', 'close', 'high', 'low'):
        px[f'adj_{c}'] = px[f'raw_{c}'] * px['adj_factor']
    r = forward_returns(px, CAL, [1], model=ReturnModel(entry='next_open'))
    # 信号 1/3 → 入场 1/4 开盘 5×1.0，出场 1/5 开盘 5×1.0 → 收益 0（不是 −50%）
    row = r.df.loc[(DATES[1], ASSET)]
    assert abs(row['forward_return_1']) < 1e-12, row['forward_return_1']


def test_rejects_missing_adj():
    px = make_prices([(d, o, o) for d, o in zip(DATES, [10, 11, 13, 15, 17])])
    with pytest.raises(ContractError) as e:
        forward_returns(px.drop(columns=['adj_open', 'adj_close']), CAL, [1])
    assert e.value.rule == 'missing_column'


def test_rejects_bad_model():
    with pytest.raises(ContractError) as e:
        ReturnModel(entry='xxx')
    assert e.value.rule == 'bad_entry'
    with pytest.raises(ContractError) as e:
        ReturnModel(policy='xxx')
    assert e.value.rule == 'bad_policy'


def test_rejects_bad_horizons():
    px = make_prices([(d, o, o) for d, o in zip(DATES, [10, 11, 13, 15, 17])])
    with pytest.raises(ContractError) as e:
        forward_returns(px, CAL, [0])
    assert e.value.rule == 'bad_horizons'


def test_tail_rows_are_nan():
    """持有期越界的末几行应留空，不能硬凑数。"""
    px = make_prices([(d, o, o) for d, o in zip(DATES, [10, 11, 13, 15, 17])])
    r = forward_returns(px, CAL, [2])
    d = r.df.reset_index()
    d = d[d['asset'] == ASSET].set_index('date')
    # h=2 且 entry=next_open：信号日后还要 1 天入场 + 2 天持有 = 3 个交易日，
    # 5 天日历下只有前两个信号日（1/2、1/3）够，其余留空。
    assert np.isfinite(d.loc[DATES[0], 'forward_return_2'])
    assert np.isfinite(d.loc[DATES[1], 'forward_return_2'])
    for i in (2, 3, 4):
        assert np.isnan(d.loc[DATES[i], 'forward_return_2']), DATES[i]
    # 越界处 exit_date 也应为空，而不是编一个日期出来
    assert pd.isna(pd.Timestamp(d.loc[DATES[4], 'exit_date_2']))
    assert pd.isna(pd.Timestamp(d.loc[DATES[2], 'exit_date_2']))


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
