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


# ── 2026-09-25 审计：calendar 参数两种写法都要收 ──────────────────
def test_calendar_accepts_bare_datetimeindex():
    """★ ``Calendar`` 与裸 ``DatetimeIndex`` 必须等价。

    此前 `forward_returns` / `compute_tradability` / `build_report` /
    `check_parity` 只认前者，传 DatetimeIndex 会抛裸
    ``AttributeError: 'DatetimeIndex' object has no attribute 'index'``；
    而 `health_check` / `align_event_windows` 一直两种都收 —— 同一个参数
    在不同入口接受度不同。现在统一走 `contract.calendar.as_calendar`。
    """
    import numpy as np
    import pandas as pd
    dates = pd.bdate_range('2023-01-02', periods=25)
    assets = [f'{i:06d}' for i in range(6)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    rng = np.random.default_rng(0)
    close = (np.tile(rng.uniform(5, 50, len(assets)), len(dates))
             * np.exp(np.cumsum(rng.normal(0, .01, len(idx)))))
    px = pd.DataFrame({'raw_close': close}, index=idx)
    for c, k in (('open', 1.0), ('high', 1.01), ('low', .99)):
        px[f'raw_{c}'] = close * k
    px['prev_close'] = px.groupby(level='asset')['raw_close'].shift(1).fillna(px['raw_close'])
    px['adj_factor'] = 1.0
    for c in ('open', 'high', 'low', 'close'):
        px[f'adj_{c}'] = px[f'raw_{c}']
    px['volume'] = 1e5

    cal = acna.Calendar(dates)
    a = acna.forward_returns(px, cal, [1, 5]).df
    b = acna.forward_returns(px, dates, [1, 5]).df
    assert a.equals(b), 'Calendar 与 DatetimeIndex 的前向收益不一致'

    t1 = acna.compute_tradability(px, calendar=cal)
    t2 = acna.compute_tradability(px, calendar=dates)
    assert t1.equals(t2), 'Calendar 与 DatetimeIndex 的可成交性不一致'


# ── 2026-09-25：出场侧可成交性（exit_policy）─────────────────────────
def _mk_exit_blocked_panel():
    """12 天；B 在 D[10] 一字跌停 —— 入场 D[9]/出场 D[10] 的那行卖不出。"""
    dates = pd.bdate_range('2024-01-02', periods=12)
    assets = ['600000', '000001']
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    px = pd.DataFrame({'raw_close': 10.0}, index=idx)
    px['raw_open'] = px['raw_high'] = px['raw_low'] = 10.0
    hit = ((px.index.get_level_values('asset') == '000001')
           & (px.index.get_level_values('date') == dates[10]))
    for c in ('raw_open', 'raw_high', 'raw_low', 'raw_close'):
        px.loc[hit, c] = 9.0
    px['prev_close'] = px.groupby(level='asset')['raw_close'].shift(1).fillna(10.0)
    px['adj_factor'] = 1.0
    for c in ('open', 'high', 'low', 'close'):
        px[f'adj_{c}'] = px[f'raw_{c}']
    px['volume'] = 1e5
    return px, dates, (dates[8], '000001')       # 该行的出场日 = dates[10]


def _run_exit(policy, px, dates, **kw):
    cal = acna.Calendar(dates)
    tr = acna.compute_tradability(px, calendar=cal)
    model = acna.ReturnModel(entry='next_open', policy='skip', exit_price='open',
                             exit_policy=policy, **kw)
    return acna.forward_returns(px, cal, [1], tradability=tr, model=model)


def test_exit_policy_assume_keeps_old_behaviour():
    """★ 默认 ``'assume'`` 必须**保持旧口径**：出场日一字跌停照样按当日价成交。

    这是"乐观"的那一支 —— 之所以留作默认，是因为改默认会静默改变所有人的结论。
    """
    px, dates, key = _mk_exit_blocked_panel()
    r = _run_exit('assume', px, dates)
    assert np.isfinite(r.df.loc[key, 'forward_return_1'])
    assert abs(r.df.loc[key, 'forward_return_1'] + 0.10) < 1e-12
    # 显式传 'assume' 与不传（默认）结果逐位相同
    cal = acna.Calendar(dates)
    tr = acna.compute_tradability(px, calendar=cal)
    d = acna.forward_returns(px, cal, [1], tradability=tr,
                             model=acna.ReturnModel(entry='next_open')).df
    assert d.equals(r.df)


def test_exit_policy_drop_removes_unsellable_exit():
    px, dates, key = _mk_exit_blocked_panel()
    r = _run_exit('drop', px, dates)
    assert not np.isfinite(r.df.loc[key, 'forward_return_1']), '卖不出的出场必须剔除'
    assert r.ledger[1].get('exit_not_sellable', 0) >= 1
    assert r.ledger[1]['tradable'] < _run_exit('assume', px, dates).ledger[1]['tradable']


def test_exit_policy_delay_shifts_to_next_sellable_day():
    px, dates, key = _mk_exit_blocked_panel()
    r = _run_exit('delay', px, dates)
    assert r.ledger[1].get('exit_delayed', 0) >= 1
    # 出场日被顺延到跌停的次日；收益按**那一天的**开盘价算（本例价格回到 10 → 0%）
    assert pd.Timestamp(r.df.loc[key, 'exit_date_1']) == dates[11]
    assert abs(r.df.loc[key, 'forward_return_1']) < 1e-12


def test_exit_policy_requires_tradability():
    """★ 没有可成交性数据就判断不出卖不卖得掉 —— 必须报错，不许静默降级。"""
    px, dates, _ = _mk_exit_blocked_panel()
    for policy in ('drop', 'delay'):
        with pytest.raises(ContractError) as e:
            acna.forward_returns(px, acna.Calendar(dates), [1],
                                 model=acna.ReturnModel(exit_policy=policy))
        assert e.value.rule == 'exit_policy_needs_tradability'


def test_exit_policy_invalid_name_rejected():
    with pytest.raises(ContractError) as e:
        acna.ReturnModel(exit_policy='whatever')
    assert e.value.rule == 'bad_exit_policy'


def test_ledger_counts_are_exact_and_dropped_is_consistent():
    """★ 台账的两条**真正成立**的性质。

    ``dropped == total − tradable`` 恒成立；而各原因键**不构成严格划分** ——
    同一行可能同时被入场侧判据与出场侧缺价命中（实测过），所以不拿 Σ 对账
    （这是既有行为，不是本次引入的）。这里钉住的是：
      · ``dropped = total − tradable``
      · ``exit_not_sellable`` **只数因出场不可卖才被剔的行**（切到 drop 时
        tradable 恰好减少这么多）—— 不能把本来就会被别的原因剔掉的行重复计入。
    """
    px, dates, _ = _mk_exit_blocked_panel()
    base = _run_exit('assume', px, dates).ledger[1]
    drop = _run_exit('drop', px, dates).ledger[1]
    for led in (base, drop):
        assert led['dropped'] == led['total'] - led['tradable']
    delta = base['tradable'] - drop['tradable']
    assert delta >= 1, '这个用例里必须真有"因出场不可卖而被剔"的行'
    assert drop.get('exit_not_sellable', 0) == delta, (
        f'exit_not_sellable={drop.get("exit_not_sellable")} 应恰好等于被剔掉的行数 {delta}')


def test_ledger_marks_delays_as_informational_not_dropped():
    """★ ``exit_delayed`` 是**处理说明**，不是剔除原因 —— 那些行仍在样本里。"""
    px, dates, _ = _mk_exit_blocked_panel()
    base = _run_exit('assume', px, dates).ledger[1]
    delay = _run_exit('delay', px, dates).ledger[1]
    assert delay.get('exit_delayed', 0) >= 1
    assert delay['tradable'] == base['tradable'], '顺延不减少样本'
    assert delay['dropped'] == base['dropped']


def test_exit_policy_does_not_override_delist_convention():
    """★ 退市行归 ``delist_policy`` 管，``exit_policy`` **不许碰**。

    退市股在出场日必然"卖不出"（早已没有价格）。若出场策略把它一并当"卖不出"
    剔掉，``delist_policy='last_price'`` 就被整个推翻了 —— 那是 0.1.1 专门修过的
    生存者偏差护栏。实测过：混在一起时 tradable 从 12 掉到 7。
    """
    dates = pd.bdate_range('2024-01-02', periods=12)
    assets = ['600000', '000001']
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    px = pd.DataFrame({'raw_close': 10.0}, index=idx)
    px['raw_open'] = px['raw_high'] = px['raw_low'] = 10.0
    gone = ((px.index.get_level_values('asset') == '000001')
            & (px.index.get_level_values('date') > dates[6]))       # B 退市
    for c in ('raw_open', 'raw_high', 'raw_low', 'raw_close'):
        px.loc[gone, c] = np.nan
    px['prev_close'] = px.groupby(level='asset')['raw_close'].shift(1).fillna(10.0)
    px['adj_factor'] = 1.0
    for c in ('open', 'high', 'low', 'close'):
        px[f'adj_{c}'] = px[f'raw_{c}']
    px['volume'] = 1e5
    cal = acna.Calendar(dates)
    tr = acna.compute_tradability(px, calendar=cal)

    def led(exit_policy):
        r = acna.forward_returns(
            px, cal, [5], tradability=tr,
            model=acna.ReturnModel(entry='next_open', policy='skip',
                                   exit_price='open', exit_policy=exit_policy,
                                   delist_policy='last_price'))
        return r.ledger[5]

    base = led('assume')
    assert base.get('delist_filled', 0) >= 1, '用例前提：真有按退市约定清算的行'
    for policy in ('drop', 'delay'):
        got = led(policy)
        assert got['tradable'] == base['tradable'], (
            f"{policy} 把退市行也剔了：tradable {got['tradable']} != {base['tradable']}")
        assert got.get('delist_filled', 0) == base.get('delist_filled', 0)
        assert got.get('exit_not_sellable', 0) == 0
