"""体检层测试 —— 防线 5（已知答案）。

每条检查都用**能手算的小数据**验一遍：故意造一个错，看它抓不抓得住；
再造一份干净数据，看它会不会误报。**误报和漏报一样是 bug。**
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna  # noqa: E402
from alphalens_cna.health import DEFAULTS, Finding, HealthReport  # noqa: E402

DATES = pd.bdate_range('2024-01-01', periods=60)
ASSETS = [f'{i:06d}' for i in range(10)]


def mk_prices(dates=DATES, assets=ASSETS, volume=1000.0):
    """干净面板：所有价 = 10 元，复权因子 = 1，收益恒为 0。

    这样任何告警都必然是"造出来的错"，不会跟数据噪声混淆。
    """
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    df = pd.DataFrame(index=idx)
    for c, col in (('raw_open', 'o'), ('raw_high', 'h'),
                   ('raw_low', 'l'), ('raw_close', 'c')):
        df[c] = 10.0
    df['prev_close'] = 10.0
    df['adj_factor'] = 1.0
    for c in ('open', 'high', 'low', 'close'):
        df[f'adj_{c}'] = df[f'raw_{c}']
    df['volume'] = volume
    return df


def mk_factor(dates=DATES, assets=ASSETS, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    return pd.DataFrame({
        'value': rng.normal(size=len(idx)),
        'available_at': idx.get_level_values('date') - pd.Timedelta(days=30),
    }, index=idx)


def mk_trad(dates=DATES, assets=ASSETS):
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    return pd.DataFrame({
        'suspended': False, 'is_st': False, 'listed_days': 500,
        'listed_days_known': True, 'is_new_stock': False,
        'open_at_limit_up': False, 'open_at_limit_down': False,
        'can_buy_open': True, 'can_sell_open': True,
    }, index=idx)


def find(rep, name):
    got = [f for f in rep.findings if f.name == name]
    assert got, f'报告里没有 {name}：{[f.name for f in rep.findings]}'
    return got[0]


def clean_prices():
    """一只"善终"的面板：两只票在样本中途结束（模拟退市）。"""
    df = mk_prices()
    d = df.index.get_level_values('date')
    a = df.index.get_level_values('asset')
    drop = np.isin(a, ['000008', '000009']) & (d >= DATES[30])
    return df[~drop]


# --------------------------------------------------------------------------- #
# 总入口语义
# --------------------------------------------------------------------------- #
def test_empty_report_is_not_ok():
    """★ "没查" 不等于 "没问题" —— 空报告必须判未通过。"""
    rep = acna.health_check()
    assert rep.ok is False
    assert rep.n_checked == 0
    assert len(rep.skips) == 6


def test_skips_are_explicit():
    """缺什么输入就记一条 skip，绝不静默省略。"""
    rep = acna.health_check(prices=clean_prices(), calendar=acna.Calendar(DATES))
    names = {f.name for f in rep.skips}
    assert '因子截面' in names and '可成交性' in names
    assert '覆盖率' not in names, '覆盖率给了输入，不该 skip'


def test_clean_panel_has_no_failures():
    """★ 干净数据不许误报硬错误。"""
    rep = acna.health_check(prices=clean_prices(), factor=mk_factor(),
                            tradability=mk_trad(),
                            calendar=acna.Calendar(DATES))
    assert not rep.fails, '\n'.join(str(f) for f in rep.fails)


def test_unknown_threshold_raises():
    """阈值键名写错要报错，不能静默忽略（否则你以为你调了）。"""
    with pytest.raises(ValueError, match='未知阈值'):
        acna.health_check(prices=mk_prices(), thresholds={'no_such_thing': 1})


def test_accepts_contract_objects():
    """传契约对象（不是裸 DataFrame）也要能跑。"""
    rep = acna.health_check(prices=acna.PricePanel(mk_prices()),
                            calendar=acna.Calendar(DATES))
    assert find(rep, 'OHLC 结构').severity == 'pass'


def test_markdown_renders():
    rep = acna.health_check(prices=clean_prices(), factor=mk_factor(),
                            calendar=acna.Calendar(DATES))
    md = rep.to_markdown('测试')
    assert md.startswith('# 测试')
    assert 'OHLC 结构' in md and '存活偏差' in md


# --------------------------------------------------------------------------- #
# 行情类
# --------------------------------------------------------------------------- #
def test_ohlc_violation_is_fail():
    """high < low → 硬错误，且报出行数。"""
    df = mk_prices()
    df.iloc[5, df.columns.get_loc('raw_high')] = 9.0
    f = find(acna.health_check(prices=df), 'OHLC 结构')
    assert f.severity == 'fail' and f.metric == 1.0


def test_ohlc_clean_is_pass():
    assert find(acna.health_check(prices=mk_prices()), 'OHLC 结构').severity == 'pass'


def test_extreme_move_explained_by_adjustment_is_fail():
    """★ 原始价腰斩但复权后不动 → 断定 `adj_factor` 没盖住除权日。

    构造：某票从第 30 天起 2.5:1 拆股，原始价 10 → 4，同时因子 1 → 2.5。
    复权价全程 10（收益 0），原始收益 −60%（> 50% 阈值）。
    """
    df = mk_prices()
    d = df.index.get_level_values('date')
    a = df.index.get_level_values('asset')
    hit = (a == '000000') & (d >= DATES[30])
    for c in ('raw_open', 'raw_high', 'raw_low', 'raw_close'):
        df.loc[hit, c] = 4.0
    df.loc[hit, 'adj_factor'] = 2.5
    for c in ('open', 'high', 'low', 'close'):
        df[f'adj_{c}'] = df[f'raw_{c}'] * df['adj_factor']
    df['prev_close'] = df.groupby(level='asset')['raw_close'].shift(1).fillna(10.0)

    f = find(acna.health_check(prices=df), '极端涨跌')
    assert f.severity == 'fail', f.summary
    assert f.metric == 1.0, '除权日只有 1 条'
    assert 'adj_factor' in f.summary


def test_adjust_factor_regression_is_fail():
    """复权因子倒退 → 硬错误（等于负分红）。"""
    df = mk_prices()
    d = df.index.get_level_values('date')
    df.loc[(df.index.get_level_values('asset') == '000003') & (d >= DATES[30]),
           'adj_factor'] = 0.5
    f = find(acna.health_check(prices=df), '复权连续性')
    assert f.severity == 'fail' and f.metric == 1.0, '只有因子变化的那一行倒退'


def test_adjust_factor_jump_is_warn():
    """单日因子翻 3 倍 → 告警（可能是送转股，需人看），不是硬错误。"""
    df = mk_prices()
    d = df.index.get_level_values('date')
    df.loc[(df.index.get_level_values('asset') == '000002') & (d >= DATES[10]),
           'adj_factor'] = 3.0
    f = find(acna.health_check(prices=df), '复权连续性')
    assert f.severity == 'warn' and f.metric == 1.0


def test_suspension_rate():
    """某日 60% 停牌 → 告警（疑似整段缺失）。"""
    df = mk_prices()
    d = df.index.get_level_values('date')
    df.loc[d == DATES[20], 'volume'] = 0.0
    # 10 只里 6 只停牌
    a = df.index.get_level_values('asset')
    df.loc[(d == DATES[20]) & (a < '000004'), 'volume'] = 1000.0
    f = find(acna.health_check(prices=df), '停牌率')
    assert f.severity == 'warn'
    assert abs(f.metric - 0.6) < 1e-12


def test_no_volume_column_skips():
    df = mk_prices().drop(columns=['volume'])
    assert find(acna.health_check(prices=df), '停牌率').severity == 'skip'


# --------------------------------------------------------------------------- #
# 面板类
# --------------------------------------------------------------------------- #
def thin(dates):
    """把这些日期的截面砍到只剩 2 只（中位数是 10 只）。"""
    df = mk_prices()
    d = df.index.get_level_values('date')
    a = df.index.get_level_values('asset')
    drop = pd.Series(d, index=df.index).isin(dates).values & (a > '000001')
    return df[~drop]


def test_coverage_thin_cross_section_warns():
    """★ 覆盖率管的是"截面变薄"（整日缺失归日历对齐管）。
    单日只剩 2 只 / 中位 10 只 → 告警。"""
    df = thin([DATES[10]])
    f = find(acna.health_check(prices=df, calendar=acna.Calendar(DATES)),
             '覆盖率')
    assert f.severity == 'warn' and f.metric == 0.2


def test_coverage_gap_widespread_is_fail():
    """超过 5% 的交易日都变薄 → 不是偶发，升级为硬错误。"""
    f = find(acna.health_check(prices=thin(DATES[:10]),
                               calendar=acna.Calendar(DATES)), '覆盖率')
    assert f.severity == 'fail'


def test_survivorship_all_survive_is_warn():
    """★ 全票善终 → 存活偏差告警（这是本层最重要的检查）。"""
    f = find(acna.health_check(prices=mk_prices(),
                               calendar=acna.Calendar(DATES)), '存活偏差')
    assert f.severity == 'warn' and f.metric == 0.0
    assert '没有一只' in f.summary


def test_survivorship_with_delistings_passes():
    """有票中途结束 → 通过，并报出比例 2/10。"""
    f = find(acna.health_check(prices=clean_prices(),
                               calendar=acna.Calendar(DATES)), '存活偏差')
    assert f.severity == 'pass'
    assert abs(f.metric - 0.2) < 1e-12


def test_calendar_alignment_off_calendar_is_fail():
    """面板日期不在交易日历上 → 硬错误。"""
    df = mk_prices()
    idx = df.index
    bad = pd.MultiIndex.from_tuples(
        [(pd.Timestamp('2024-01-06'), '000000')], names=['date', 'asset'])
    df = pd.concat([df, pd.DataFrame(10.0, index=bad, columns=df.columns)])
    f = find(acna.health_check(prices=df, calendar=acna.Calendar(DATES)),
             '日历对齐')
    assert f.severity == 'fail'


def test_interior_hole_warns():
    """日频面板中间缺一天 → 告警。"""
    df = mk_prices()
    d = df.index.get_level_values('date')
    df = df[d != DATES[30]]
    f = find(acna.health_check(prices=df, calendar=acna.Calendar(DATES)),
             '日历对齐')
    assert f.severity == 'warn' and f.metric == 1.0


def test_monthly_panel_not_flagged():
    """★ 月频面板天然"缺"大量交易日 —— 不许误报空洞。"""
    dates = pd.DatetimeIndex(DATES[::21])
    cal = acna.Calendar(DATES)
    df = mk_prices(dates=dates)
    f = find(acna.health_check(prices=df, calendar=cal), '日历对齐')
    assert f.severity == 'info', f.summary


# --------------------------------------------------------------------------- #
# 因子类
# --------------------------------------------------------------------------- #
def test_factor_freeze_detected():
    """某票因子值连续 30 期不变 → 告警。"""
    f_df = mk_factor()
    d = f_df.index.get_level_values('date')
    f_df.loc[f_df.index.get_level_values('asset') == '000001', 'value'] = 7.0
    f = find(acna.health_check(factor=f_df), '因子冻结')
    assert f.severity == 'warn' and f.metric == 60.0


def test_factor_thin_cross_section():
    """某日因子只有一个取值 → 那天算不出 IC。"""
    f_df = mk_factor()
    d = f_df.index.get_level_values('date')
    f_df.loc[d == DATES[5], 'value'] = 1.0
    f = find(acna.health_check(factor=f_df), '因子截面')
    assert f.severity == 'warn' and f.metric == 1.0


def test_factor_staleness_warns_when_stale():
    f_df = mk_factor()
    f_df['staleness_days'] = 900
    assert find(acna.health_check(factor=f_df), '因子陈旧度').severity == 'warn'
    f_df['staleness_days'] = 90
    assert find(acna.health_check(factor=f_df), '因子陈旧度').severity == 'pass'


def test_factor_staleness_skipped_without_column():
    assert find(acna.health_check(factor=mk_factor()), '因子陈旧度').severity == 'skip'


# --------------------------------------------------------------------------- #
# 可成交性
# --------------------------------------------------------------------------- #
def test_tradability_rates_passes():
    f = find(acna.health_check(tradability=mk_trad()), '可成交性')
    assert f.severity == 'pass' and f.metric == 1.0


def test_tradability_zero_buyable_is_fail():
    """某日一只都买不进 → 硬错误。"""
    td = mk_trad()
    d = td.index.get_level_values('date')
    td.loc[d == DATES[7], 'can_buy_open'] = False
    f = find(acna.health_check(tradability=td), '可成交性')
    assert f.severity == 'fail' and f.metric < 1.0


def test_listed_days_unknown_warns():
    """★ 回归：上市日期未知 → 新股过滤没生效，必须告警。

    这正是之前真踩过的坑（整个面板被当成新股，可买 0）。
    """
    td = mk_trad()
    td['listed_days_known'] = False
    f = find(acna.health_check(tradability=td), '新股过滤')
    assert f.severity == 'warn' and '没有生效' in f.summary


def test_listed_days_known_passes():
    assert find(acna.health_check(tradability=mk_trad()),
                '新股过滤').severity == 'pass'


# --------------------------------------------------------------------------- #
# 类型本身
# --------------------------------------------------------------------------- #
def test_bad_severity_rejected():
    with pytest.raises(ValueError):
        Finding('x', 'terrible', 'y')


def test_defaults_are_documented():
    assert 'extreme_move' in DEFAULTS and DEFAULTS['extreme_move'] == 0.5


def test_warnings_list_is_human_readable():
    rep = HealthReport(findings=[Finding('存活偏差', 'warn', '全票善终')])
    assert rep.warnings == ['存活偏差：全票善终']
    assert rep.ok is True, '只有 warn 时不算未通过'
