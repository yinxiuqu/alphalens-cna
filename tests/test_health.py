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


def test_extreme_move_explained_by_adjustment_is_pass():
    """★ 原始价腰斩但复权后不动 → **复权是对的**，必须判 pass。

    构造：某票从第 30 天起 2.5:1 拆股，原始价 10 → 4，同时因子 1 → 2.5。
    复权价全程 10（收益 0），原始收益 −60%（> 50% 阈值）。

    ⚠️ 2026-09-25 修正：这条测试此前断言 `fail`（"adj_factor 没盖住除权日"），
    方向正好反了 —— `adj = raw × factor`，因子从 1 涨到 2.5 恰好把腰斩抹平，
    这正是**正确处理**除权日的签名。
    真实数据佐证：2016-2022 的 400 只样本报出 38 条，其中 31 条经 `stock_xdxr`
    核对全部是 `category==1` 的正常除权除息日（高送转 10送10/12/15）。
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
    assert f.severity == 'pass', f.summary
    assert f.metric == 1.0, '除权日只有 1 条'
    assert '已覆盖' in f.summary


def test_extreme_move_not_covered_by_adjustment_is_warn():
    """原始价腰斩且**复权后也腰斩** → 因子没跟上（或真实跳变），必须判 warn。

    构造：原始价 10 → 4，且 `adj_factor` **保持 1.0 不变**（等于除权日没更新因子）。
    此时 adj 与 raw 同步下跌，复权价并没有把跳变抹平 —— 这才是该报警的情形。
    """
    df = mk_prices()
    d = df.index.get_level_values('date')
    a = df.index.get_level_values('asset')
    hit = (a == '000000') & (d >= DATES[30])
    for c in ('raw_open', 'raw_high', 'raw_low', 'raw_close'):
        df.loc[hit, c] = 4.0
    # adj_factor 故意不动（默认就是 1.0）→ adj 同步腰斩
    for c in ('open', 'high', 'low', 'close'):
        df[f'adj_{c}'] = df[f'raw_{c}'] * df['adj_factor']
    df['prev_close'] = df.groupby(level='asset')['raw_close'].shift(1).fillna(10.0)

    f = find(acna.health_check(prices=df), '极端涨跌')
    assert f.severity == 'warn', f.summary
    assert f.metric == 1.0, '未被覆盖的只有 1 条'
    assert '没盖住' in f.summary or 'adj_factor' in f.summary


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


def test_extreme_move_ignores_suspension_days():
    """★ 停牌日不许被前值填充伪装成 0 收益（体检层自己也不能造假数据）。

    停牌 → 复牌暴涨这个真实跳变**要**被抓到；
    停牌**期间**那些假 0 收益不该出现在统计里。

    ⚠️ 2026-09-25 修正：原夹具改完 ``raw_*`` 后**没有同步 ``adj_*``**，
    于是复权价没跟着跳，判读上变成"复权后正常"（= 除权已被覆盖）。
    真实面板必须满足 ``adj_* == raw_* × adj_factor``（契约层硬校验），
    这里补齐同步，让场景回到本意：**复权后也涨** → 属于
    "adj_factor 没盖住 / 停牌复牌这类真实跳变" → warn。
    """
    df = mk_prices()
    d = df.index.get_level_values('date')
    a = df.index.get_level_values('asset')
    # 000000 停牌 3 天，复牌当天价格翻倍
    gap = (a == '000000') & (d >= DATES[10]) & (d <= DATES[12])
    df.loc[gap, ['raw_open', 'raw_high', 'raw_low', 'raw_close']] = np.nan
    df.loc[(a == '000000') & (d == DATES[13]),
           ['raw_open', 'raw_high', 'raw_low', 'raw_close']] = 20.0
    # 同步复权价，保持 adj_* == raw_* × adj_factor（契约要求）
    for c in ('open', 'high', 'low', 'close'):
        df[f'adj_{c}'] = df[f'raw_{c}'] * df['adj_factor']
    f = find(acna.health_check(prices=df), '极端涨跌')
    # 复牌 +100% 必须被抓到（复权后同样是大跳变）
    assert f.severity == 'warn', f.summary
    assert f.metric >= 1


# ── 2026-09-25 审计：极端涨跌的三档归因 ──────────────────────────────
def _mk_extreme(n_evt, n_good, n_missing_adj=0, no_adj_col=False):
    """造 n_evt 条 |原始收益|>50%，其中 n_good 条被复权抹平、n_missing_adj 条复权价缺失。

    每只票在自己的第 3 天腰斩（10 → 4，−60%）；被抹平的票同日把因子抬到 2.5。
    """
    dates = pd.bdate_range('2020-01-02', periods=n_evt * 2 + 2)
    codes = [f'{i:06d}.SZ' for i in range(n_evt)]
    idx = pd.MultiIndex.from_product([dates, codes], names=['date', 'asset'])
    px = pd.DataFrame({'raw_close': 10.0}, index=idx)
    px['adj_factor'] = 1.0
    dd = px.index.get_level_values('date')
    aa = px.index.get_level_values('asset')
    for k in range(n_evt):
        hit = (aa == codes[k]) & (dd >= dates[2 * k + 1])
        px.loc[hit, 'raw_close'] = 4.0
        if k < n_good:
            px.loc[hit, 'adj_factor'] = 2.5
    px['adj_close'] = px['raw_close'] * px['adj_factor']
    for k in range(n_good, n_good + n_missing_adj):
        hit = (aa == codes[k]) & (dd >= dates[2 * k + 1])
        px.loc[hit, 'adj_close'] = np.nan
    if no_adj_col:
        px = px.drop(columns=['adj_close'])
    return px


def test_extreme_move_pass_still_reports_unexplained_remainder():
    """★ pass 也要交代没被覆盖的那部分 —— 不然汇总行等于把它们藏起来。

    31/38 是测试者在 2016-2022 上实测到的比例；剩下 7 条需要人看，
    以前这一支输出 pass 且**一个字都不提**它们。
    """
    f = find(acna.health_check(prices=_mk_extreme(38, 31)), '极端涨跌')
    assert f.severity == 'pass', f.summary
    assert '另有 7 条复权后仍大' in f.summary, f.summary
    # 明细要能自己说清哪几条是"没被覆盖"的
    assert '问题' in f.detail.columns
    assert dict(f.detail['问题'].value_counts()) == {'已覆盖（因子已跟上）': 31, '复权后仍大': 7}


def test_extreme_move_missing_adj_is_not_pass():
    """★ 复权价缺失 ≠ 复权后正常。

    `ar` 是**前值填充**后算的，缺失行算出 0% 收益 —— 看着"正常"，
    其实"没有复权价可判断"。此前它被并进 explained，占比一过 80% 就报
    pass＋"复权价连续"，是假 all-clear。现在缺失不为 0 就不给 pass。
    """
    f = find(acna.health_check(prices=_mk_extreme(3, 0, no_adj_col=True)), '极端涨跌')
    assert f.severity == 'warn', f.summary
    assert '复权价缺失' in f.summary
    # ★ 缺失绝不能被读成"已覆盖" —— 这三句都能真失败（此前这里写的是一条
    #   `... or '缺失' in f.summary` 的断言，恒为真，等于没测）
    assert '已覆盖' not in f.summary, f.summary
    assert '复权价连续' not in f.summary, f.summary
    assert '复权后正常' not in f.summary, f.summary
    assert f.metric == 3.0
    assert set(f.detail['问题']) == {'复权价缺失'}

    # 大多数被抹平、少数缺失 —— 依然是 warn（"没检验"与"检验通过"必须分开）
    f2 = find(acna.health_check(prices=_mk_extreme(38, 31, n_missing_adj=7)), '极端涨跌')
    assert f2.severity == 'warn', f2.summary
    assert '7 条复权价缺失' in f2.summary


def test_extreme_move_metric_is_total_flagged():
    """★ metric 与文案首数必须是同一个量（此前 warn 支报的是未覆盖数，报告里
    渲染成「13 条 … [11]」）。pass 支与 warn 支口径也要一致。"""
    f = find(acna.health_check(prices=_mk_extreme(13, 2)), '极端涨跌')   # 2 抹平 + 11 仍大
    assert f.severity == 'warn'
    assert f.metric == 13.0, f'应报总触发数 13，实际 {f.metric}'
    assert f.summary.startswith('13 条')
    g = find(acna.health_check(prices=_mk_extreme(38, 31)), '极端涨跌')  # pass 支
    assert g.metric == 38.0 and g.summary.startswith('38 条')


def test_extreme_move_warn_text_omits_zero_buckets():
    """某一档为 0 时不该还念它（"0 条复权后仍然很大"只会让人分心）。"""
    f = find(acna.health_check(prices=_mk_extreme(3, 0, no_adj_col=True)), '极端涨跌')
    assert '0 条复权后' not in f.summary, f.summary
    assert '0 条已被复权抹平' not in f.summary, f.summary


def test_extreme_move_buckets_partition_and_align():
    """★ 三档必须**恰好划分**全部触发行，且明细与触发行一一对齐。

    这是这次归因重写的核心不变量：任何一行的标签都必须能从
    「已覆盖 / 复权后仍大 / 复权价缺失」里唯一确定，不许重叠、不许漏。
    用随机组合扫（含无 adj_close 列、混合缺失）—— 以前那种"从 ar 上判缺失"
    的错误写法会在这里露馅（ffill 之后缺失行的收益是 0%，会被算进"已覆盖"）。
    """
    rng = np.random.default_rng(0)
    for trial in range(12):
        n_evt = int(rng.integers(2, 30))
        n_good = int(rng.integers(0, n_evt + 1))
        n_mis = int(rng.integers(0, n_evt - n_good + 1))
        no_col = bool(rng.integers(0, 2))
        f = find(acna.health_check(
            prices=_mk_extreme(n_evt, n_good, n_missing_adj=n_mis, no_adj_col=no_col)),
            '极端涨跌')
        n_flag = int(f.metric)
        d = f.detail
        assert len(d) == n_flag, f'明细 {len(d)} 行 ≠ 触发 {n_flag} 行'
        assert d['raw_return'].notna().all()
        lab = d['问题']
        # ① 三档齐全且互斥（value_counts 之和 = 触发数）
        assert set(lab) <= {'已覆盖（因子已跟上）', '复权后仍大', '复权价缺失'}
        assert len(d) == int((lab == '已覆盖（因子已跟上）').sum()) \
            + int((lab == '复权后仍大').sum()) + int((lab == '复权价缺失').sum())
        # ② 标签与 adj_return 自洽。**只查反向**：正向查不了 —— 缺失档的
        #    adj_return 会被 ffill 填成 0%（正是当初误判的来源），所以
        #    "缺失档的 adj_return 必须是 NaN"是**错**的断言，别写；
        #    同理也不能用 `... or True` 糊过去（那等于没测）。
        assert d.loc[lab != '复权价缺失', 'adj_return'].notna().all()
        # ③ 无 adj_close 列时，全部必须归到"缺失"
        if no_col:
            assert set(lab) == {'复权价缺失'}
