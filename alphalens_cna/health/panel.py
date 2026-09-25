"""体检 · 面板类 —— 覆盖率 / 存活偏差 / 日历对齐 / 因子截面。

这三项合起来回答一个 alphalens 从不问的问题：
**"我手里这份面板，是不是一个能支撑结论的样本？"**
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail
from .core import Finding

__all__ = [
    'coverage_by_date', 'check_coverage', 'check_survivorship',
    'check_calendar_alignment', 'check_factor_panel',
]


def coverage_by_date(prices):
    """每日有行情的股票数。"""
    return prices.groupby(level='date').size().rename('n')


# --------------------------------------------------------------------------- #
def check_coverage(prices, calendar, th):
    """覆盖率 —— 单日股票数相对中位数骤降就是数据缺口。

    判据用**相对**中位数而不是绝对股票数：库里的票数随年份变化
    （1995 年几百只、2026 年五千多只），只有"跟自己比"才有意义。
    """
    cov = coverage_by_date(prices)
    if not len(cov):
        return Finding('覆盖率', 'fail', '面板一行都没有', 0.0)
    med = float(cov.median())
    ratio = cov / med
    low = ratio[ratio < th['min_coverage_ratio']]
    metric = float(ratio.min())
    detail = None
    if len(low):
        detail = pd.DataFrame({'date': low.index, 'n': cov.loc[low.index].values,
                               'ratio': low.values})
    if len(low) > 0.05 * len(cov):
        # 超过 5% 的交易日都缺 → 不是偶发
        return Finding('覆盖率', 'fail',
                       f'{len(low):,}/{len(cov):,} 个交易日的股票数不足中位数的 '
                       f'{th["min_coverage_ratio"]:.0%}', metric, detail,
                       th['min_coverage_ratio'])
    if len(low):
        return Finding('覆盖率', 'warn',
                       f'{len(low)} 个交易日股票数骤降（最少 {metric:.1%} 中位数）',
                       metric, detail, th['min_coverage_ratio'])
    return Finding('覆盖率', 'pass',
                   f'每日 {med:,.0f} 只，无异常缺口（最低 {metric:.1%} 中位数）',
                   metric, None, th['min_coverage_ratio'])


# --------------------------------------------------------------------------- #
def check_survivorship(prices, calendar, th):
    """存活偏差 —— **样本里有没有"中途消失"的股票**。

    这是整层里最重要的一个检查。做法不需要任何外部数据：
    每只票在面板里的**最后一个交易日**，如果全都压在样本末尾，
    那说明这份面板只装了活到今天的公司 —— 中途退市的一只都没有。

    后果：回测里所有股票都"善终"，收益率被系统性高估。
    """
    cal = pd.DatetimeIndex(calendar)
    idx = prices.index
    last = pd.Series(idx.get_level_values('date')).groupby(
        idx.get_level_values('asset')).max()
    pos = cal.get_indexer(last.values)
    end_pos = len(cal) - 1
    early = (end_pos - pos) > th['survivorship_grace']
    n_early = int(early.sum())
    n_all = int(len(last))
    frac = n_early / n_all if n_all else 0.0

    detail = None
    if n_early:
        e = last[early].sort_values()
        detail = pd.DataFrame({
            'asset': e.index, 'last_date': e.values,
            '提前结束交易日': (end_pos - cal.get_indexer(e.values)),
        })

    if n_early == 0:
        return Finding(
            '存活偏差', 'warn',
            f'{n_all:,} 只票**没有一只**在样本中途消失 —— 这通常意味着'
            f'股票池是"今天的成分股"或已退市股未入库，历史收益会被系统性高估',
            0.0, None, th['survivorship_grace'])
    if frac < 0.01:
        return Finding(
            '存活偏差', 'warn',
            f'中途结束的股票仅 {n_early}/{n_all:,}（{frac:.2%}）—— 偏低，'
            f'退市股可能只入库了一部分', frac, detail, th['survivorship_grace'])
    return Finding(
        '存活偏差', 'pass',
        f'{n_early:,}/{n_all:,} 只票在中途结束（{frac:.2%}），'
        f'含退市股，无存活偏差', frac, detail, th['survivorship_grace'])


# --------------------------------------------------------------------------- #
def check_calendar_alignment(prices, calendar, th):
    """日历对齐 —— 面板日期必须落在交易日历上；日频面板不该有内部空洞。

    月频/周频面板天然"缺"大量交易日，那不是问题，所以只有**日频面板**
    （相邻日期中位间隔 ≤ 4 个自然日）才检查内部空洞。
    """
    cal = pd.DatetimeIndex(calendar)
    pd_dates = pd.DatetimeIndex(
        prices.index.get_level_values('date').unique()).sort_values()
    off = pd_dates.difference(cal)
    if len(off):
        detail = pd.DataFrame({'date': off[:20]})
        return Finding('日历对齐', 'fail',
                       f'{len(off):,} 个面板日期不在交易日历上（如 {off[0]:%Y-%m-%d}）',
                       float(len(off)), detail)

    if len(pd_dates) < 3:
        return Finding('日历对齐', 'pass', f'面板仅 {len(pd_dates)} 个日期', None)

    gaps = np.diff(pd_dates.values).astype('timedelta64[D]').astype(float)
    med_gap = float(np.median(gaps))
    if med_gap > 4:
        return Finding('日历对齐', 'info',
                       f'非日频面板（相邻日期中位间隔 {med_gap:.0f} 天），'
                       f'不检查内部空洞', med_gap)

    inside = cal[(cal >= pd_dates[0]) & (cal <= pd_dates[-1])]
    missing = inside.difference(pd_dates)
    metric = float(len(missing))
    if len(missing):
        detail = pd.DataFrame({'date': missing[:20]})
        return Finding('日历对齐', 'warn',
                       f'日频面板内部缺 {len(missing):,} 个交易日'
                       f'（区间内共 {len(inside):,} 个）', metric, detail)
    return Finding('日历对齐', 'pass',
                   f'日频面板 {len(pd_dates):,} 个交易日，与日历完全对齐', 0.0)


# --------------------------------------------------------------------------- #
def _longest_run(x):
    """一维数组里最长"连续相同值"的长度（≥1）。"""
    v = np.asarray(x)
    if len(v) == 0:
        return 0
    change = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
    return int(np.diff(np.r_[change, len(v)]).max())


def check_factor_panel(factor, th):
    """因子截面体检 —— 截面够不够、有没有冻结、陈旧度多高。

    返回**两条**结论：``因子截面``（每日股票数 + 唯一值）与 ``因子陈旧度``。
    """
    out = []
    # ★ 缺列要说人话。此前是裸 `factor['value']` → KeyError: 'value'，
    #   看不出该改什么（health_check 是诊断工具，故意不走契约全量校验，
    #   但**结构约定**仍要明确报出来）。
    if not isinstance(factor, pd.DataFrame):
        # ★ 传 Series 时说"实际列：[]"是错的（Series 没有列这回事）——
        #   报错必须说准，否则读者会去数一个不存在的列清单。
        fail('health', 'factor_panel',
             f'`factor` 需要 DataFrame（MultiIndex(date, asset) + `value` 列），'
             f'收到 {type(factor).__name__}。')
    if 'value' not in factor.columns:
        fail('health', 'factor_column',
             f"因子面板缺少 `value` 列，实际列：{list(getattr(factor, 'columns', []))}。\n"
             f"  约定：因子列名固定为 `value`（`FactorPanel` 要求 `value` + `available_at`）。\n"
             f"  修法：`factor = factor.rename(columns={{'你的因子列': 'value'}})`。")
    v = factor['value']
    n_by_date = v.groupby(level='date').size()
    u_by_date = v.groupby(level='date').nunique()

    thin = u_by_date[u_by_date < th['min_cross_section']]
    metric = float(u_by_date.min()) if len(u_by_date) else 0.0
    if len(thin):
        detail = pd.DataFrame({'date': thin.index, 'n_unique': thin.values,
                               'n_asset': n_by_date.loc[thin.index].values})
        out.append(Finding('因子截面', 'warn',
                           f'{len(thin)} 个截面唯一值 < {th["min_cross_section"]}'
                           f'（这些日子算不出 IC）', metric, detail,
                           th['min_cross_section']))
    else:
        out.append(Finding('因子截面', 'pass',
                           f'{len(n_by_date)} 个截面，'
                           f'每日 {n_by_date.median():,.0f} 只、'
                           f'最少 {u_by_date.min():,} 个不同取值', metric))

    # ── 冻结：某只票的因子值长期一模一样 ────────────────────────────
    run = pd.Series(v.to_numpy(),
                    index=factor.index.get_level_values('asset')) \
        .groupby(level=0).apply(_longest_run)
    frozen = run[run >= th['freeze_days']]
    if len(frozen):
        detail = pd.DataFrame({'asset': frozen.index, '最长不变期数': frozen.values})
        out.append(Finding('因子冻结', 'warn',
                           f'{len(frozen):,} 只票的因子值连续 '
                           f'≥{th["freeze_days"]} 期不变（疑似数据未更新）',
                           float(frozen.max()), detail, th['freeze_days']))
    elif not len(run):
        # ★ 空因子面板（0 行）→ run 为空 → `run.max()` 是 NaN → `int(NaN)` 直接
        #   ValueError（格式化字符串里崩，最难看）。空输入要说人话，别崩。
        out.append(Finding('因子冻结', 'skip', '无因子观测，跳过',
                           0.0, None, th['freeze_days']))
    else:
        out.append(Finding('因子冻结', 'pass',
                           f'无长期不变的票（最长 {int(run.max())} 期）',
                           float(run.max()), None, th['freeze_days']))

    # ── 陈旧度（有才查） ────────────────────────────────────────────
    if 'staleness_days' in factor.columns:
        s = pd.to_numeric(factor['staleness_days'], errors='coerce').dropna()
        if len(s):
            med, mx = float(s.median()), float(s.max())
            if med > th['stale_days']:
                out.append(Finding(
                    '因子陈旧度', 'warn',
                    f'PIT 陈旧度中位数 {med:.0f} 天 > {th["stale_days"]:.0f} —— '
                    f'因子可能长期没更新', med, None, th['stale_days']))
            else:
                out.append(Finding(
                    '因子陈旧度', 'pass',
                    f'中位 {med:.0f} 天 / 最大 {mx:.0f} 天', med, None,
                    th['stale_days']))
    else:
        out.append(Finding('因子陈旧度', 'skip',
                           '因子面板没有 `staleness_days` 列，跳过'))
    return out
