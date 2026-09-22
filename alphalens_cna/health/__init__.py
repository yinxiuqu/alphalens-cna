"""数据体检（**六道防线 · 第 2 条**）—— "这份数据能不能用"。

为什么要有这一层
----------------
契约层（防线 1）管的是**结构**：索引对不对、列全不全、有没有前视。
它管不了**内容**：复权因子算错、整段行情缺失、股票池只有活着的公司、
因子其实半年没更新过 —— 这些数据**结构完全合法**，结论却全错。

`alphalens` 对这类问题一律沉默：丢几行、算个 IC、给你一个数字。
本层要在**出数字之前**先把数据的问题摆到桌面上。

严重度五档、阈值依据见 :mod:`.core`。

用法
----
>>> rep = acna.health_check(prices=px, factor=f, tradability=t, calendar=cal)
>>> print(rep)                       # 终端速览
>>> print(rep.warnings)              # 直接塞进报告头部的告警列表
>>> open('health.md', 'w').write(rep.to_markdown('ROE 因子 · 数据体检'))
"""

from __future__ import annotations

from .core import DEFAULTS, Finding, HealthReport, fmt_pct
from .panel import (
    check_calendar_alignment,
    check_coverage,
    check_factor_panel,
    check_survivorship,
    coverage_by_date,
)
from .prices import (
    check_adjust_continuity,
    check_extreme_moves,
    check_ohlc,
    check_suspension,
)
from .tradability import check_tradability

__all__ = [
    'Finding', 'HealthReport', 'DEFAULTS', 'check', 'fmt_pct',
    'coverage_by_date',
]


def check(prices=None, factor=None, tradability=None, calendar=None, *,
          thresholds=None, name=None):
    """跑一遍数据体检。

    Parameters
    ----------
    prices : PricePanel | DataFrame, optional
        行情（含 ``raw_*`` / ``adj_*`` / ``adj_factor``）。给了才查行情类项目。
    factor : FactorPanel | DataFrame, optional
        因子面板。给了才查因子类项目。
    tradability : Tradability | DataFrame, optional
        可成交性。给了才会报停牌/涨跌停/ST/新股比例。
    calendar : Calendar, optional
        交易日历。覆盖率、存活偏差、日历对齐都要它。
    thresholds : dict, optional
        覆盖 :data:`DEFAULTS` 里的阈值（键名写错会报错，不静默忽略）。
    name : str, optional
        数据集名称，只影响输出文案。

    Returns
    -------
    HealthReport

    Notes
    -----
    **缺什么输入就记一条 ``skip``** —— 报告里看得见"这项没查"，
    不会被误读成"这项没问题"。
    """
    th = dict(DEFAULTS)
    if thresholds:
        unknown = set(thresholds) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'未知阈值 {sorted(unknown)}；'
                             f'可用：{sorted(DEFAULTS)}')
        th.update(thresholds)

    rep = HealthReport()
    px = getattr(prices, 'df', prices)
    fp = getattr(factor, 'df', factor)
    td = getattr(tradability, 'df', tradability)
    cal = getattr(calendar, 'index', calendar)

    # ── 行情类 ──────────────────────────────────────────────────────
    if px is None:
        rep.findings.append(Finding('行情完整性', 'skip', '未提供 prices，跳过'))
    else:
        rep.findings += [
            check_ohlc(px),
            check_extreme_moves(px, th),
            check_adjust_continuity(px, th),
            check_suspension(px, th),
        ]

    # ── 面板类（需要日历） ──────────────────────────────────────────
    if px is None or cal is None:
        why = '未提供 prices' if px is None else '未提供 calendar'
        rep.findings += [
            Finding('覆盖率', 'skip', f'{why}，跳过'),
            Finding('存活偏差', 'skip', f'{why}，跳过'),
            Finding('日历对齐', 'skip', f'{why}，跳过'),
        ]
    else:
        rep.findings += [
            check_coverage(px, cal, th),
            check_survivorship(px, cal, th),
            check_calendar_alignment(px, cal, th),
        ]

    # ── 因子类 ──────────────────────────────────────────────────────
    if fp is None:
        rep.findings.append(Finding('因子截面', 'skip', '未提供 factor，跳过'))
    else:
        rep.findings += check_factor_panel(fp, th)

    # ── 可成交性 ────────────────────────────────────────────────────
    if td is None:
        rep.findings.append(
            Finding('可成交性', 'skip',
                    '未提供 tradability —— 停牌/涨跌停/ST/新股比例未查'))
    else:
        rep.findings += check_tradability(td, th)

    if name:
        rep.findings.insert(0, Finding('数据集', 'info', name))
    return rep
