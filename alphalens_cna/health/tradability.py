"""体检 · 可成交性 —— 停牌 / 涨跌停 / ST / 新股的实际占比。

设计意图：**打开 A 股规则的同时，把代价量化出来。**
"剔除涨停买不进的"听着无害，但它到底剔掉了多少？这一层给出数字。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Finding, fmt_pct

__all__ = ['tradability_rates', 'check_tradability']


def tradability_rates(td):
    """按日统计各类占比。只在有对应列时才产出。"""
    out = {}
    pairs = {
        'can_buy': 'can_buy_open',
        'can_sell': 'can_sell_open',
        'suspended': 'suspended',
        'is_st': 'is_st',
        'is_new': 'is_new_stock',
        'limit_up_open': 'open_at_limit_up',
        'limit_down_open': 'open_at_limit_down',
    }
    for label, col in pairs.items():
        if col in td.columns:
            out[label] = td[col].astype(bool).groupby(level='date').mean()
    return pd.DataFrame(out)


def check_tradability(td, th):
    """可成交性体检。返回两条：``可成交性``（占比）与 ``新股过滤``。"""
    out = []
    rates = tradability_rates(td)
    if not len(rates.columns):
        return [Finding('可成交性', 'skip',
                        'Tradability 里没有 `can_buy_open` / `suspended` 等列，跳过')]

    overall = rates.mean()
    detail = rates.describe().T[['mean', 'min', 'max']].reset_index()
    detail.columns = ['项目', '均值', '最小', '最大']

    if 'can_buy' in rates.columns:
        buy = rates['can_buy']
        metric = float(buy.mean())
        zero_days = buy[buy <= 0]
        if len(zero_days):
            return out + [Finding(
                '可成交性', 'fail',
                f'{len(zero_days)} 个交易日**一只都买不进**（可买比例 0）',
                metric,
                pd.DataFrame({'date': zero_days.index, 'can_buy': zero_days.values}),
                th['min_buyable_rate'])]
        if metric < th['min_buyable_rate']:
            out.append(Finding(
                '可成交性', 'warn',
                f'平均可买比例 {fmt_pct(metric)} < {th["min_buyable_rate"]:.0%}，'
                f'剔除偏多', metric, detail, th['min_buyable_rate']))
        else:
            out.append(Finding(
                '可成交性', 'pass',
                '平均可买 ' + fmt_pct(metric)
                + '，可卖 ' + (fmt_pct(float(rates["can_sell"].mean()))
                               if 'can_sell' in rates.columns else '—')
                + '，停牌 ' + (fmt_pct(float(rates["suspended"].mean()))
                               if 'suspended' in rates.columns else '—')
                + '，ST ' + (fmt_pct(float(rates["is_st"].mean()))
                             if 'is_st' in rates.columns else '—'),
                metric, detail, th['min_buyable_rate']))
    else:
        out.append(Finding('可成交性', 'info',
                           '；'.join(f'{k} {fmt_pct(float(v))}'
                                     for k, v in overall.items()), None, detail))

    # ── 新股过滤是否真的生效 ────────────────────────────────────────
    if 'listed_days_known' in td.columns:
        known = bool(td['listed_days_known'].astype(bool).all())
        if not known:
            out.append(Finding(
                '新股过滤', 'warn',
                '`listed_days_known` 为假 —— 未提供上市日期，'
                '**新股过滤没有生效**（上市不足 60 个交易日的新股仍在样本里）。'
                '修法：把 Calendar 覆盖到上市首日之前，或直接给 `listed_days`。',
                0.0, None, None))
        else:
            nb = (float(rates['is_new'].mean())
                  if 'is_new' in rates.columns else None)
            out.append(Finding('新股过滤', 'pass',
                               '上市日期已知，新股过滤生效'
                               + ('' if nb is None else f'（新股占比 {fmt_pct(nb)}）'),
                               nb))
    else:
        out.append(Finding('新股过滤', 'skip',
                           'Tradability 无 `listed_days_known` 列，'
                           '无法确认新股过滤是否生效'))
    return out
