"""契约校验（六道防线 · 第 1 条）。

分工：
* **单对象校验**在 ``panels.py`` 的构造函数里（构造即校验）
* **本模块做跨对象校验** —— 对象之间对不上，同样拒绝运行

设计原则：**错误信息必须能定位到具体是哪一行、哪个字段。**
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .errors import ContractError, fail
from .panels import (
    Events,
    Exposures,
    FactorPanel,
    Grouping,
    PricePanel,
    Tradability,
    Universe,
)

__all__ = ['validate_inputs', 'check_adjust_agreement', 'ContractError']

# 参数名 → 契约类型。用于把裸 DataFrame 自动包装成契约对象。
_WRAP = {
    'factor': FactorPanel,
    'prices': PricePanel,
    'universe': Universe,
    'tradability': Tradability,
    'grouping': Grouping,
    'exposures': Exposures,
    'events': Events,
}


def _as_panel(name, obj):
    """裸 DataFrame 自动包装成对应契约对象；已是契约对象则原样返回。

    这样 ``validate_inputs(factor=df, prices=df2, calendar=cal)`` 也能用 ——
    参数名本身就确定了契约类型，不存在歧义。
    """
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return _WRAP[name](obj)
    if not hasattr(obj, 'df'):
        fail('validate', 'type',
             f'`{name}` 需要契约对象或 DataFrame，收到 {type(obj).__name__}')
    return obj


def validate_inputs(factor, prices, calendar, universe=None,
                    tradability=None, grouping=None, exposures=None,
                    events=None, strict=True):
    """跨对象契约校验。任何一条不过就拒绝运行。

    裸 ``DataFrame`` 会自动包装成对应契约对象（按参数名判断类型）。

    Parameters
    ----------
    factor, prices, calendar : 契约对象或 DataFrame
        必填三项。
    universe, tradability, grouping, exposures, events : 契约对象或 DataFrame, 可选
    strict : bool
        ``True``（默认）时全部检查；``False`` 时跳过"覆盖度"这类软检查，
        只保留硬性错误。

    Raises
    ------
    ContractError
    """
    factor = _as_panel('factor', factor)
    prices = _as_panel('prices', prices)
    universe = _as_panel('universe', universe)
    grouping = _as_panel('grouping', grouping)
    exposures = _as_panel('exposures', exposures)

    _check_same_asset_type(factor, prices)
    _check_dates_on_calendar(calendar, factor=factor, prices=prices,
                             universe=universe, grouping=grouping,
                             exposures=exposures, strict=strict)
    _check_factor_in_prices(factor, prices)
    if universe is not None:
        _check_universe_covers_factor(factor, universe)
    if strict:
        _check_price_coverage(factor, prices)
    return True


# --------------------------------------------------------------------------- #
def _check_same_asset_type(factor, prices):
    """factor 与 prices 的 asset 层级必须同类且能对齐。"""
    af = factor.df.index.get_level_values('asset')
    ap = prices.df.index.get_level_values('asset')
    overlap = set(af.unique()) & set(ap.unique())
    if not overlap:
        fail('validate', 'asset_mismatch',
             f'factor 的 asset 与 prices 完全没有交集。\n'
             f'  factor 例：{list(af.unique()[:3])}\n'
             f'  prices 例：{list(ap.unique()[:3])}\n'
             f'  常见原因：一边带交易所后缀（600519.SH），一边不带（600519）。')


def _check_dates_on_calendar(calendar, strict=True, **panels):
    """所有出现在数据里的日期都必须在日历上。"""
    for name, p in panels.items():
        if p is None:
            continue
        d = p.df.index.get_level_values('date')
        off = pd.DatetimeIndex(d.unique())
        off = off[~off.isin(calendar.index)]
        if len(off):
            fail(name, 'off_calendar',
                 f'{len(off)} 个日期不在交易日历上：'
                 f'{"; ".join(str(x)[:10] for x in off[:3])}'
                 f'{" …" if len(off) > 3 else ""}\n'
                 f'  含义：拿非交易日当交易日，前向收益会算错。')


def _check_factor_in_prices(factor, prices):
    """因子的每个 (date, asset) 都要能在 prices 里找到 —— 否则算不出收益。"""
    fi = factor.df.index
    pi = prices.df.index
    miss = fi.difference(pi)
    if len(miss):
        ratio = len(miss) / max(len(fi), 1)
        fail('validate', 'factor_not_in_prices',
             f'{len(miss)} 个 (date, asset) 在 prices 里找不到'
             f'（占因子 {ratio:.1%}）：\n'
             f'  {" ; ".join(f"{d:%Y-%m-%d} {c}" for d, c in miss[:3])}\n'
             f'  含义：这些样本算不出前向收益。\n'
             f'  修法：补行情，或用 Universe 把它们排除掉。')


def _check_universe_covers_factor(factor, universe):
    """因子的样本应当落在股票池内 —— 否则生存者偏差会溜进来。"""
    u = universe.df['in_universe']
    if not u.dtype == bool:
        u = u.astype(bool)
    inside = u.reindex(factor.df.index)
    # 池子里没记录的样本按"不在池"处理，但要在错误信息里说清楚
    unknown = inside.isna()
    outside = (~inside.fillna(False)) & (~unknown)
    if outside.any():
        n = int(outside.sum())
        fail('validate', 'factor_outside_universe',
             f'{n} 个因子样本不在股票池内（占 {n / max(len(factor.df), 1):.1%}）：\n'
             f'  {"; ".join(f"{d:%Y-%m-%d} {c}" for d, c in factor.df.index[outside][:3])}\n'
             f'  含义：这些是当时不该被选中的股票 —— 典型来源是**生存者偏差**\n'
             f'  （用今天的成分股回测历史）。\n'
             f'  修法：把 Universe 换成 as-of 的（退市股在它还活着时应入池）。')
    unknown_n = int(unknown.sum())
    if unknown_n > 0.2 * len(factor.df):
        fail('validate', 'universe_coverage',
             f'股票池只覆盖了 {1 - unknown_n / len(factor.df):.1%} 的因子样本，'
             f'缺口过大。\n'
             f'  含义：Universe 与 factor 对不齐，很可能是股票池的日期/代码格式不对。')


def _check_price_coverage(factor, prices, min_ratio=0.5):
    """价格对因子的覆盖度过低时提醒（软检查）。"""
    fi = set(factor.df.index.get_level_values('asset').unique())
    pi = set(prices.df.index.get_level_values('asset').unique())
    ratio = len(fi & pi) / max(len(fi), 1)
    if ratio < min_ratio:
        fail('validate', 'price_coverage',
             f'prices 只覆盖了 factor 中 {ratio:.1%} 的股票（阈值 {min_ratio:.0%}）。\n'
             f'  含义：可能取错了行情区间或代码格式。')


# --------------------------------------------------------------------------- #
def check_adjust_agreement(adj_a, adj_b, tol=1e-6, calendar=None,
                           name_a='computed', name_b='stored'):
    """两种复权口径的收益必须一致（防线 3 的不变量）。

    依据（D5）：任何常数缩放都不影响收益 ——
    ``adj(t2)/adj(t1)`` 里的常数会约掉，所以「后复权」与「前复权」
    **在收益上完全等价**。既然两者应当等价，就能互相验证。

    Parameters
    ----------
    adj_a, adj_b : pd.Series
        两个口径的复权因子，同一个 MultiIndex(date, asset)。
    tol : float
        收益差的容差，**默认 1e-6**。

        为什么不是 1e-10：若一方是**存储值**，它通常被舍入过
        （本库对接的库内 ``stock_adj`` 存 10 位小数）。小因子（如 0.003）保留
        10 位小数的相对误差就有 ~1.6e-8，乘进收益后到 ~2e-8 ——
        **1e-10 会把纯粹的存储舍入误报成数据错误**。
        实测：把计算值也 ``round(10)`` 后差异归零（0.00e+00），
        证明差异确实只来自舍入。

        1e-6 足够松（容得下舍入），也足够紧（**漏掉一次送转会产生
        ~1e-1 的收益差**，量级差 5 个数量级）。

    Returns
    -------
    dict
        ``{'max_diff':..., 'n_bad':..., 'ok':...}``

    Raises
    ------
    ContractError
        差异超过容差。
    """
    a = adj_a.rename('a')
    b = adj_b.rename('b')
    m = pd.concat([a, b], axis=1, join='inner').sort_index()
    if len(m) < 2:
        return {'max_diff': 0.0, 'n_bad': 0, 'ok': True, 'note': '重叠不足，跳过'}
    diff_max, bad = 0.0, 0
    for _, g in m.groupby(level='asset'):
        if len(g) < 2:
            continue
        d = (g['a'].pct_change() - g['b'].pct_change()).abs().dropna()
        if len(d):
            diff_max = max(diff_max, float(d.max()))
            bad += int((d > tol).sum())
    if bad:
        fail('adjust_agreement', 'two_sources_differ',
             f'两种复权口径的收益不一致：{bad} 处差异 > {tol:g}，'
             f'最大差 {diff_max:.3e}（{name_a} vs {name_b}）。\n'
             f'  含义：复权数据或算法有问题 —— 两者在数学上应当完全等价。\n'
             f'  常见原因：除权事件表缺记录（漏一次送转 ⇒ 收益差 ~1e-1，量级远超本容差）、\n'
             f'            或一方被舍入过（存储舍入只会到 ~1e-8，不会触发本错误）。\n'
             f'  修法：补齐 xdxr 事件表，或核对复权因子是否与原始价自洽。')
    return {'max_diff': diff_max, 'n_bad': bad, 'ok': True}
