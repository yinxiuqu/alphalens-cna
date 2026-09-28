"""契约校验（六道防线 · 第 1 条）。

分工：
* **单对象校验**在 ``panels.py`` 的构造函数里（构造即校验）
* **本模块做跨对象校验** —— 对象之间对不上，同样拒绝运行
* **入口层接线**走 :func:`ensure_contract` —— 一条龙入口（``build_report`` /
  ``check_parity``）必须调它。⚠️ 这一条曾经是缺的：防线 1 一度只在"走 loader
  或用户显式构造面板"时生效，裸 DataFrame 直通一条龙入口时会**整层绕过**
  （实测前视、非交易日、复权缺行、``+inf``、因子多出行五类坏数据静默跑通）。

设计原则：**错误信息必须能定位到具体是哪一行、哪个字段。**
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calendar import as_calendar
from .errors import ContractError, fail
from .panels import (
    DATE,
    Events,
    Exposures,
    FactorPanel,
    Grouping,
    PricePanel,
    Tradability,
    Universe,
)

__all__ = ['ensure_contract', 'validate_inputs', 'check_adjust_agreement',
           'ContractError']

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


# --------------------------------------------------------------------------- #
# 唯一值：**不要**用 `set(index.get_level_values(i).unique())` ——
# 960,000 行的面板上，光 `_check_same_asset_type` 与 `_check_price_coverage`
# 各算一遍就要 593 ms + 533 ms（实测），把校验总开销推到 2.3 s。
# 走整数 code 去重，只在唯一值上取标签，是 O(行数) 次整数扫描 + O(唯一值) 次取值。
# ⚠️ 不能用 `index.levels[i]` 直接当唯一值：切片/过滤后的索引可能残留**过时**的
#    level 值（已不在数据里），拿它做覆盖度检查会误报。
def _unique_level(df, level):
    """该表**实际出现过**的第 ``level`` 层取值（去重，已丢掉缺失）。"""
    mi = df.index
    codes = np.asarray(mi.codes[level])
    ok = codes >= 0
    if not ok.any():
        return pd.Index([], dtype=mi.levels[level].dtype)
    # 用 bincount 数"哪些槽位出现过"：O(行数) 且**不排序**，
    # 比 np.unique（内部排序 1,000,000 个整数）快一个量级。
    seen = np.bincount(codes[ok], minlength=len(mi.levels[level])) > 0
    return mi.levels[level].take(np.flatnonzero(seen))


def _unique_dates(df):
    return _unique_level(df, 0)


def _unique_assets(df):
    return _unique_level(df, 1)


def _factor_validation_frame(df, assume_available_at, notice):
    """把因子表归一成 ``FactorPanel`` 要的形状（``value`` + ``available_at``）。

    ⚠️ 归一规则必须与 ``engine.clean._factor_frame`` **完全一致** —— 否则会出现
    "**校验的列**和**分析的列**不是同一列"这种最难查的不一致（都跑绿了，数字是错的）。

    ``assume_available_at=True`` 时缺列按 loader 的既有约定合成 ``= date``
    （"收盘后可知"），并把这件事写进 ``notice`` —— **合成等于前视闸门失效**，
    所以不许静默。
    """
    if isinstance(df, pd.Series):         # clean 也收 Series（to_frame('factor')）
        df = df.to_frame('value')
    if 'factor' in df.columns:            # clean 优先取 factor，这里必须跟着
        col = 'factor'
    elif 'value' in df.columns:
        col = 'value'
    else:
        cand = [c for c in df.columns if c not in ('available_at', 'date', 'asset')]
        if len(cand) != 1:
            fail('validate', 'ambiguous_factor',
                 f'因子表看不出哪列是因子值（候选 {cand}）。\n'
                 f'  修法：把列命名为 `value` 或 `factor`。')
        col = cand[0]
    base = df[[col]].rename(columns={col: 'value'})
    if 'available_at' in df.columns:
        base = base.assign(available_at=df['available_at'])
    elif assume_available_at:
        base = base.assign(available_at=base.index.get_level_values(DATE))
        if notice is not None:
            notice.append('因子未提供 `available_at`，已按 `= date` 合成'
                          '（与 loader 约定一致）→ **前视检查因此未生效**')
    else:
        fail('validate', 'available_at_missing',
             '因子表缺 `available_at`（该值**何时可知**）。\n'
             '  它是前视硬闸门：`available_at > date` 会被拒绝。\n'
             '  修法：补上这一列；或显式传 `assume_available_at=True` '
             '（按 `= date` 合成 —— **前视检查将不生效**，报告里会写明）。')
    return base


def _wrap_for_check(name, obj, validate=True, **kw):
    """包装成契约对象。已经是**校验过**的契约对象则原样返回。"""
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return _WRAP[name](obj, validate=validate, **kw)
    if not hasattr(obj, 'df'):
        fail('validate', 'type',
             f'`{name}` 需要契约对象或 DataFrame，收到 {type(obj).__name__}')
    if validate and not getattr(obj, '_validated', False):
        # 用户可能是 `XxxPanel(df, validate=False)` 构造的 —— 那这里就不是"已校验"，
        # 补一次，避免"上游关掉校验、下游以为校验过了"的静默漏洞。
        # ⚠️ 但别假设它一定有 `validate()`：库里不少对象也带 `.df`（例如 `Returns`）。
        #    直接调会漏出 `AttributeError`，那比"传错对象"本身更难懂。
        meth = getattr(obj, 'validate', None)
        if meth is None:
            fail('validate', 'type',
                 f'`{name}` 既不是 DataFrame，也不像契约对象：'
                 f'{type(obj).__name__} 有 `.df` 但没有 `validate()`。\n'
                 f'  修法：传 DataFrame，或传对应的契约对象'
                 f'（如 PricePanel / FactorPanel）。')
        meth()
    return obj


def ensure_contract(factor, prices, calendar, *, universe=None, tradability=None,
                    grouping=None, exposures=None, events=None, strict=True,
                    assume_available_at=True, notice=None):
    """**入口层**统一契约校验：包装成契约对象 + 跨对象校验，并**把包装结果交回来**。

    与 :func:`validate_inputs` 的分工：那个只回答"过不过"（返回 ``True``），
    这个返回包装好的对象，供一条龙入口（``build_report`` / ``check_parity``）
    复用，免得二次包装。

    为什么需要它：防线 1 原先靠"构造即校验"，于是**只在走 loader 或用户显式构造
    面板时生效**；一条龙入口把裸 DataFrame 解包后直接往下传，契约层从未被调用。
    实测：前视、非交易日、复权缺行、``+inf``、因子多出行这五类坏数据在
    ``build_report`` 下**静默跑通**并生成一份看着完全正常的报告。

    Parameters
    ----------
    factor, prices, calendar : 契约对象或 DataFrame
        ``calendar`` 也接受裸 ``DatetimeIndex``（与其余入口口径一致）。
    universe, tradability, grouping, exposures, events : 契约对象或 DataFrame, 可选
    strict : bool
        ``True``（默认）时连"覆盖度"这类软检查一起做。
    assume_available_at : bool
        默认 ``True``：因子缺 ``available_at`` 时按 ``= date`` 合成
        （与 loader 一致）。**合成会让前视闸门失效** —— 这个事实会写进
        ``notice``，由调用方带进报告。
    notice : list, optional
        传一个列表进来，函数把"兜底/降级"这类事实 append 进去。

    Returns
    -------
    dict
        ``{'factor','prices','universe','tradability','grouping','exposures','events',
        'checked','cross','notices','strict'}``：
        ``checked`` 是构造并校验通过的契约名，``cross`` 是跑过的跨表检查名。
    """
    cal = as_calendar(calendar)
    checked, cross = [], []
    notices = notice if notice is not None else []

    fp = (factor if isinstance(factor, FactorPanel)
          else FactorPanel(_factor_validation_frame(_unwrap_or_frame(factor),
                                                    assume_available_at, notices)))
    pp = _wrap_for_check('prices', prices)
    up = _wrap_for_check('universe', universe)
    tp = _wrap_for_check('tradability', tradability)
    gp = _wrap_for_check('grouping', grouping)
    ep = _wrap_for_check('exposures', exposures)
    vt = _wrap_for_check('events', events)
    for nm, obj in (('FactorPanel', fp), ('PricePanel', pp), ('Universe', up),
                    ('Tradability', tp), ('Grouping', gp), ('Exposures', ep),
                    ('Events', vt)):
        if obj is not None:
            checked.append(nm)

    # 资产唯一值只算一次，两处检查共用（见 `_unique_level` 的说明）
    ua_f = _unique_assets(fp.df)
    ua_p = _unique_assets(pp.df)
    _check_same_asset_type(fp, pp, ua_f, ua_p)
    cross.append('same_asset_type')
    _check_dates_on_calendar(cal, factor=fp, prices=pp, universe=up,
                            grouping=gp, exposures=ep, strict=strict)
    cross.append('dates_on_calendar')
    _check_factor_in_prices(fp, pp)
    cross.append('factor_in_prices')
    if up is not None:
        _check_universe_covers_factor(fp, up)
        cross.append('universe_covers_factor')
    if strict:
        _check_price_coverage(fp, pp, ua_f=ua_f, ua_p=ua_p)
        cross.append('price_coverage')
    return {'factor': fp, 'prices': pp, 'universe': up, 'tradability': tp,
            'grouping': gp, 'exposures': ep, 'events': vt,
            'checked': checked, 'cross': cross, 'notices': notices,
            'strict': bool(strict)}


def _unwrap_or_frame(obj):
    """契约对象 → 其 DataFrame；DataFrame/Series 原样返回。"""
    return getattr(obj, 'df', obj)


def validate_inputs(factor, prices, calendar, universe=None,
                    tradability=None, grouping=None, exposures=None,
                    events=None, strict=True):
    """跨对象契约校验。任何一条不过就拒绝运行。

    裸 ``DataFrame`` 会自动包装成对应契约对象（按参数名判断类型）。
    **本函数只回答"过不过"（返回 ``True``）**；要拿包装好的对象（一条龙入口
    复用），用 :func:`ensure_contract`。

    Parameters
    ----------
    factor, prices, calendar : 契约对象或 DataFrame
        必填三项。``calendar`` 也接受裸 ``DatetimeIndex``。
    universe, tradability, grouping, exposures, events : 契约对象或 DataFrame, 可选
    strict : bool
        ``True``（默认）时全部检查；``False`` 时跳过"覆盖度"这类软检查，
        只保留硬性错误。

    Raises
    ------
    ContractError
    """
    ensure_contract(factor, prices, calendar, universe=universe,
                    tradability=tradability, grouping=grouping,
                    exposures=exposures, events=events, strict=strict)
    return True


# --------------------------------------------------------------------------- #
def _check_same_asset_type(factor, prices, ua_f=None, ua_p=None):
    """factor 与 prices 的 asset 层级必须同类且能对齐。"""
    af = _unique_assets(factor.df) if ua_f is None else ua_f
    ap = _unique_assets(prices.df) if ua_p is None else ua_p
    if not len(af.intersection(ap)):
        fail('validate', 'asset_mismatch',
             f'factor 的 asset 与 prices 完全没有交集。\n'
             f'  factor 例：{list(af[:3])}\n'
             f'  prices 例：{list(ap[:3])}\n'
             f'  常见原因：一边带交易所后缀（600519.SH），一边不带（600519）。')


def _check_dates_on_calendar(calendar, strict=True, **panels):
    """所有出现在数据里的日期都必须在日历上。"""
    for name, p in panels.items():
        if p is None:
            continue
        d = _unique_dates(p.df)
        off = d[~d.isin(calendar.index)]
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


def _check_price_coverage(factor, prices, min_ratio=0.5, ua_f=None, ua_p=None):
    """价格对因子的覆盖度过低时提醒（软检查）。"""
    fi = _unique_assets(factor.df) if ua_f is None else ua_f
    pi = _unique_assets(prices.df) if ua_p is None else ua_p
    ratio = len(fi.intersection(pi)) / max(len(fi), 1)
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
        # 两侧用同一口径（都前值填充）再比收益 —— 手写而非 pct_change()，
        # 见 analysis/event.py 里的说明：pandas 3.0 会把默认改成不填充并移除该关键字。
        # 组内已按日期排序，且每组只有一只票，shift(1) 与 pct_change 等价（逐位相同）。
        _a = g['a'].ffill()
        _b = g['b'].ffill()
        d = ((_a / _a.shift(1) - 1) - (_b / _b.shift(1) - 1)).abs().dropna()
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
