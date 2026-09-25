"""可成交性判定 —— A 股制度约束。

判定"某天开盘能不能买 / 能不能卖"。**全部基于原始不复权价与当日之前的信息，零前视。**

移植自一套经实盘检验的 A 股执行过滤逻辑，并补齐其缺口：

===========================================  ==========================================
缺口                                          本模块的处理
===========================================  ==========================================
``board_pct`` 不认北交所（返回 10%，应 30%）   补 ``43x/83x/87x/88x/92x`` → 30%
``limit_status`` 的 2/3、5/6 语义未定          实测破解：**3/6 = 一字板**（见下）
无面板级接口                                  提供 ``compute_tradability``
新股窗口只有布尔开关                          改用天数（默认 60），见下
===========================================  ==========================================

``limit_status`` 语义（本会话实测，各 250 样本）
------------------------------------------------
===========  ==============  ==================
值            含义            开盘即在板的比例
===========  ==============  ==================
2             盘中封涨停      **1.2%**
3             **一字涨停**    **71.6%**
5             盘中跌停        2.4%
6             **一字跌停**    **62.7%**
===========  ==============  ==================

2 与 3 相差 **60 倍** —— 所以「收盘在板」分两种，
只有 **3/6 才意味着开盘买不进/卖不出**。

⚠️ 但 ``limit_status`` 是**收盘**状态，用它门控**当日开盘**成交属于**用了未来信息**。
所以：**首选自算**（用 ``raw_prev_close`` 与当日 ``raw_open``，开盘时点信息完备）；
仅在拿不到 ST 名称等必要信息时，才降级用 ``limit_status``，并通过
``source`` 字段标明本行判定来自哪条路径。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.calendar import as_calendar
from ..contract.errors import fail

__all__ = [
    'board_pct', 'is_st_name', 'limit_prices', 'limit_ratio_of',
    'compute_tradability', 'BOARD_LIMIT', 'EPS',
]

#: 涨跌停判定容差。
#: 用 1e-6 而非 0.01 —— A 股最小变动 0.01 元，开盘价**恰好等于**涨停价才算封死；
#: 用 0.01 会把"低于涨停价 1 分"误判为买不进。
EPS = 1e-6

#: 板块涨跌幅上限
BOARD_LIMIT = {
    'main': 0.10,        # 沪深主板
    'star': 0.20,        # 科创板 688/689
    'gem': 0.20,         # 创业板 300/301/302
    'bse': 0.30,         # 北交所 43x/83x/87x/88x/92x
}
#: ST 股票的涨跌幅（主板 5%；创业/科创仍 20%；北交所仍 30%）
ST_LIMIT = 0.05

_BSE_PREFIX = ('43', '83', '87', '88', '92')
_STAR_PREFIX = ('688', '689')
_GEM_PREFIX = ('300', '301', '302')


# --------------------------------------------------------------------------- #
# 板块与 ST
# --------------------------------------------------------------------------- #
def board_of(code) -> str:
    """代码 → 板块标签（``main`` / ``star`` / ``gem`` / ``bse``）。"""
    c = str(code).zfill(6)
    if c.startswith(_STAR_PREFIX):
        return 'star'
    if c.startswith(_GEM_PREFIX):
        return 'gem'
    if c.startswith(_BSE_PREFIX):
        return 'bse'
    return 'main'


def board_pct(code) -> float:
    """板块涨跌幅上限（不含 ST）。"""
    return BOARD_LIMIT[board_of(code)]


def is_st_name(name) -> bool:
    """名称以 ``ST`` / ``*ST`` / ``SST`` / ``S*ST`` 开头 → ST。

    只用前缀，避免误伤名称中间含 ST 的正常股票（如"华ST科技"）。
    必须传 **PIT 名称**（当日名称），不是最新名称。
    """
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return False
    n = str(name).upper().replace(' ', '')
    return n.startswith(('ST', '*ST', 'SST', 'S*ST'))


def limit_ratio_of(code, name=None) -> float:
    """该股票当日适用的涨跌幅比例。

    * 主板 ST → 5%
    * 创业/科创 ST → **仍 20%**（不变）
    * 北交所 → 30%（ST 亦不改）
    """
    pct = board_pct(code)
    if is_st_name(name) and pct == BOARD_LIMIT['main']:
        return ST_LIMIT
    return pct


def limit_prices(prev_close, code, name=None, unlimited=False):
    """涨跌停价 ``(limit_up, limit_down)``。

    Parameters
    ----------
    prev_close : float
        **原始不复权**前收盘价（不是复权价！）。
    code : str
    name : str, 可选
        当日 PIT 名称，用于判 ST。
    unlimited : bool
        是否处于无涨跌幅窗口（新股上市前 5 个交易日）。
        默认 ``new_stock_days=60`` 已把新股整段剔除，正常用不到。

    Returns
    -------
    (float | None, float | None)

    ⚠️ **必须传原始价**。实测 600519 在 2016-08-26 的涨停价：
    用原始价算 **333.31**（对），用当时的前复权价算 **270.01** —— **差 23%**。
    """
    if unlimited or prev_close is None or not np.isfinite(prev_close) or prev_close <= 0:
        return None, None
    pct = limit_ratio_of(code, name)
    up = round(float(prev_close) * (1 + pct), 2)
    down = round(float(prev_close) * (1 - pct), 2)
    return up, down


# --------------------------------------------------------------------------- #
# 面板级判定
# --------------------------------------------------------------------------- #
def compute_tradability(prices, calendar=None, *, names=None,
                        limit_status=None, list_dates=None,
                        new_stock_days=60, use_limit_status=False,
                        validate=True):
    """逐 (date, asset) 判定「当日开盘」能否成交。

    Parameters
    ----------
    prices : PricePanel | DataFrame
        必须含 ``raw_open`` / ``raw_close``（或 ``adj_*`` 对应）。**只用 raw**。
    calendar : Calendar, 可选
        用于算 ``listed_days``。
    names : Series, 可选
        索引 ``MultiIndex(date, asset)``，值为当日名称。用于判 ST。
        **拿不到就退化为"按板块比例"，主板 ST 的 5% 会被算成 10%。**
    limit_status : Series, 可选
        指数同上的 ``stock_daily_basic.limit_status``。
    list_dates : Series | dict, 可选
        ``asset -> 上市日``。不给则用行情里该股首个交易日近似。
    new_stock_days : int
        上市后多少个交易日内视为新股（默认 **60**）。见 Notes。
    use_limit_status : bool
        ``True`` 时**优先**用 ``limit_status ∈ {3,6}`` 判一字板（有轻度前视）。
        默认 ``False``：优先自算。

    Returns
    -------
    DataFrame
        索引 ``MultiIndex(date, asset)``，列：

        ===================  ======  ====================================
        列                   类型    说明
        ===================  ======  ====================================
        ``suspended``        bool    停牌（无开盘价 / limit_status==0）
        ``is_st``            bool    名称含 ST
        ``listed_days``      int32   上市第几个交易日
        ``limit_up_price``   f64     涨停价（原始价口径）
        ``limit_down_price`` f64     跌停价
        ``open_at_limit_up`` bool    开盘即在涨停（买不进）
        ``open_at_limit_down`` bool  开盘即在跌停（卖不出）
        ``can_buy_open``     bool
        ``can_sell_open``    bool
        ``reason``           str     不可成交原因；可成交为 ``'ok'``
        ``calc_source``      str     ``'price'``（自算）/ ``'limit_status'``（查表）
        ===================  ======  ====================================

    Notes
    -----
    **新股默认剔除 60 个交易日**（D3）。这带来一个附带好处：
    A 股新股涨跌幅规则复杂（2023-04-10 前主板首日 ±44%/−36% 次日起 10%；
    注册制板块前 5 日无限制），**60 > 5 使这些历史规则不必建模**。
    """
    df = getattr(prices, 'df', prices)
    required = ('raw_open',)
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail('tradability', 'missing_column',
             f'缺 {missing}。可成交性判定**必须**用原始不复权价 —— '
             f'复权价算出的涨跌停价会差 23%。')
    if 'raw_close' not in df.columns:
        fail('tradability', 'missing_column',
             '缺 `raw_close`（用于算 prev_close / 判停牌）')

    idx = df.index
    out = pd.DataFrame(index=idx)

    # ── prev_close：优先用数据里的，否则同股票前移一位 ──
    if 'prev_close' in df.columns:
        prev = pd.to_numeric(df['prev_close'], errors='coerce')
    else:
        prev = df.sort_index().groupby(level='asset')['raw_close'].shift(1)
        prev = prev.reindex(idx)
    out['prev_close'] = prev

    # ── 停牌 ──
    open_px = pd.to_numeric(df['raw_open'], errors='coerce')
    suspended = open_px.isna() | (open_px <= 0)
    if limit_status is not None:
        ls = pd.to_numeric(limit_status.reindex(idx), errors='coerce')
        suspended = suspended | (ls == 0)
    else:
        ls = pd.Series(np.nan, index=idx)
    out['suspended'] = suspended.fillna(True).astype(bool)

    # ── ST ──
    if names is not None:
        nm = names.reindex(idx)
        out['is_st'] = nm.map(is_st_name).fillna(False).astype(bool)
    else:
        nm = pd.Series(None, index=idx, dtype=object)
        out['is_st'] = False

    # ── 上市天数 ──
    listed, known = _listed_days(idx, list_dates, calendar)
    out['listed_days'] = listed
    # 「已知」= 调用方给了上市日。没给就只能靠面板内序号，
    # 那个数在截断面板上是错的 —— 见下方新股过滤的安全默认。
    out['listed_days_known'] = known

    # ── 涨跌停价（自算，只用 raw）──
    ratio = pd.Series(
        [limit_ratio_of(a, n) for a, n in zip(idx.get_level_values('asset'), nm)],
        index=idx, dtype=float)
    lim_up = (prev * (1 + ratio)).round(2)
    lim_dn = (prev * (1 - ratio)).round(2)
    out['limit_ratio'] = ratio
    out['limit_up_price'] = lim_up
    out['limit_down_price'] = lim_dn

    # ── 开盘是否在板 ──
    at_up = (open_px.notna() & lim_up.notna() & (open_px >= lim_up - EPS))
    at_dn = (open_px.notna() & lim_dn.notna() & (open_px <= lim_dn + EPS))
    src = pd.Series('price', index=idx, dtype=object)

    # 降级：拿不到 PIT 名称 / 缺前收 → 用 limit_status 查表（3=一字涨停, 6=一字跌停）
    need_fallback = (names is None) | prev.isna()
    if use_limit_status or need_fallback.any():
        use = (ls.isin([3, 6])) & (need_fallback | use_limit_status)
        at_up = at_up.where(~use, ls == 3)
        at_dn = at_dn.where(~use, ls == 6)
        src = src.where(~use, 'limit_status')

    out['open_at_limit_up'] = at_up.fillna(False).astype(bool)
    out['open_at_limit_down'] = at_dn.fillna(False).astype(bool)
    out['calc_source'] = src

    # ── 新股窗口 ──
    # ⚠️ 安全默认：**上市日不可靠时，不启用新股过滤**。
    # 理由：`listed_days` 的兜底是"组内序号"，隐含假设面板从上市首日开始。
    # 截断面板（如只取一天）会让所有股票都变成"第 1 天"，
    # 于是**静默排除全部数据** —— 这比"不排除"危险得多。
    if new_stock_days and not known:
        newbie = pd.Series(False, index=idx)
        excluded_note = ('上市日未知（未提供 list_dates），已跳过新股过滤。'
                         '如需启用，请传 list_dates=asset→上市日')
    else:
        newbie = (listed <= new_stock_days).fillna(False)
        excluded_note = None
    out['is_new_stock'] = newbie.astype(bool)

    # ── 结论 ──
    out['can_buy_open'] = ~(out['suspended'] | out['open_at_limit_up'] | out['is_new_stock'])
    out['can_sell_open'] = ~(out['suspended'] | out['open_at_limit_down'])

    # 买卖原因分开 —— 新股只是**不能买**，持有的话照样能卖，
    # 共用一个 reason 字段会把卖出的原因错报成 new_stock。
    out['reason_buy'] = np.select(
        [out['suspended'], out['is_new_stock'], out['open_at_limit_up']],
        ['suspended', 'new_stock', 'limit_up'], default='ok')
    out['reason_sell'] = np.select(
        [out['suspended'], out['open_at_limit_down']],
        ['suspended', 'limit_down'], default='ok')

    if calendar is not None:
        od = pd.DatetimeIndex(idx.get_level_values('date').unique())
        as_calendar(calendar).validate_dates(od, contract='tradability')
    if excluded_note:
        out.attrs['new_stock_filter'] = excluded_note
    return out


def _listed_days(idx, list_dates, calendar):
    """上市第几个**交易日**（1 = 首日）。

    Returns
    -------
    (Series, bool)
        ``(上市天数, 是否精确)``。

    **只有给了 ``list_dates`` 才算精确。** 兜底用的"组内序号"隐含假设
    面板从上市首日开始 —— 截断面板（例如只取一天）会让所有股票都变成第 1 天，
    进而被新股过滤**静默全部排除**。所以调用方在 ``exact=False`` 时必须
    放弃新股过滤（见 ``compute_tradability``）。
    """
    assets = pd.Index(idx.get_level_values('asset'), name='asset')
    dates = pd.DatetimeIndex(idx.get_level_values('date'))
    df = pd.DataFrame({'d': dates}, index=assets)
    first_obs = df.groupby(level='asset')['d'].transform('min').values

    # 没有 list_dates → 无法知道真实上市日，只能给"面板内序号"（不可靠）
    if list_dates is None:
        return pd.Series(df.groupby(level='asset').cumcount().values + 1,
                         index=idx, dtype='Int32'), False

    ld = list_dates if isinstance(list_dates, pd.Series) else pd.Series(list_dates)
    mapped = pd.to_datetime(
        pd.Series(assets.values).map(ld).values, errors='coerce')
    use = ~pd.isna(mapped)
    if not use.any():
        return pd.Series(df.groupby(level='asset').cumcount().values + 1,
                         index=idx, dtype='Int32'), False

    start = np.asarray(first_obs).copy()
    start[use] = mapped[use].astype('datetime64[ns]')
    start = pd.DatetimeIndex(start)

    if calendar is not None:
        # ⚠️ calendar.sessions() 只在日历**覆盖范围内**数得准。
        # 若只传了分析窗口（如 2024 一年）的日历，而上市日在 1991 年，
        # 数出来是「窗口内的交易日数」而非真实上市天数 ——
        # 实测会把老股票误判成新股（120/484 行）。
        # 所以上市日早于日历起点的，退回自然日折算（对 60 日过滤足够准）。
        cal_lo = np.datetime64(calendar.index[0], 'ns')
        start_arr = np.asarray(start.values)
        in_cal = start_arr >= cal_lo
        days = np.empty(len(dates), dtype=float)
        if in_cal.any():
            # sessions() 是闭区间计数：上市当天 = 1
            days[in_cal] = [calendar.sessions(a, b)
                            for a, b in zip(start[in_cal], dates[in_cal])]
        if (~in_cal).any():
            span = (dates.values[~in_cal] - start_arr[~in_cal]) / np.timedelta64(1, 'D')
            days[~in_cal] = np.rint(span * 250 / 365) + 1
        return pd.Series(days.astype(int), index=idx, dtype='Int32'), bool(use.all())
    # 没有日历时按自然日折算 —— 是估算，但"上市日已知"，过滤仍有意义
    span = (dates.values - start.values) / np.timedelta64(1, 'D')
    days = np.rint(span * 250 / 365).astype(int) + 1
    return pd.Series(days, index=idx, dtype='Int32'), bool(use.all())
