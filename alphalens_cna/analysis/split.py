"""样本内 / 样本外拆分（稳健性校验的第一道：泛化能力）。

两条纪律，缺一条这个拆分就没意义：

1. **时序数据不能随机切。** 随机切会让相邻期的重叠收益跨到两边 ——
   "样本外"于是变成"样本内的近似复制"，乐观偏差照旧，还多了一层"我做过样本外"的错觉。
   所以这里**只按时间顺序切**。
2. **切分点会泄漏。** 持有期 ``h`` 的样本，其前向收益**跨过切分点** ——
   等于让样本内"看到"样本外的价格。``embargo`` 用来在切分点前挖掉一段
   （通常给最大持有期），把这类样本从样本内里剔掉。

第 2 条是文章没提、但机构里必备的细节（purge / embargo）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail

__all__ = ['split_is_oos']


def _purge_label(mode, emb, purge_horizons, entry_lag):
    """报错里说清"挖掉的那一段"是怎么算出来的 —— 别让人猜口径。"""
    if mode == 'exact':
        return (f'精确 purge：入场滞后 {int(entry_lag)} + 最大持有期 '
                f'{int(np.max(list(purge_horizons) or [0]))} 个交易日')
    return f'embargo={emb} 期分析日期'


def _split_slices(d, *, ratio, cut, embargo, calendar, purge_horizons, entry_lag):
    """算好样本内 / 样本外两段与 purge 量 —— **只做算术，不判期数、不报错**。

    单独抽出来是为了让 :func:`_feasible_ratio_hint` 能扫 ratio 而**不重入**
    :func:`split_is_oos` 的报错路径：后者在期数不足时会回头调用提示函数，
    提示函数再调它，就是无限递归（0.4.2 修）。

    Returns
    -------
    (is_dates, oos_dates, cut_date, embargo_used, mode, purged)
    """
    if cut is not None:
        c = pd.Timestamp(cut)
        k = int(np.searchsorted(d.values, np.datetime64(c), side='left'))
        cut_date = d[k]
    else:
        k = int(round(len(d) * float(ratio)))
        cut_date = d[min(max(k, 1), len(d) - 1)]

    oos_dates = d[d >= cut_date]
    if purge_horizons is None:
        emb = int(embargo)
        is_dates = d[:max(len(d[d < cut_date]) - emb, 0)]
        mode, purged = 'positions', None
    else:
        # ★ 精确 purge：按**交易日历**算每个候选样本的出场日，剔掉碰过切分日的。
        #   `embargo` 数的是"分析日期期数"，而持有期是"交易日数" —— 非日频面板上
        #   两者差一个数量级（月末：1 期 = 1 个月），所以不能拿 embargo 表达持有期。
        cal = pd.DatetimeIndex(getattr(calendar, 'index', calendar))
        lag = int(entry_lag)
        hmax = int(np.max(list(purge_horizons) or [0]))
        cand = d[d < cut_date]
        pos = cal.get_indexer(cand)
        pos_cut = int(cal.get_indexer([cut_date])[0])
        bad = (pos < 0) | (pos + lag + hmax >= pos_cut)
        is_dates = cand[~bad]
        emb = embargo if embargo else 0
        mode, purged = 'exact', int(bad.sum())
    return is_dates, oos_dates, cut_date, emb, mode, purged


def _feasible_ratio_hint(d, cut_date, purge_horizons, calendar, entry_lag,
                         embargo, min_periods, n_try=200):
    """扫一遍 ratio，给出**仍然可行**的区间（这就是"我到底需要多长样本"的答案）。

    ⚠️ 这里必须调 :func:`_split_slices` 而**不是** :func:`split_is_oos` ——
    后者在期数不足时会回头调用本函数，用公开入口就会无限递归（0.4.2 修）。
    """
    ok = []
    for r in np.linspace(0.05, 0.95, n_try):
        is_d, oos_d, _, _, _, _ = _split_slices(
            d, ratio=float(r), cut=None, embargo=int(embargo),
            calendar=calendar, purge_horizons=purge_horizons, entry_lag=entry_lag)
        if len(is_d) >= min_periods and len(oos_d) >= min_periods:
            ok.append(float(r))
    if not ok:
        return (f'当前数据下**没有可行的 ratio**（N={len(d)} 太短）—— '
                f'只能放宽数据区间。')
    return f'可行的 ratio 约 [{min(ok):.2f}, {max(ok):.2f}]（N={len(d)}）。'


def _as_dates(dates):
    """任意日期序列 → 去重升序的 ``DatetimeIndex``。"""
    d = pd.DatetimeIndex(pd.Series(dates).dropna().unique())
    if getattr(d.dtype, 'tz', None) is not None:
        fail('split', 'tz', '`dates` 必须 tz-naive（本库全部日期都不带时区）')
    if not len(d):
        fail('split', 'empty', '`dates` 是空的，切不出样本内/外')
    return d.sort_values()


def split_is_oos(dates, *, ratio=0.7, cut=None, embargo=0, min_periods=8,
                 calendar=None, purge_horizons=None, entry_lag=1):
    """按**时间顺序**把样本切成样本内 / 样本外两段。

    Parameters
    ----------
    dates : DatetimeIndex | Series | list
        参与分析的交易日（通常取清洗后 ``CleanResult.data`` 的日期层）。
        内部会去重升序。
    ratio : float
        样本内占比，``(0, 1)``，默认 ``0.7``。给了 ``cut`` 时忽略。
    cut : str | Timestamp, 可选
        直接指定切分日（**切分日归样本外**）。给了它就用它，而不是按比例。
        事前冻结的样本外通常用这个：先写下日期，再跑数据。
    embargo : int
        在切分点**前**挖掉多少**期分析日期**（purge）。⚠️ 单位是"你在 ``dates``
        里数多少期"，**不是交易日** —— 月频调仓时 1 期 = 1 个月。
        默认 0（不挖，但你应当知道自己没挖）。

        ⚠️ 正因为它数的是"期数"，用它去表达"持有期那么长"在非日频面板上会失真：
        月末面板上 `embargo=22`（本意 22 个交易日）会挖掉 22 **个月**。
        要按持有期精确挖，请给 ``calendar`` + ``purge_horizons``。
    calendar : Calendar | DatetimeIndex, 可选
        交易日历。给了它 + ``purge_horizons`` 就走**精确 purge**（推荐）：
        直接剔掉"前向收益窗口碰到切分日之后"的那些样本内样本。
    purge_horizons : int | Sequence[int], 可选
        持有期（**交易日数**，与 ``forward_returns`` 的 horizons 同口径）。
        精确 purge 会按 ``entry_lag + max(purge_horizons)`` 判断出场日。
    entry_lag : int
        入场滞后（交易日）：``entry='next_open'`` 是 1，``'close'``/``'same_open'`` 是 0。
        请跟着 ``ReturnModel`` 走，别硬编码。
    min_periods : int
        两段各至少要多少期，默认 8。不足就报错而不是给一段空洞的统计。

    Returns
    -------
    dict
        ``{'is', 'oos', 'cut', 'embargo', 'n_is', 'n_oos'}``：
        ``is`` / ``oos`` 是 ``DatetimeIndex``，``cut`` 是切分日。
    """
    d = _as_dates(dates)
    if not (0.0 < float(ratio) < 1.0):
        fail('split', 'bad_ratio', f'`ratio` 必须在 (0,1) 内，收到 {ratio!r}')
    if int(embargo) < 0:
        fail('split', 'bad_embargo', f'`embargo` 不能为负，收到 {embargo!r}')

    if purge_horizons is not None and calendar is None:
        fail('split', 'need_calendar',
             '给了 `purge_horizons` 就必须同时给 `calendar` —— '
             '精确 purge 要按**交易日**算出场日，没有日历算不了。\n'
             '  修法：传 calendar=…；或改用 `embargo=期数`（按分析日期期数挖）。')
    if purge_horizons is not None and int(embargo):
        fail('split', 'both_purge_modes',
             '`embargo`（按分析日期期数）与 `purge_horizons`（按交易日精确）'
             '是两种口径，不能同时给 —— 会互相打架。\n'
             '  修法：二选一。推荐 `calendar` + `purge_horizons`。')

    if cut is not None:
        c = pd.Timestamp(cut)
        if getattr(getattr(c, 'tzinfo', None), 'tzinfo', None) is not None:
            fail('split', 'tz', '`cut` 必须 tz-naive')
        k = int(np.searchsorted(d.values, np.datetime64(c), side='left'))
        if k == 0 or k == len(d):
            fail('split', 'cut_outside',
                 f'切分日 {c:%Y-%m-%d} 落在数据范围 '
                 f'{d[0]:%Y-%m-%d} ~ {d[-1]:%Y-%m-%d} 之外（或正好在边界）。\n'
                 f'  修法：换成区间内的日期，或改用 `ratio=`。')
        cut_date = d[k]
    else:
        k = int(round(len(d) * float(ratio)))
        cut_date = d[min(max(k, 1), len(d) - 1)]

    is_dates, oos_dates, cut_date, emb, mode, purged = _split_slices(
        d, ratio=ratio, cut=cut_date, embargo=embargo, calendar=calendar,
        purge_horizons=purge_horizons, entry_lag=entry_lag)

    if len(is_dates) < min_periods or len(oos_dates) < min_periods:
        fail('split', 'too_few_periods',
             f'样本内 {len(is_dates)} 期 / 样本外 {len(oos_dates)} 期，'
             f'至少各需 {min_periods} 期（{_purge_label(mode, emb, purge_horizons, entry_lag)}）。\n'
             f'  含义：段太短时 IC / IR 本身就是噪声，"样本外有效"无从谈起。\n'
             f'  下界：可用期数 N 至少要 ≈ purge 期数 + 2×min_periods'
             f'（当前 N={len(d)}，purge≈{len(d) - len(is_dates)}）。\n'
             f'  ⚠️ 提高 `ratio` 会**同时压缩样本外**，两头互相挤 —— 可行区间往往比想象窄。\n'
             f'  {_feasible_ratio_hint(d, cut_date, purge_horizons, calendar, entry_lag, embargo, min_periods)}\n'
             f'  修法：放宽数据区间（最有效）、减小 purge（精确模式下换更短持有期）、'
             f'或把 `ratio` 挪进上面的可行区间。')

    return {'is': is_dates, 'oos': oos_dates, 'cut': cut_date, 'embargo': emb,
            'n_is': int(len(is_dates)), 'n_oos': int(len(oos_dates)),
            'mode': mode, 'purged': purged}
