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


def _as_dates(dates):
    """任意日期序列 → 去重升序的 ``DatetimeIndex``。"""
    d = pd.DatetimeIndex(pd.Series(dates).dropna().unique())
    if getattr(d.dtype, 'tz', None) is not None:
        fail('split', 'tz', '`dates` 必须 tz-naive（本库全部日期都不带时区）')
    if not len(d):
        fail('split', 'empty', '`dates` 是空的，切不出样本内/外')
    return d.sort_values()


def split_is_oos(dates, *, ratio=0.7, cut=None, embargo=0, min_periods=8):
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
        在切分点**前**挖掉多少个交易日（purge）。通常给最大持有期 ——
        那些样本的前向收益会跨过切分点。默认 0（不挖，但你应当知道自己没挖）。
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

    emb = int(embargo)
    is_dates = d[:max(len(d[d < cut_date]) - emb, 0)]
    oos_dates = d[d >= cut_date]

    if len(is_dates) < min_periods or len(oos_dates) < min_periods:
        fail('split', 'too_few_periods',
             f'样本内 {len(is_dates)} 期 / 样本外 {len(oos_dates)} 期，'
             f'至少各需 {min_periods} 期（embargo={emb}）。\n'
             f'  含义：段太短时 IC / IR 本身就是噪声，"样本外有效"无从谈起。\n'
             f'  修法：放宽数据区间、调小 `ratio` 的两端差异、或减小 `embargo`。')

    return {'is': is_dates, 'oos': oos_dates, 'cut': cut_date, 'embargo': emb,
            'n_is': int(len(is_dates)), 'n_oos': int(len(oos_dates))}
