"""交易日历。

**自持，不依赖 pandas 的 ``freq`` 推断。**

为什么不用 ``pd.DatetimeIndex.freq``
------------------------------------
alphalens 在 ``df.index.levels[0].freq = freq`` 那一步会崩：
月频因子配日频价格时，pandas 推断出 ``None``，赋值直接抛
``ValueError: Inferred frequency None from passed values does not conform
to passed frequency C``。

根因是**把节奏判断交给了 pandas 的推断**。本库改为**显式持有交易日序列**，
所有"前进 N 个交易日"的操作都走日历查表，不做推断。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd

from .errors import fail


class Calendar:
    """交易日历。

    Parameters
    ----------
    dates : DatetimeIndex | array-like
        交易日序列。会排序、去重。
        如果传进来的日期本身有重复，会去重而**不报错** —— 因为日历的语义就是
        "哪些日子开市"，重复无意义。

    Examples
    --------
    >>> cal = Calendar(pd.bdate_range('2024-01-01', periods=10))
    >>> cal.shift(pd.Timestamp('2024-01-03'), 2)
    Timestamp('2024-01-05 00:00:00')
    """

    def __init__(self, dates):
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
        if getattr(idx.dtype, 'tz', None) is not None:
            fail('Calendar', 'timezone',
                 f'日历必须 tz-naive（全库统一），收到 tz={idx.dtype.tz}')
        if len(idx) == 0:
            fail('Calendar', 'empty', '交易日历为空')
        idx = idx.unique().sort_values()
        self._idx = idx
        self._arr = idx.values.astype('datetime64[ns]')

    # -- 构造 -----------------------------------------------------------------
    @classmethod
    def from_panels(cls, *panels, extra=None):
        """从若干契约对象的日期层级汇总出日历。

        用途：手上没有官方交易日历时，退而求其次用"数据里出现过的日期"。
        ⚠️ 这只在**股票池覆盖足够全**时才等价于真日历；停市日会缺失。
        """
        parts = []
        for p in panels:
            if p is None:
                continue
            df = getattr(p, 'df', p)
            if isinstance(df, pd.DataFrame) and isinstance(df.index, pd.MultiIndex):
                parts.append(pd.DatetimeIndex(df.index.get_level_values('date')))
        if extra is not None:
            parts.append(pd.DatetimeIndex(pd.to_datetime(extra)))
        if not parts:
            fail('Calendar', 'from_panels', '没有可用来构造日历的对象')
        return cls(pd.DatetimeIndex(np.concatenate([p.values for p in parts])).unique())

    # -- 基本 -----------------------------------------------------------------
    @property
    def index(self) -> pd.DatetimeIndex:
        return self._idx

    def __len__(self):
        return len(self._idx)

    def __contains__(self, d) -> bool:
        return pd.Timestamp(d) in self._idx

    def __repr__(self):
        return (f'<Calendar {len(self._idx):,} 个交易日 · '
                f'{self._idx[0]:%Y-%m-%d} ~ {self._idx[-1]:%Y-%m-%d}>')

    # -- 查表运算（不依赖 freq）-----------------------------------------------
    def _pos(self, d) -> int:
        """日期在日历中的位置；不在日历上则返回应插入的位置。"""
        t = np.datetime64(pd.Timestamp(d), 'ns')
        return bisect_left(self._arr, t)

    def contains(self, d) -> bool:
        p = self._pos(d)
        return p < len(self._arr) and self._arr[p] == np.datetime64(pd.Timestamp(d), 'ns')

    def shift(self, d, n: int):
        """从日期 ``d`` 前进（``n>0``）或后退（``n<0``）``n`` 个交易日。

        ``d`` 不必是交易日 —— 会先对齐到"之后最近的交易日"（``n>0``）
        或"之前最近的交易日"（``n<0``）。
        越界返回 ``None``。
        """
        if n == 0:
            return pd.Timestamp(d)
        if n > 0:
            p = bisect_right(self._arr, np.datetime64(pd.Timestamp(d), 'ns')) - 1 + n
        else:
            p = bisect_left(self._arr, np.datetime64(pd.Timestamp(d), 'ns')) + n
        if p < 0 or p >= len(self._arr):
            return None
        return pd.Timestamp(self._arr[p])

    def next(self, d):
        """下一个交易日。"""
        p = bisect_right(self._arr, np.datetime64(pd.Timestamp(d), 'ns'))
        return pd.Timestamp(self._arr[p]) if p < len(self._arr) else None

    def prev(self, d):
        """上一个交易日。"""
        p = bisect_left(self._arr, np.datetime64(pd.Timestamp(d), 'ns')) - 1
        return pd.Timestamp(self._arr[p]) if p >= 0 else None

    def between(self, start, end) -> pd.DatetimeIndex:
        """闭区间内的交易日。"""
        lo = bisect_left(self._arr, np.datetime64(pd.Timestamp(start), 'ns'))
        hi = bisect_right(self._arr, np.datetime64(pd.Timestamp(end), 'ns'))
        return self._idx[lo:hi]

    def sessions(self, start, end) -> int:
        """区间内有几个交易日 —— 用于把"自然日"换成"交易日"。"""
        return len(self.between(start, end))

    def validate_dates(self, dates, contract='Calendar', sample=3):
        """校验一批日期都在日历上；不在则抛错。"""
        d = pd.DatetimeIndex(pd.to_datetime(dates)).unique()
        off = d[~d.isin(self._idx)]
        if len(off):
            fail(contract, 'off_calendar',
                 f'{len(off)} 个日期不在交易日历上：'
                 f'{"; ".join(str(x)[:10] for x in off[:sample])}'
                 f'{" …" if len(off) > sample else ""}\n'
                 f'  含义：拿非交易日当交易日，前向收益会算错。')
