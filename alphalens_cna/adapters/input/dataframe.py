"""DataFrame 适配器 —— **默认路径，零依赖**。

直接把手上已有的 DataFrame 喂进来。这是最通用的一种：
不管你数据从哪来（CSV、数据库、API），先读成 DataFrame 就能用。

Examples
--------
>>> import alphalens_cna as acna
>>> src = acna.DataFrameAdapter(prices=px_df, factors={'roe': roe_df},
...                             calendar=cal_idx)
>>> prices = acna.load_prices(source=src)
>>> roe    = acna.load_factor('roe', source=src)
"""

from __future__ import annotations

import pandas as pd

from ...contract.errors import fail
from .base import InputAdapter, as_multiindex, register


class DataFrameAdapter(InputAdapter):
    """把已有的 DataFrame 包成数据源。

    Parameters
    ----------
    prices : DataFrame, 可选
        行情，列见 docs/输入数据规格.md §2.2。
    factors : dict[str, DataFrame], 可选
        因子表：``{'roe': df, ...}``。
    calendar : DatetimeIndex, 可选
        交易日历。不给则从 prices/factors 的日期推断。
    universe, tradability, exposures, events : DataFrame, 可选
    grouping : dict[str, DataFrame] 或 DataFrame, 可选
    name : str
        数据源名，只用于报错信息。
    """

    def __init__(self, prices=None, factors=None, calendar=None,
                 universe=None, tradability=None, exposures=None,
                 events=None, grouping=None, name='dataframe'):
        self.name = name
        self._prices = as_multiindex(prices, 'DataFrameAdapter')
        self._factors = {k: as_multiindex(v, 'DataFrameAdapter')
                         for k, v in (factors or {}).items()}
        self._calendar = pd.DatetimeIndex(calendar) if calendar is not None else None
        self._universe = as_multiindex(universe, 'DataFrameAdapter')
        self._tradability = as_multiindex(tradability, 'DataFrameAdapter')
        self._exposures = as_multiindex(exposures, 'DataFrameAdapter')
        self._events = as_multiindex(events, 'DataFrameAdapter')
        if grouping is None:
            self._grouping = {}
        elif isinstance(grouping, dict):
            self._grouping = {k: as_multiindex(v, 'DataFrameAdapter')
                              for k, v in grouping.items()}
        else:
            self._grouping = {'default': as_multiindex(grouping, 'DataFrameAdapter')}

        if self._prices is None and not self._factors:
            fail('DataFrameAdapter', 'empty',
                 'prices 和 factors 至少要给一个')

    # -- 协议实现 -------------------------------------------------------------
    def prices(self, start=None, end=None, **kw):
        return _slice(self._prices, start, end, 'DataFrameAdapter', 'prices')

    def calendar(self, start=None, end=None):
        if self._calendar is not None:
            c = self._calendar
            if start is not None:
                c = c[c >= pd.Timestamp(start)]
            if end is not None:
                c = c[c <= pd.Timestamp(end)]
            return c
        return None

    def factor(self, name, start=None, end=None, **kw):
        if name not in self._factors:
            fail('DataFrameAdapter', 'unknown_factor',
                 f'没有因子 `{name}`。已有：{sorted(self._factors)}\n'
                 f'  提示：构造时用 factors={{"{name}": df}} 传入。')
        return _slice(self._factors[name], start, end, 'DataFrameAdapter', name)

    def universe(self, start=None, end=None, **kw):
        return _slice(self._universe, start, end, 'DataFrameAdapter', 'universe')

    def tradability(self, start=None, end=None, **kw):
        return _slice(self._tradability, start, end, 'DataFrameAdapter', 'tradability')

    def exposures(self, start=None, end=None, **kw):
        return _slice(self._exposures, start, end, 'DataFrameAdapter', 'exposures')

    def grouping(self, name='default', start=None, end=None, **kw):
        if name not in self._grouping:
            if len(self._grouping) == 1:
                name = next(iter(self._grouping))
            else:
                fail('DataFrameAdapter', 'unknown_grouping',
                     f'没有分组 `{name}`。已有：{sorted(self._grouping)}')
        return _slice(self._grouping[name], start, end, 'DataFrameAdapter', name)

    def events(self, name=None, start=None, end=None, **kw):
        return _slice(self._events, start, end, 'DataFrameAdapter', 'events')


def _slice(df, start, end, contract, what):
    """按日期区间切片。索引 level 0 是 date。"""
    if df is None:
        return None
    if start is None and end is None:
        return df
    d = df.index.get_level_values('date')
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= d >= pd.Timestamp(start)
    if end is not None:
        mask &= d <= pd.Timestamp(end)
    out = df[mask.values]
    if len(out) == 0:
        fail(contract, 'empty_range',
             f'`{what}` 在 {start} ~ {end} 区间内没有任何数据。')
    return out


register('dataframe', DataFrameAdapter)
register('df', DataFrameAdapter)
