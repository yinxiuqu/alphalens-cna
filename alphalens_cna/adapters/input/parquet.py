"""Parquet 适配器 —— 认【公开目录约定】。

任何人把数据摆成下面这样，零代码即可读：

.. code-block:: text

    mydata/
    ├── prices.parquet        # 行情（列见 docs/输入数据规格.md §2.2）
    ├── xdxr.parquet          # 除权事件（用于现算复权）
    ├── tradability.parquet   # 可选
    ├── universe.parquet      # 可选
    ├── exposures.parquet     # 可选
    ├── events.parquet        # 可选
    └── factors/
        ├── roe.parquet       # date, asset, value, available_at
        └── <name>.parquet

**磁盘上用普通列**（``date`` / ``asset`` 是列，不是索引）——
方便用 DuckDB / pandas 直接看。**读进来后才转 MultiIndex。**

Examples
--------
>>> src = acna.ParquetAdapter('~/mydata')
>>> prices = acna.load_prices(source=src)
>>> roe    = acna.load_factor('roe', source=src)
"""

from __future__ import annotations

import os

import pandas as pd

from ...contract.errors import fail
from .base import InputAdapter, as_multiindex, register

PRICES = 'prices.parquet'
XDXR = 'xdxr.parquet'
FACTOR_DIR = 'factors'
OPTIONAL = {
    'universe': 'universe.parquet',
    'tradability': 'tradability.parquet',
    'exposures': 'exposures.parquet',
    'events': 'events.parquet',
}


class ParquetAdapter(InputAdapter):
    """按公开目录约定读 parquet。

    Parameters
    ----------
    root : str | Path
        数据根目录。
    strict : bool
        默认 ``True``：连 ``prices.parquet`` 都没有就直接报错并说明约定。
        ``False`` 时缺失的文件安静地返回 ``None``。
    """

    name = 'parquet'

    def __init__(self, root, strict=True, name=None):
        self.root = os.path.expanduser(str(root))
        self.strict = strict
        self.name = name or f'parquet:{os.path.basename(self.root)}'
        self._cache = {}

        if not os.path.isdir(self.root):
            fail('ParquetAdapter', 'no_root',
                 f'目录不存在：{self.root}\n'
                 f'  约定见 docs/输入数据规格.md §3。')
        if strict and not os.path.exists(os.path.join(self.root, PRICES)):
            fail('ParquetAdapter', 'no_prices',
                 f'{self.root} 下没有 `{PRICES}`。\n'
                 f'  目录约定：\n'
                 f'    prices.parquet      行情（必需）\n'
                 f'    xdxr.parquet        除权事件（现算复权时需要）\n'
                 f'    factors/<name>.parquet   因子\n'
                 f'    universe / tradability / exposures / events .parquet  可选\n'
                 f'  详见 docs/输入数据规格.md §3。')

    # -- 读文件 ---------------------------------------------------------------
    def _read(self, relpath):
        """读一个 parquet，带缓存。不存在返回 None。"""
        if relpath in self._cache:
            return self._cache[relpath]
        p = os.path.join(self.root, relpath)
        if not os.path.exists(p):
            self._cache[relpath] = None
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            fail('ParquetAdapter', 'read_failed',
                 f'读不了 {p}：{type(e).__name__}: {e}\n'
                 f'  提示：装了 pyarrow 吗？（pip install alphalens-cna[io]）')
        if not len(df):
            fail('ParquetAdapter', 'empty_file', f'{p} 是空表。')
        try:
            df = as_multiindex(df, 'ParquetAdapter')
        except Exception as e:
            raise type(e)(f'{p}：{e}') from None
        self._cache[relpath] = df
        return df

    def _optional(self, key):
        df = self._read(OPTIONAL[key])
        if df is None and self.strict:
            return None
        return df

    # -- 协议实现 -------------------------------------------------------------
    def prices(self, start=None, end=None, **kw):
        df = self._read(PRICES)
        if df is None:
            return None
        return _slice(df, start, end, 'prices')

    def xdxr(self):
        """除权事件表 —— 现算复权因子时需要。"""
        return self._read(XDXR)

    def calendar(self, start=None, end=None):
        c = self._read('calendar.parquet') if os.path.exists(
            os.path.join(self.root, 'calendar.parquet')) else None
        if c is not None:
            c = pd.DatetimeIndex(c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c)
        else:
            p = self.prices()
            c = pd.DatetimeIndex(p.index.get_level_values('date').unique())
        if start is not None:
            c = c[c >= pd.Timestamp(start)]
        if end is not None:
            c = c[c <= pd.Timestamp(end)]
        return c.sort_values()

    def factor(self, name, start=None, end=None, **kw):
        rel = f'{FACTOR_DIR}/{name}.parquet'
        df = self._read(rel)
        if df is None:
            avail = self._list_factors()
            fail('ParquetAdapter', 'unknown_factor',
                 f'找不到 {os.path.join(self.root, rel)}\n'
                 f'  已有的因子：{avail or "（无）"}\n'
                 f'  约定：每个因子一个文件 factors/<name>.parquet，'
                 f'列为 date, asset, value, available_at')
        return _slice(df, start, end, f'factor:{name}')

    def _list_factors(self):
        d = os.path.join(self.root, FACTOR_DIR)
        if not os.path.isdir(d):
            return []
        return sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                      if f.endswith('.parquet'))

    def universe(self, start=None, end=None, **kw):
        return _slice(self._optional('universe'), start, end, 'universe')

    def tradability(self, start=None, end=None, **kw):
        return _slice(self._optional('tradability'), start, end, 'tradability')

    def exposures(self, start=None, end=None, **kw):
        return _slice(self._optional('exposures'), start, end, 'exposures')

    def events(self, name=None, start=None, end=None, **kw):
        return _slice(self._optional('events'), start, end, 'events')


def _slice(df, start, end, what):
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
        fail('ParquetAdapter', 'empty_range',
             f'`{what}` 在 {start} ~ {end} 区间内没有数据。')
    return out


register('parquet', ParquetAdapter)
