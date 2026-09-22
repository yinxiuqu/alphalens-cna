"""输入适配器协议。

**核心包不认识任何具体数据源。** 适配器的职责只有一件：
把某个数据源翻译成契约层要求的 DataFrame。

私有数据源怎么接
----------------
写一个类实现本协议，放在 ``examples/``（**不进核心包**）：:

    from alphalens_cna.adapters.input.base import InputAdapter

    class MyAdapter(InputAdapter):
        def prices(self, start=None, end=None, **kw):
            df = ...          # 从你的库读
            return df         # 列名/索引符合 docs/输入数据规格.md

库更新时只要协议不变，私有适配器一行都不用改。
"""

from __future__ import annotations

import pandas as pd

from ...contract.errors import fail

__all__ = ['InputAdapter', 'AdapterBase', 'register', 'get_source', 'list_sources']

# 内置数据源注册表：名字 → 工厂
_SOURCES = {}


def register(name, factory):
    """注册一个数据源。"""
    _SOURCES[name] = factory
    return factory


def list_sources():
    """已注册的数据源名。"""
    return sorted(_SOURCES)


def get_source(name, **kw):
    """按名字取数据源实例。"""
    if name not in _SOURCES:
        fail('loader', 'unknown_source',
             f'未知数据源 `{name}`。已注册：{list_sources()}\n'
             f'  提示：私有数据源请自己写适配器（见 examples/），'
             f'或直接用 acna.PricePanel(df) 传 DataFrame。')
    return _SOURCES[name](**kw)


class InputAdapter:
    """数据源协议（基类，子类按需覆盖方法）。

    所有方法返回 **DataFrame / DatetimeIndex**，不是契约对象 ——
    包装与校验由 ``load_*`` 统一负责，适配器不重复干活。

    返回 ``None`` 表示"这个数据源没有这类数据"，调用方会跳过对应检查。
    """

    #: 数据源名（用于报错信息）
    name = 'base'

    def prices(self, start=None, end=None, **kw) -> pd.DataFrame:
        """行情。列见 docs/输入数据规格.md §2.2。**必须实现。**"""
        fail(self.name, 'not_implemented',
             f'{type(self).__name__} 没有实现 prices()。\n'
             f'  行情是必填输入 —— 至少给出 raw_* / adj_factor / adj_* 三组。')

    def calendar(self, start=None, end=None) -> pd.DatetimeIndex:
        """交易日历。不给就由库从行情日期推断（精度略低）。"""
        return None

    def factor(self, name, start=None, end=None, **kw) -> pd.DataFrame:
        """因子。列见 §2.1。"""
        fail(self.name, 'not_implemented',
             f'{type(self).__name__} 没有实现 factor({name!r})')

    def universe(self, start=None, end=None, **kw) -> pd.DataFrame:
        return None

    def tradability(self, start=None, end=None, **kw) -> pd.DataFrame:
        return None

    def names(self, start=None, end=None, **kw) -> pd.Series:
        """**PIT 名称** —— 该 (date, asset) 当日的证券简称。

        用于判 ST（决定涨跌停是 5% 还是 10%/20%）。
        索引 ``MultiIndex(date, asset)``，值为当日名称。

        **务必是当日名称，不是最新名称** —— 用最新名称会把
        "现在是 ST" 套到历史上所有日期，反之亦然。
        """
        return None

    def list_dates(self, **kw) -> pd.Series:
        """``asset -> 上市日``。用于算上市天数、过滤新股。

        不给则**跳过新股过滤**（安全默认）——
        没有真实上市日时，退而用"面板内序号"会把截断面板里的
        所有股票都当成第 1 天，从而静默排除全部数据。
        """
        return None

    def grouping(self, name, start=None, end=None, **kw) -> pd.DataFrame:
        return None

    def exposures(self, start=None, end=None, **kw) -> pd.DataFrame:
        return None

    def events(self, name=None, start=None, end=None, **kw) -> pd.DataFrame:
        return None

    def __repr__(self):
        return f'<{type(self).__name__} name={self.name!r}>'


# 便于子类继承的别名（语义更清楚）
AdapterBase = InputAdapter


# --------------------------------------------------------------------------- #
def as_multiindex(df, contract='loader', date_col='date', asset_col='asset'):
    """把 ``date``/``asset`` 两列变成 MultiIndex，并统一 dtype。

    磁盘上的 parquet / CSV 用普通列存；进库前在这里转索引。
    """
    if df is None:
        return None
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex) and df.index.names == [date_col, asset_col]:
        pass                                  # 已经是契约索引
    else:
        for c in (date_col, asset_col):
            if c not in df.columns:
                fail(contract, 'missing_column',
                     f'缺少列 `{c}`；实际列：{list(df.columns)[:12]}\n'
                     f'  提示：磁盘上的数据用普通列存 date/asset，读进来后再转索引。')
        df[date_col] = pd.to_datetime(df[date_col])
        df[asset_col] = df[asset_col].astype(str)
        df = df.set_index([date_col, asset_col])
    d = df.index.get_level_values(date_col)
    if getattr(d.dtype, 'tz', None) is not None:
        df.index = df.index.set_levels(d.tz_localize(None), level=date_col)
    return df.sort_index()
