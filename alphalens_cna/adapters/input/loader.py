"""载入函数 —— 把数据源转成契约对象。

这是使用者最常碰的入口：

.. code-block:: python

    import alphalens_cna as acna

    px  = acna.load_prices(source='parquet', root='~/mydata')
    roe = acna.load_factor('roe', source=px_source)

``adjust_source``（D5）
-----------------------
``'computed'``（默认）
    从 ``raw_*`` + 除权事件**现算**后复权因子。可复现 ——
    加新数据不会让历史因子回溯改变。
``'stored'``
    直接读数据里现成的 ``adj_factor``。快，但若是前复权快照，
    重建数据后历史价格水平会变（**收益不受影响**）。

两者在**收益上完全等价**（整体缩放会约掉），可互相验证 ——
见 ``acna.check_adjust_agreement``。
"""

from __future__ import annotations

import pandas as pd

from ...contract.errors import fail
from ...contract.panels import (
    Exposures,
    FactorPanel,
    Grouping,
    PricePanel,
    Tradability,
    Universe,
)
from ...engine.adjust import compute_adj_factor
from .base import InputAdapter, get_source

__all__ = ['load_prices', 'load_factor', 'load_calendar', 'load_universe',
           'load_tradability', 'load_exposures', 'load_grouping', 'resolve']

RAW = ('raw_open', 'raw_high', 'raw_low', 'raw_close')
ADJ = ('adj_open', 'adj_high', 'adj_low', 'adj_close')


def resolve(source):
    """把 source 参数解析成适配器实例。

    * ``None``        → 报错并提示怎么给
    * ``str``         → 查注册表（``'parquet'`` / ``'dataframe'`` / …）
    * 适配器实例/类    → 原样返回
    """
    if source is None:
        fail('loader', 'no_source',
             '没有给数据源。三选一：\n'
             '  1. 直接传 DataFrame：acna.PricePanel(df)\n'
             "  2. 公开目录约定：acna.load_prices(source='parquet', root='~/mydata')\n"
             '  3. 自己写适配器：见 examples/（库不绑任何私有目录）')
    if isinstance(source, str):
        return get_source(source)
    if isinstance(source, type):
        return source()
    if isinstance(source, InputAdapter) or hasattr(source, 'prices'):
        return source
    fail('loader', 'bad_source',
         f'source 需要是字符串或适配器实例，收到 {type(source).__name__}')


# --------------------------------------------------------------------------- #
def load_prices(source=None, start=None, end=None, adjust_source='computed',
                xdxr=None, validate=True, **kw) -> PricePanel:
    """载入行情并包装成 :class:`PricePanel`。

    Parameters
    ----------
    source : str | InputAdapter
    start, end : 日期, 可选
    adjust_source : {'computed', 'stored'}
        复权因子怎么来。见模块文档。
    xdxr : DataFrame, 可选
        除权事件。``adjust_source='computed'`` 且数据源没提供时，必须显式给。
    validate : bool
        是否构造时校验（默认 True）。

    Returns
    -------
    PricePanel
    """
    src = resolve(source)
    df = src.prices(start=start, end=end, **kw)
    if df is None or not len(df):
        fail('loader', 'no_prices', f'{src} 没返回行情数据')

    df = df.copy()
    missing = [c for c in RAW if c not in df.columns]
    if missing:
        fail('loader', 'missing_raw',
             f'行情缺原始价列 {missing}。\n'
             f'  为什么必需：涨跌停判定**只能**用原始不复权价 ——\n'
             f'  实测 600519 在 2016-08-26 的涨停价，用原始价算 333.31（对），\n'
             f'  用当时的前复权价算 270.01 —— 差 23%。')

    if adjust_source == 'computed':
        x = xdxr
        if x is None and hasattr(src, 'xdxr'):
            x = src.xdxr()
        if x is None or not len(x):
            fail('loader', 'no_xdxr',
                 "adjust_source='computed' 需要除权事件表，但没拿到。\n"
                 "  三选一：\n"
                 "   1. 数据源提供 xdxr（ParquetAdapter 会读 xdxr.parquet）\n"
                 "   2. 显式传入：acna.load_prices(..., xdxr=df)\n"
                 "   3. 改用现成因子：acna.load_prices(..., adjust_source='stored')")
        factor = compute_adj_factor(df, x, method='hfq')
        df['adj_factor'] = factor.reindex(df.index).astype(float)
    elif adjust_source == 'stored':
        if 'adj_factor' not in df.columns:
            fail('loader', 'no_stored_factor',
                 "adjust_source='stored' 但行情里没有 `adj_factor` 列。\n"
                 "  改用 adjust_source='computed' 并提供除权事件表。")
    else:
        fail('loader', 'bad_adjust_source',
             f"adjust_source 只能是 'computed' 或 'stored'，收到 {adjust_source!r}")

    # 复权价统一由 raw × factor 导出 —— 保证契约自洽
    for raw, adj in zip(RAW, ADJ):
        if raw in df.columns:
            df[adj] = df[raw] * df['adj_factor']

    if 'prev_close' not in df.columns:
        df = _fill_prev_close(df)

    keep = [c for c in list(RAW) + ['prev_close', 'adj_factor'] + list(ADJ)
            + ['volume', 'amount'] if c in df.columns]
    return PricePanel(df[keep], validate=validate)


def _fill_prev_close(df):
    """不给 prev_close 时，用同股票的 raw_close 前移一位补上。"""
    df = df.sort_index()
    df['prev_close'] = df.groupby(level='asset')['raw_close'].shift(1)
    return df


# --------------------------------------------------------------------------- #
def load_factor(name, source=None, start=None, end=None, validate=True,
                **kw) -> FactorPanel:
    """载入因子并包装成 :class:`FactorPanel`。"""
    src = resolve(source)
    df = src.factor(name, start=start, end=end, **kw)
    if df is None or not len(df):
        fail('loader', 'no_factor', f'{src} 没返回因子 `{name}`')
    if 'value' not in df.columns:
        cand = [c for c in df.columns if c not in ('date', 'asset', 'available_at')]
        if len(cand) == 1:
            df = df.rename(columns={cand[0]: 'value'})
        else:
            fail('loader', 'no_value',
                 f'因子 `{name}` 找不到 `value` 列；实际列：{list(df.columns)}')
    if 'available_at' not in df.columns:
        # 缺省：收盘后可知 → available_at = date
        df = df.copy()
        df['available_at'] = df.index.get_level_values('date')
    return FactorPanel(df[['value', 'available_at']], validate=validate)


def load_calendar(source=None, start=None, end=None, **kw):
    """载入交易日历。数据源不给就退回"从行情日期推断"。"""
    from ...contract.calendar import Calendar
    src = resolve(source)
    c = src.calendar(start=start, end=end, **kw) if hasattr(src, 'calendar') else None
    if c is None:
        px = src.prices(start=start, end=end, **kw)
        if px is None or not len(px):
            fail('loader', 'no_calendar',
                 '数据源没给日历，也没行情可供推断。')
        c = pd.DatetimeIndex(px.index.get_level_values('date').unique())
    return Calendar(c)


def _load_optional(source, attr, cls, start, end, validate, **kw):
    src = resolve(source)
    fn = getattr(src, attr, None)
    if fn is None:
        return None
    df = fn(start=start, end=end, **kw)
    if df is None or not len(df):
        return None
    return cls(df, validate=validate)


def load_universe(source=None, start=None, end=None, validate=True, **kw):
    return _load_optional(source, 'universe', Universe, start, end, validate, **kw)


def load_tradability(source=None, start=None, end=None, validate=True, **kw):
    return _load_optional(source, 'tradability', Tradability, start, end, validate, **kw)


def load_exposures(source=None, start=None, end=None, validate=True, **kw):
    return _load_optional(source, 'exposures', Exposures, start, end, validate, **kw)


def load_grouping(name='default', source=None, start=None, end=None,
                  validate=True, **kw):
    return _load_optional(source, 'grouping', Grouping, start, end, validate,
                          name=name, **kw)
