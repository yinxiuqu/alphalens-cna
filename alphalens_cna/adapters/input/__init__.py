"""输入适配层 —— 把任意数据源翻译成契约对象。

    base.py        InputAdapter 协议 + 数据源注册表
    dataframe.py   直接吃 DataFrame（零依赖，默认）
    parquet.py     按【公开目录约定】读 parquet
    loader.py      load_prices / load_factor / ... 载入函数

核心包**不认识任何具体数据源**。私有数据源写 examples/ 里的适配器，不进核心。
"""

from .base import AdapterBase, InputAdapter, as_multiindex, get_source, list_sources, register
from .dataframe import DataFrameAdapter
from .loader import (
    load_calendar,
    load_exposures,
    load_factor,
    load_grouping,
    load_prices,
    load_tradability,
    load_universe,
    resolve,
)
from .parquet import ParquetAdapter

__all__ = [
    'InputAdapter', 'AdapterBase', 'register', 'get_source', 'list_sources',
    'as_multiindex', 'DataFrameAdapter', 'ParquetAdapter',
    'load_prices', 'load_factor', 'load_calendar', 'load_universe',
    'load_tradability', 'load_exposures', 'load_grouping', 'resolve',
]
