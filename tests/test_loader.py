"""loader 的参数路由 —— 字符串源收构造参数，实例源把 kw 交给方法。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna
from alphalens_cna.adapters.input.base import InputAdapter

D = pd.bdate_range('2024-01-02', periods=6)
A = ['600000', '000001']
IDX = pd.MultiIndex.from_product([D, A], names=['date', 'asset'])


def mk_prices():
    return pd.DataFrame(
        {c: [10.0] * len(IDX) for c in
         ['raw_open', 'raw_high', 'raw_low', 'raw_close', 'prev_close',
          'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_factor',
          'volume']}, index=IDX)


def mk_factor():
    return pd.DataFrame({'value': np.random.default_rng(0).normal(size=len(IDX)),
                         'available_at': IDX.get_level_values('date')}, index=IDX)


def test_string_source_forwards_constructor_kwargs():
    """★ 文档/Docstring 里写的 `load_prices(source='parquet', root=…)` 必须真能用。

    此前 `resolve()` 调 `get_source(name)` 时**没转发 **kw** —— 适配器被空构造，
    参数给对了也拿不到数据（`DataFrameAdapter` 直接报 `empty`）。
    """
    px = acna.load_prices(source='dataframe', prices=mk_prices(),
                          adjust_source='stored')
    assert isinstance(px, acna.PricePanel)
    assert len(px.df) == len(IDX)


def test_string_source_forwards_kwargs_to_factor_and_calendar():
    f = acna.load_factor('roe', source='dataframe', factors={'roe': mk_factor()})
    assert isinstance(f, acna.FactorPanel)
    cal = acna.load_calendar(source='dataframe', prices=mk_prices())
    assert isinstance(cal, acna.Calendar)


def test_instance_source_kwargs_go_to_method_not_ctor():
    """反向：实例已经构造好了，`**kw` 该交给它的方法，不能被吞掉。"""
    seen = {}

    class My(InputAdapter):
        def prices(self, start=None, end=None, **kw):
            seen.update(kw)
            return mk_prices()

    acna.load_prices(source=My(), adjust_source='stored', probe=123)
    assert seen.get('probe') == 123, f'实例源的 kw 没传到 prices(): {seen}'


def test_string_source_with_bad_ctor_kwarg_fails_loudly():
    """字符串源多给一个构造函数不认的参数 → 明确报错，而不是静默吞掉。"""
    import pytest
    with pytest.raises(TypeError):
        acna.load_prices(source='dataframe', prices=mk_prices(),
                         adjust_source='stored', nonexistent_arg=1)
