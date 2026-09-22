"""复权因子测试 —— 防线 5（已知答案）+ 两个真 bug 的回归。

本文件锁住两个**静默出错**的 bug（不报错、只是数错）：

* **bug A**：``compute_adj_factor`` 曾用 ``swaplevel()``，导致索引**名字是**
  ``(date, asset)`` 而**值是** ``(asset, date)``。下游按 ``level='asset'`` 分组
  会分到日期上去 —— 一个看似通过的验证其实完全无效。

* **bug B**：曾用 ``reindex`` 对齐除权事件，**丢弃了不在行情索引里的除权日**
  （停牌日、上市前、未来公告）。实测 600519 漏掉 2006-05-19/05-24 两次除权，
  前 1078 天的因子偏大 **2.262 倍**。正确做法是 ``pd.concat`` 取并集。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna
from alphalens_cna.contract.errors import ContractError
from alphalens_cna.engine.adjust import compute_adj_factor

D = pd.to_datetime
ASSET = '600000'


def make_prices(dates, closes):
    idx = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(dates), [ASSET] * len(dates)], names=['date', 'asset'])
    return pd.DataFrame({'raw_close': closes}, index=idx)


def make_xdxr(rows):
    """rows: [(date, category, fenhong, peigu, peigujia, songzhuangu)]"""
    idx = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([r[0] for r in rows]), [ASSET] * len(rows)],
        names=['date', 'asset'])
    return pd.DataFrame([{'category': r[1], 'fenhong': r[2], 'peigu': r[3],
                          'peigujia': r[4], 'songzhuangu': r[5]} for r in rows],
                        index=idx)


# --------------------------------------------------------------------------- #
# 防线 5：已知答案 —— 手算得出来
# --------------------------------------------------------------------------- #
def test_known_answer_10for10_split():
    """10 送 10：价格减半，复权后应无跳变。

    手算：1/4 除权，10 送 10 → preclose = close[1/3]×10/(10+10) = 10/2 = 5。
    ⚠️ 除权【当天】价格就反映送转（1/4 收 5），不是次日 —— 这一点搞错会让
       测试数据和事件对不上。
    """
    dates = D(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08'])
    px = make_prices(dates, [10.0, 10.0, 5.0, 5.0, 5.0])
    xr = make_xdxr([(dates[2], 1, 0.0, 0.0, 0.0, 10.0)])

    hfq = compute_adj_factor(px, xr, method='hfq')
    qfq = compute_adj_factor(px, xr, method='qfq')

    # 后复权：首日=1；除权日【当天】起变为 2（不是次日）
    assert np.allclose(hfq.values, [1.0, 1.0, 2.0, 2.0, 2.0]), hfq.values
    # 前复权：末位=1；除权日【之前】为 0.5
    assert np.allclose(qfq.values, [0.5, 0.5, 1.0, 1.0, 1.0]), qfq.values

    # 复权价应连续（10 送 10 后仍是 10 元）
    for s in (hfq, qfq):
        adjpx = px['raw_close'] * s
        assert np.allclose(adjpx.values, adjpx.values[0]), adjpx.values


def test_known_answer_cash_dividend():
    """纯现金分红：10 派 5 → preclose = (close×10 − 5)/10。"""
    dates = D(['2024-01-02', '2024-01-03', '2024-01-04'])
    px = make_prices(dates, [10.0, 10.0, 9.5])
    xr = make_xdxr([(dates[2], 1, 5.0, 0.0, 0.0, 0.0)])
    hfq = compute_adj_factor(px, xr, method='hfq')
    # preclose[2] = (10×10−5)/10 = 9.5 → close[1]/preclose[2] = 10/9.5
    assert hfq.iloc[0] == 1.0
    assert abs(hfq.iloc[2] - 10 / 9.5) < 1e-12


def test_hfq_starts_at_one_qfq_ends_at_one():
    dates = D(['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'])
    px = make_prices(dates, [10.0, 10.0, 5.0, 5.0])
    xr = make_xdxr([(dates[2], 1, 0.0, 0.0, 0.0, 10.0)])
    assert compute_adj_factor(px, xr, 'hfq').iloc[0] == 1.0
    assert abs(compute_adj_factor(px, xr, 'qfq').iloc[-1] - 1.0) < 1e-12


def test_no_xdxr_factor_is_all_one():
    dates = D(['2024-01-02', '2024-01-03', '2024-01-04'])
    px = make_prices(dates, [10.0, 11.0, 12.0])
    empty = pd.DataFrame(columns=['category', 'fenhong', 'peigu', 'peigujia',
                                  'songzhuangu'],
                         index=pd.MultiIndex.from_arrays([[], []], names=['date', 'asset']))
    for m in ('hfq', 'qfq'):
        assert np.allclose(compute_adj_factor(px, empty, m).values, 1.0)


# --------------------------------------------------------------------------- #
# bug A 回归：索引【值和名字】必须都是 (date, asset)
# --------------------------------------------------------------------------- #
def test_index_value_matches_name():
    """bug A：曾出现名字是 (date, asset) 而值是 (asset, date)。"""
    dates = D(['2024-01-02', '2024-01-03'])
    px = make_prices(dates, [10.0, 10.0])
    xr = make_xdxr([(dates[1], 1, 0.0, 0.0, 0.0, 0.0)])
    s = compute_adj_factor(px, xr, 'hfq')

    assert s.index.names == ['date', 'asset']
    # 名字对了还不够 —— 值也必须对得上：date 层得是时间戳
    assert pd.api.types.is_datetime64_any_dtype(s.index.get_level_values('date'))
    assert (s.index.get_level_values('asset') == ASSET).all()
    # 真按名字分组，能拿到完整序列（若值反了，这里只会分到 1 行）
    assert len(s.groupby(level='asset').get_group(ASSET)) == 2


def test_reject_string_index():
    """字符串索引会静默产生 10/11 偏差 —— 必须拦住。"""
    idx = pd.MultiIndex.from_arrays(
        [['2024-01-02', '2024-01-03'], [ASSET] * 2], names=['date', 'asset'])
    px = pd.DataFrame({'raw_close': [10.0, 10.0]}, index=idx)
    with pytest.raises(ContractError) as e:
        compute_adj_factor(px, None, 'hfq')
    assert e.value.rule == 'index_not_datetime'
    assert 'DatetimeIndex' in str(e.value)


# --------------------------------------------------------------------------- #
# bug B 回归：除权日不在行情索引里时，绝不能漏
# --------------------------------------------------------------------------- #
def test_xdxr_date_not_in_price_index():
    """bug B：除权日不在行情里（停牌/未来公告）也要生效。

    构造：交易日 [1/2, 1/3, 1/8]，除权日在 1/5（**不在行情索引里**）。
    漏掉它 ⇒ 1/3 之前的因子会偏大。
    """
    dates = D(['2024-01-02', '2024-01-03', '2024-01-08'])
    px = make_prices(dates, [10.0, 10.0, 5.0])
    xr = make_xdxr([(D('2024-01-05'), 1, 0.0, 0.0, 0.0, 10.0)])   # 停牌日

    hfq = compute_adj_factor(px, xr, 'hfq')
    assert len(hfq) == 3, '只应保留真实交易日'
    # 1/8 那天价格减半，因子必须翻倍 —— 若漏掉这次除权，末位会是 1 而非 2
    assert abs(hfq.iloc[-1] - 2.0) < 1e-12, hfq.values
    assert abs(hfq.iloc[-2] - 1.0) < 1e-12, hfq.values


def test_xdxr_before_first_trading_day():
    """上市前的除权记录（如 000001 的 1990-03-01）不应崩，且被正确处理。"""
    dates = D(['2024-01-02', '2024-01-03'])
    px = make_prices(dates, [10.0, 10.0])
    xr = make_xdxr([(D('2010-03-01'), 1, 0.0, 1.0, 3.56, 0.0)])
    s = compute_adj_factor(px, xr, 'hfq')
    assert len(s) == 2 and np.isfinite(s.values).all()


# --------------------------------------------------------------------------- #
# D5：两口径收益必须一致
# --------------------------------------------------------------------------- #
def test_two_methods_agree_on_returns():
    dates = pd.bdate_range('2024-01-02', periods=12)
    px = make_prices(dates, [10, 10.2, 9.8, 10.5, 5.25, 5.3, 5.1, 5.4, 5.2, 5.6, 5.5, 5.9])
    xr = make_xdxr([(dates[4], 1, 0.0, 0.0, 0.0, 10.0)])
    hfq = compute_adj_factor(px, xr, 'hfq')
    qfq = compute_adj_factor(px, xr, 'qfq')
    r = acna.check_adjust_agreement(hfq, qfq, tol=1e-12)
    assert r['ok'] and r['max_diff'] < 1e-12


def test_agreement_catches_missing_event():
    """漏一次送转 → 收益差达 ~1e-1，必须被抓到（容差 1e-6 量级差 5 个数量级）。"""
    dates = pd.bdate_range('2024-01-02', periods=6)
    px = make_prices(dates, [10, 10, 5, 5, 5, 5])
    full = compute_adj_factor(px, make_xdxr([(dates[2], 1, 0, 0, 0, 10)]), 'hfq')
    # 人为删掉一次除权
    partial = compute_adj_factor(px, make_xdxr([]), 'hfq')
    with pytest.raises(ContractError) as e:
        acna.check_adjust_agreement(full, partial, tol=1e-6)
    assert e.value.rule == 'two_sources_differ'


def test_bad_method_rejected():
    dates = D(['2024-01-02'])
    with pytest.raises(ContractError) as e:
        compute_adj_factor(make_prices(dates, [10.0]), None, 'xxx')
    assert e.value.rule == 'bad_method'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
