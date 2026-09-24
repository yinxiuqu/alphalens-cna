"""清洗测试 —— 防线 3（不变量对账）+ 归因完整性。

核心断言：**输入 = 输出 + 各类剔除之和**。
账对不上必须抛异常，而不是默默少几行。
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
from alphalens_cna.engine.clean import DropLedger, clean
from alphalens_cna.engine.returns import ReturnModel, forward_returns
from alphalens_cna.engine.tradability import compute_tradability

D = pd.to_datetime
ASSETS = ['600000', '000001', '300750']
DATES = pd.bdate_range('2024-01-02', periods=8)


def make_prices(n=8, assets=None):
    assets = assets or ASSETS
    idx = pd.MultiIndex.from_product([DATES[:n], assets], names=['date', 'asset'])
    k = len(idx)
    base = np.tile([10.0, 12.0, 30.0], n)[:k]
    drift = np.repeat(np.arange(n) * 0.1, len(assets))[:k]
    o = base + drift
    c = o * 1.01
    df = pd.DataFrame({'raw_open': o, 'raw_close': c, 'raw_high': c, 'raw_low': o,
                       'prev_close': np.r_[np.full(len(assets), np.nan), c[:-len(assets)]],
                       'adj_factor': 1.0}, index=idx)
    for col in ('open', 'close', 'high', 'low'):
        df[f'adj_{col}'] = df[f'raw_{col}'] * df['adj_factor']
    return df


def make_factor(n=8, assets=None, na_at=None):
    assets = assets or ASSETS
    idx = pd.MultiIndex.from_product([DATES[:n], assets], names=['date', 'asset'])
    v = np.tile([0.5, 0.3, 0.8], n)[:len(idx)].astype(float)
    f = pd.DataFrame({'value': v,
                      'available_at': idx.get_level_values('date')}, index=idx)
    for spec in (na_at or []):
        f.loc[spec, 'value'] = np.nan
    return f


def pipeline(prices=None, factor=None, horizons=(1,), **kw):
    px = prices if prices is not None else make_prices()
    fac = factor if factor is not None else make_factor()
    cal = acna.Calendar(DATES)
    r = forward_returns(px, cal, list(horizons))
    return clean(fac, r, **kw)


# --------------------------------------------------------------------------- #
# 防线 3：不变量
# --------------------------------------------------------------------------- #
def test_ledger_balances_on_happy_path():
    res = pipeline()
    assert res.ledger.n_input == res.ledger.n_output + res.ledger.total_dropped
    # 有 8 天、持有 1 日、入场要 +1 日 → 最后两天（信号日 +1 与末尾）没收益
    assert res.ledger.total_dropped > 0
    assert 'no_return' in res.ledger.counts


def test_invariant_holds_with_every_reason():
    """把所有剔除原因都造出来，账仍要平。"""
    fac = make_factor(na_at=[(DATES[0], '600000')])           # 因子缺失
    px = make_prices()
    cal = acna.Calendar(DATES)

    # 造一个入场日停牌的样本
    px2 = px.copy()
    px2.loc[(DATES[2], '000001'), ['raw_open', 'adj_open']] = np.nan
    r = forward_returns(px2, cal, [1])

    u = pd.DataFrame({'in_universe': True}, index=px.index)
    u.loc[(DATES[1], '300750'), 'in_universe'] = False        # 不在池内

    res = clean(fac, r, universe=u)

    led = res.ledger
    assert led.n_input == led.n_output + led.total_dropped
    assert 'no_factor' in led.counts
    assert 'no_return' in led.counts
    assert 'not_in_universe' in led.counts


def test_unbalanced_ledger_raises():
    """★ 账不平必须抛，不能默默算了。"""
    led = DropLedger()
    led.n_input = 100
    led.counts = {'x': 10}
    led.n_output = 95          # 100 ≠ 95 + 10
    with pytest.raises(ContractError) as e:
        led.check()
    assert e.value.rule == 'ledger_not_balanced'
    assert '账不平' in str(e.value)


def test_clean_result_checks_on_construction():
    """CleanResult 构造时就对账。"""
    from alphalens_cna.engine.clean import CleanResult
    led = DropLedger(n_input=10, counts={'a': 3}, n_output=5)   # 10 ≠ 5+3
    with pytest.raises(ContractError):
        CleanResult(data=pd.DataFrame(index=pd.MultiIndex.from_arrays(
            [[], []], names=['date', 'asset'])), ledger=led)


# --------------------------------------------------------------------------- #
# 归因
# --------------------------------------------------------------------------- #
def test_no_factor_recorded():
    fac = make_factor(na_at=[(DATES[0], '600000')])
    res = pipeline(factor=fac)
    assert res.ledger.counts.get('no_factor') == 1
    assert (DATES[0], '600000') in res.ledger.examples['no_factor']


def test_not_in_universe_recorded():
    u = pd.DataFrame({'in_universe': True}, index=make_prices().index)
    u.loc[(DATES[1], '300750'), 'in_universe'] = False
    res = pipeline(universe=u)
    assert res.ledger.counts.get('not_in_universe') == 1


def test_universe_blocks_survivorship():
    """★ 生存者偏差护栏：留空的股票池必须挡住样本。"""
    u = pd.DataFrame({'in_universe': False}, index=make_prices().index)
    res = pipeline(universe=u)
    assert len(res.data) == 0
    assert res.ledger.n_input == res.ledger.total_dropped


def test_limit_up_reason_copied_from_returns():
    px = make_prices()
    px.loc[(DATES[2], '600000'), ['raw_open', 'adj_open']] = 11.0   # 涨停
    px.loc[(DATES[2], '600000'), 'prev_close'] = 10.0
    cal = acna.Calendar(DATES)
    t = compute_tradability(px, names=pd.Series('正常股', index=px.index),
                            new_stock_days=0)
    r = forward_returns(px, cal, [1], tradability=t,
                        model=ReturnModel(entry='next_open', policy='skip'))
    res = clean(make_factor(), r)
    # 信号日 1/2 的入场日正是 1/3（涨停那天）
    assert res.ledger.counts.get('limit_up', 0) >= 1


def test_group_and_exposure_missing_recorded():
    g = pd.DataFrame({'group': '银行'}, index=make_prices().index)
    g.loc[(DATES[1], '000001'), 'group'] = np.nan
    e = pd.DataFrame({'ln_mv': 10.0}, index=make_prices().index)
    e.loc[(DATES[0], '300750'), 'ln_mv'] = np.nan
    res = pipeline(groupby=g, exposures=e)
    assert res.ledger.counts.get('no_group') == 1
    assert res.ledger.counts.get('no_exposure') == 1
    assert res.ledger.n_input == res.ledger.n_output + res.ledger.total_dropped


def test_reason_text_is_human_readable():
    from alphalens_cna.engine.clean import REASON_TEXT
    for r in ('no_factor', 'no_return', 'not_in_universe', 'limit_up'):
        assert r in REASON_TEXT and REASON_TEXT[r]


def test_ledger_str_and_frame():
    res = pipeline()
    s = str(res.ledger)
    assert '输入' in s and '保留' in s
    df = res.ledger.to_frame()
    assert set(['reason', 'count', 'pct', 'meaning', 'sample']) <= set(df.columns)
    assert df['count'].sum() == res.ledger.n_input      # 剔除 + 保留 = 输入


# --------------------------------------------------------------------------- #
# 输出形状
# --------------------------------------------------------------------------- #
def test_output_columns_and_index():
    res = pipeline(horizons=(1, 3))
    assert res.data.index.names == ['date', 'asset']
    assert 'factor' in res.data.columns
    for h in (1, 3):
        assert f'forward_return_{h}' in res.data.columns
        assert f'overnight_gap_{h}' in res.data.columns
        assert f'total_return_{h}' in res.data.columns
    assert res.horizons == [1, 3]
    assert not res.data['factor'].isna().any()
    assert not res.data['forward_return_1'].isna().any()


def test_factor_column_normalised():
    """`value` / 单列 / Series 都能进，统一成 `factor`。"""
    r = forward_returns(make_prices(), acna.Calendar(DATES), [1])
    for f in (make_factor(),
              make_factor().rename(columns={'value': 'roe'}),
              make_factor()['value']):
        res = clean(f, r)
        assert 'factor' in res.data.columns


def test_ambiguous_factor_rejected():
    f = make_factor().rename(columns={'value': 'a'})
    f['b'] = 1.0
    r = forward_returns(make_prices(), acna.Calendar(DATES), [1])
    with pytest.raises(ContractError) as e:
        clean(f, r)
    assert e.value.rule == 'ambiguous_factor'


def test_rejects_missing_forward_returns():
    with pytest.raises(ContractError) as e:
        clean(make_factor(), pd.DataFrame({'x': 1.0}, index=make_prices().index))
    assert e.value.rule == 'no_horizons'


def test_horizons_inferred_from_returns():
    r = forward_returns(make_prices(), acna.Calendar(DATES), [2, 5])
    res = clean(make_factor(), r)
    assert res.horizons == [2, 5]


def test_accepts_contract_objects():
    fac = acna.FactorPanel(make_factor())
    px = acna.PricePanel(make_prices())
    cal = acna.Calendar(DATES)
    r = forward_returns(px, cal, [1])
    res = clean(fac, r)
    assert len(res.data) > 0


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))


def test_sample_column_is_human_readable():
    """★ 第十节「剔除明细」的 sample 列不许出现 Python repr。

    修前是 str(list) → 公众号报告里直接出现
    ``[(Timestamp('2023-01-31 00:00:00'), '000000'), ...]``。
    """
    import pandas as pd
    from alphalens_cna.engine.clean import _fmt_sample
    d = pd.bdate_range('2023-01-31', periods=6, freq='ME')
    s = _fmt_sample(list(zip(d, [f'{i:06d}' for i in range(6)])))
    assert 'Timestamp(' not in s and '[' not in s
    assert s.startswith('2023-01-31 000000')
    assert '共 6 条' in s                      # 超长截断并说明总数
    assert _fmt_sample([]) == '' and _fmt_sample(None) == ''
    assert _fmt_sample('原样') == '原样'        # 非列表原样返回
