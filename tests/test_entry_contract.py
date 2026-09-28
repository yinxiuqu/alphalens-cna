"""入口层契约校验（防线 1 · 端到端）：**算任何数字之前**先把输入校验掉。

背景（实测，不是推测）：防线 1 原先靠"构造即校验"实现，只在走 loader 或用户
显式构造面板时生效。``build_report`` 把裸 DataFrame 解包后直接往下传，契约层
**从未被调用** —— 前视（``available_at > date``）、因子日期不在交易日历、
``raw`` 有值而 ``adj_*`` 缺行、价格含 ``±inf``、因子 ``(date, asset)`` 在
prices 里找不到、代码带后缀导致资产无交集，这六类坏数据都会**静默跑通**，
并生成一份看着完全正常的报告。

这个文件把六类场景钉死，另外钉住"不误伤"（正常数据、正常停牌、只有 value 的
因子、Series 因子都必须照跑）与逃生门 ``validate=False``。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna                                          # noqa: E402
from alphalens_cna.contract.errors import ContractError               # noqa: E402


def mk(n_date=40, n_assets=6, seed=0):
    """最小但有真实收益的面板：prices（三价并存）+ factor（含 available_at）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2023-01-02', periods=n_date)
    assets = [f'{600000 + i:06d}' for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    close = pd.Series(
        np.tile(rng.uniform(8, 40, n_assets), n_date)
        * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    px = pd.DataFrame({'raw_close': close}, index=idx)
    px['raw_open'] = close * (1 + rng.normal(0, 0.003, len(idx)))
    px['raw_high'] = px[['raw_open', 'raw_close']].max(axis=1) * 1.004
    px['raw_low'] = px[['raw_open', 'raw_close']].min(axis=1) * 0.996
    px['prev_close'] = (px.groupby(level='asset')['raw_close'].shift(1)
                        .fillna(px['raw_close']))
    px['adj_factor'] = 1.0
    for c in ('open', 'high', 'low', 'close'):
        px[f'adj_{c}'] = px[f'raw_{c}']
    px['volume'] = 1e5
    f = pd.DataFrame({'value': rng.normal(size=len(idx))}, index=idx)
    f['available_at'] = f.index.get_level_values('date')
    return dates, px, f


def run(dates, px, f, **kw):
    return acna.build_report(f, px, acna.Calendar(dates), horizons=(1, 5),
                             quantiles=2, n_trials=2, **kw)


def _move_row_date(f, key, new_date):
    """把某行的索引日期挪到 ``new_date``（同时把 available_at 挪过去，
    免得先被"前视"那条闸门拦下、测不到"非交易日"这条）。"""
    d = f.index.get_level_values('date').to_numpy().copy()
    i = list(f.index).index(key)
    d[i] = np.datetime64(new_date)
    out = f.copy()
    out.index = pd.MultiIndex.from_arrays(
        [d, out.index.get_level_values('asset')], names=['date', 'asset'])
    out.loc[(new_date, key[1]), 'available_at'] = pd.Timestamp(new_date)
    return out.sort_index()


# ── ① 六类坏数据：必须在算数字之前被拒 ────────────────────────────────────
@pytest.mark.parametrize('case,rule', [
    ('lookahead', 'lookahead'),
    ('off_calendar', 'off_calendar'),
    ('adjust_incomplete', 'adjust_incomplete'),
    ('non_finite', 'non_finite'),
    ('asset_suffix', 'asset_mismatch'),
    ('factor_not_in_prices', 'factor_not_in_prices'),
])
def test_bad_data_is_refused_before_any_number(case, rule):
    """★ 这六条以前是**静默跑通**的 —— 每一条都会让报告数字变成错的。"""
    dates, px, f = mk()
    if case == 'lookahead':
        f = f.copy()
        f.loc[(dates[10], '600000'), 'available_at'] = dates[30]
    elif case == 'off_calendar':
        f = _move_row_date(f, (dates[10], '600000'), pd.Timestamp('2023-01-07'))
        px = px.sort_index()
    elif case == 'adjust_incomplete':
        px = px.copy()
        px.loc[(dates[7], '600000'), 'adj_close'] = np.nan      # raw 有值、adj 缺
    elif case == 'non_finite':
        px = px.copy()
        px.loc[(dates[9], '600001'), 'adj_close'] = np.inf
    elif case == 'asset_suffix':
        f = f.copy()
        f.index = pd.MultiIndex.from_arrays(
            [f.index.get_level_values('date'),
             ['SH' + a for a in f.index.get_level_values('asset')]],
            names=['date', 'asset'])
    elif case == 'factor_not_in_prices':
        extra = pd.DataFrame(
            {'value': [1.0], 'available_at': [dates[20]]},
            index=pd.MultiIndex.from_tuples([(dates[20], '600999')],
                                            names=['date', 'asset']))
        f = pd.concat([f, extra]).sort_index()                 # prices 里没这只
    with pytest.raises(ContractError) as e:
        run(dates, px, f)
    assert e.value.rule == rule, f'{case}: 期望 {rule}，收到 {e.value.rule}'


# ── ② 不误伤：这些都必须照跑 ──────────────────────────────────────────────
def test_normal_data_runs_and_prints_receipt():
    dates, px, f = mk()
    rep = run(dates, px, f)
    assert rep.contract['enabled'] is True
    assert rep.contract['checked'] == ['FactorPanel', 'PricePanel']
    assert 'factor_in_prices' in rep.contract['cross']
    line = [l for l in rep.to_markdown().splitlines() if '契约校验' in l]
    assert len(line) == 1, '报告首屏必须有且只有一行契约收据'
    assert 'FactorPanel ✓' in line[0] and 'PricePanel ✓' in line[0]


def test_bare_value_factor_is_synthesized_with_loud_notice():
    """缺 ``available_at`` 不报错（与 loader 约定一致），但**不许静默**。"""
    dates, px, f = mk()
    rep = run(dates, px, f[['value']])
    notices = rep.contract['notices']
    assert len(notices) == 1 and 'available_at' in notices[0]
    assert '前视检查因此未生效' in notices[0]
    line = [l for l in rep.to_markdown().splitlines() if '契约校验' in l][0]
    assert '前视检查因此未生效' in line, '收据必须把"闸门失效"印在首屏'


def test_bare_value_factor_still_catches_lookahead_when_column_given():
    """给了 available_at 就必须真查前视。"""
    dates, px, f = mk()
    f = f.copy()
    f.loc[(dates[5], '600001'), 'available_at'] = dates[6]
    with pytest.raises(ContractError) as e:
        run(dates, px, f)
    assert e.value.rule == 'lookahead'


def test_normal_suspension_is_allowed():
    """停牌（raw 与 adj 两边都缺）按输入规格允许，别误伤。"""
    dates, px, f = mk()
    px = px.copy()
    for c in ('raw_open', 'raw_high', 'raw_low', 'raw_close',
              'adj_open', 'adj_high', 'adj_low', 'adj_close'):
        px.loc[(dates[12], '600002'), c] = np.nan
    rep = run(dates, px, f)
    assert rep.contract['enabled']


def test_factor_missing_some_dates_is_allowed():
    """因子只覆盖部分日期是正常的（不是"在 prices 里找不到"）。"""
    dates, px, f = mk()
    rep = run(dates, px, f.loc[f.index.get_level_values('date') < dates[25]])
    assert rep.contract['enabled']


def test_series_factor_is_accepted():
    """``clean`` / ``check_parity`` 都收 Series，入口层不能把它拒了。"""
    dates, px, f = mk()
    rep = run(dates, px, f['value'])
    assert rep.contract['checked'] == ['FactorPanel', 'PricePanel']


# ── ③ 逃生门 ──────────────────────────────────────────────────────────────
def test_validate_false_reproduces_permissive_behaviour_and_says_so():
    """``validate=False`` 恢复"什么都不查"的旧行为，但报告首屏必须写明已关闭。"""
    dates, px, f = mk()
    f = f.copy()
    f.loc[(dates[10], '600000'), 'available_at'] = dates[30]      # 前视
    rep = run(dates, px, f, validate=False)                        # 不报错
    assert rep.contract['enabled'] is False
    line = [l for l in rep.to_markdown().splitlines() if '契约校验' in l][0]
    assert '已关闭' in line and '未检查' in line


# ── ④ 面板 vs 裸 DataFrame：数字必须逐字节相同 ────────────────────────────
def test_panel_and_raw_dataframe_give_identical_output():
    dates, px, f = mk()
    cal = acna.Calendar(dates)
    a = acna.build_report(f, px, cal, horizons=(1, 5), quantiles=2, n_trials=2)
    b = acna.build_report(acna.FactorPanel(f), acna.PricePanel(px), cal,
                          horizons=(1, 5), quantiles=2, n_trials=2)
    assert a.to_markdown() == b.to_markdown()
    for name, fr in a.frames().items():
        g = b.frames()[name]
        if fr is None:
            assert g is None
        else:
            pd.testing.assert_frame_equal(fr, g)


# ── ⑤ validate_inputs / ensure_contract 的契约面 ──────────────────────────
def test_validate_inputs_still_returns_true():
    dates, px, f = mk()
    assert acna.validate_inputs(f, px, acna.Calendar(dates)) is True


def test_validate_inputs_accepts_bare_datetimeindex():
    """以前传裸 DatetimeIndex 会 AttributeError，现在与其余入口一个口径。"""
    dates, px, f = mk()
    assert acna.validate_inputs(f, px, dates) is True


def test_ensure_contract_returns_panels_and_notices():
    dates, px, f = mk()
    ck = acna.ensure_contract(f[['value']], px, dates)
    assert isinstance(ck['factor'], acna.FactorPanel)
    assert isinstance(ck['prices'], acna.PricePanel)
    assert ck['cross'] and ck['checked'] == ['FactorPanel', 'PricePanel']
    assert ck['notices'], '缺 available_at 的兜底必须留痕'
    assert ck['factor'].df['available_at'].notna().all()


def test_ensure_contract_revalidates_unvalidated_panel():
    """``XxxPanel(df, validate=False)`` 造的对象不能被当成"已校验"放行。"""
    dates, px, f = mk()
    px = px.copy()
    px.loc[(dates[9], '600001'), 'adj_close'] = np.inf
    p = acna.PricePanel(px, validate=False)
    assert p._validated is False
    with pytest.raises(ContractError) as e:
        acna.ensure_contract(f, p, dates)
    assert e.value.rule == 'non_finite'


# ── ⑥ check_parity 也过防线 1 ─────────────────────────────────────────────
def test_check_parity_is_gated_too():
    pytest.importorskip('alphalens', reason='对拍需要可选依赖 alphalens-reloaded')
    dates, px, f = mk(n_date=60, n_assets=8)
    f = f.copy()
    f.loc[(dates[10], '600000'), 'available_at'] = dates[40]
    with pytest.raises(ContractError) as e:
        acna.check_parity(f, px, acna.Calendar(dates), horizons=(1,))
    assert e.value.rule == 'lookahead'


# ── ⑦ 性能护栏（松，只挡量级退化）─────────────────────────────────────────
def test_validation_cost_stays_linear():
    """跨表检查曾是 ``set(index.get_level_values(...).unique())``：

    960,000 行面板上两项就吃掉 1.1 s。改成整数 code 去重后是个位数毫秒。
    这条护栏**故意留得很松**（只挡量级退化，不追精确值）：20 万行面板上
    整套 ``validate_inputs`` 不应超过 2 s。
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range('2020-01-02', periods=400)
    assets = [f'{600000 + i:06d}' for i in range(500)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    px = pd.DataFrame({'raw_close': rng.uniform(8, 40, len(idx))}, index=idx)
    px['raw_open'] = px['raw_close']
    px['raw_high'] = px['raw_close'] * 1.004
    px['raw_low'] = px['raw_close'] * 0.996
    px['prev_close'] = px['raw_close']
    px['adj_factor'] = 1.0
    for c in ('open', 'high', 'low', 'close'):
        px[f'adj_{c}'] = px[f'raw_{c}']
    px['volume'] = 1e5
    f = pd.DataFrame({'value': rng.normal(size=len(idx))}, index=idx)
    f['available_at'] = f.index.get_level_values('date')
    t = time.perf_counter()
    acna.validate_inputs(f, px, dates)
    dt = time.perf_counter() - t
    assert dt < 2.0, f'20 万行面板校验耗时 {dt:.2f}s —— 量级退化了'


def test_wrong_object_gets_actionable_error_not_attributeerror():
    """传"有 ``.df`` 但不是契约对象"的东西（例如 ``Returns``）要给可执行报错。"""
    dates, px, f = mk()
    ret = acna.forward_returns(px, acna.Calendar(dates), [1])
    with pytest.raises(ContractError) as e:
        acna.ensure_contract(f, ret, dates)          # Returns 有 .df、没有 validate()
    assert e.value.rule == 'type'
    assert 'validate()' in str(e.value)
