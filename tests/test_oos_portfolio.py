"""样本内外拆分（B）与组合绩效（C）—— 0.4.0 新增的评估闭环。

覆盖三件事：
1. `split_is_oos` 的**顺序切 / embargo / 期数不足**（时序数据的切分纪律）；
2. 报告集成：不要求时不出现、要求时出现，且**不改任何数字**；
3. 组合绩效节的**口径诚实性**（必须写明"非可交易净值"）与已知答案。
另外钉住报告节次编号的**连续性**（编号已改成自动生成）。
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna                                          # noqa: E402
from alphalens_cna.analysis.split import split_is_oos                 # noqa: E402
from alphalens_cna.contract.errors import ContractError               # noqa: E402


def mk(n_date=120, n_assets=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2023-01-02', periods=n_date)
    assets = [f'{600000 + i:06d}' for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    close = pd.Series(
        np.tile(rng.uniform(8, 40, n_assets), n_date)
        * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    px = pd.DataFrame({'raw_close': close}, index=idx)
    px['raw_open'] = close
    px['raw_high'] = close * 1.004
    px['raw_low'] = close * 0.996
    px['prev_close'] = px.groupby(level='asset')['raw_close'].shift(1).fillna(close)
    px['adj_factor'] = 1.0
    for c in ('open', 'high', 'low', 'close'):
        px[f'adj_{c}'] = px[f'raw_{c}']
    px['volume'] = 1e5
    f = pd.DataFrame({'value': rng.normal(size=len(idx))}, index=idx)
    f['available_at'] = f.index.get_level_values('date')
    return dates, px, f


def build(dates, px, f, **kw):
    kw.setdefault('horizons', (1, 5))
    kw.setdefault('quantiles', 3)
    return acna.build_report(f, px, acna.Calendar(dates), **kw)


def sections(md):
    """报告里所有节标题（去掉 `## ` 前缀）。"""
    return re.findall(r'^## (.+)$', md, re.M)


# ── 1. split_is_oos：顺序切 + embargo + 期数不足 ───────────────────────────
def test_split_is_chronological_and_respects_embargo():
    d = pd.bdate_range('2023-01-02', periods=100)
    sp = split_is_oos(d, ratio=0.7, embargo=5)
    # 顺序切：样本内整体早于样本外（时序数据不能随机切）
    assert sp['is'].max() < sp['oos'].min()
    assert sp['cut'] == sp['oos'][0], '切分日应归样本外'
    # embargo：样本内尾部被挖掉 5 期
    assert sp['n_is'] == 70 - 5, sp
    assert sp['n_oos'] == 30
    assert sp['embargo'] == 5


def test_split_cut_mode_uses_the_frozen_date():
    d = pd.bdate_range('2023-01-02', periods=100)
    sp = split_is_oos(d, cut='2023-03-01', embargo=0)
    assert sp['cut'] == pd.Timestamp('2023-03-01') or sp['cut'] >= pd.Timestamp('2023-03-01')
    assert (sp['is'] < sp['cut']).all() and (sp['oos'] >= sp['cut']).all()


@pytest.mark.parametrize('kw,rule', [
    ({'ratio': 1.5}, 'bad_ratio'),
    ({'ratio': 0.7, 'embargo': -1}, 'bad_embargo'),
    ({'ratio': 0.98, 'min_periods': 30}, 'too_few_periods'),   # 样本外只有 2 期
    ({'cut': '2030-01-01'}, 'cut_outside'),
])
def test_split_rejects_bad_input(kw, rule):
    d = pd.bdate_range('2023-01-02', periods=100)
    with pytest.raises(ContractError) as e:
        split_is_oos(d, **kw)
    assert e.value.rule == rule


# ── 2. 报告集成：不要就不出现；要了也不改数字 ──────────────────────────────
def test_no_is_oos_section_by_default():
    dates, px, f = mk()
    rep = build(dates, px, f)
    assert not rep.is_oos
    assert '样本内 / 样本外对照' not in rep.to_markdown()
    assert rep.is_oos == {}
    # ⚠️ `frames()` 会把值为 None 的表**丢掉** —— 所以默认情况下这两个键不存在，
    #    而不是"存在但为 None"。
    assert 'is_oos_is' not in rep.frames()
    assert 'is_oos_oos' not in rep.frames()


def test_is_oos_section_appears_and_does_not_change_any_number():
    """★ 切分只**多出一节**，绝不能动 IC / 结论里的任何数字。"""
    dates, px, f = mk()
    a = build(dates, px, f, n_trials=3)
    b = build(dates, px, f, n_trials=3, is_oos=0.7)
    pd.testing.assert_frame_equal(a.frames()['ic'], b.frames()['ic'])
    pd.testing.assert_frame_equal(a.frames()['ic_summary'], b.frames()['ic_summary'])
    pd.testing.assert_frame_equal(a.frames()['quantile_stats'], b.frames()['quantile_stats'])
    assert a.verdict.p_adjusted == b.verdict.p_adjusted
    assert a.verdict.net_return == b.verdict.net_return

    md = b.to_markdown()
    assert '样本内 / 样本外对照' in md
    assert 'embargo' in md, '必须写明挖掉了多少期（purge），不能让人猜'
    for seg in ('样本内', '样本外'):
        tbl = b.is_oos[seg]
        assert tbl is not None and len(tbl)
        for col in ('IC 均值', 'ICIR', 'IC 胜率', '期数', '单调性'):
            assert col in tbl.columns, (seg, col)


# ── 3. 组合绩效：口径诚实性 + 已知答案 ────────────────────────────────────
def test_portfolio_section_states_its_scope():
    """★ 最大的风险是"看着像回测" —— 必须写明它不是可交易净值。"""
    dates, px, f = mk()
    md = build(dates, px, f).to_markdown()
    assert '组合绩效' in md
    assert '不是可交易净值' in md
    assert '未扣成本' in md, '本表未扣成本，必须说出来（扣成本的口径在「结论」里）'


def test_portfolio_metrics_match_hand_computation():
    """已知答案：最大回撤 / 夏普 / 总收益与手算一致。"""
    dates, px, f = mk()
    rep = build(dates, px, f)
    port = rep.portfolio
    assert port is not None and len(port)
    for h in port.index:
        r = rep.extra['long_short'][h].dropna()
        eq = (1 + r).cumprod()
        assert port.loc[h, 'n_periods'] == len(r)
        assert port.loc[h, 'total_return'] == pytest.approx(eq.iloc[-1] - 1, abs=1e-12)
        assert port.loc[h, 'max_drawdown'] == pytest.approx(
            float((eq / eq.cummax() - 1).min()), abs=1e-12)
        if np.isfinite(port.loc[h, 'annual_vol']) and port.loc[h, 'annual_vol'] > 0:
            assert port.loc[h, 'sharpe'] == pytest.approx(
                port.loc[h, 'annual_return'] / port.loc[h, 'annual_vol'], rel=1e-12)


# ── 4. 节次编号的连续性（编号已自动生成，别再出跳号/重号）──────────────────
def test_section_numbers_are_sequential_and_unique():
    dates, px, f = mk()
    for kw in ({}, {'is_oos': 0.7}, {'health': False}, {'preprocess': ('winsorize',)}):
        md = build(dates, px, f, **kw).to_markdown()
        nums = [s.split('、')[0] for s in sections(md) if '、' in s]
        cn = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
              '十一', '十二', '十三', '十四']
        assert nums == cn[:len(nums)], f'{kw} 节次编号不连续/不唯一：{nums}'


def test_nonpositive_equity_does_not_emit_runtime_warning():
    """★ 净值被亏成非正时不能开分数次方（否则 NaN + RuntimeWarning）。

    这条路径 0.4.0 起会被 `build_report` **默认**走到（组合绩效节），
    所以不能留一个往 stdout 喷警告的静默 NaN。
    """
    import warnings as _w

    r = pd.Series([-1.5, 0.1, 0.1], index=pd.bdate_range('2023-01-02', periods=3))
    with _w.catch_warnings():
        _w.simplefilter('error', RuntimeWarning)          # 有警告就算失败
        s = acna.portfolio_summary(r)
    assert np.isnan(s['annual_return']), '净值非正 → 年化无定义，应为 NaN'
    assert np.isnan(s['sharpe'])
    assert s['total_return'] < 0


def test_default_embargo_actually_purges_the_cut_boundary():
    """★ 自检发现的 off-by-one：默认 embargo 少一天时，样本内**最后一个**样本的
    前向收益出场日正好落在切分日上 —— 那就是把样本外的价格读进了样本内。

    这条测试直接按日历验算，而不是断言一个魔数：只要默认口径或 `split_is_oos`
    的切片算法变了，它就会失败。
    """
    dates, px, f = mk()
    horizons = (1, 5, 21)
    rep = build(dates, px, f, horizons=horizons, is_oos=0.7)
    info = rep.is_oos['info']
    cut = pd.Timestamp(info['cut'])
    assert info['mode'] == 'exact', info   # 0.4.2 起按日历精确 purge

    cal = pd.DatetimeIndex(acna.Calendar(dates).index)
    all_d = pd.DatetimeIndex(sorted(set(rep.clean.data.index.get_level_values('date'))))
    pos = cal.get_indexer(all_d)
    leak = [d for d, p in zip(all_d, pos) if d < cut and p + 1 + max(horizons) < len(cal)
            and cal[p + 1 + max(horizons)] >= cut]
    assert len(leak) <= info['purged'], f'样本内仍有 {len(leak)} 个样本的出场日碰到切分日'


def test_embargo_zero_is_allowed_but_says_nothing_was_purged():
    """显式给 0 是允许的（研究决策），但那意味着没挖 —— 报告仍要写出来。"""
    dates, px, f = mk()
    rep = build(dates, px, f, horizons=(1, 5), is_oos=0.7, embargo=0)
    assert rep.is_oos['info']['embargo'] == 0
    assert 'embargo=0' in rep.to_markdown()
