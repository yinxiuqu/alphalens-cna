"""preprocess 接进 build_report 的测试。

两条最重要：
1. **默认不许改变任何数字**（保证对拍基线与既有结论不被悄悄动过）
2. **给了暴露却不中性化时必须告警**（实测该场景会导致符号翻转）
"""

from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphalens_cna as acna  # noqa: E402

D = pd.bdate_range('2024-01-01', periods=80)
A = [f'{i:06d}' for i in range(60)]
IDX = pd.MultiIndex.from_product([D, A], names=['date', 'asset'])


def mk(seed=0):
    rng = np.random.default_rng(seed)
    n = len(IDX)
    px = pd.DataFrame({
        'raw_open': rng.uniform(5, 30, n), 'adj_factor': 1.0,
        'volume': 1e6, 'prev_close': rng.uniform(5, 30, n)}, index=IDX)
    px['raw_close'] = px['raw_open'] * (1 + rng.normal(0, 0.02, n))
    px['raw_high'] = np.maximum(px['raw_open'], px['raw_close']) * 1.01
    px['raw_low'] = np.minimum(px['raw_open'], px['raw_close']) * 0.99
    for c in ('open', 'close', 'high', 'low'):
        px[f'adj_{c}'] = px[f'raw_{c}']
    f = pd.DataFrame({'value': rng.normal(size=n),
                      'available_at': IDX.get_level_values('date')}, index=IDX)
    ex = pd.DataFrame({'ln_mv': rng.normal(10, 1, n)}, index=IDX)
    grp = pd.DataFrame({'group': rng.choice(list('ABC'), n)}, index=IDX)
    return px, f, ex, grp


def test_default_changes_nothing():
    """★ 不传 preprocess 时，报告必须与加参数之前**逐位一致**。"""
    px, f, ex, grp = mk()
    a = acna.build_report(f, px, acna.Calendar(D), horizons=(5, 21), name='x')
    b = acna.build_report(f, px, acna.Calendar(D), horizons=(5, 21), name='x',
                          preprocess=None)
    assert a.preprocess == [] and b.preprocess == []
    for col in a.ic.columns:
        assert np.array_equal(a.ic[col].values, b.ic[col].values, equal_nan=True)


def test_steps_recorded_and_log():
    px, f, _, _ = mk()
    r = acna.build_report(f, px, acna.Calendar(D), horizons=(5,),
                          preprocess=('winsorize', 'standardize'))
    assert r.preprocess == ['winsorize', 'standardize']
    log = r.frames()['preprocess_log']
    assert list(log['step']) == ['winsorize(mad,n=3.0)', 'standardize(zscore)']
    assert set(log['n_rows']) == {len(r.clean.data)}


def test_neutralize_by_exposures():
    px, f, ex, _ = mk()
    r = acna.build_report(f, px, acna.Calendar(D), horizons=(5,),
                          exposures=ex, preprocess=('neutralize',))
    assert r.preprocess == ['neutralize']
    # 残差对暴露的截面相关性应≈0
    d = r.clean.data
    for _, g in d.groupby(level='date'):
        assert abs(np.corrcoef(g['factor'], ex['ln_mv'].reindex(g.index))[0, 1]) < 1e-6


def test_neutralize_by_group_zeros_group_means():
    px, f, _, grp = mk()
    r = acna.build_report(f, px, acna.Calendar(D), horizons=(5,),
                          groupby=grp, preprocess=('neutralize',))
    d = r.clean.data
    m = d.groupby([d.index.get_level_values('date'),
                   grp['group'].reindex(d.index)]).mean()['factor'].abs().max()
    assert m < 1e-12


def test_neutralize_requires_input():
    px, f, _, _ = mk()
    with pytest.raises(acna.ContractError, match='没有暴露'):
        acna.build_report(f, px, acna.Calendar(D), horizons=(5,),
                          preprocess=('neutralize',))


def test_bad_step_rejected():
    px, f, _, _ = mk()
    with pytest.raises(acna.ContractError, match='preprocess 只支持'):
        acna.build_report(f, px, acna.Calendar(D), horizons=(5,),
                          preprocess=('winsorize', 'nope'))


def test_warns_when_exposures_given_but_not_neutralized():
    """★ 给了暴露/行业却不中性化 → 必须在结论里告警（实测会符号翻转）。"""
    px, f, ex, grp = mk()
    r = acna.build_report(f, px, acna.Calendar(D), horizons=(5,), exposures=ex)
    assert any('没有做中性化' in w for w in r.verdict.warnings)
    # 做了中性化就不该再告警
    r2 = acna.build_report(f, px, acna.Calendar(D), horizons=(5,), exposures=ex,
                           preprocess=('neutralize',))
    assert not any('没有做中性化' in w for w in r2.verdict.warnings)
    # 可以不告警
    r3 = acna.build_report(f, px, acna.Calendar(D), horizons=(5,), exposures=ex,
                           warn_unnormalized=False)
    assert not any('没有做中性化' in w for w in r3.verdict.warnings)


def test_no_warning_without_exposures():
    px, f, _, _ = mk()
    r = acna.build_report(f, px, acna.Calendar(D), horizons=(5,))
    assert not any('没有做中性化' in w for w in r.verdict.warnings)
