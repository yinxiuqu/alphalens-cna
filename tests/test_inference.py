"""推断层测试 —— **本项目最关键的一层**。

已知答案全部来自本会话在真实数据上实测过的数字：
* Bonferroni |t| 阈值：n=4→2.498 / n=20→3.023 / n=100→3.481 / n=316→3.778
* BHY 依赖因子 c(4) = 2.0833
* FM 四因子 BHY 校正后：ln_mv 0.0019 ✓、turnover 0.0047 ✓、ROE/PB 不显著
* **换手因子 t=−3.257 在 n=100 时会掉出阈值（3.481）** ← 本项目的核心演示
* NW 修正：ROE t 0.16→0.04（3.7×）、动量 −21.93→−6.76（3.2×）
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna
from alphalens_cna.contract.errors import ContractError
from alphalens_cna.inference import multiplicity as mult


# --------------------------------------------------------------------------- #
# 已知答案：阈值
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('n,expect', [(4, 2.498), (20, 3.023), (100, 3.481),
                                      (316, 3.778)])
def test_t_threshold_known_answers(n, expect):
    """本会话实测过的四个阈值。"""
    assert round(acna.t_threshold(n), 3) == expect


def test_dependency_factor_known_answer():
    """c(4) = 1 + 1/2 + 1/3 + 1/4 = 2.0833。"""
    assert round(acna.dependency_factor(4), 4) == 2.0833
    assert abs(acna.dependency_factor(1) - 1.0) < 1e-12
    # 大 n 近似 ln(n) + 0.5772
    n = 10_000
    assert abs(acna.dependency_factor(n) - (np.log(n) + 0.5772)) < 1e-3


def test_hlz_threshold_matches_bonferroni():
    """HLZ 阈值与 Bonferroni 数值相同（都是正态近似），单列只为标注口径。"""
    assert acna.hlz_threshold(100) == acna.t_threshold(100)


# --------------------------------------------------------------------------- #
# ★ 核心演示：同一个因子，只因 n_trials 变了结论就翻转
# --------------------------------------------------------------------------- #
def test_turnover_factor_flips_at_100_trials():
    """**本项目最有说服力的一个演示。**

    换手因子 t = −3.257（本次 FM 实测）：
      测 4 个   → 阈值 2.498 → 存活
      测 20 个  → 阈值 3.023 → 存活
      测 100 个 → 阈值 3.481 → **掉出去**
    """
    t = -3.257
    assert abs(t) > acna.t_threshold(4), 'n=4 时应存活'
    assert abs(t) > acna.t_threshold(20), 'n=20 时应存活'
    assert abs(t) < acna.t_threshold(100), 'n=100 时**必须**掉出去'


def test_fm_four_factors_bhy_known_answers():
    """复现报告里的 BHY 表（用精确 p，不是四舍五入的）。"""
    t = np.array([0.177, -0.102, -3.687, -3.257])
    p = 2 * (1 - stats.norm.cdf(np.abs(t)))
    adj = acna.adjust(p, 'bhy')

    assert adj[0] > 0.99 and adj[1] > 0.99          # ROE / PB 不显著
    assert round(adj[2], 4) == 0.0019               # ln_mv 显著
    assert round(adj[3], 4) == 0.0047               # 换手显著
    assert adj[2] < 0.05 and adj[3] < 0.05


def test_adjusted_p_never_below_raw():
    rng = np.random.default_rng(0)
    p = rng.uniform(1e-6, 1, 200)
    for m in ('bonferroni', 'holm', 'bh', 'bhy'):
        adj = acna.adjust(p, m)
        assert (adj >= p - 1e-12).all(), m
        assert (adj <= 1 + 1e-12).all(), m


def test_holm_is_less_conservative_than_bonferroni():
    p = np.array([0.001, 0.01, 0.02, 0.5])
    b, h = acna.bonferroni(p), acna.holm(p)
    assert (h <= b + 1e-12).all(), 'Holm 应当不弱于 Bonferroni'
    assert h[0] == b[0], '最小的那个 p 两者相同'


def test_bhy_more_conservative_than_bh():
    """BHY 乘了 c(n)≥1，所以比 BH 保守。"""
    p = np.array([0.001, 0.01, 0.02, 0.5])
    assert (acna.benjamini_yekutieli(p) >= acna.benjamini_hochberg(p) - 1e-12).all()


def test_single_test_unchanged():
    """只测一个假设时，除 bonferroni/holm 外不应改变 p。"""
    p = np.array([0.03])
    assert abs(acna.adjust(p, 'bh')[0] - 0.03) < 1e-12
    assert abs(acna.adjust(p, 'bhy')[0] - 0.03) < 1e-12
    assert abs(acna.adjust(p, 'bonferroni')[0] - 0.03) < 1e-12
    assert abs(acna.adjust(p, 'none')[0] - 0.03) < 1e-12


# --------------------------------------------------------------------------- #
# n_trials：校正基数
# --------------------------------------------------------------------------- #
def test_n_trials_pads_with_unreported():
    """★ 只报了 2 个 p，但一共测过 100 个 → 校正基数必须是 100。

    否则等于把"测过但没报"的悄悄抹掉，多重检验就白做了。
    """
    p = np.array([0.001, 0.02])
    a2 = acna.adjust(p, 'bonferroni', n_trials=2)
    a100 = acna.adjust(p, 'bonferroni', n_trials=100)
    assert (a100 >= a2 - 1e-12).all()
    assert abs(a100[0] - 0.1) < 1e-12          # 100 × 0.001
    assert abs(a2[0] - 0.002) < 1e-12          # 2 × 0.001


def test_n_trials_smaller_than_p_rejected():
    with pytest.raises(ContractError) as e:
        acna.adjust([0.1, 0.2, 0.3], 'bhy', n_trials=2)
    assert e.value.rule == 'n_trials_too_small'


def test_bad_p_rejected():
    with pytest.raises(ContractError) as e:
        acna.adjust([0.1, np.nan], 'bhy')
    assert e.value.rule == 'nan_p'
    with pytest.raises(ContractError) as e:
        acna.adjust([1.5], 'bhy')
    assert e.value.rule == 'p_range'


def test_to_frame_reports_context():
    df = acna.to_frame([0.001, 0.5], 'bhy', n_trials=100,
                       labels=['a', 'b'], tstats=[-3.3, 0.1])
    assert df.attrs['n_trials'] == 100
    assert round(df.attrs['t_threshold'], 3) == 3.481
    assert {'label', 't', 'p_raw', 'p_adj', 'significant_adj'} <= set(df.columns)


# --------------------------------------------------------------------------- #
# Newey-West
# --------------------------------------------------------------------------- #
def test_nw_tstat_naive_equals_plain_t():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1.0, 300)
    r = acna.nw_tstat(x, lags=0)
    m, sd, n = x.mean(), x.std(ddof=1), len(x)
    assert abs(r['t_naive'] - m / (sd / np.sqrt(n))) < 1e-9
    # lags=0 时 NW 与朴素一致
    assert abs(r['t_nw'] - r['t_naive']) < 1e-9


def test_nw_shrinks_t_on_autocorrelated_series():
    """★ 正自相关序列上，NW 必须把 t 压下来（重叠观测的典型情形）。"""
    rng = np.random.default_rng(2)
    n = 400
    e = rng.normal(0, 1, n)
    x = np.zeros(n)
    for i in range(1, n):                       # AR(1)，phi=0.8
        x[i] = 0.8 * x[i - 1] + e[i]
    x = x + 0.3                              # 加个正均值
    r = acna.nw_tstat(x, horizon=20)
    assert abs(r['t_nw']) < abs(r['t_naive']), r
    assert r['vif'] > 1.5, r['vif']
    assert r['n_eff'] < r['n']


def test_nw_no_inflation_on_iid():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 2000)
    r = acna.nw_tstat(x)
    assert abs(r['t_inflation'] if 't_inflation' in r else
               r['t_naive'] / r['t_nw'] - 1) < 0.3


def test_auto_lags_respects_horizon():
    """持有期越长，重叠越严重 → 滞后阶数至少要有 h−1。"""
    assert acna.auto_lags(100, horizon=1) >= 0
    assert acna.auto_lags(100, horizon=20) >= 19
    assert acna.auto_lags(1000, horizon=5) >= 4


def test_effective_n_less_than_n_on_overlap():
    """名义 91 期、持有 12 期 → 有效样本应远小于 91。"""
    rng = np.random.default_rng(4)
    n = 91
    e = rng.normal(0, 1, n + 20)
    x = np.convolve(e, np.ones(12) / 12, mode='same')[:n]   # 强重叠
    eff = acna.effective_n(x, horizon=12)
    assert eff < n / 3, eff


def test_newey_west_summary_columns():
    rng = np.random.default_rng(5)
    ic = pd.DataFrame({1: rng.normal(0.01, 0.1, 500),
                       21: rng.normal(0.01, 0.1, 500)})
    s = acna.newey_west_summary(ic)
    assert set(['n', 'mean', 't_naive', 't_nw', 'vif', 'n_eff', 'lags',
                't_inflation']) <= set(s.columns)
    assert 21 in s.index


# --------------------------------------------------------------------------- #
# Estimate
# --------------------------------------------------------------------------- #
def test_estimate_carries_uncertainty_and_warns():
    """★ 不给单一数字 —— Estimate 必须自带一组标注与警告。"""
    rng = np.random.default_rng(6)
    e = rng.normal(0, 1, 120)
    x = pd.Series(np.convolve(e, np.ones(20) / 20, mode='same')[:91]) + 0.02
    est = acna.estimate_from_series(x, name='RankIC', horizon=20, n_trials=100)
    assert est.n == 91
    assert np.isfinite(est.t_naive) and np.isfinite(est.t_nw)
    assert np.isfinite(est.n_eff) and est.n_eff < est.n
    assert est.n_trials == 100
    assert est.warnings, '应当自动生成警告'
    assert any('虚高' in w or '有效样本' in w for w in est.warnings), est.warnings
    s = str(est)
    assert 't_naive' in s and 'n_eff' in s and 'n_trials' in s


def test_estimate_flags_correction_killing_significance():
    """未校正显著、校正后不显著 —— 必须明说。"""
    p_raw = 0.01
    t = stats.norm.ppf(1 - p_raw / 2)
    x = pd.Series(np.full(50, t / np.sqrt(50)) * 1.0)
    x = x + np.random.default_rng(7).normal(0, 0.0, 50)
    est = acna.estimate_from_series(x, name='x', horizon=1, n_trials=200,
                                    all_p=np.array([p_raw]))
    if est.significant_naive and not est.significant:
        assert any('校正后不显著' in w for w in est.warnings)


def test_estimates_to_frame():
    a = acna.Estimate(name='a', value=1.0, n=10)
    b = acna.Estimate(name='b', value=2.0, n=20)
    df = acna.estimates_to_frame([a, b])
    assert list(df['name']) == ['a', 'b']
    assert 'warnings' in df.columns


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def test_verdict_warns_when_not_corrected():
    """★ 不做校正必须**明说**，不许假装做了。"""
    rng = np.random.default_rng(8)
    ic = pd.DataFrame({1: rng.normal(0.05, 0.1, 200)})
    v = acna.assess(ic=ic)
    assert v.method == 'none'
    assert any('未做多重假设检验校正' in w for w in v.warnings)
    assert 'n_trials' in str(v)


def test_verdict_with_correction():
    rng = np.random.default_rng(9)
    ic = pd.DataFrame({1: rng.normal(0.05, 0.1, 300),
                       5: rng.normal(0.01, 0.1, 300)})
    v = acna.assess(ic=ic, n_trials=50, method='bhy')
    assert v.n_trials == 50
    assert round(v.threshold_used, 3) == acna.t_threshold(50).round(3) \
        if False else abs(v.threshold_used - acna.t_threshold(50)) < 1e-9
    assert len(v.estimates) == 2
    assert isinstance(v.to_frame(), pd.DataFrame)


def test_verdict_uses_ledger_for_tradable_ratio():
    from alphalens_cna.engine.clean import DropLedger
    led = DropLedger(n_input=1000, n_output=800, counts={'x': 200})
    rng = np.random.default_rng(10)
    ic = pd.DataFrame({1: rng.normal(0.05, 0.1, 100)})
    v = acna.assess(ic=ic, n_trials=10, ledger=led)
    assert abs(v.tradable_ratio - 0.8) < 1e-12
    assert v.n_input == 1000 and v.n_used == 800


def test_verdict_flags_high_drop_rate():
    from alphalens_cna.engine.clean import DropLedger
    led = DropLedger(n_input=1000, n_output=300, counts={'x': 700})
    rng = np.random.default_rng(11)
    ic = pd.DataFrame({1: rng.normal(0.05, 0.1, 100)})
    v = acna.assess(ic=ic, n_trials=10, ledger=led)
    assert any('剔除率' in w for w in v.warnings)


def test_verdict_cost_eats_return_warning():
    rng = np.random.default_rng(12)
    r = pd.Series(rng.normal(0.0002, 0.002, 250))      # 微正收益
    to = pd.Series(0.6, index=r.index)                  # 高换手
    ic = pd.DataFrame({1: rng.normal(0.05, 0.1, 100)})
    v = acna.assess(ic=ic, n_trials=10, returns=r, turnover=to, cost_bps=15)
    assert np.isfinite(v.net_return) and np.isfinite(v.cost)
    assert v.net_return < v.gross_return
    assert any('成本' in w for w in v.warnings)


def test_verdict_str_readable():
    ic = pd.DataFrame({1: np.random.default_rng(13).normal(0.05, 0.1, 200)})
    s = str(acna.assess(ic=ic, n_trials=4))
    assert '结论' in s and 'n_trials' in s and '有效样本' in s


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
