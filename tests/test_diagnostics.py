"""RRE（排名熵稳定性）与 PFS（扰动鲁棒性）测试。"""

from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphalens_cna as acna  # noqa: E402

D = pd.bdate_range('2024-01-01', periods=30)
A = [f'{i:06d}' for i in range(100)]
IDX = pd.MultiIndex.from_product([D, A], names=['date', 'asset'])


def panel(fn, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(fn(rng), index=IDX)


# ---------------------------------------------------------------- RRE ------
def test_rre_identical_ranking_is_one():
    """排名完全不变 → RRE = 1。"""
    x = panel(lambda r: np.tile(np.arange(100), len(D)).astype(float))
    assert abs(acna.rank_stability(x) - 1.0) < 1e-12


def test_rre_independent_ranking_is_zero():
    """每期独立重排 → RRE = 0（归一化后）。"""
    x = panel(lambda r: r.normal(size=len(IDX)))
    assert abs(acna.rank_stability(x)) < 1e-9


def test_rre_increases_with_persistence():
    """自相关越强，RRE 越大 —— 方向不能反。"""
    rng = np.random.default_rng(5)
    def mk(rho):
        base = np.tile(rng.normal(size=100), len(D))
        noise = rng.normal(size=len(IDX))
        return base * rho + noise * np.sqrt(max(1 - rho ** 2, 0))
    a, b = mk(0.0), mk(0.8)
    assert acna.rank_stability(pd.Series(a, index=IDX)) < \
        acna.rank_stability(pd.Series(b, index=IDX))


def test_entropy_constant_factor_is_zero():
    """★ 退化因子（常数）→ 熵 0，而且必须被 note 点出来，不许是 NaN。"""
    x = panel(lambda r: np.ones(len(IDX)))
    assert acna.rank_entropy(x) == 0.0
    s = acna.rank_entropy_summary(x)
    assert s['min_entropy'] == 0.0 and '退化' in s['note']


def test_entropy_continuous_factor_is_high():
    x = panel(lambda r: r.normal(size=len(IDX)))
    assert acna.rank_entropy(x) > 0.6


def test_rre_summary_shape():
    x = panel(lambda r: r.normal(size=len(IDX)))
    s = acna.rank_entropy_summary(x)
    for k in ('rre', 'mean_entropy', 'min_entropy', 'n_periods', 'bins', 'note'):
        assert k in s
    assert s['n_periods'] == len(D)


def test_rre_bad_inputs():
    with pytest.raises(acna.ContractError):
        acna.rank_stability(pd.Series([1.0, 2.0]))
    with pytest.raises(acna.ContractError):
        acna.rank_entropy(panel(lambda r: r.normal(size=len(IDX))), bins=1)


# ---------------------------------------------------------------- PFS ------
def test_pfs_robust_series_is_one():
    x = np.full(100, -0.04) + np.random.default_rng(2).normal(0, 0.002, 100)
    assert acna.pfs(x, n_draws=300, frac=0.8, seed=0) > 0.95


def test_pfs_detects_outlier_driven_conclusion():
    """★★ 结论靠**一期**撑着 → PFS 归零。

    99 期小负 + 1 期巨大正 → 全样本均值翻正，但抽掉那一期就变负。
    ``same_sign`` 仍然高（多数抽样里那期还在），
    ``pfs`` 却为 0（幅度完全不可复现）—— 两个数一起看才不会被骗。
    """
    rng = np.random.default_rng(2)
    x = np.r_[1.5, rng.normal(-0.01, 0.002, 99)]
    assert x.mean() > 0                       # 基准是正的
    r = acna.perturb_series(x, n_draws=300, frac=0.8, seed=0)
    assert r['pfs'] < 0.05, r['pfs']
    assert r['p5'] < 0 < r['p95']             # 扰动分布跨过 0
    assert r['same_sign'] > 0.5               # 符号多半还在 —— 所以不能只看它


def test_pfs_bootstrap_mode():
    x = np.full(60, -0.03) + np.random.default_rng(1).normal(0, 0.002, 60)
    r = acna.perturb_series(x, mode='bootstrap', n_draws=100, frac=1.0)
    assert r['mode'] == 'bootstrap' and r['pfs'] > 0.9


def test_pfs_drop_assets_needs_panel():
    with pytest.raises(acna.ContractError, match='面板'):
        acna.perturb_series(np.arange(50.0), mode='drop_assets')


def test_pfs_bad_inputs():
    with pytest.raises(acna.ContractError):
        acna.perturb_series(np.arange(4.0))               # 期数太少
    with pytest.raises(acna.ContractError):
        acna.perturb_series(np.arange(50.0), mode='nope')
    with pytest.raises(acna.ContractError):
        acna.perturb_series(np.arange(50.0), frac=0.0)


def test_robustness_report_shape():
    x = np.full(100, -0.04) + np.random.default_rng(3).normal(0, 0.003, 100)
    t = acna.robustness_report(x, fracs=(0.5, 0.9), n_draws=50)
    assert t.index.names == ['mode', 'frac']
    assert {'base', 'p5', 'p95', 'pfs'} <= set(t.columns)
    # 保留比例越高 → 越贴近基准 → PFS 不降
    assert t.loc[('subsample', 0.9), 'pfs'] >= t.loc[('subsample', 0.5), 'pfs'] - 1e-9
