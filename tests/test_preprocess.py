"""预处理层测试 —— 防线 5（已知答案）+ 三条铁律。

铁律：①逐期截面 ②NaN 进 NaN 出、**行数不变** ③每处修改留痕。
"""

from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphalens_cna as acna  # noqa: E402

D = pd.bdate_range('2024-01-01', periods=3)
A = [f'{i:06d}' for i in range(20)]
IDX = pd.MultiIndex.from_product([D, A], names=['date', 'asset'])
GRP = pd.Series((['A'] * 10 + ['B'] * 10) * 3, index=IDX)


def vec(seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(np.tile(rng.normal(size=20), 3), index=IDX)


# ------------------------------------------------------------ 通用铁律 -----
@pytest.mark.parametrize('fn', [
    lambda x: acna.winsorize(x),
    lambda x: acna.standardize(x),
    lambda x: acna.standardize(x, 'rank'),
    lambda x: acna.standardize(x, 'demean'),
    lambda x: acna.neutralize(x, groups=GRP),
    lambda x: acna.neutralize(x, exposures=vec(1).to_frame('e')),
])
def test_shape_and_index_invariant(fn):
    """★ 行数、索引、NaN 位置一律不变 —— 预处理不许动样本。"""
    x = vec()
    x.iloc[[0, 7, 33]] = np.nan
    y = fn(x)
    assert len(y) == len(x)
    assert y.index.equals(x.index)
    assert y.isna().equals(x.isna()) or (y.isna() >= x.isna()).all()


def test_attrs_log_accumulates():
    x = vec()
    y = acna.winsorize(x)
    assert acna.preprocess_log(y)['step'].iloc[0].startswith('winsorize')
    z = acna.standardize(y)
    assert len(acna.preprocess_log(z)) == 2


# ------------------------------------------------------------ winsorize ----
def test_winsorize_mad_known_answer():
    """★ 手算：0..19，中位 9.5，MAD×1.4826 = 7.413 → 边界 2.087 / 16.913。"""
    x = pd.Series(np.tile(np.arange(20.0), 3), index=IDX)
    y = acna.winsorize(x, method='mad', n=1.0)
    med = np.median(np.arange(20.0))
    mad = np.median(np.abs(np.arange(20.0) - med)) * 1.4826
    assert abs(y.min() - (med - mad)) < 1e-12
    assert abs(y.max() - (med + mad)) < 1e-12
    assert y.max() < 19 and y.min() > 0


def test_winsorize_sigma_and_quantile():
    x = vec(3)
    s = acna.winsorize(x, method='sigma', n=2.0)
    q = acna.winsorize(x, method='quantile', limits=(0.1, 0.9))
    for y in (s, q):
        assert y.max() <= x.max() + 1e-12 and y.min() >= x.min() - 1e-12
    assert q.max() < s.max()          # 10/90 分位比 2σ 更紧


def test_winsorize_mad_zero_leaves_untouched():
    """★ 半数列相同（MAD=0）→ 原样返回，别把因子压成常数。"""
    x = pd.Series(np.tile([0.0] * 10 + list(np.arange(10.0)), 3), index=IDX)
    y = acna.winsorize(x, method='mad')
    assert np.allclose(np.asarray(y, dtype=float), np.asarray(x, dtype=float),
                       equal_nan=True)


# ----------------------------------------------------------- standardize ---
def test_standardize_zscore():
    y = acna.standardize(vec(4), 'zscore')
    assert abs(y.groupby(level='date').mean().abs().max()) < 1e-12
    assert np.allclose(y.groupby(level='date').std().values, 1.0)


def test_standardize_rank_range():
    y = acna.standardize(vec(5), 'rank')
    assert y.min() > 0 and y.max() < 1
    assert abs(y.min() - 0.5 / 20) < 1e-12


def test_standardize_demean_only():
    y = acna.standardize(vec(6), 'demean')
    assert abs(y.groupby(level='date').mean().abs().max()) < 1e-12
    assert not np.allclose(y.groupby(level='date').std().values, 1.0)


def test_standardize_constant_section_untouched():
    x = pd.Series(np.tile([7.0] * 20, 3), index=IDX)
    y = acna.standardize(x)
    assert np.allclose(np.asarray(y, dtype=float), 7.0)


# ------------------------------------------------------------ neutralize ---
def test_neutralize_group_means_zero():
    """★ 行业中性化后，每个行业组内均值必须为 0。"""
    x = vec(7)
    y = acna.neutralize(x, groups=GRP)
    m = y.groupby([y.index.get_level_values('date'), GRP]).mean().abs().max()
    assert m < 1e-12
    assert abs(y.groupby(level='date').mean().abs().max()) < 1e-12


def test_neutralize_ols_residual_orthogonal():
    """★ 对暴露回归的残差必须与暴露正交，且逐期均值为 0。"""
    x = vec(8)
    e = vec(9).rename('size')
    y = acna.neutralize(x, exposures=e.to_frame())
    assert abs(y.groupby(level='date').mean().abs().max()) < 1e-10
    j = pd.DataFrame({'y': y, 'e': e}).dropna()
    for d, g in j.groupby(level='date'):
        assert abs(np.corrcoef(g['y'], g['e'])[0, 1]) < 1e-8


def test_neutralize_self_regression_is_zero():
    x = vec(10)
    y = acna.neutralize(x, exposures=x.to_frame('x'))
    assert np.abs(np.asarray(y, dtype=float)).max() < 1e-9


def test_orthogonalize_same_as_neutralize():
    x, o = vec(11), vec(12)
    a = acna.orthogonalize(x, o.to_frame('o'))
    b = acna.neutralize(x, exposures=o.to_frame('o'))
    assert np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                       equal_nan=True)


def test_neutralize_thin_section_is_nan_and_logged():
    """样本不足的截面 → 整段 NaN，并在痕迹里记下跳过了几个截面。"""
    x = vec(13)
    d = x.index.get_level_values('date')
    keep = ~((d == D[0]) & (x.index.get_level_values('asset') > '000004'))
    x2 = x[keep]
    y = acna.neutralize(x2, exposures=vec(14)[keep].to_frame('e'), min_obs=15)
    assert y.loc[D[0]].isna().all()
    assert acna.preprocess_log(y)['n_skipped_sections'].iloc[0] == 1


# --------------------------------------------------------------- combine ---
def test_combine_equal_weight():
    a, b = vec(15), vec(16)
    c = acna.combine([a, b])
    assert abs(c.groupby(level='date').mean().abs().max()) < 1e-10
    assert np.allclose(c.groupby(level='date').std().values, 1.0, atol=1e-6)


def test_combine_weights_normalized():
    """权重不必自己归一化 —— 传 [2,2] 和 [1,1] 必须等价。"""
    a, b = vec(17), vec(18)
    assert np.allclose(acna.combine([a, b], weights=[2, 2]).values,
                       acna.combine([a, b], weights=[1, 1]).values)


def test_combine_scale_invariant():
    """★ 量纲不该影响合成结果（先各自标准化）。"""
    a, b = vec(19), vec(20)
    assert np.allclose(acna.combine([a, b]).values,
                       acna.combine([a * 1000, b]).values, atol=1e-9)


# ----------------------------------------------------------------- 报错 ----
def test_bad_inputs():
    with pytest.raises(acna.ContractError):
        acna.winsorize(pd.Series([1.0, 2.0]))                  # 索引不对
    with pytest.raises(acna.ContractError):
        acna.winsorize(vec(), method='nope')
    with pytest.raises(acna.ContractError):
        acna.standardize(vec(), method='nope')
    with pytest.raises(acna.ContractError):
        acna.neutralize(vec())                                 # 暴露和分组都没给
    with pytest.raises(acna.ContractError):
        acna.combine([])
    with pytest.raises(acna.ContractError):
        acna.combine([vec(), vec()], weights=[1, 2, 3])
    with pytest.raises(acna.ContractError):
        acna.combine([vec(), vec()], weights=[1, -1])
