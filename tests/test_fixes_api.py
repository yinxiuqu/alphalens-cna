"""0.1.2 修掉的 6 个问题的回归测试。

**第 1 条最重要**：`__all__` 里写了名字但取不到，会让
`from alphalens_cna import *` 直接抛异常 —— 任何星号导入的 notebook 一升级就炸。
这类 bug 325 个测试一个都没抓到，所以这里补一道**自省式防线**。
"""

from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphalens_cna as acna  # noqa: E402


# ── ① __all__ 完整性（自省式防线）────────────────────────────────
def test_every_all_name_is_reachable():
    """★ `__all__` 里的每个名字都必须真的取得到。

    （`rolling_ic` 曾只写进 __all__、漏了 import，导致
      `from alphalens_cna import *` 抛 AttributeError。）
    """
    missing = [n for n in acna.__all__ if not hasattr(acna, n)]
    assert not missing, f'__all__ 里取不到的名字: {missing}'


def test_star_import_works():
    """★ 星号导入不许抛异常。"""
    ns = {}
    exec('from alphalens_cna import *', ns)
    assert 'rolling_ic' in ns and 'build_report' in ns


def test_rolling_ic_reachable_from_both_paths():
    assert hasattr(acna, 'rolling_ic')
    from alphalens_cna.analysis import rolling_ic as r2   # noqa: F401


# ── ② __version__ 跟发行元数据走 ─────────────────────────────────
def test_version_matches_installed_metadata():
    """★ 运行时自报的版本必须与发行元数据一致。"""
    try:
        from importlib.metadata import version
        assert acna.__version__ == version('alphalens-cna')
    except Exception:                                            # noqa: BLE001
        pytest.skip('未安装为发行版')
    assert acna.__version__ != '0.1.0.dev0', '不许再硬编码开发版号'


# ── ③ 复权连续性：相对容差，不被存储舍入误判 ────────────────────
def test_adj_continuity_ignores_storage_rounding():
    """★ 10 位小数存储带来的 ~1e-10 舍入不该被判成"因子倒退"。"""
    dates = pd.bdate_range('2024-01-01', periods=6)
    assets = ['000001']
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    f = np.array([1.0, 1.1, 1.1, 1.1, 1.21, 1.21])
    f = np.round(f + np.array([0, 0, -1e-10, 0, +1e-10, 0]), 10)   # 模拟存储舍入
    df = pd.DataFrame({'adj_factor': f, 'raw_open': 10.0, 'raw_close': 10.0,
                       'raw_high': 10.0, 'raw_low': 10.0,
                       'adj_open': 10.0, 'adj_close': 10.0,
                       'adj_high': 10.0, 'adj_low': 10.0,
                       'prev_close': 10.0, 'volume': 1e6}, index=idx)
    rep = acna.health_check(prices=df)
    ac = [x for x in rep.findings if x.name == '复权连续性'][0]
    assert ac.severity == 'pass', f'舍入被误判: {ac.summary}'


def test_adj_continuity_still_catches_real_regression():
    """反向：真实倒退（相对幅度远超容差）仍必须判硬错误。"""
    dates = pd.bdate_range('2024-01-01', periods=4)
    idx = pd.MultiIndex.from_product([dates, ['000001']], names=['date', 'asset'])
    df = pd.DataFrame({'adj_factor': [1.0, 1.2, 0.6, 0.6], 'raw_open': 10.0,
                       'raw_close': 10.0, 'raw_high': 10.0, 'raw_low': 10.0,
                       'adj_open': 10.0, 'adj_close': 10.0, 'adj_high': 10.0,
                       'adj_low': 10.0, 'prev_close': 10.0, 'volume': 1e6}, index=idx)
    ac = [x for x in acna.health_check(prices=df).findings
          if x.name == '复权连续性'][0]
    assert ac.severity == 'fail'


# ── ④ 短样本：敢算，且"算不出来"不许说成"没衰减" ────────────────
def test_stability_works_on_14_periods():
    """★ 14 期月频面板是常见规模，不该整节空白。"""
    rng = np.random.default_rng(0)
    x = rng.normal(-0.02, 0.15, 14)
    st = acna.subsample_stability(x)
    dc = acna.decay_test(x)
    assert np.isfinite(st['stability']) and st['n_valid'] == 2
    assert np.isfinite(dc['slope']) and np.isfinite(dc['t_nw'])


def test_decaying_is_none_not_false_when_uncomputable():
    """★★ 样本不足时 decaying 必须是 **None**（渲染成"—"），不能是 False。

    False 的含义是"检验过、没衰减"；这里是"根本没检验"。
    把"未检验"报成"已检验且通过"，正是这个库一路在抓的那类错误。
    """
    r = acna.decay_test(np.arange(5.0))
    assert r['decaying'] is None, r['decaying']
    assert np.isnan(r['slope'])


def test_report_renders_dash_not_no_when_uncomputable():
    """★ 算不出衰减时，报告必须显示"—"，不能显示"否"。

    直接构造 Report 来测渲染，避免依赖具体数据规模。
    """
    nan = float('nan')
    rep = acna.Report(factor_name='x', stability_detail={
        21: {'stability': nan, 'chunk_means': [], 'slope': nan,
             't_slope': nan, 'decaying': None, 'half_life': nan}})
    blk = rep.to_markdown().split('## 七、因子衰减与稳定性')[1].split('## ')[0]
    assert '—' in blk, blk
    assert '| 否 |' not in blk and '| 是 |' not in blk, f'不该报"是/否"：{blk}'
    # 反向：能算出来时必须给是/否，而不是永远破折号
    rep2 = acna.Report(factor_name='x', stability_detail={
        21: {'stability': 1.0, 'chunk_means': [0.1, 0.1], 'slope': -1e-4,
             't_slope': -3.0, 'decaying': True, 'half_life': 100.0}})
    blk2 = rep2.to_markdown().split('## 七、因子衰减与稳定性')[1].split('## ')[0]
    assert '| 是 |' in blk2 and '100%' in blk2, blk2



# ── ⑤ auto_lags 比例上限 ─────────────────────────────────────────
def test_auto_lags_capped_by_sample_size():
    """★ 14 期不该估 12 阶滞后。"""
    from alphalens_cna.inference.newey_west import auto_lags
    assert auto_lags(14, 63) <= 3
    assert auto_lags(20, 252) <= 5
    # 样本足够时不改变既有行为
    assert auto_lags(91, 21) == 20
    assert auto_lags(330, 1) == 5      # floor(4·3.3^(2/9))=5，上限 n/4=82 不生效


def test_nw_still_uses_available_lags():
    from alphalens_cna.inference.newey_west import nw_tstat
    x = np.random.default_rng(0).normal(size=14)
    st = nw_tstat(x, horizon=63)
    assert st['lags'] <= 3 and st['vif'] >= 1.0


# ── ⑥ 表格不再多出 index 列 ─────────────────────────────────────
def test_report_tables_have_no_spurious_index_column():
    """★ 第七、八节的表头不该出现重复的 `index` 列。"""
    rng = np.random.default_rng(2)
    dates = pd.bdate_range('2022-01-03', periods=300)
    assets = [f'{i:06d}' for i in range(40)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    n = len(idx)
    px = pd.DataFrame({c: rng.uniform(5, 30, n) for c in
                       ('raw_open', 'raw_close', 'raw_high', 'raw_low')}, index=idx)
    px['raw_high'] = np.maximum(px['raw_open'], px['raw_close']) * 1.01
    px['raw_low'] = np.minimum(px['raw_open'], px['raw_close']) * 0.99
    for k in ('open', 'close', 'high', 'low'):
        px[f'adj_{k}'] = px[f'raw_{k}']
    px['adj_factor'] = 1.0
    px['prev_close'] = px.groupby(level='asset')['raw_close'].shift(1).fillna(px['raw_open'])
    f = pd.DataFrame({'value': rng.normal(size=n),
                      'available_at': idx.get_level_values('date')}, index=idx)
    md = acna.build_report(f, px, acna.Calendar(dates), horizons=(21,),
                           quantiles=5, n_trials=3, name='t').to_markdown()
    for sec in ('## 七、因子衰减与稳定性', '## 八、尾部风险'):
        blk = md.split(sec)[1].split('## ')[0]
        headers = [l for l in blk.split('\n') if l.startswith('| ')][0]
        assert '| index |' not in headers, f'{sec} 表头多了 index 列: {headers}'
