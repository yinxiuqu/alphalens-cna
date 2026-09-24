"""DSR（紧缩夏普）与因子失效监控（稳定性 / 衰减）测试。"""

from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphalens_cna as acna  # noqa: E402


# ─────────────────────────────── DSR ───────────────────────────────
def test_psr_zero_sharpe_is_half():
    """★ 已知答案：SR=0 → PSR = 0.5（精确）。"""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    x = x - x.mean()                      # 强制样本均值恰为 0
    assert abs(acna.psr(x) - 0.5) < 1e-12


def test_dsr_with_one_trial_equals_psr():
    """N=1（没挑过）→ 期望最大夏普为 0 → DSR 必须精确等于 PSR。"""
    y = np.random.default_rng(1).normal(0.05, 1, 500)
    d = acna.dsr(y, n_trials=1)
    assert abs(d.dsr - d.psr) < 1e-12
    assert acna.inference.expected_max_sharpe(1, 0.01) == 0.0


def test_dsr_shrinks_with_more_trials():
    """★ 挑得越多，扣得越狠 —— 方向不能反。"""
    rng = np.random.default_rng(2)
    # 把样本夏普**控制成恰好 0.05**，否则随机抽样的均值可能恰好≈0，断言不稳
    z = rng.normal(0, 1, 500)
    y = (z - z.mean()) / z.std(ddof=1) + 0.08   # SR=0.08 → PSR≈0.96
    v = [acna.dsr(y, n).dsr for n in (1, 10, 100, 1000, 2850)]
    assert all(a > b for a, b in zip(v, v[1:])), v
    assert v[0] > 0.9 and v[-1] < 0.1


def test_dsr_2850_scenario():
    """★ 审阅者点名的场景：年化夏普 ~1、PSR 挺好看，但挑过 2850 次就什么都不剩。"""
    z = np.random.default_rng(3).normal(0, 1, 250)
    y = (z - z.mean()) / z.std(ddof=1) + 0.06      # 受控夏普
    d = acna.dsr(y, n_trials=2850)
    assert d.psr > 0.7                      # 不做校正时"看着挺好"
    assert d.dsr < 0.05                     # 校正后什么都不剩
    assert not d.significant
    assert d.expected_max_sr > d.sr         # 噪声门槛比你的夏普还高
    assert '扣不掉选择偏差' in str(d)


def test_dsr_more_data_helps():
    """同样单期夏普下，样本越长 DSR 越高（用同一份序列截断，避免抽样差异）。"""
    rng = np.random.default_rng(4)
    y = rng.normal(0.08, 1, 3000)
    a = acna.dsr(y[:200], 50).dsr
    b = acna.dsr(y[:1500], 50).dsr
    assert b > a


def test_dsr_bad_inputs():
    with pytest.raises(acna.ContractError):
        acna.dsr([0.1, 0.2, 0.3], n_trials=1)          # 观测太少
    with pytest.raises(acna.ContractError):
        acna.dsr(np.random.default_rng(0).normal(size=100), n_trials=0)
    with pytest.raises(acna.ContractError):
        acna.dsr(np.zeros(100), n_trials=1)            # 零波动


def test_min_track_record_length():
    rng = np.random.default_rng(5)
    y = rng.normal(0.15, 1, 300)
    n = acna.min_track_record_length(y, n_trials=10)
    assert np.isfinite(n) and n > 0
    # 挑得越多，需要越长的轨长
    assert acna.min_track_record_length(y, 500) > n
    # 夏普太低（打不过噪声门槛）→ 再多数据也没用
    assert acna.min_track_record_length(rng.normal(0.001, 1, 300), 5000) == np.inf


# ──────────────────────── 稳定性 / 衰减 ────────────────────────
def test_stability_known_answers():
    rng = np.random.default_rng(6)
    flip = np.r_[rng.normal(0.05, 0.01, 60), rng.normal(-0.05, 0.01, 60)]
    stable = rng.normal(0.05, 0.02, 120)
    assert acna.subsample_stability(flip)['stability'] == 0.5
    assert acna.subsample_stability(stable)['stability'] == 1.0
    # 四段：三段与全样本同号
    x = np.r_[np.full(30, 0.05), np.full(30, 0.05), np.full(30, 0.05), np.full(30, -0.05)]
    assert acna.subsample_stability(x, n_splits=4)['stability'] == 0.75


def test_stability_reports_chunk_means():
    x = np.r_[np.full(20, 1.0), np.full(20, -1.0)]
    r = acna.subsample_stability(x)
    assert len(r['chunk_means']) == 2 and r['n_valid'] == 2


def test_stability_too_short_is_nan():
    assert np.isnan(acna.subsample_stability(np.arange(5.0), min_obs=10)['stability'])


def test_decay_detects_decay():
    """★ 人工衰减序列必须判为 decaying，且 HAC t 与经典公式同量级。"""
    rng = np.random.default_rng(7)
    t = np.arange(240)
    x = 0.08 - 0.0006 * t + rng.normal(0, 0.02, 240)
    r = acna.decay_test(pd.Series(x))
    assert r['decaying'] is True
    assert r['t_nw'] < -10
    assert r['slope'] < 0 and r['half_life'] > 0
    # 与经典同方差公式交叉验证（HAC 只做自相关修正，不该差一个量级）
    tc = t - t.mean()
    sxx = (tc ** 2).sum()
    b = (tc * (x - x.mean())).sum() / sxx
    e = x - (x.mean() + b * tc)
    se_cls = np.sqrt((e ** 2).sum() / (240 - 2) / sxx)
    assert abs(r['t_nw'] / (b / se_cls) - 1) < 0.25


def test_decay_no_false_positive_on_noise():
    rng = np.random.default_rng(8)
    r = acna.decay_test(pd.Series(rng.normal(0.03, 0.02, 300)))
    assert r['decaying'] is False


def test_decay_too_short():
    # 门槛已从 20 降到 8：10 期现在**能算**（这正是修短样本问题的目的）
    assert np.isfinite(acna.decay_test(pd.Series(np.arange(10.0)))['slope'])
    # 真短到算不出时，decaying 必须是 None 而不是 False
    r = acna.decay_test(pd.Series(np.arange(5.0)))
    assert np.isnan(r['slope']) and r['decaying'] is None


# ──────────────────────── 接进报告 ────────────────────────
def _mk(seed=9, n_days=300, n_assets=60):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2022-01-03', periods=n_days)
    assets = [f'{i:06d}' for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    n = len(idx)
    cl = 10 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, (n_days, n_assets)), axis=0))
    c = pd.DataFrame(cl.ravel(), index=idx, columns=['c'])['c']
    o = c * (1 + rng.normal(0, 0.004, n))
    px = pd.DataFrame({'raw_open': o, 'raw_close': c,
                       'raw_high': np.maximum(o, c) * 1.005,
                       'raw_low': np.minimum(o, c) * 0.995,
                       'adj_factor': 1.0, 'volume': 1e6}, index=idx)
    for k in ('open', 'close', 'high', 'low'):
        px[f'adj_{k}'] = px[f'raw_{k}']
    px['prev_close'] = px['raw_close'].groupby(level='asset').shift(1).fillna(px['raw_open'])
    fwd = pd.Series(px['adj_close'].groupby(level='asset')
                    .pct_change(fill_method=None).shift(-21), index=idx)
    f = pd.DataFrame({'value': 0.5 * fwd.fillna(0).to_numpy()
                               + rng.normal(0, 0.05, n),
                      'available_at': idx.get_level_values('date')}, index=idx)
    return px, f, acna.Calendar(dates)


def test_report_has_stability_and_dsr():
    px, f, cal = _mk()
    r = acna.build_report(f, px, cal, horizons=(21, 63), n_trials=2850, name='t')
    assert set(r.stability_detail) == {21, 63}
    assert r.dsr is not None and r.dsr.n_trials == 2850
    # ★ 悬空字段不再悬空
    assert np.isfinite(r.verdict.stability)
    assert r.verdict.stability == r.stability_detail[21]['stability']
    md = r.to_markdown()
    assert '## 七、因子衰减与稳定性' in md and '紧缩夏普比率' in md
    fr = r.frames()
    assert 'stability' in fr and 'dsr' in fr


def test_verdict_renders_stability():
    y = np.full(60, -0.04) + np.random.default_rng(0).normal(0, 0.01, 60)
    ic = pd.DataFrame({'21': y}, index=pd.bdate_range('2024-01-01', periods=60))
    st = acna.subsample_stability(y)['stability']
    v = acna.assess(ic=ic, n_trials=3, stability=st)
    assert '稳定性' in str(v) and '100% 子样本成立' in str(v)
