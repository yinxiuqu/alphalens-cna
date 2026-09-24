"""子样本稳定性 + 衰减检验 —— **因子失效监控**的判定层。

为什么必须有
------------
``Verdict.stability`` 这个字段一直存在、报告里也会渲染成
"稳定性 X% 子样本成立"，但**全库没有任何地方计算它** —— 一个悬空字段。
看起来有这个能力、实际永远是 NaN，比缺失更糟。

本模块把它接上，并给出配套的"因子是不是在失效"的检验。

两个量
------
* :func:`subsample_stability` —— 把序列切成若干**连续**子段，看有多少段的
  均值与全样本**同号**。这就是 ``Verdict`` 里"X% 子样本成立"的字面含义。
* :func:`decay_test` —— 对序列与时间做回归，看**斜率**是否显著为负
  （即因子在衰减）。标准误用 Newey-West，重叠观测下朴素 t 会虚高。

⚠️ 与 :func:`alphalens_cna.analysis.ic_decay` 的区别：那个是**跨持有期**的衰减
（选调仓频率用），本模块是**跨时间**的衰减（监控因子失效）。名字像，用途不同。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail
from .newey_west import auto_lags, nw_variance, nw_tstat

__all__ = ['subsample_stability', 'decay_test']


def _clean(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.iloc[:, 0] if isinstance(x, pd.DataFrame) else x
        x = x.dropna()
    v = np.asarray(x, dtype=float)
    return v[np.isfinite(v)]


def subsample_stability(series, n_splits=2, min_obs=10):
    """**子样本一致性** —— 有多少比例的连续子段与全样本同号。

    Parameters
    ----------
    series : Series | array
        逐期 IC（或任何逐期估计量）。
    n_splits : int
        切成几段。``2`` = 前后半段；分段越多越严格。
    min_obs : int
        子段样本不足则**不参与**统计（并从 ``n_valid`` 里如实反映）。

    Returns
    -------
    dict
        ``stability`` ∈ [0,1]（= 同号子段占比）、``n_splits`` / ``n_valid`` /
        ``base_mean`` / ``base_sign`` / ``chunk_means``（各段均值，便于看路径）。

    Notes
    -----
    只判**符号**不判幅度：因子从 −0.05 衰减到 −0.01 仍然是"同号"，
    但那是失效的过程 —— 幅度的事交给 :func:`decay_test`。
    两个一起看才完整。
    """
    if n_splits < 2:
        fail('stability', 'bad_splits', f'n_splits 必须 >= 2，收到 {n_splits}')
    x = _clean(series)
    if len(x) < n_splits * min_obs:
        return {'stability': np.nan, 'n_splits': int(n_splits), 'n_valid': 0,
                'base_mean': float(x.mean()) if len(x) else np.nan,
                'base_sign': np.nan, 'chunk_means': []}
    base = float(x.mean())
    sign = np.sign(base)
    chunks = np.array_split(x, n_splits)
    means = [float(c.mean()) for c in chunks if len(c) >= min_obs]
    if not means or sign == 0:
        return {'stability': np.nan, 'n_splits': int(n_splits),
                'n_valid': len(means), 'base_mean': base,
                'base_sign': float(sign), 'chunk_means': means}
    frac = float(np.mean([np.sign(m) == sign for m in means]))
    return {'stability': frac, 'n_splits': int(n_splits),
            'n_valid': len(means), 'base_mean': base,
            'base_sign': float(sign), 'chunk_means': means}


def decay_test(series, lags=None, horizon=1):
    """**衰减检验** —— 序列对时间回归，斜率是否显著为负。

    Parameters
    ----------
    series : Series | array
        逐期 IC（或因子收益）。
    lags : int, optional
        Newey-West 滞后阶数。IC 序列常有自相关，不给就自动定。
    horizon : int
        持有期，用于自动定滞后阶数（重叠观测的跨度）。

    Returns
    -------
    dict
        ``slope``（每期变化量）、``t_nw`` / ``t_naive``、``annual_slope``
        （按 252 期折算）、``decaying``（斜率 < 0 且 |t| > 2）、
        ``n`` / ``lags`` / ``vif``、``half_life``（若在衰减，多少期后效应减半）。
    """
    x = _clean(series)
    n = len(x)
    if n < 20:
        return {'slope': np.nan, 't_nw': np.nan, 't_naive': np.nan,
                'annual_slope': np.nan, 'decaying': False, 'n': n,
                'lags': 0, 'vif': np.nan, 'half_life': np.nan}
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    sxx = float((tc ** 2).sum())
    b = float((tc * (x - x.mean())).sum() / sxx)          # OLS 斜率
    a = float(x.mean() - b * t.mean())
    e = x - (a + b * t)
    # HAC：score 序列的长期方差 → Var(b) = S / sxx
    u = tc * e
    if lags is None:
        lags = auto_lags(n, horizon)
    S = nw_variance(u, lags=lags)
    # ⚠️ 这里是 Var(β) = Ŝ_22 / sxx²，不是 S/sxx。
    #    推导（X=[[1,tc]]，Σtc=0 → (X'X)⁻¹=diag(1/n, 1/sxx)）：
    #        Var(β) = Ŝ_22 / sxx²，而 Ŝ_22 = Σ tc_t²e_t² ≈ T·Var(tc·e)
    #    nw_variance 返回的是**平均**量（已除 T−1），所以要乘回 T。
    #    自检：同方差下它应退化为经典公式 σ²/Σtc² —— 这正是抓出下面这个 bug 的方式
    #    （修前 t=−0.50，修后 −29.9，与经典公式一致）。
    se_hac = np.sqrt(len(x) * S / sxx ** 2)
    se_ols = np.sqrt((e ** 2).sum() / (n - 2) / sxx)
    t_nw = b / se_hac if se_hac > 0 else np.nan
    t_naive = b / se_ols if se_ols > 0 else np.nan
    st = nw_tstat(x, lags=lags, horizon=horizon)
    half = (abs(x.mean() / b) if b < 0 and x.mean() != 0 else np.nan)
    return {
        'slope': b, 't_nw': float(t_nw), 't_naive': float(t_naive),
        'annual_slope': b * 252.0,
        'decaying': bool(b < 0 and np.isfinite(t_nw) and abs(t_nw) > 2.0),
        'n': n, 'lags': int(lags), 'vif': st['vif'],
        'half_life': float(half) if np.isfinite(half) else np.nan,
    }
