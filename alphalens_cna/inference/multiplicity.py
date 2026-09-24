"""多重假设检验 —— **本书的灵魂，也是本项目区别于所有 A 股 alphalens 改造的地方。**

为什么必须做
------------
alphalens 只给一个 p 值。但你测的因子越多，**假阳性必然出现**：

    测试 4 个因子   → Bonferroni 阈值 |t| > 2.498
    测试 20 个      → |t| > 3.023
    测试 100 个     → |t| > 3.481
    316 个（HLZ 统计的已发表因子数） → |t| > 3.778

**同一因子、同一数据，只因"你一共测了多少个"，显著性结论就翻转。**

本次实测（FM 四因子，NW lag=3）
--------------------------------
===============  ========  ========  ========  ========
因子              t_NW      原始 p    BH        **BHY**
===============  ========  ========  ========  ========
roe_z            +0.177    0.8595    0.9188    1.0000 ✗
pb_z             −0.102    0.9188    0.9188    1.0000 ✗
**ln_mv_z**      **−3.687** 0.0002   0.0009    **0.0019 ✓**
**turnover_z**   **−3.257** 0.0011   0.0023    **0.0047 ✓**
===============  ========  ========  ========  ========

但**换手因子的 t=−3.257 在 n=100 时会掉出阈值（3.481）**。

选哪个方法
----------
* **Bonferroni** —— 最简单、最保守（控制 FWER）。适合"宁可漏，不可错"
* **Holm** —— 比 Bonferroni 强（同样控 FWER，但更不保守），**优先用它替代 Bonferroni**
* **BH** —— 控制 FDR，**假设检验间独立或正相关**
* **BHY** —— 控制 FDR，**不假设独立性**（乘以 c(n)=Σ1/i）

> 因子之间天然相关（本次实测 ROE↔规模相关 **0.360**），
> **所以默认用 BHY**，而不是 BH。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail

__all__ = ['bonferroni', 'holm', 'benjamini_hochberg', 'benjamini_yekutieli',
           'adjust', 't_threshold', 'p_from_t', 'hlz_threshold', 'METHODS',
           'dependency_factor']

METHODS = ('bonferroni', 'holm', 'bh', 'bhy', 'none')


# --------------------------------------------------------------------------- #
def dependency_factor(n):
    """BHY 的依赖修正因子 ``c(n) = Σ_{i=1..n} 1/i``。

    ``c(4) = 2.0833``；n 大时 ≈ ``ln(n) + 0.5772``。
    **这是 BH 与 BHY 的唯一区别。**
    """
    n = int(n)
    if n < 1:
        fail('multiplicity', 'bad_n', f'n 必须 >= 1，收到 {n}')
    return float(np.sum(1.0 / np.arange(1, n + 1)))


# --------------------------------------------------------------------------- #
def _as_array(p):
    a = np.asarray(p, dtype=float)
    if a.ndim != 1:
        a = a.ravel()
    if not np.all(np.isfinite(a)):
        fail('multiplicity', 'nan_p',
             'p 值里有 NaN/Inf —— 多重检验无法在有缺失时进行。\n'
             '  最可能的原因不是 p 值本身，而是**上游样本不足**：'
             '持有期超过样本跨度、或期数太少导致 t 算不出来。\n'
             '  先看 clean() 的台账与 IC 面板，再回来做校正。')
    if np.any((a < 0) | (a > 1)):
        fail('multiplicity', 'p_range', f'p 值必须落在 [0,1]，收到 {a.min()}~{a.max()}')
    return a


def bonferroni(p):
    """Bonferroni：``p_adj = min(1, n·p)``。控 FWER，最保守。"""
    p = _as_array(p)
    return np.minimum(1.0, p * len(p))


def holm(p):
    """Holm step-down：控 FWER，**在同等保证下比 Bonferroni 更不保守**。

    ``p_(i)_adj = min(1, max_{j<=i} (n-j+1)·p_(j))``（按 p 升序排序后累进取最大）
    """
    p = _as_array(p)
    n = len(p)
    order = np.argsort(p, kind='stable')
    ranked = p[order]
    factors = n - np.arange(n)                 # n, n-1, ..., 1
    adj = np.minimum(1.0, factors * ranked)
    adj = np.maximum.accumulate(adj)           # 保证单调不减
    out = np.empty(n)
    out[order] = adj
    return out


def _stepup(p, c=1.0):
    """BH / BHY 共用的 step-up 过程（只差一个常数 c）。"""
    p = _as_array(p)
    n = len(p)
    order = np.argsort(p, kind='stable')
    ranked = p[order]
    factors = n * c / np.arange(1, n + 1)      # n·c/1, n·c/2, ...
    adj = np.minimum(1.0, factors * ranked)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # 从大到小取最小 → 单调不减
    out = np.empty(n)
    out[order] = adj
    return out


def benjamini_hochberg(p):
    """BH：控 FDR。**假设检验间独立或正相关。**"""
    return _stepup(p, c=1.0)


def benjamini_yekutieli(p):
    """BHY：控 FDR，**不假设独立性**（乘 ``c(n)=Σ1/i``）。

    因子之间天然相关，所以**这是本库的默认方法**。
    """
    p = _as_array(p)
    return _stepup(p, c=dependency_factor(len(p)))


_DISPATCH = {
    'bonferroni': bonferroni,
    'holm': holm,
    'bh': benjamini_hochberg,
    'bhy': benjamini_yekutieli,
    'none': lambda p: _as_array(p).copy(),
}


def adjust(p, method='bhy', n_trials=None):
    """统一入口。

    Parameters
    ----------
    p : array-like
        原始 p 值。
    method : {'bhy', 'bh', 'holm', 'bonferroni', 'none'}
        默认 ``'bhy'``（不假设独立性 —— 因子间相关性是常态）。
    n_trials : int, 可选
        **你一共测过多少个假设。** 给了就用它做校正基数，
        而不是 ``len(p)`` —— 这一点很关键：

        > 你测了 100 个因子，只有 4 个的 p 值留下来了（其余算不出），
        > 校正基数**仍应是 100**，不是 4。
        > 否则等于把"测过但没报"的悄悄抹掉，多重检验就白做了。

    Returns
    -------
    ndarray
        校正后 p 值（与输入等长）。
    """
    if method not in METHODS:
        fail('multiplicity', 'bad_method',
             f'method 只能是 {METHODS}，收到 {method!r}')
    p = _as_array(p)
    if n_trials is not None and n_trials < len(p):
        fail('multiplicity', 'n_trials_too_small',
             f'n_trials={n_trials} 小于实际检验数 {len(p)}。\n'
             f'  含义：n_trials 是"一共测过几次"，不可能比给出的 p 值还少。')
    if n_trials is None or n_trials == len(p):
        return _DISPATCH[method](p)

    # 用 n_trials 做基数：把 p 值补上 (n_trials - len(p)) 个"未报告"的 1.0
    pad = np.ones(int(n_trials) - len(p))
    adj = _DISPATCH[method](np.concatenate([p, pad]))
    return adj[:len(p)]


# --------------------------------------------------------------------------- #
def p_from_t(t, df=None):
    """双侧 p 值。``df=None`` 用正态近似（大样本）。"""
    from scipy import stats
    t = np.asarray(t, dtype=float)
    if df is None:
        return 2 * (1 - stats.norm.cdf(np.abs(t)))
    return 2 * (1 - stats.t.cdf(np.abs(t), df))


def t_threshold(n_trials, alpha=0.05):
    """给定检验次数，**Bonferroni 校正下的 |t| 阈值**（正态近似、双侧）。

    >>> round(t_threshold(4), 3)
    2.498
    >>> round(t_threshold(100), 3)
    3.481
    """
    from scipy import stats
    if n_trials < 1:
        fail('multiplicity', 'bad_n', f'n_trials 必须 >= 1，收到 {n_trials}')
    p = alpha / n_trials
    return float(stats.norm.ppf(1 - p / 2))


def hlz_threshold(n_trials, alpha=0.05):
    """Harvey-Liu-Zhu 式阈值，**均值 0、方差 1 的 t 统计量**。

    与 :func:`t_threshold` 数值相同（都是 Bonferroni 的正态近似），
    单列出来是为了在报告里能明确标注口径来源。

    参考基准：HLZ (2016) 建议 **t > 3.0**；
    316 个已发表因子对应 **3.778**。
    """
    return t_threshold(n_trials, alpha)


# --------------------------------------------------------------------------- #
def to_frame(p, method='bhy', n_trials=None, labels=None, tstats=None):
    """整理成可读的表 —— 报告里直接贴这个。

    Returns
    -------
    DataFrame
        columns: ``label`` / ``p_raw`` / ``p_adj`` / （有 tstats 时）``t`` /
        ``significant_raw`` / ``significant_adj``
    """
    p = _as_array(p)
    adj = adjust(p, method, n_trials)
    n = int(n_trials or len(p))
    df = pd.DataFrame({
        'label': labels if labels is not None else [f'h{i}' for i in range(len(p))],
        'p_raw': p,
        'p_adj': adj,
        'significant_raw': p < 0.05,
        'significant_adj': adj < 0.05,
    })
    if tstats is not None:
        df.insert(1, 't', np.asarray(tstats, dtype=float))
    df.attrs['method'] = method
    df.attrs['n_trials'] = n
    df.attrs['t_threshold'] = t_threshold(n)
    df.attrs['note'] = (f'{method.upper()} 校正，n_trials={n}，'
                        f'Bonferroni |t| 阈值 {t_threshold(n):.3f}')
    return df.sort_values('p_raw').reset_index(drop=True)
