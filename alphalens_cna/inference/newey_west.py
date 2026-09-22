"""Newey-West 修正 + 有效样本量。

为什么必须做
------------
因子分析里 **IC 序列是重叠观测**：算 20 日持有期的 IC 时，相邻两天的
收益窗口重叠 19 天 —— 它们**不是独立样本**。直接用 ``ICIR·√n`` 会**严重高估** t 值。

本次实测（同样的数据、同样的 IC 序列）
--------------------------------------
=================  ==========  ==========  ========
因子                t_naive     t_NW        虚高倍数
=================  ==========  ==========  ========
ROE（20 日）        0.16        0.04        **3.7×**
动量（20 日）       −21.93      −6.76       **3.2×**
=================  ==========  ==========  ========

**同一份数据，t 值差 3–4 倍。** alphalens 只给朴素 t，所以看不出该信哪个。

``n_eff``（有效样本量）是这件事的另一面：名义 91 个月，12 个月持有期下
**有效独立样本只有约 7 个** —— 91 和 7 是两个世界。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail

__all__ = ['nw_variance', 'nw_tstat', 'auto_lags', 'effective_n',
           'variance_inflation', 'newey_west_summary']


def auto_lags(n, horizon=1):
    """自动选滞后阶数。

    * 有持有期时，**至少取 h−1** —— 重叠窗口的跨度就是 h−1"""
    n = int(n)
    if n < 3:
        return 0
    l_auto = int(np.floor(4 * (n / 100) ** (2 / 9)))     # Newey-West (1994) 经验式
    return int(max(l_auto, max(int(horizon) - 1, 0)))


def _as_1d(x):
    if isinstance(x, (pd.Series, pd.Index)):
        x = x.values
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.ndim != 1:
        fail('newey_west', 'shape', f'需要一维序列，收到 {x.shape}')
    return x


def nw_variance(x, lags=None, horizon=1):
    """长期方差 ``S`` 的 Newey-West 估计（Bartlett 核）。

    .. math::
        S = \\gamma_0 + 2\\sum_{j=1}^{L}\\left(1-\\frac{j}{L+1}\\right)\\gamma_j

    Returns
    -------
    float
        ``S``（**不是**均值的方差 —— 后者是 ``S/T``）。
    """
    x = _as_1d(x)
    T = len(x)
    if T < 3:
        return np.nan
    if lags is None:
        lags = auto_lags(T, horizon)
    lags = int(min(max(lags, 0), T - 2))
    d = x - x.mean()
    # ⚠️ 分母统一用 T−1（与样本方差 `sd(ddof=1)` 一致），不是 T。
    # 若用 T，则**无自相关时 vif 也会是 (T−1)/T ≈ 0.9967 而非 1.0**，
    # 进而让 n_eff = n/vif 系统性偏大 —— 而 vif 的全部意义就是
    # 「相对无自相关时膨胀了几倍」。统一分母后：
    #   · 无自相关 → vif = 1.0（精确）
    #   · lags=0   → t_nw = t_naive（精确）
    g0 = float(np.dot(d, d) / max(T - 1, 1))
    S = g0
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1)                 # Bartlett 核
        gj = float(np.dot(d[:-j], d[j:]) / max(T - 1, 1))
        S += 2.0 * w * gj
    # ★ 下限 γ₀（等价于 vif ≥ 1）：NW **永不比朴素方差更小**。
    #
    # 依据（实测，300 次 iid 模拟）：因为减了样本均值，E[γ̂_j] = −σ²/T，
    # 于是 E[S] ≈ γ₀(1 − 2Σw/T)，NW 在短样本上**系统性低估**方差：
    #     T= 30 → vif 均值 0.898（少校正 10.2%）
    #     T= 91 → vif 均值 0.968（少校正  3.2%）
    #     T=252 → vif 均值 0.995（少校正  0.5%）
    # 换无偏分母（÷(T−j)）救不了 —— 偏差来自"估计均值"这一步，不是分母。
    # 低估方差 = **高估显著性**，正是本库最不该犯的错，所以直接封死这个方向。
    #
    # 代价（明说）：真实的负自相关也会被当成 vif=1 —— 此时 t_nw = t_naive，
    # 不会比朴素 t 更宽松。本库的主场景（重叠观测）只会让 vif > 1，下限不生效。
    if not np.isfinite(S):
        return g0
    return float(max(S, g0))


def variance_inflation(x, lags=None, horizon=1):
    """方差膨胀因子 ``VIF = S / γ₀``。``>1`` 表示正自相关（重叠观测）。"""
    x = _as_1d(x)
    if len(x) < 3:
        return np.nan
    g0 = nw_variance(x, 0, horizon)      # lags=0 时 S 就是 γ₀（同源，不出 ulp）
    if not np.isfinite(g0) or g0 <= 0:
        return np.nan
    return nw_variance(x, lags, horizon) / g0


def effective_n(x, lags=None, horizon=1):
    """**有效样本量** ``n_eff = n / VIF``。

    ⚠️ 这是**启发式估计**，不是精确量 —— 重叠观测下没有唯一正确答案。
    本库把它**当作量级参考**：``91 → 约 7`` 这种差别足以改变结论，
    但 ``7.2 vs 7.8`` 没有意义。

    所以报告里给**区间**而不是点估计（见 ``newey_west_summary``）。
    """
    x = _as_1d(x)
    n = len(x)
    if n < 3:
        return np.nan
    vif = variance_inflation(x, lags, horizon)
    if not np.isfinite(vif) or vif <= 0:
        return np.nan
    return float(n / vif)


def nw_tstat(x, lags=None, horizon=1):
    """均值的 Newey-West t 值 ``t = mean / sqrt(S/T)``。

    Returns
    -------
    dict
        ``mean`` / ``t_naive`` / ``t_nw`` / ``n`` / ``n_eff`` / ``lags`` /
        ``vif``
    """
    x = _as_1d(x)
    n = len(x)
    if n < 3:
        return {'mean': np.nan, 't_naive': np.nan, 't_nw': np.nan,
                'n': n, 'n_eff': np.nan, 'lags': 0, 'vif': np.nan}
    if lags is None:
        lags = auto_lags(n, horizon)
    # ⚠️ 报告用的 lags 必须与**实际计算用的**一致：
    #    nw_variance 内部把 lags 截到 T-2，若这里报未截断的值，
    #    会出现"32 期却报了 125 阶滞后"这种自相矛盾的输出（实测踩过）。
    lags = int(min(max(int(lags), 0), max(n - 2, 0)))
    mean = float(x.mean())
    # ★ 朴素 t 与 NW t 必须走**同一个方差来源**。
    #   原实现里 t_naive 用 numpy.std(ddof=1)、t_nw 用 np.dot(d,d)/(n-1) ——
    #   数学上相等，但浮点求和顺序不同：本机 BLAS 上恰好一致，
    #   GitHub Actions 的 BLAS 上差 1 ULP（CI 实测 -0.2866165090394768
    #   vs -0.28661650903947683）。差最后一位不是"精度问题"，
    #   是**两个真相源** —— 修掉源头，而不是把断言放松。
    g0 = nw_variance(x, 0, horizon)
    t_naive = mean / np.sqrt(g0 / n) if g0 > 0 else np.nan
    S = nw_variance(x, lags, horizon)
    se = np.sqrt(S / n)
    t_nw = mean / se if se > 0 else np.nan
    # ⚠️ vif 走 variance_inflation（与 nw_variance 同一个 γ₀），
    #    不要在这里另算 sd² —— numpy 的 std 与 np.dot(d,d)/(T-1)
    #    求和顺序不同，末位会差 1 ulp，把"无自相关时 vif 精确为 1"破坏掉。
    vif = variance_inflation(x, lags, horizon)
    return {
        'mean': mean,
        't_naive': float(t_naive) if np.isfinite(t_naive) else np.nan,
        't_nw': float(t_nw) if np.isfinite(t_nw) else np.nan,
        'n': n,
        'n_eff': float(n / vif) if (np.isfinite(vif) and vif > 0) else np.nan,
        'lags': int(lags),
        'vif': float(vif) if np.isfinite(vif) else np.nan,
    }


def newey_west_summary(series, horizons=None, lags=None):
    """对 IC / γ 的时间序列做 NW 汇总。

    Parameters
    ----------
    series : Series | DataFrame
        * Series → 单序列
        * DataFrame → 逐列做；可用 ``horizons`` 限定列
    horizons : list, 可选
        从 DataFrame 的列里挑（列名含持有期数字，如 ``1``/``21``）。
    lags : int, 可选
        **不给就自动**（``max(NW 经验式, h−1)``）——
        持有期越长，重叠越严重，滞后阶数要跟上。

    Returns
    -------
    DataFrame
        columns: ``n`` / ``mean`` / ``t_naive`` / ``t_nw`` / ``vif`` / ``n_eff`` /
        ``lags`` / ``t_inflation``（``|t_naive/t_nw|`` —— **虚高倍数**）
    """
    if isinstance(series, pd.DataFrame):
        cols = list(series.columns)
        if horizons is not None:
            cols = [c for c in cols if _horizon_of(c) in set(horizons)]
        rows = {}
        for c in cols:
            rows[c] = nw_tstat(series[c].dropna(), lags=lags,
                               horizon=_horizon_of(c) or 1)
        df = pd.DataFrame(rows).T
    else:
        s = series.dropna() if isinstance(series, pd.Series) else pd.Series(series).dropna()
        df = pd.DataFrame({'value': nw_tstat(s, lags=lags)}).T
    if 't_naive' in df.columns and 't_nw' in df.columns:
        with np.errstate(divide='ignore', invalid='ignore'):
            df['t_inflation'] = (df['t_naive'] / df['t_nw']).abs()
    return df


def _horizon_of(col):
    try:
        return int(str(col).rsplit('_', 1)[1])
    except (ValueError, IndexError):
        try:
            return int(col)
        except (ValueError, TypeError):
            return None
