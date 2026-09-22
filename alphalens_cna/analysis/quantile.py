"""分层与双重排序。

设计 §6.1：**分层是一个可组合的原语**。

============  ============  ===================================
用法          ``by``        ``method``
============  ============  ===================================
单变量分层     ``None``      —
行业中性分层   行业          ``'conditional'``（行业内分层）
独立双重排序   控制变量      ``'independent'``（两者分位点都取自全样本）
条件双重排序   控制变量      ``'conditional'``（因子分位点取自组内）
============  ============  ===================================

**为什么双重排序必要**
本次 Fama-MacBeth 证明 ROE 与规模相关 **0.360**，且 ROE 的负向是规模混淆出来的 ——
但那**只靠参数方法（回归）支撑**。双重排序是**非参数交叉验证**：

> **可证伪预测**：若"ROE 负向 = 规模混淆"成立，
> 则**同一规模组内** ROE 的单调性应当消失。

两种排序的差别
--------------
* **独立**：两套分位点都按全样本算 ⇒ 各格样本数可能很不均
  （小盘股在高因子组里会扎堆）
* **条件**：先按 ``by`` 分组，因子分位点在**组内**算 ⇒ 每格样本数均衡，
  且能干净地回答"控制掉 by 之后，因子还有没有区分度"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail
from .ic import horizon_of, return_cols

__all__ = ['quantize', 'quantile_returns', 'quantile_stats', 'double_sort',
           'monotonicity_test']


# --------------------------------------------------------------------------- #
def _qcut(s, n):
    """等频分位；重复值过多时退回等距，避免 qcut 抛 bin 不唯一。"""
    s = pd.Series(s).dropna()
    if len(s) < n:
        return pd.Series(np.nan, index=s.index)
    try:
        q = pd.qcut(s, n, labels=False, duplicates='drop')
        return q.astype('float')
    except (ValueError, IndexError):
        if np.ptp(s.values) == 0:
            return pd.Series(np.nan, index=s.index)
        edges = np.linspace(s.min(), s.max(), n + 1)[1:-1]
        return pd.Series(np.digitize(s.values, edges).astype(float), index=s.index)


def _quantile_within(index, values, group_keys, n):
    """按 ``group_keys`` 的每个组合分组，组内做 ``_qcut``。

    用显式分组赋值，不用 ``groupby(...).apply(...)`` ——
    后者在多键分组时返回的索引可能是组键而非原索引，
    ``reindex`` 之后会**静默变成全 NaN**（本会话踩过）。
    """
    out = pd.Series(np.nan, index=index, dtype=float)
    if not group_keys:
        out.loc[values.dropna().index] = _qcut(values, n).values
        return out
    key_df = pd.DataFrame({f'_k{i}': np.asarray(k) for i, k in enumerate(group_keys)},
                          index=index)
    for _, g in key_df.groupby(list(key_df.columns), sort=False, dropna=False):
        sub = values.reindex(g.index).dropna()
        if len(sub) < n:
            continue
        out.loc[sub.index] = _qcut(sub, n).values
    return out


def quantize(data, n=5, by=None, method='independent', factor_col='factor'):
    """给因子打分层标签。

    Parameters
    ----------
    data : DataFrame
        索引 ``(date, asset)``。
    n : int
        层数（默认 5 分位）。
    by : Series | DataFrame | str, 可选
        控制变量（如行业、市值）。给定时做双重排序。
    method : {'independent', 'conditional'}
        见模块文档。
    factor_col : str

    Returns
    -------
    DataFrame
        索引同 ``data``，列：

        * ``q`` —— 因子分层标签，**1..n**（1 = 因子值最小）
        * ``q_by`` —— 控制变量分层标签（仅在给了 ``by`` 时存在）
    """
    df = _unwrap(data)
    if factor_col not in df.columns:
        fail('quantile', 'no_factor', f'缺 `{factor_col}` 列')
    if n < 2:
        fail('quantile', 'bad_n', f'层数至少 2，收到 {n}')
    if method not in ('independent', 'conditional'):
        fail('quantile', 'bad_method',
             f"method 只能是 'independent' / 'conditional'，收到 {method!r}")

    dates = pd.Index(df.index.get_level_values('date'))
    f = pd.to_numeric(df[factor_col], errors='coerce')
    out = pd.DataFrame(index=df.index)

    if by is None:
        out['q'] = _quantile_within(df.index, f, [dates], n) + 1
        return out

    b = _as_series(by, df.index, 'by')
    # 控制变量本身按"每期全样本"分位（两种 method 都一样）
    out['q_by'] = _quantile_within(df.index, b, [dates], n) + 1
    if method == 'independent':
        # 因子分位点也取自全样本
        out['q'] = _quantile_within(df.index, f, [dates], n) + 1
    else:
        # 因子分位点取自 **by 组内**
        out['q'] = _quantile_within(df.index, f, [dates, out['q_by']], n) + 1
    return out


def _as_series(v, index, what):
    if isinstance(v, str):
        fail('quantile', 'by_missing', f'by={v!r} 需要传实际的 Series/DataFrame')
    s = v.iloc[:, 0] if isinstance(v, pd.DataFrame) else pd.Series(v)
    s = pd.to_numeric(s, errors='coerce') if np.issubdtype(
        s.dtype, np.number) else s
    s = s.reindex(index)
    if s.isna().all():
        fail('quantile', 'by_align', f'`{what}` 与 data 的索引对不上（全空）')
    return s


# --------------------------------------------------------------------------- #
def quantile_returns(data, quantiles=5, by=None, method='independent',
                     horizons=None, quantile_labels=None):
    """各分位、各持有期的**平均前向收益**。

    Returns
    -------
    DataFrame
        索引 ``(date, q)``（给了 ``by`` 时是 ``(date, q_by, q)``），
        columns = 各持有期的平均收益。
        另含 ``count`` 列（每格样本数 —— **格子里样本太少时结论不可信**）。
    """
    df = _unwrap(data)
    cols = return_cols(df, horizons)
    q = quantiles if isinstance(quantiles, pd.DataFrame) else \
        quantize(df, quantiles, by=by, method=method)
    t = df[cols].join(q, how='inner')

    keys = [k for k in ('q_by', 'q') if k in t.columns]
    g = t.groupby([t.index.get_level_values('date')] + keys)
    res = g[cols].mean()
    res['count'] = g[cols[0]].size()
    res.index.names = ['date'] + keys
    return res.sort_index()


def quantile_stats(data, quantiles=5, by=None, method='independent',
                   horizons=None):
    """分层统计摘要 —— 每层的均值、多空差、单调性。

    Returns
    -------
    DataFrame
        每个持有期一行：``q1_mean`` … ``qN_mean``、``spread``（QN − Q1）、
        ``t_naive``、``monotonicity``（层序与收益的 Spearman）、``mean_count``。
    """
    qr = quantile_returns(data, quantiles, by=by, method=method, horizons=horizons)
    cols = [c for c in qr.columns if c != 'count']
    qs = sorted(qr.index.get_level_values('q').dropna().unique())
    rows = []
    for c in cols:
        m = qr[c].groupby('q').mean()
        x = np.array(qs, dtype=float)
        y = np.array([m.get(qq, np.nan) for qq in qs], dtype=float)
        ok = np.isfinite(y)
        mono = (float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method='spearman'))
                if ok.sum() >= 3 else np.nan)
        # 多空组合：逐日 QN − Q1，再对时间序列取统计
        ls = _long_short_series(qr, c, qs)
        rows.append({
            'horizon': horizon_of(c),
            **{f'q{int(qq)}_mean': float(m.get(qq, np.nan)) for qq in qs},
            'spread': float(ls.mean()) if len(ls) else np.nan,
            't_naive': (float(ls.mean() / ls.std(ddof=1) * np.sqrt(len(ls)))
                        if len(ls) > 2 and ls.std(ddof=1) > 0 else np.nan),
            'monotonicity': mono,
            'mean_count': float(qr['count'].groupby('q').mean().mean()),
            'n_periods': len(ls),
        })
    return pd.DataFrame(rows).set_index('horizon')


def _long_short_series(qr, col, qs):
    """逐期 QN − Q1 序列。"""
    if len(qs) < 2:
        return pd.Series(dtype=float)
    top, bot = qs[-1], qs[0]
    try:
        hi = qr[col].xs(top, level='q')
        lo = qr[col].xs(bot, level='q')
    except KeyError:
        return pd.Series(dtype=float)
    hi = hi.groupby(level='date').mean() if hi.index.nlevels > 1 else hi
    lo = lo.groupby(level='date').mean() if lo.index.nlevels > 1 else lo
    return (hi - lo).dropna()


def _unwrap(obj):
    """接受 DataFrame / Series / 任何带 ``.data`` / ``.df`` 的结果对象。"""
    for attr in ('data', 'df'):
        v = getattr(obj, attr, None)
        if isinstance(v, (pd.DataFrame, pd.Series)):
            return v
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj
    fail('quantile', 'bad_input',
         f'需要 DataFrame / Series / 清洗结果，收到 {type(obj).__name__}')


# --------------------------------------------------------------------------- #
def double_sort(data, by, n=5, method='conditional', horizons=None):
    """双重排序 —— 返回 ``(date, q_by, q)`` 的平均收益立方。

    **核心用法（可证伪预测）**：先按规模分组，再看**同一规模组内**
    因子的单调性是否还在。若消失，说明因子效应是规模混淆的产物。
    """
    return quantile_returns(data, quantiles=n, by=by, method=method,
                            horizons=horizons)


def monotonicity_test(data, quantiles=5, by=None, method='conditional',
                      horizons=None):
    """分层单调性检验。

    Returns
    -------
    DataFrame
        columns: ``horizon`` / ``spearman`` / ``n_layers`` / ``is_monotonic``

    单调性失败往往意味着因子定义有问题 —— 例如只有两端有效、
    中间层乱序（本次 ROE 就是"两端清晰、中间平坦"）。
    """
    st = quantile_stats(data, quantiles, by=by, method=method, horizons=horizons)
    out = st[['monotonicity', 'n_periods']].copy()
    out.columns = ['spearman', 'n_periods']
    out['is_monotonic'] = out['spearman'].abs() >= 0.9
    return out
