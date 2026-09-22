"""IC 系列 —— 信息系数、ICIR、衰减、秩自相关。

**本模块只算，不下结论。** 显著性判定（Newey-West、多重检验、有效样本量）
在 ``inference/`` 层 —— 那条分层是设计原则 1：**推断与计算分离**。

为什么秩自相关要自己写
----------------------
alphalens 整条链路都建立在 ``index.levels[0].freq`` 上。真实 A 股月频面板
（每月最后一个交易日）推断出的 ``freq`` 是 ``None``，于是：

* ``get_clean_factor_and_forward_returns`` **在清洗阶段就抛 ValueError**
  （``utils.py:358`` 硬写 ``df.index.levels[0].freq = freq``）——
  月频分析**根本跑不起来**，这是 D9 的真相；
* 就算绕过清洗，``factor_rank_autocorrelation`` / ``quantile_turnover``
  也会因 ``asfreq`` 失败而全空。

本模块**完全不依赖 ``freq``**。实测同一份 ROE 月度面板（91 期 × 5,004 只）：
秩自相关 **0.9227**（79 期全有值）、月换手 **20.03%**（alphalens 口径）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail

__all__ = [
    'information_coefficient', 'ic_summary', 'ic_decay',
    'rank_autocorrelation', 'quantile_turnover', 'return_cols', 'horizon_of',
]


# --------------------------------------------------------------------------- #
def return_cols(data, horizons=None):
    """从面板里挑出前向收益列。"""
    cols = [c for c in data.columns if c.startswith('forward_return_')]
    if horizons is not None:
        want = {f'forward_return_{h}' for h in horizons}
        cols = [c for c in cols if c in want]
    if not cols:
        fail('ic', 'no_returns',
             '数据里没有 `forward_return_*` 列；'
             '先跑 forward_returns() 与 clean()。')
    return sorted(cols, key=horizon_of)


def horizon_of(col):
    """``'forward_return_21'`` → ``21``。"""
    try:
        return int(str(col).rsplit('_', 1)[1])
    except (ValueError, IndexError):
        return 0


# --------------------------------------------------------------------------- #
def information_coefficient(data, horizons=None, method='spearman'):
    """逐日横截面 IC。

    Parameters
    ----------
    data : DataFrame
        索引 ``(date, asset)``，含 ``factor`` 与 ``forward_return_*``。
        通常是 :class:`~alphalens_cna.engine.clean.CleanResult` 的 ``data``
        （也可直接传 ``CleanResult``）。
    horizons : list[int], 可选
    method : {'spearman', 'pearson'}
        ``'spearman'``（默认，即 RankIC）对极值稳健，是业界惯例。

    Returns
    -------
    DataFrame
        index = 日期，columns = 各持有期（值即 IC）。

    Notes
    -----
    **算的是因子值与前向收益的横截面相关** —— 前向收益已由
    ``engine/returns.py`` 按"次日开盘成交"口径算好，
    所以 IC 的经济含义是「T 日收盘的因子值预测 T+1 开盘起的收益」。
    """
    df = _unwrap(data)
    cols = return_cols(df, horizons)
    if 'factor' not in df.columns:
        fail('ic', 'no_factor', f'缺 `factor` 列；实际列：{list(df.columns)[:12]}')

    g = df.groupby(level='date')
    out = {}
    for c in cols:
        out[horizon_of(c)] = g.apply(
            lambda x, c=c: _corr(x['factor'].values, x[c].values, method),
            include_groups=False)
    res = pd.DataFrame(out).sort_index()
    res.columns.name = 'horizon'
    return res


def _corr(a, b, method):
    """一对数组的相关系数；样本不足或方差为 0 时返回 NaN。"""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan              # 常数因子没有 IC 可言（不是 0）
    s = pd.Series(a).corr(pd.Series(b), method=method)
    return float(s) if np.isfinite(s) else np.nan


# --------------------------------------------------------------------------- #
def ic_summary(ic, horizons=None):
    """IC 的描述统计。

    Returns
    -------
    DataFrame
        columns: ``n`` / ``mean`` / ``std`` / ``icir`` / ``t_naive`` /
        ``positive_rate`` / ``abs_mean``

    ⚠️ ``t_naive`` 是**朴素 t 值**（``icir × sqrt(n)``）——
    重叠观测下会虚高。实测 20 日持有期上虚高 **3–4 倍**
    （动量因子 t 从 −21.93 修到 −6.76）。
    **要下结论请走 ``inference/`` 的 Newey-West 修正。**
    """
    ic = _unwrap(ic)
    if horizons is not None:
        ic = ic[[h for h in ic.columns if h in set(horizons)]]
    rows = []
    for h in ic.columns:
        s = ic[h].dropna()
        n = len(s)
        mean = float(s.mean()) if n else np.nan
        std = float(s.std(ddof=1)) if n > 1 else np.nan
        icir = mean / std if std and np.isfinite(std) and std > 0 else np.nan
        rows.append({
            'horizon': h, 'n': n, 'mean': mean, 'std': std,
            'icir': icir,
            't_naive': icir * np.sqrt(n) if np.isfinite(icir) else np.nan,
            'positive_rate': float((s > 0).mean()) if n else np.nan,
            'abs_mean': float(s.abs().mean()) if n else np.nan,
        })
    return pd.DataFrame(rows).set_index('horizon')


# --------------------------------------------------------------------------- #
def ic_decay(data, horizons=None, method='spearman'):
    """IC 衰减 —— 各持有期的 IC 一览。

    用途：选调仓频率。若 |IC| 随持有期**单调下降**，说明信号是短时效的；
    若反而上升（实测本库的财务因子就是这样），说明它不是"衰减型"因子。
    """
    ic = information_coefficient(data, horizons, method)
    return ic_summary(ic)


# --------------------------------------------------------------------------- #
def rank_autocorrelation(data, factor_col='factor', lag=1, min_n=3):
    """因子**秩**的时序自相关 —— 因子有多"黏"。

    ``> 0.9`` 说明排名一天/一期基本不动，换手低。

    ⚠️ alphalens 在非日频索引上返回全 NaN（D9）；本实现不依赖 ``freq``。

    Returns
    -------
    Series
        index = 日期（每个可用期），值 = 与上一期的秩相关。
    """
    df = _unwrap(data)
    if factor_col not in df.columns:
        fail('ic', 'no_factor', f'缺 `{factor_col}` 列')
    # 因子是**逐期**的：同一期（date）内样本构成一个横截面
    by_date = df.groupby(level='date')[factor_col]
    ranks = by_date.rank()
    wide = ranks.unstack('asset')
    if wide.shape[1] < min_n:
        return pd.Series(dtype=float)
    out = {}
    prev = None
    for d, row in wide.iterrows():
        if prev is not None:
            m = row.notna() & prev.notna()
            if m.sum() >= min_n and row[m].nunique() > 1 and prev[m].nunique() > 1:
                out[d] = float(row[m].corr(prev[m], method='spearman'))
        prev = row
    s = pd.Series(out, dtype=float)
    s.name = 'rank_autocorr'
    return s


def quantile_turnover(quantiles, n_quantiles=None, lag=1, method='alphalens'):
    """分位组合的**换手率** —— 相邻期成员变动的比例。

    Parameters
    ----------
    quantiles : Series
        索引 ``(date, asset)``，值为分位标签。
    lag : int
        比较间隔（默认 1 = 相邻期）。
    method : {'alphalens', 'symmetric'}
        口径。**进出对称时两者数值相同**；名单整体重构或规模变化时不同：

        ``'alphalens'``（默认）
            文献与 alphalens 的标准口径 —— **单边**：

            ``tau_t = |Q_t \\ Q_{t-1}| / |Q_t|``

            即"本期名单里有百分之多少是本期新买入的"。分母是**本期**名单，
            分子只数**新进**、不数**退出**。名单完全换血 → 1.0。
            与 ``alphalens.performance.quantile_turnover`` **逐点一致**
            （由 :func:`alphalens_cna.compat.check_parity` 守住）。
        ``'symmetric'``
            对称口径：``|Q_t △ Q_{t-1}| / |Q_t ∪ Q_{t-1}| / 2``。
            进出都数、除以并集，完全换血 → 0.5。单边规模剧变时更稳健，
            但与 alphalens 及多数文献**不可直接比数**。

    Returns
    -------
    DataFrame
        index = 日期（从第 ``lag`` 期起，共 ``n - lag`` 行），
        columns = 各分位；值为该分位换手比例。
        某期该分位名单为空 → ``NaN``（与 alphalens 一致：0/0 无定义）。

    Notes
    -----
    ⚠️ alphalens 的 ``quantile_turnover`` 在非日频索引上返回全 NaN（D9），
    本函数在月频 / 周频下正常返回值。

    Examples
    --------
    >>> to = quantile_turnover(q['q'])                            # alphalens 口径
    >>> to_sym = quantile_turnover(q['q'], method='symmetric')    # 对称口径
    """
    if method not in ('alphalens', 'symmetric'):
        fail('ic', 'bad_method',
             f"method 只能是 'alphalens' 或 'symmetric'，收到 {method!r}")
    q = _unwrap(quantiles)
    if isinstance(q, pd.DataFrame):
        q = q.iloc[:, 0]
    wide = q.unstack('asset')
    labels = sorted(wide.stack().dropna().unique()) if wide.size else []
    if not labels:
        return pd.DataFrame(dtype=float)
    dates = wide.index
    rows = {}
    for i in range(lag, len(dates)):
        prev, cur = wide.iloc[i - lag], wide.iloc[i]
        rec = {}
        for lab in labels:
            a = set(prev.index[prev == lab])
            b = set(cur.index[cur == lab])
            if method == 'alphalens':
                # 本期名单为空 → 0/0，alphalens 亦为 NaN；否则只数新进
                rec[lab] = np.nan if not b else len(b - a) / len(b)
            else:
                if not a and not b:
                    continue
                rec[lab] = len(a ^ b) / max(len(a | b), 1) / 2   # 对称口径
        rows[dates[i]] = rec
    return pd.DataFrame(rows).T.sort_index()


# --------------------------------------------------------------------------- #
def _unwrap(obj):
    """接受 DataFrame / Series / 任何带 ``.data`` / ``.df`` 的结果对象。"""
    for attr in ('data', 'df'):
        v = getattr(obj, attr, None)
        if isinstance(v, (pd.DataFrame, pd.Series)):
            return v
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj
    fail('ic', 'bad_input',
         f'需要 DataFrame / Series / 清洗结果，收到 {type(obj).__name__}')
