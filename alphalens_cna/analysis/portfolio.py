"""组合层 —— 多空收益、累计曲线、换手。

与 ``quantile.py`` 的分工：
* ``quantile.py`` 回答"各层分别赚多少"
* 本模块回答"**多空组合作为一条策略**表现如何"——收益序列、累计、回撤、换手
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail
from .ic import quantile_turnover
from .quantile import quantize, quantile_returns, _unwrap

__all__ = ['factor_returns', 'cumulative_returns', 'portfolio_summary',
           'turnover_summary', 'weighted_returns']


# --------------------------------------------------------------------------- #
def factor_returns(data, quantiles=5, horizons=None, by=None, method='independent',
                   long_short=True):
    """多空（或纯多头）组合的**逐期收益序列**。

    Parameters
    ----------
    data : DataFrame | CleanResult
    quantiles : int | DataFrame
        层数，或直接给 ``quantize()`` 的结果。
    long_short : bool
        ``True``（默认）→ 做多最高层、做空最低层，返回 (QN − Q1)。
        ``False`` → 只返回最高层（A 股做空不易，纯多头更贴近实际）。

    Returns
    -------
    DataFrame
        index = 日期，columns = 各持有期。**未扣成本** ——
        成本用 :func:`turnover_summary` 与 ``inference`` 层的净收益另行处理。
    """
    df = _unwrap(data)
    q = quantiles if isinstance(quantiles, pd.DataFrame) else \
        quantize(df, quantiles, by=by, method=method)
    qr = quantile_returns(df, quantiles=q, horizons=horizons)
    cols = [c for c in qr.columns if c != 'count']
    qs = sorted(qr.index.get_level_values('q').dropna().unique())
    if len(qs) < 2 and long_short:
        fail('portfolio', 'too_few_quantiles',
             f'多空至少要 2 层，实际 {len(qs)} 层')

    out = {}
    for c in cols:
        try:
            top = qr[c].xs(qs[-1], level='q')
        except KeyError:
            continue
        top = top.groupby(level='date').mean() if top.index.nlevels > 1 else top
        if not long_short:
            out[c] = top
            continue
        bot = qr[c].xs(qs[0], level='q')
        bot = bot.groupby(level='date').mean() if bot.index.nlevels > 1 else bot
        out[c] = (top - bot)
    res = pd.DataFrame(out).sort_index()
    res.columns = [int(c.rsplit('_', 1)[1]) for c in res.columns]
    res.columns.name = 'horizon'
    return res


def cumulative_returns(returns):
    """累计净值 —— ``(1+r).cumprod()``。"""
    r = _unwrap(returns)
    if isinstance(r, pd.DataFrame):
        return (1 + r.fillna(0)).cumprod()
    return (1 + r.fillna(0)).cumprod()


def portfolio_summary(returns, periods_per_year=None):
    """组合绩效摘要。

    Parameters
    ----------
    periods_per_year : int, 可选
        年化用。**不给就按观测频率推断**（用相邻日期中位数间隔），
        推断不出就留 NaN —— 不硬编 252，那对月频调仓是错的。
    """
    r = _unwrap(returns)
    if isinstance(r, pd.DataFrame):
        if r.shape[1] == 1:
            r = r.iloc[:, 0]
        else:
            return pd.DataFrame({c: portfolio_summary(r[c], periods_per_year)
                                 for c in r.columns}).T
    r = r.dropna()
    if len(r) < 2:
        return pd.Series(dtype=float)

    if periods_per_year is None:
        periods_per_year = _infer_periods(r.index)
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    ann = np.nan
    if periods_per_year:
        ann = float(eq.iloc[-1] ** (periods_per_year / len(r)) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if periods_per_year else np.nan
    sharpe = (ann / vol) if (periods_per_year and vol and vol > 0) else np.nan
    return pd.Series({
        'n_periods': len(r),
        'total_return': float(eq.iloc[-1] - 1),
        'annual_return': ann,
        'annual_vol': vol,
        'sharpe': sharpe,
        'max_drawdown': float(dd.min()),
        'win_rate': float((r > 0).mean()),
        'mean_period': float(r.mean()),
        'periods_per_year': periods_per_year or np.nan,
    })


def _infer_periods(index):
    """从日期间隔推断每年多少期。推断不出返回 None（**不硬编 252**）。"""
    idx = pd.DatetimeIndex(index)
    if len(idx) < 3:
        return None
    gaps = np.diff(idx.values) / np.timedelta64(1, 'D')
    med = float(np.median(gaps))
    if not np.isfinite(med) or med <= 0:
        return None
    # 自然日间隔 → 年频次；交易日间隔按 252 折算
    if med <= 4:
        return int(round(252 / max(med, 1))) if med > 1 else 252
    return int(round(365 / med)) if med else None


# --------------------------------------------------------------------------- #
def turnover_summary(quantiles, lag=1):
    """各分位换手率摘要 + 成本估算。

    Returns
    -------
    DataFrame
        index = 分位，columns: ``turnover``（平均单边换手）、
        ``cost_per_period_15bp``（按单边 15bp 估的成本）。

    > 实测（ROE 月度面板，91 期 × 5,004 只）：
    > **alphalens 口径 20.03%/月**（Q1 10.4% … Q4 26.7%），
    > symmetric 口径 14.06%/月。
    > 年化成本（12 次调仓、单边 15bp、买卖各一次）：**≈ 0.72%/年**。
    > 详见 ``outputs/换手口径与D9核实.md``。
    """
    to = quantile_turnover(quantiles, lag=lag)
    if not len(to):
        return pd.DataFrame()
    out = pd.DataFrame({
        'turnover': to.mean(),
        'cost_per_period_15bp': to.mean() * 2 * 0.0015,   # 买卖双边
        'n_periods': to.notna().sum(),
    })
    out.index.name = 'q'
    return out


def weighted_returns(data, weights, horizons=None):
    """按给定权重合成组合收益（**截面加权**）。

    Parameters
    ----------
    weights : Series
        索引 ``(date, asset)``，权重。**每期会自动归一化**。
    """
    df = _unwrap(data)
    from .ic import return_cols
    cols = return_cols(df, horizons)
    w = pd.Series(weights).reindex(df.index).astype(float)
    if w.isna().all():
        fail('portfolio', 'weight_align', '权重与数据索引对不上（全空）')
    w = w.fillna(0.0)
    g = w.groupby(level='date').transform('sum')
    w = (w / g.replace(0, np.nan)).fillna(0.0)
    contrib = df[cols].mul(w, axis=0)
    res = contrib.groupby(level='date').sum()
    res.columns = [int(c.rsplit('_', 1)[1]) for c in res.columns]
    res.columns.name = 'horizon'
    return res
