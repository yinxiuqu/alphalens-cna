"""截面回归与 Fama-MacBeth —— 用户点名要的"石川书里的截面回归"。

和 IC / 分层的关系
------------------
IC 和分层回答"因子值高的股票，后面收益是不是也高"；
截面回归回答的是**"控制住别的变量之后，还高不高"**，
以及**"这个因子的风险溢价是多少"**（Fama-MacBeth 的 λ）。

Fama-MacBeth 两步法
-------------------
1. **每期做一次截面回归** ``r_it = λ_t' x_it + ε_it``（对 i 回归，得到 λ_t）
2. **对 λ_t 的时间序列求均值与 t 值**

第二步的 t 值是全网最容易被高估的地方：λ_t 序列有自相关和重叠观测，
朴素 t 会虚高。所以本模块**不自己算 t**，直接调
:func:`alphalens_cna.inference.newey_west.nw_tstat` —— 与库里其它地方同一把尺子。

计算与推断分离
--------------
本模块只算系数（``coef`` / 均值 / 标准差），
**显著性判定统一走 ``inference/``**。这也是为什么 ``fama_macbeth`` 的
``summary`` 里同时给出 ``t_naive`` 与 ``t_nw``：让人看见朴素 t 虚高了几倍。

用法
----
>>> res = fama_macbeth(y=forward_return, X=exposures)   # X 列 = 因子/暴露
>>> print(res)
>>> res.summary        # 每个回归元的均值、朴素 t、NW t、有效样本
>>> res.coef           # 每期系数（可画时序图）
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contract.errors import fail
from ..inference.newey_west import nw_tstat

__all__ = [
    'cross_sectional_regression', 'fama_macbeth', 'FMResult',
    'shanken_inflation',
]


def _check_index(*objs):
    """y 与 X 必须是同一个 ``MultiIndex(date, asset)``。"""
    idx = None
    for o in objs:
        if not isinstance(o.index, pd.MultiIndex):
            fail('regression', 'index',
                 f'需要 MultiIndex(date, asset)，收到 {type(o.index).__name__}。'
                 f'提示：df.set_index(["date","asset"])')
        if list(o.index.names[:2]) != ['date', 'asset']:
            fail('regression', 'index_names',
                 f'索引层级名必须是 ["date","asset"]，收到 {o.index.names}')
        if idx is None:
            idx = o.index
        elif not o.index.equals(idx):
            fail('regression', 'index_mismatch',
                 'y 与 X 的索引必须完全一致（长度不同或顺序不同都不行）')
    return idx


def cross_sectional_regression(y, X, *, add_constant=True, min_obs=10,
                               weights=None):
    """**逐期**截面 OLS —— 每个日期单独回归，返回该期的系数。

    Parameters
    ----------
    y : Series
        因变量（通常是前向收益），索引 ``(date, asset)``。
    X : DataFrame
        自变量（因子值 / 暴露），索引同 ``y``。
    add_constant : bool
        是否加截距（默认 ``True``）。
    min_obs : int
        当期有效样本少于这个数就**不回归**（记 NaN），
        不硬凑一个系数出来。
    weights : Series, optional
        加权最小二乘的权重（如市值平方根），索引同 ``y``。

    Returns
    -------
    DataFrame
        index = 日期，columns = ``const`` + ``X`` 的列名；值为该期系数。
        样本不足的日期**整行为 NaN**（不是 0）。

    Notes
    -----
    用 ``numpy.linalg.lstsq`` 解，**不做任何统计推断** ——
    t 值请走 :func:`fama_macbeth` 或 ``inference/``。
    """
    idx = _check_index(y, X)
    if isinstance(X, pd.Series):
        X = X.to_frame()
    cols = list(X.columns)
    dates = idx.get_level_values('date')
    yv_all = np.asarray(y, dtype=float)
    Xv_all = np.asarray(X, dtype=float)
    wv_all = None if weights is None else np.asarray(weights, dtype=float)

    rows, out_index = [], []
    # groupby 保序：用 unique 保持时间顺序
    for d, pos in pd.Series(np.arange(len(dates)), index=dates).groupby(
            level=0, sort=True):
        p = np.asarray(pos)
        yy, XX = yv_all[p], Xv_all[p]
        ok = np.isfinite(yy) & np.isfinite(XX).all(axis=1)
        if wv_all is not None:
            ok &= np.isfinite(wv_all[p])
        n = int(ok.sum())
        out_index.append(d)
        if n < max(min_obs, len(cols) + (1 if add_constant else 0)):
            rows.append([np.nan] * (len(cols) + (1 if add_constant else 0)))
            continue
        A, b = XX[ok], yy[ok]
        if add_constant:
            A = np.column_stack([np.ones(len(A)), A])
        if wv_all is not None:
            w = np.sqrt(np.maximum(wv_all[p][ok], 0))
            A, b = A * w[:, None], b * w
        beta, *_ = np.linalg.lstsq(A, b, rcond=None)
        rows.append(list(beta))

    names = (['const'] if add_constant else []) + [str(c) for c in cols]
    return pd.DataFrame(rows, index=pd.DatetimeIndex(out_index), columns=names)


@dataclass
class FMResult:
    """Fama-MacBeth 结果。

    Attributes
    ----------
    coef : DataFrame
        每期截面系数（index=日期，columns=回归元）。
    summary : DataFrame
        每个回归元一行：``mean`` / ``std`` / ``t_naive`` / ``t_nw`` /
        ``n_periods`` / ``n_eff`` / ``lags`` / ``vif``。
    """

    coef: pd.DataFrame = None
    summary: pd.DataFrame = None
    extra: dict = field(default_factory=dict)

    def __str__(self):
        L = ['Fama-MacBeth 截面回归']
        if self.summary is None or not len(self.summary):
            return L[0] + '（无有效截面）'
        n_ok = 0 if self.summary['n_periods'].isna().all() \
            else int(self.summary['n_periods'].max())
        if n_ok == 0:
            L.append('  （无有效截面 —— 每期样本数都不够，'
                     '一个系数都没估出来。检查 min_obs 或数据覆盖）')
            return '\n'.join(L)
        L.append(f'  有效截面 {n_ok} 期')
        L.append(f'  {"回归元":<12}{"均值":>12}{"朴素t":>10}{"NW t":>10}'
                 f'{"虚高":>8}{"有效n":>8}')
        for name, r in self.summary.iterrows():
            infl = (abs(r['t_naive'] / r['t_nw'])
                    if np.isfinite(r['t_nw']) and r['t_nw'] != 0 else np.nan)
            L.append(f'  {name:<12}{r["mean"]:>12.5f}{r["t_naive"]:>10.2f}'
                     f'{r["t_nw"]:>10.2f}{infl:>8.2f}{r["n_eff"]:>8.1f}')
        L.append('  ⚠️ 只用 `t_nw` 下结论。`t_naive` 在重叠观测下会虚高，'
                 '`虚高` 列就是倍数。')
        return '\n'.join(L)

    def to_frame(self):
        return self.summary


def fama_macbeth(y, X, *, add_constant=True, min_obs=10, lags=None,
                 horizon=1, weights=None):
    """Fama-MacBeth 两步法。

    Parameters
    ----------
    y, X : Series / DataFrame
        索引 ``(date, asset)``。``X`` 的列就是因子/暴露。
    min_obs : int
        单期最少样本数，不足则该期不回归（见 :func:`cross_sectional_regression`）。
    lags : int, optional
        Newey-West 滞后阶数；不给按 :func:`auto_lags` 自动定。
    horizon : int
        前向收益的持有期 —— **决定自动滞后阶数**（重叠观测的长度）。

    Returns
    -------
    FMResult

    Notes
    -----
    第二步的 t 值有两个：``t_naive``（直接对 λ_t 求 t）和 ``t_nw``
    （Newey-West 修正）。**两者差多少，就是重叠观测把显著性吹大了多少。**
    """
    if horizon < 1:
        fail('regression', 'horizon', f'horizon 必须 >= 1，收到 {horizon}')
    coef = cross_sectional_regression(y, X, add_constant=add_constant,
                                      min_obs=min_obs, weights=weights)
    rows = {}
    for c in coef.columns:
        s = coef[c].dropna()
        if not len(s):
            rows[c] = {'mean': np.nan, 'std': np.nan, 't_naive': np.nan,
                       't_nw': np.nan, 'n_periods': 0, 'n_eff': np.nan,
                       'lags': 0, 'vif': np.nan}
            continue
        st = nw_tstat(s.to_numpy(), lags=lags, horizon=horizon)
        rows[c] = {
            'mean': st['mean'], 'std': float(s.std(ddof=1)) if len(s) > 1
            else np.nan,
            't_naive': st['t_naive'], 't_nw': st['t_nw'],
            'n_periods': int(len(s)), 'n_eff': st['n_eff'],
            'lags': st['lags'], 'vif': st['vif'],
        }
    summary = pd.DataFrame(rows).T
    summary.index.name = 'regressor'
    return FMResult(coef=coef, summary=summary)


def shanken_inflation(mean_coef, factor_cov):
    """Shanken (1992) 误差变量修正的**膨胀因子** ``1 + λ' Σ_f^{-1} λ``。

    Parameters
    ----------
    mean_coef : array-like
        第二步得到的平均风险溢价 λ̂（**不要包含截距**）。
    factor_cov : DataFrame | ndarray
        因子的协方差矩阵 Σ_f（同期、同顺序）。

    Returns
    -------
    float
        把 FM 标准误乘上 ``sqrt(该值)`` 即得修正后的标准误。

    Notes
    -----
    ⚠️ 这是**一阶近似**，成立前提是因子近似平稳、T 远大于 1。
    它修正的是"因子暴露本身是估计出来的"这一层误差；
    如果 ``λ̂ ≈ 0``，膨胀因子 ≈ 1（无修正）。

    本库**默认不做这个修正** —— 想用就显式调，并且知道自己在做什么。
    """
    lam = np.asarray(mean_coef, dtype=float).ravel()
    S = np.asarray(factor_cov, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        fail('regression', 'factor_cov', 'Σ_f 必须是方阵')
    if S.shape[0] != len(lam):
        fail('regression', 'factor_cov_shape',
             f'Σ_f 是 {S.shape}，但 λ 有 {len(lam)} 个 —— 维数必须对上')
    if not np.isfinite(lam).all() or not np.isfinite(S).all():
        return np.nan
    try:
        inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return np.nan
    return float(1.0 + lam @ inv @ lam)
