"""Verdict —— 结论收口。

.. code-block:: text

    alphalens 告诉你「因子 IC 是 0.004」
    alphalens-cna 告诉你「这个 0.004 可不可信」

:class:`Verdict` 把散在各层的证据收成**一份可判定的结论**：
显著吗（校正后）、在多少可交易样本上、有效样本几个、扣成本还剩多少、
哪些地方要打问号。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ..contract.errors import fail
import pandas as pd

from . import multiplicity as mult
from .estimate import Estimate, estimates_to_frame

__all__ = ['Verdict', 'assess']


@dataclass
class Verdict:
    """一份结论。

    Attributes
    ----------
    significant : bool
        **校正后**是否显著。这是唯一该拿去做决策的字段。
    p_adjusted : float
        校正后 p 值。
    n_trials : int
        校正基数 —— 你一共测过多少个假设。
    threshold_used : float
        对应的 Bonferroni |t| 阈值（说明"门槛有多高"）。
    effective_n : float
        有效样本量（启发式）。
    tradable_ratio : float
        可交易样本占比 = 1 − 剔除率。**回答"结论建立在多少样本上"。**
    net_return : float
        扣成本后的多空收益（给了才填）。
    cost : float
        估算的交易成本。
    stability : float
        子样本稳定性 0–1（M2 才有，M0 为 NaN）。
    estimates : list[Estimate]
        各持有期/各口径的详细估计。
    warnings : list[str]
    ledger : DropLedger | None
    """

    significant: bool = False
    p_adjusted: float = np.nan
    p_raw: float = np.nan
    n_trials: int = 0
    method: str = 'none'
    threshold_used: float = np.nan
    effective_n: float = np.nan
    nominal_n: int = 0
    tradable_ratio: float = np.nan
    n_input: int = 0
    n_used: int = 0
    net_return: float = np.nan
    gross_return: float = np.nan
    cost: float = np.nan
    stability: float = np.nan
    crash: object = None
        # ``crash_spread`` 的输出（尾部风险：QN−Q1 的崩盘命中率差 + NW t）。
        # **与 `significant` 回答不同的问题**：均值/IC 说不显著，
        # 不代表两组的下行风险没有差别 —— 实测 ROE 就是这个情况。
    estimates: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    ledger: object = None

    def to_frame(self):
        """一行摘要，便于汇总多个因子。"""
        return pd.DataFrame([{
            'significant': self.significant,
            'p_raw': self.p_raw,
            'p_adjusted': self.p_adjusted,
            'n_trials': self.n_trials,
            'method': self.method,
            't_threshold': self.threshold_used,
            'nominal_n': self.nominal_n,
            'effective_n': self.effective_n,
            'tradable_ratio': self.tradable_ratio,
            'gross_return': self.gross_return,
            'cost': self.cost,
            'net_return': self.net_return,
            'stability': self.stability,
            'tail': self._tail_line(),
            'warnings': ' / '.join(self.warnings),
        }])

    def _tail_line(self):
        """尾部风险摘要（**列出全部持有期，不挑最好的那个**）。"""
        if self.crash is None or not len(self.crash):
            return ''
        parts = []
        for h, r in self.crash.iterrows():
            t = r.get('t_hit', np.nan)
            ts = f't={t:+.2f}' if np.isfinite(t) else 't=—'
            parts.append(f'h={int(h)}: Q1 {r["hit_lo"]:.1%} vs QN {r["hit_hi"]:.1%} ({ts})')
        return '；'.join(parts)

    def __str__(self):
        flag = '✅ 显著' if self.significant else '❌ 不显著'
        _tail = self._tail_line()
        lines = [f'【结论】{flag}（{self.method.upper()} 校正）',
                 f'  p 值      {self.p_raw:.4f}（原始） → {self.p_adjusted:.4f}（校正后）'
                 if np.isfinite(self.p_raw) else '  p 值      —',
                 (f'  尾部风险  {_tail}' if _tail else
                  '  尾部风险  —（未提供 crash_spread 结果）'),
                 f'  校正基数  n_trials = {self.n_trials}'
                 + (f'，对应 |t| 阈值 {self.threshold_used:.3f}'
                    if np.isfinite(self.threshold_used) else ''),
                 f'  样本      {self.n_used:,} / {self.n_input:,} 可用'
                 + (f'（可交易占比 {self.tradable_ratio:.1%}）'
                    if np.isfinite(self.tradable_ratio) else ''),
                 f'  有效样本  {self.effective_n:.1f}（名义 {self.nominal_n}）'
                 if np.isfinite(self.effective_n) else f'  有效样本  —（名义 {self.nominal_n}）']
        if np.isfinite(self.net_return):
            lines.append(f'  收益      毛 {self.gross_return:+.4%} → '
                         f'净 {self.net_return:+.4%}（成本 {self.cost:.4%}）')
        if np.isfinite(self.stability):
            lines.append(f'  稳定性    {self.stability:.0%} 子样本成立')
        if self.warnings:
            lines.append('  注意')
            lines += [f'    ⚠️ {w}' for w in self.warnings]
        return '\n'.join(lines)


# --------------------------------------------------------------------------- #
def assess(estimates=None, *, ic=None, returns=None, turnover=None,
           ledger=None, n_trials=None, method='bhy', horizon=None,
           cost_bps=15.0, stability=None, crash=None):
    """把各层证据收成一份 :class:`Verdict`。

    Parameters
    ----------
    estimates : Estimate | list[Estimate], 可选
        已有的估计量。不给则从 ``ic`` / ``returns`` 现算。
    ic : DataFrame, 可选
        ``information_coefficient()`` 的输出（逐期 IC）。
    returns : DataFrame | Series, 可选
        多空收益序列（逐期）。用于算毛/净收益。
    turnover : Series | DataFrame, 可选
        逐期换手率。给了才能算成本。
    ledger : DropLedger, 可选
        清洗账 —— 填 ``tradable_ratio``。
    n_trials : int, 可选
        **一共测过多少个假设。** 不给就**不做多重检验校正**，
        并在 warnings 里明确写出"未校正" —— 不假装做了。
    horizon : int, 可选
        选哪个持有期作为主结论（默认取第一个）。
    cost_bps : float
        单边成本（基点，默认 15bp = 0.15%）。成本 = 换手 × 2 × cost_bps。

    Returns
    -------
    Verdict
    """
    # ★ 入口前置判断：IC 全空或没有任何有限值 —— 必须在下游（多重检验）
    #   之前拦住，否则用户拿到的是 "p 值里有 NaN/Inf"，那信息对他毫无用处。
    if ic is not None:
        _ic = pd.DataFrame(getattr(ic, 'data', ic))
        if not len(_ic.columns):
            fail('verdict', 'empty_ic',
                 'IC 面板是空的（没有任何持有期列）—— 没有可判断的东西。\n'
                 '  排查：看 clean() 的台账；若有效观测为 0，多半是**持有期超过样本跨度**'
                 '（如月频价格却要 63 个交易日的前向收益），缩短 horizons 即可。')
        _v = _ic.to_numpy(dtype=float)
        if not np.isfinite(_v).any():
            fail('verdict', 'all_nan_ic',
                 f'IC 全是 NaN（{_ic.shape[0]} 期 × {_ic.shape[1]} 个持有期）—— 样本不足以判断。\n'
                 '  常见原因：① 期数太少（每个截面算不出相关）；'
                 '② 持有期超过样本跨度；③ 因子在截面上是常数。\n'
                 '  排查：看 clean() 的台账，或缩短 horizons。')
    ests = _collect(estimates, ic, n_trials, method, horizon)
    v = Verdict(estimates=ests, n_trials=int(n_trials or 0),
                method=method if n_trials else 'none', ledger=ledger,
                crash=crash)

    # ── 主结论取第一个估计（通常是主持有期） ──
    if ests:
        e = ests[0]
        v.significant = e.significant
        v.p_adjusted = e.p_adj
        v.p_raw = e.p_raw
        v.nominal_n = e.n
        v.effective_n = e.n_eff
        v.n_trials = e.n_trials or v.n_trials
        v.method = e.method
        v.warnings.extend(e.warnings)
    if v.n_trials:
        v.threshold_used = mult.t_threshold(v.n_trials)
    else:
        v.warnings.append(
            '未做多重假设检验校正 —— 结论只在"本次只看这一个因子"时成立。'
            '要下可发表的结论，必须提供 n_trials（一共测过多少个）')

    # ── 样本账 ──
    if ledger is not None:
        v.n_input = int(getattr(ledger, 'n_input', 0) or 0)
        v.n_used = int(getattr(ledger, 'n_output', 0) or 0)
        if v.n_input:
            v.tradable_ratio = v.n_used / v.n_input
            drop = getattr(ledger, 'total_dropped', 0)
            if drop / v.n_input > 0.5:
                v.warnings.append(
                    f'剔除率 {drop / v.n_input:.0%} 偏高 —— '
                    f'结论建立在一半以下的样本上')

    # ── 收益与成本 ──
    if returns is not None:
        r = _series(returns)
        r = r.dropna()
        if len(r) >= 2:
            v.gross_return = float((1 + r).prod() - 1)
            c = _cost(r, turnover, cost_bps)
            v.cost = float(c.sum())          # 累计成本（报告里显示这个）
            v.net_return = float((1 + r - c).prod() - 1)
            if v.cost and abs(v.cost) > abs(v.gross_return) * 0.3:
                v.warnings.append(
                    f'成本 {v.cost:.2%} 吃掉了毛收益的 '
                    f'{abs(v.cost) / max(abs(v.gross_return), 1e-12):.0%}')

    if stability is not None:
        v.stability = float(stability)
        if v.stability < 0.5:
            v.warnings.append(f'仅 {v.stability:.0%} 的子样本中结论成立')

    v.warnings = _dedup(v.warnings)
    return v


# --------------------------------------------------------------------------- #
def _collect(estimates, ic, n_trials, method, horizon):
    from .estimate import estimate_from_series
    if estimates is not None:
        return list(estimates) if isinstance(estimates, (list, tuple)) else [estimates]
    if ic is None:
        return []
    ic = getattr(ic, 'df', ic)
    cols = list(ic.columns)
    if horizon is not None:
        cols = [c for c in cols if _h(c) == horizon] or cols
    # 同一族的全部 p 值：先用 NW t 换成 p，再一起校正 —— 这样 n_trials 有据可依
    out = []
    for c in cols:
        out.append(estimate_from_series(ic[c].dropna(), name=f'IC({_h(c)}d)',
                                        horizon=_h(c) or 1,
                                        n_trials=n_trials, method=method))
    return out


def _h(col):
    try:
        return int(col)
    except (ValueError, TypeError):
        try:
            return int(str(col).rsplit('_', 1)[1])
        except (ValueError, IndexError):
            return None


def _series(x):
    for attr in ('data', 'df'):
        v = getattr(x, attr, None)
        if isinstance(v, (pd.DataFrame, pd.Series)):
            x = v
            break
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0] if x.shape[1] == 1 else x.mean(axis=1)
    return pd.Series(x).astype(float)


def _cost(r, turnover, cost_bps):
    """逐期成本 = 换手 × 2（买卖）× 单边费率。**总是返回 Series**（无换手时全 0）。"""
    if turnover is None:
        return pd.Series(0.0, index=r.index)
    t = turnover
    for attr in ('data', 'df'):
        t = getattr(t, attr, t)
    if isinstance(t, pd.DataFrame):
        t = t.mean(axis=1)
    t = pd.Series(t).astype(float)
    if t.index.nlevels > 1:
        t = t.groupby(level=-1).mean()
    t = t.reindex(r.index).fillna(0.0)
    per_period = t * 2 * (cost_bps / 10000.0)
    per_period.name = 'cost'
    return per_period          # 逐期成本（调用方按需聚合）


def _dedup(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
