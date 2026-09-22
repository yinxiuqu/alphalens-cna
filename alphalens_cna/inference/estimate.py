"""Estimate —— 带不确定性标注的估计量。

设计原则（§7.1）
----------------
> **绝不给单一数字。** alphalens 给出 ``IC = 0.004``，但不告诉你这个数可不可信。
> 本次实测：同一个 ROE，朴素 t = 0.16，Newey-West 后 t = 0.04 —— **差 4 倍**。
> 库里只给一个数，使用者无从判断。

所以本库的每个统计量都是一个 :class:`Estimate`，**自带一组标注**：
名义样本量、**有效样本量**、朴素 t、NW t、原始 p、校正后 p、
用了多少次检验做校正、以及一条条警告。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import multiplicity as mult
from .newey_west import nw_tstat

__all__ = ['Estimate', 'estimate_from_series', 'estimates_to_frame']


@dataclass
class Estimate:
    """一个带标注的估计量。

    Attributes
    ----------
    name : str
        叫什么（如 ``'RankIC'`` / ``'gamma'``）。
    value : float
        点估计。
    horizon : int
        持有期（交易日）。重叠观测的严重程度由它决定。
    t_naive, t_nw : float
        朴素 t 与 Newey-West t。**两个都给** —— 差别本身就是信息。
    n, n_eff : int, float
        名义样本量、**有效样本量**（启发式，见 :mod:`.newey_west`）。
    p_raw, p_adj : float
        原始 p、多重检验校正后 p。
    n_trials : int
        **做校正时用了几次检验** —— 这是"可不可信"的关键上下文。
    method : str
        校正方法。
    warnings : list[str]
        自动生成的提醒（样本太少、t 虚高严重、有效样本不足…）。
    """

    name: str
    value: float = np.nan
    horizon: int = 1
    t_naive: float = np.nan
    t_nw: float = np.nan
    n: int = 0
    n_eff: float = np.nan
    p_raw: float = np.nan
    p_adj: float = np.nan
    lags: int = 0
    vif: float = np.nan
    n_trials: int = 0
    method: str = 'bhy'
    warnings: list = field(default_factory=list)

    # -- 派生 ---------------------------------------------------------------
    @property
    def t_inflation(self):
        """朴素 t 相对 NW t 的**虚高倍数**。"""
        if not np.isfinite(self.t_naive) or not np.isfinite(self.t_nw) or self.t_nw == 0:
            return np.nan
        return abs(self.t_naive / self.t_nw)

    @property
    def significant(self):
        """**校正后**是否显著。"""
        return bool(np.isfinite(self.p_adj) and self.p_adj < 0.05)

    @property
    def significant_naive(self):
        """未校正是否显著 —— 用来展示"校正掉了什么"。"""
        return bool(np.isfinite(self.p_raw) and self.p_raw < 0.05)

    def _auto_warnings(self, min_n=20, n_eff_floor=10, inflation_flag=1.5):
        w = list(self.warnings)
        if self.n and self.n < min_n:
            w.append(f'样本仅 {self.n} 期，统计功效低')
        if np.isfinite(self.n_eff) and self.n_eff < n_eff_floor:
            w.append(f'有效样本仅 {self.n_eff:.0f}（名义 {self.n}），'
                     f'重叠观测严重')
        if np.isfinite(self.t_inflation) and self.t_inflation >= inflation_flag:
            w.append(f'朴素 t 虚高 {self.t_inflation:.1f} 倍 —— '
                     f'只看 t_naive 会误判')
        if self.significant_naive and not self.significant:
            w.append(f'未校正显著（p={self.p_raw:.4f}），'
                     f'但 {self.method.upper()} 校正后不显著'
                     f'（p={self.p_adj:.4f}，n_trials={self.n_trials}）')
        if self.n_trials and self.n_trials > max(self.n, 1):
            w.append(f'校正基数 n_trials={self.n_trials} 大于样本量 '
                     f'{self.n} —— 因子测得多而样本少')
        # 去重保序
        seen, out = set(), []
        for x in w:
            if x not in seen:
                seen.add(x)
                out.append(x)
        self.warnings = out
        return self

    def __str__(self):
        s = (f'{self.name}: {self.value:+.6g}  '
             f'[t_naive {self.t_naive:+.2f} → t_NW {self.t_nw:+.2f}'
             f'{f", 虚高 {self.t_inflation:.1f}×" if np.isfinite(self.t_inflation) and self.t_inflation >= 1.5 else ""}'
             f' | n {self.n} → n_eff {self.n_eff:.0f}'
             f' | p {self.p_raw:.4f} → {self.p_adj:.4f}'
             f' ({self.method.upper()}, n_trials={self.n_trials})]')
        for w in self.warnings:
            s += f'\n    ⚠️ {w}'
        return s


# --------------------------------------------------------------------------- #
def estimate_from_series(series, name='RankIC', horizon=1, lags=None,
                         n_trials=None, method='bhy', all_p=None, labels=None):
    """从一条时间序列（IC / γ 等）造一个 :class:`Estimate`。

    Parameters
    ----------
    series : Series | array
        逐期的时间序列（**不是** 面板）。
    horizon : int
        持有期 —— 决定自动滞后阶数与重叠程度判断。
    n_trials : int, 可选
        **一共测过多少个假设。** 给了才做多重检验校正；
        不给则 ``p_adj = p_raw``，并在结果里标明"未校正"。
    all_p : array, 可选
        **同一族里所有检验的 p 值。** 给了就用它做校正基数
        （比 ``n_trials`` 更准 —— 它来自真实的检验台账）。
    """
    s = pd.Series(series).dropna() if not isinstance(series, pd.Series) else series.dropna()
    st = nw_tstat(s, lags=lags, horizon=horizon)
    p_raw = (float(mult.p_from_t(st['t_nw']))
             if np.isfinite(st.get('t_nw', np.nan)) else np.nan)

    p_adj, used_method, used_n = p_raw, 'none', 0
    if all_p is not None:
        arr = np.asarray(all_p, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) and np.isfinite(p_raw):
            idx = int(np.argmin(np.abs(arr - p_raw)))
            p_adj = float(mult.adjust(arr, method)[idx])
            used_method, used_n = method, len(arr)
    elif n_trials:
        p_adj = float(mult.adjust([p_raw], method, n_trials=n_trials)[0])
        used_method, used_n = method, int(n_trials)

    return Estimate(
        name=name, value=st['mean'], horizon=int(horizon),
        t_naive=st['t_naive'], t_nw=st['t_nw'],
        n=st['n'], n_eff=st['n_eff'],
        p_raw=p_raw, p_adj=p_adj,
        lags=st['lags'], vif=st['vif'],
        n_trials=used_n, method=used_method,
    )._auto_warnings()


def estimates_to_frame(estimates):
    """一组 Estimate → 可读的表。"""
    rows = []
    for e in (estimates if isinstance(estimates, (list, tuple)) else [estimates]):
        rows.append({
            'name': e.name, 'horizon': e.horizon, 'value': e.value,
            't_naive': e.t_naive, 't_nw': e.t_nw, 't_inflation': e.t_inflation,
            'n': e.n, 'n_eff': e.n_eff, 'lags': e.lags,
            'p_raw': e.p_raw, 'p_adj': e.p_adj,
            'n_trials': e.n_trials, 'method': e.method,
            'significant_naive': e.significant_naive,
            'significant': e.significant,
            'warnings': ' / '.join(e.warnings),
        })
    return pd.DataFrame(rows)
