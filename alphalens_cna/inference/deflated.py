"""DSR —— **夏普比率的多重检验校正**。

为什么这个库必须有它
--------------------
`multiplicity.py` 治的是 **t 值 / IC 的多重检验**；但**挑组合挑出来的夏普**
从来没被治过。这是同一件事只做了一半：

    试了 2,850 个参数组合，挑夏普最高的那个 —— 哪怕全是噪声，
    期望最大夏普也会随 N 增长。不校正，就是把选择偏差当成 alpha。

而且它比 p-hacking **更隐蔽**：夏普长得不像统计检验，人不会本能地想去校正它。

这个库刚好有算 DSR 最难拿到的那一环 —— **N**：:class:`~alphalens_cna.ledger.ResearchLedger`。

三个量
------
* :func:`probabilistic_sharpe_ratio` —— PSR：给定基准 SR*，真实 SR 超过它的概率
* :func:`deflated_sharpe_ratio` —— DSR：把 SR* 换成**N 次尝试下的期望最大夏普**
* :func:`min_track_record_length` —— 要多少样本才能把 DSR 抬到给定置信度

公式（Bailey & López de Prado, 2014）
-------------------------------------
.. math::

    E[\\max SR] \\approx \\sqrt{V(SR)}\\left[(1-\\gamma)\\Phi^{-1}\\!\\left(1-\\tfrac1N\\right)
                 + \\gamma\\,\\Phi^{-1}\\!\\left(1-\\tfrac1{Ne}\\right)\\right]

.. math::

    DSR = \\Phi\\!\\left(
      \\frac{(SR - E[\\max SR])\\sqrt{T-1}}
           {\\sqrt{1 - \\gamma_3 SR + \\frac{\\gamma_4-1}{4}SR^2}}\\right)

其中 :math:`\\gamma_3` 是偏度、:math:`\\gamma_4` 是**非超额**峰度（正态 = 3），
:math:`SR` 是**单期**（非年化）夏普。

⚠️ 两个常见错法：① 拿**年化**夏普代进去；② 峰度少加 3。本模块内部统一处理，
但如果你手工算，务必对齐。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contract.errors import fail

__all__ = ['psr', 'dsr', 'expected_max_sharpe', 'min_track_record_length',
           'DSRResult']

EULER_GAMMA = 0.5772156649015329


def _stats(returns):
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        fail('deflated', 'too_few',
             f'至少 4 个观测才能估偏度/峰度，收到 {len(x)}')
    sd = x.std(ddof=1)
    if sd == 0:
        fail('deflated', 'zero_vol', '收益序列标准差为 0，夏普无定义')
    sr = float(x.mean() / sd)                       # 单期夏普，不年化
    skew = float(((x - x.mean()) ** 3).mean() / sd ** 3)
    kurt = float(((x - x.mean()) ** 4).mean() / sd ** 4)   # 非超额，正态 = 3
    return sr, len(x), skew, kurt


def _phi(z):
    from scipy import stats
    return float(stats.norm.cdf(z))


def _ppf(q):
    from scipy import stats
    return float(stats.norm.ppf(q))


def _se_factor(sr, skew, kurt):
    """PSR/DSR 分母里那个方差因子。"""
    v = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    return np.sqrt(max(v, 1e-12))


def psr(returns, sr_benchmark=0.0):
    """**概率化夏普**：真实（单期）夏普大于 ``sr_benchmark`` 的概率。

    ``sr_benchmark=0`` 时，问的就是"这个夏普是真本事还是运气"。
    """
    sr, T, skew, kurt = _stats(returns)
    z = (sr - sr_benchmark) * np.sqrt(T - 1) / _se_factor(sr, skew, kurt)
    return _phi(z)


def expected_max_sharpe(n_trials, sr_variance):
    """**N 次尝试下的期望最大夏普**（纯噪声时）。

    Parameters
    ----------
    n_trials : int
        试了多少个组合/参数。``1`` 表示没挑过 → 返回 0。
    sr_variance : float
        各次尝试夏普的**方差**。不知道时用
        :func:`sharpe_variance_estimate` 的理论估计。
    """
    N = int(n_trials)
    if N < 1:
        fail('deflated', 'bad_n', f'n_trials 必须 >= 1，收到 {n_trials}')
    if N == 1:
        return 0.0
    if not np.isfinite(sr_variance) or sr_variance <= 0:
        fail('deflated', 'bad_var', f'sr_variance 必须为正，收到 {sr_variance}')
    s = np.sqrt(sr_variance)
    g = EULER_GAMMA
    return float(s * ((1 - g) * _ppf(1 - 1.0 / N)
                      + g * _ppf(1 - 1.0 / (N * np.e))))


def sharpe_variance_estimate(sr, T):
    """单次尝试夏普的方差的理论估计 ``V(SR) ≈ (1 + SR²/2) / T``。

    这是 iid 正态下的近似。**有 N 个组合的实际夏普时，直接用它们的样本方差更好** ——
    那种情况请把 ``sr_variance`` 显式传给 :func:`dsr`。
    """
    return float((1.0 + 0.5 * sr ** 2) / max(T, 2))


@dataclass
class DSRResult:
    """DSR 的完整结果 —— **不给单一数字**。"""

    dsr: float
    psr: float
    sr: float                 # 单期
    sr_annual: float          # 年化（仅供参考，不参与计算）
    n_trials: int
    n_obs: int
    skew: float
    kurt: float
    expected_max_sr: float    # 期望最大夏普（单期）
    sr_variance: float
    periods_per_year: float = 252.0

    @property
    def significant(self):
        """DSR > 0.95 才认为"扣掉选择偏差后仍站得住"（惯例门槛）。"""
        return bool(self.dsr > 0.95)

    def __str__(self):
        return (f'【DSR】{"✅ 扣掉选择偏差仍显著" if self.significant else "❌ 扣不掉选择偏差"}'
                f'（DSR={self.dsr:.3f}，门槛 0.95）\n'
                f'  单期 SR {self.sr:.4f}（年化 {self.sr_annual:.2f}）  '
                f'T={self.n_obs}  N={self.n_trials}\n'
                f'  期望最大 SR {self.expected_max_sr:.4f}（纯噪声下挑 N 次就能挑到这么多）\n'
                f'  PSR(0)={self.psr:.3f}  偏度 {self.skew:+.2f}  峰度 {self.kurt:.2f}')

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame([{k: getattr(self, k) for k in
                              ('sr', 'sr_annual', 'n_obs', 'n_trials', 'skew',
                               'kurt', 'expected_max_sr', 'psr', 'dsr',
                               'significant')}])


def dsr(returns, n_trials, sr_variance=None, periods_per_year=252.0,
        sr_benchmark=0.0):
    """**紧缩夏普比率** —— 把"挑过 N 次"这件事从夏普里扣掉。

    Parameters
    ----------
    returns : Series | array
        策略收益序列（**单期**，不需要年化）。
    n_trials : int
        挑过多少个组合。**这是最关键也最容易被忘掉的输入** ——
        可以直接用 ``ResearchLedger.n_trials(goal)``。
    sr_variance : float, optional
        各次尝试夏普的方差。不给则用 :func:`sharpe_variance_estimate` 的理论近似。
        **如果你保留了所有组合的夏普，请显式传进来** —— 那个估计更准。
    periods_per_year : float
        仅用于展示年化夏普，不参与 DSR 计算。
    sr_benchmark : float
        保留给 PSR 的基准（DSR 用期望最大夏普，不用这个）。

    Returns
    -------
    DSRResult
    """
    sr, T, skew, kurt = _stats(returns)
    var = (sharpe_variance_estimate(sr, T) if sr_variance is None
           else float(sr_variance))
    e_max = expected_max_sharpe(n_trials, var)
    z = (sr - e_max) * np.sqrt(T - 1) / _se_factor(sr, skew, kurt)
    return DSRResult(
        dsr=_phi(z), psr=psr(returns, sr_benchmark), sr=sr,
        sr_annual=sr * np.sqrt(periods_per_year), n_trials=int(n_trials),
        n_obs=T, skew=skew, kurt=kurt, expected_max_sr=e_max,
        sr_variance=var, periods_per_year=periods_per_year)


def min_track_record_length(returns, n_trials, target=0.95,
                            sr_variance=None, periods_per_year=252.0):
    """**最短轨长** —— 要多少观测才能让 DSR 达到 ``target``。

    问的是"样本还不够，还是策略不行"。返回值单位是观测数
    （``/periods_per_year`` 就是年）。
    """
    if not 0.5 < target < 1:
        fail('deflated', 'bad_target', f'target 必须落在 (0.5,1)，收到 {target}')
    sr, T, skew, kurt = _stats(returns)
    var = (sharpe_variance_estimate(sr, T) if sr_variance is None
           else float(sr_variance))
    e_max = expected_max_sharpe(n_trials, var)
    denom = sr - e_max
    if denom <= 0:
        return np.inf          # 夏普压根没超过噪声门槛，再多样本也没用
    z = _ppf(target)
    need = 1.0 + (z * _se_factor(sr, skew, kurt) / denom) ** 2
    return float(need)
