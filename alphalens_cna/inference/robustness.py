"""扰动鲁棒性（**PFS**，Perturbation Fragility Score）—— 结论怕不怕挪一下椅子。

它回答的问题
------------
"你这个 IC 是 −0.0395，换了样本区间 / 少几只票 / 换个持有期，还剩多少？"

**一个只在恰好这组参数下成立的结论，不是结论。**
本模块把同一个评估在**扰动**下重跑很多次，报告结果分布 —— 而不是一个数字。

⚠️ **它测的不是显著性。** 实测：一个纯噪声序列（均值≈0）的
``subsample`` PFS 可以到 0.85 —— 因为"抽掉 20% 的期数"对 100 期的均值
本来就没多大影响。PFS 真正擅长抓的是另一种病：

    结论**靠少数几期（或少数极端样本）撑着** —— 把那一期抽掉，符号就翻。

所以 PFS 必须与 IC / NW t / 尾部统计**并列**读，不能单独下结论（设计原则 6）。

定义
----
``PFS`` = **扰动后仍与基准同号（且幅度在给定容差内）的比例**，取值 ``[0, 1]``。
``1.0`` = 怎么折腾都不变；``0.5`` = 掷硬币；``< 0.5`` = 基准结论本身不可靠。

三种扰动（各自独立，也可组合）
------------------------------
* ``subsample`` —— 随机抽 ``frac`` 比例的**期数**（时间上的稳定性）
* ``bootstrap`` —— 有放回重抽期数（抽样不确定性）
* ``drop_assets`` —— 随机丢掉 ``frac`` 比例的**股票**（截面上的稳定性）

⚠️ **不给单一数字当结论。** 返回的是分布 + 得分 + 每一抽的明细，
由使用者和 IC / t / 尾部一起并列看（设计原则 6：稳健性报告）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail

__all__ = ['perturb_series', 'pfs', 'robustness_report']

MODES = ('subsample', 'bootstrap', 'drop_assets')


def _as_1d(x):
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    return v[np.isfinite(v)]


def perturb_series(series, *, mode='subsample', n_draws=200, frac=0.8,
                   agg=np.mean, seed=0, tol=0.5):
    """对一个时间序列做扰动，返回每次抽样的 ``agg``。

    Parameters
    ----------
    series : Series | array
        通常是逐期 IC / 逐期多空收益 / 逐期崩盘率差。
    mode : {'subsample', 'bootstrap', 'drop_assets'}
        ``subsample`` / ``bootstrap`` 作用在**期数**上；
        ``drop_assets`` 只在 ``series`` 是 ``(date, asset)`` 面板时可用 ——
        那说明传进来的其实是一列逐期值，此时会报错提示改用面板入口。
    n_draws : int
    frac : float
        每次保留的比例。
    agg : callable
        每次抽样后的汇总函数（默认均值）。
    tol : float
        与基准同号之外，幅度还要落在这个相对容差内（默认 0.5 = 一半以内）。
        只判同号就传 ``np.inf``。

    Returns
    -------
    dict
        ``base`` / ``draws``（ndarray）/ ``mean`` / ``p5`` / ``p95`` /
        ``same_sign`` / ``within_tol`` / ``pfs`` / ``n_draws`` / ``mode`` /
        ``frac``
    """
    if mode not in MODES:
        fail('robustness', 'bad_mode', f'mode 只能是 {MODES}，收到 {mode!r}')
    if mode == 'drop_assets':
        fail('robustness', 'need_panel',
             "drop_assets 需要面板入口 robustness_report(factor=..., returns=...)；"
             "对一列逐期值做 drop_assets 没有意义")
    x = _as_1d(series)
    if len(x) < 5:
        fail('robustness', 'too_few',
             f'只有 {len(x)} 期，扰动没有意义（至少 5 期）')
    if not 0 < frac <= 1:
        fail('robustness', 'bad_frac', f'frac 必须落在 (0,1]，收到 {frac}')
    rng = np.random.default_rng(seed)
    k = max(2, int(round(frac * len(x))))
    base = float(agg(x))
    draws = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        if mode == 'bootstrap':
            idx = rng.integers(0, len(x), size=k)
        else:
            idx = rng.choice(len(x), size=k, replace=False)
        draws[i] = float(agg(x[idx]))
    same = np.sign(draws) == np.sign(base) if base != 0 else np.zeros(n_draws, bool)
    if np.isfinite(tol) and base != 0:
        within = same & (np.abs(draws / base - 1) <= tol)
    else:
        within = same
    return {
        'base': base, 'draws': draws, 'mean': float(draws.mean()),
        'p5': float(np.percentile(draws, 5)),
        'p95': float(np.percentile(draws, 95)),
        'same_sign': float(same.mean()), 'within_tol': float(within.mean()),
        'pfs': float(within.mean()),
        'n_draws': int(n_draws), 'mode': mode, 'frac': float(frac),
        'tol': float(tol),
    }


def pfs(series, **kw):
    """:func:`perturb_series` 的 ``pfs`` 得分（``[0,1]``）。"""
    return perturb_series(series, **kw)['pfs']


def robustness_report(series, *, modes=('subsample', 'bootstrap'),
                      n_draws=200, fracs=(0.5, 0.7, 0.9), seed=0, tol=0.5):
    """多组扰动 → 一张表（**并列展示，不合成单一结论**）。

    Returns
    -------
    DataFrame
        index = ``(mode, frac)``；columns = ``base`` / ``mean`` / ``p5`` /
        ``p95`` / ``same_sign`` / ``pfs`` / ``n_draws``。
    """
    rows = {}
    for m in modes:
        for f in fracs:
            try:
                r = perturb_series(series, mode=m, n_draws=n_draws, frac=f,
                                   seed=seed, tol=tol)
            except Exception as e:                                # noqa: BLE001
                rows[(m, f)] = {'base': np.nan, 'pfs': np.nan,
                                'note': f'{type(e).__name__}: {e}'}
                continue
            rows[(m, f)] = {k: r[k] for k in
                            ('base', 'mean', 'p5', 'p95', 'same_sign', 'pfs',
                             'n_draws')}
    out = pd.DataFrame(rows).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=['mode', 'frac'])
    return out
