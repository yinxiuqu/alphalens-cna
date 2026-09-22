"""排名熵稳定性（**RRE**，Rank Rank-Entropy）—— 因子"新鲜度"的诊断。

它回答的问题
------------
因子每天把股票重新排一遍。**如果每天的排名分布几乎一样**，说明这个因子
几乎没在更新（可能是一份静态快照，或者 PIT 陈旧度没管住）；
**如果排名分布剧烈跳变**，说明因子是噪声。

两个极端都不能用，中间才是"活着且稳定"的因子。

定义
----
* **排名熵** ``H_t``：把当日因子的**排名**切成 ``bins`` 个桶，取分布的
  香农熵，除以 ``log(bins)`` 归一化到 ``[0, 1]``。
  全挤在一个桶（因子退化成常数）→ 0；均匀铺开 → 1。
* **RRE**：把相邻两期的排名分布做 KL 散度，取 ``1 − 平均归一化 KL``：

  .. math:: RRE = 1 - \\frac{1}{T-1}\\sum_t \\frac{KL(P_t \\| P_{t-1})}{\\log bins}

  ``1`` = 排名结构完全不变，``0`` = 每期都在大洗牌。

⚠️ **这是诊断量，不是显著性检验。** 它进 :class:`~alphalens_cna.inference.verdict.Verdict`
的 ``rre`` 字段是为了**并列展示**，不要拿它单独下结论。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail

__all__ = ['rank_entropy', 'rank_entropy_series', 'rank_stability',
           'rank_entropy_summary']

DEFAULT_BINS = 10


def _ranks(factor):
    """按日切分并转向量；返回 (日期数组, 每个日期的排名分布矩阵)。"""
    s = factor['value'] if isinstance(factor, pd.DataFrame) else factor
    if not isinstance(s.index, pd.MultiIndex) or \
            list(s.index.names[:2]) != ['date', 'asset']:
        fail('rank_entropy', 'index',
             '需要 MultiIndex(date, asset)；'
             '提示：df.set_index(["date","asset"])')
    return s


def _zscore_by_date(s):
    """逐期标准化（截面 z 分数）—— 让不同期的分布可比。"""
    v = pd.Series(np.asarray(s, dtype=float),
                  index=s.index.get_level_values('date'))
    g = v.groupby(level=0)
    z = (v - g.transform('mean')) / g.transform('std')
    # 常数截面（std=0）→ z 全 0：落在中间那个箱里 → 熵 0。
    # 这是**退化因子**该有的读数，不能变成 NaN（NaN 会被当成"没算"）。
    return z.fillna(0.0)


def _value_hist(s, bins):
    """逐期的**取值分布**直方图（对截面 z 分数等宽分箱）。

    ⚠️ 不要按"排名"分桶 —— 排名分桶后每期都必然均匀，熵恒等于 1、
    KL 恒等于 0，什么都测不出来（这是本模块第一版的真实 bug）。
    """
    z = _zscore_by_date(s)
    edges = np.linspace(-3, 3, bins + 1)
    out = {}
    for d, g in z.groupby(level=0):
        a = np.asarray(g.dropna(), dtype=float)
        if len(a) < bins:
            continue
        cnt = np.histogram(np.clip(a, -3, 3), bins=edges)[0].astype(float)
        if cnt.sum() <= 0:
            continue
        out[d] = cnt / cnt.sum()
    return out


def _rank_bins(s, bins):
    """逐期把资产按排名分成 ``bins`` 档，返回 {日期: {asset: 档位}}。"""
    out = {}
    for d, g in s.groupby(level='date'):
        a = pd.Series(np.asarray(g, dtype=float),
                      index=g.index.get_level_values('asset')).dropna()
        if len(a) < bins:
            continue
        r = a.rank(method='first')
        out[d] = np.clip(((r - 1) / len(r) * bins).astype(int), 0, bins - 1)
    return out


def _hist(ranks, date_vals, bins):
    """每期排名的归一化直方图。"""
    out = {}
    for d, g in pd.Series(ranks).groupby(date_vals):
        g = np.asarray(g, dtype=float)
        g = g[np.isfinite(g)]
        if len(g) < bins:
            continue
        # 排名 → [0, bins) 的桶；用秩而不是原值，避免分布形状干扰
        r = pd.Series(g).rank(method='average').to_numpy()
        b = np.clip(((r - 1) / len(r) * bins).astype(int), 0, bins - 1)
        cnt = np.bincount(b, minlength=bins).astype(float)
        out[d] = cnt / cnt.sum()
    return out


def rank_entropy_series(factor, bins=DEFAULT_BINS):
    """逐期**排名熵**（归一化到 [0,1]）。"""
    if bins < 2:
        fail('rank_entropy', 'bad_bins', f'bins 必须 >= 2，收到 {bins}')
    s = _ranks(factor)
    h = _value_hist(s, bins)
    logb = np.log(bins)
    out = {d: float(-(p[p > 0] * np.log(p[p > 0])).sum() / logb)
           for d, p in h.items()}
    ser = pd.Series(out).sort_index()
    ser.name = 'rank_entropy'
    return ser


def rank_entropy(factor, bins=DEFAULT_BINS):
    """:func:`rank_entropy_series` 的均值（单个数字）。"""
    ser = rank_entropy_series(factor, bins=bins)
    return float(ser.mean()) if len(ser) else np.nan


def rank_stability(factor, bins=DEFAULT_BINS):
    """**RRE** —— 相邻两期"排名档位不变"的比例，归一化到 ``[0, 1]``。

    .. math:: RRE = \frac{\bar{d} - 1/bins}{1 - 1/bins}

    其中 :math:`\bar{d}` 是相邻两期**留在同一档**的资产占比均值。
    完全稳定 → 1；与上一期独立 → 0（此时 :math:`\bar d = 1/bins`）。

    ⚠️ 用**转移矩阵的对角线**，不用"排名直方图的 KL" ——
    后者按排名分桶后每期必然均匀，KL 恒为 0、RRE 恒为 1，什么都测不出来。
    """
    if bins < 2:
        fail('rank_entropy', 'bad_bins', f'bins 必须 >= 2，收到 {bins}')
    s = _ranks(factor)
    h = _rank_bins(s, bins)
    dates = sorted(h)
    if len(dates) < 2:
        return np.nan
    diags = []
    for a, b in zip(dates[:-1], dates[1:]):
        pa, pb = h[a], h[b]
        common = pa.index.intersection(pb.index)
        if len(common) < bins:
            continue
        diags.append(float((pa.loc[common].to_numpy()
                            == pb.loc[common].to_numpy()).mean()))
    if not diags:
        return np.nan
    d_bar = float(np.mean(diags))
    lo = 1.0 / bins
    return float(np.clip((d_bar - lo) / (1.0 - lo), 0.0, 1.0))


def rank_entropy_summary(factor, bins=DEFAULT_BINS):
    """打包诊断 —— 供报告并列展示。

    Returns
    -------
    dict
        ``rre`` / ``mean_entropy`` / ``min_entropy`` / ``mean_kl`` /
        ``n_periods`` / ``note``
    """
    ser = rank_entropy_series(factor, bins=bins)
    rre = rank_stability(factor, bins=bins)
    note = ''
    if len(ser):
        if float(ser.min()) < 0.05:
            note = ('有截面的排名熵接近 0 —— 因子在该期退化成常数'
                    '（算不出有意义的 IC）')
        elif rre == rre and rre < 0.3:
            note = 'RRE 很低：排名结构每期剧烈变化，因子可能是噪声'
        elif rre == rre and rre > 0.99:
            note = 'RRE 接近 1：排名几乎不变，留神因子没在更新（见体检层"因子冻结"）'
    return {
        'rre': rre,
        'mean_entropy': float(ser.mean()) if len(ser) else np.nan,
        'min_entropy': float(ser.min()) if len(ser) else np.nan,
        'n_periods': int(len(ser)),
        'bins': int(bins),
        'note': note,
    }
