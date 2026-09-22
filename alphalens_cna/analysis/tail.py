"""尾部统计 —— **IC 看不见的那一块**。

为什么需要这一层
----------------
实测（`outputs/退市收益约定实现与结论修正.md`）：把 250 只退市股放回样本后，
新增的 1,405 条观测里 **88.6% 落在 ROE 最低分位**、252 天收益均值 **−62.9%**，
可 RankIC 反而**变弱**了 —— 因为 Spearman 度量的是全样本成对单调性，
一团挤在"双低角"的样本彼此同序，会把 ρ 往正方向拉。

**结论：秩相关对"灾难性下行"这类效应存在结构性盲区。**
本模块补三个专门看左尾的量：

============  ==========================================================
``VaR``       历史在险价值：左尾分位点（最差 5% 的门槛）
``CVaR``      条件在险价值 / 期望损失：左尾的**均值**（比 VaR 更看尾部厚度）
崩盘命中率      收益跌破门槛的**比例** —— 直接回答"这组票炸得多不多"
============  ==========================================================

符号约定（重要）
----------------
**统一用收益率符号，负数 = 亏损，不翻转成正数。**
理由：本模块的产物要和均值、分位差、IC 并排放；
如果 CVaR 报正数、均值报负数，同表里方向就分不清了。
需要"损失为正"的表述时用 ``-cvar(...)``。

崩盘门槛怎么定
--------------
默认**不拍脑袋定绝对阈值**，而是用**当日截面的最差 level 比例**
（默认 5%）—— 自我校准、随持有期自动缩放。
也可以传绝对阈值（如 ``-0.5`` 表示"一年腰斩"）做压力测试，
两种口径都由 ``threshold`` 参数显式选择，不隐式混用。

推断仍然走 ``inference/``
-------------------------
``crash_spread`` 逐期算 QN−Q1 的命中率差，再用
:func:`~alphalens_cna.inference.newey_west.nw_tstat` 给 t 值 ——
与库里其它地方同一把尺子。**本模块只算，判定统一在 inference 层。**
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail
from ..inference.newey_west import nw_tstat
from .ic import horizon_of, return_cols

__all__ = [
    'value_at_risk', 'cvar', 'expected_shortfall', 'downside_deviation',
    'tail_ratio', 'crash_stats', 'tail_by_quantile', 'crash_spread',
]

DEFAULT_LEVEL = 0.05


def _with_q(data):
    """统一分位列名：``q``（quantize 的输出）→ ``factor_quantile``。"""
    if isinstance(data, pd.DataFrame) and 'factor_quantile' not in data.columns \
            and 'q' in data.columns:
        data = data.rename(columns={'q': 'factor_quantile'})
    if not isinstance(data, pd.DataFrame) or 'factor_quantile' not in data.columns:
        fail('tail', 'no_quantile',
             '数据里没有 `factor_quantile`（或 `q`）；先跑 quantize()。')
    return data


def _clean(x):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    return x[np.isfinite(x)]


# --------------------------------------------------------------------------- #
def _tail_size(n, level):
    """尾部样本个数 ``k = max(1, ceil(level·n))``。

    ⚠️ **用顺序统计量，不用分位数插值。**
    若用 ``np.quantile``（线性插值），小样本下"最差 5%"会落到两个样本**中间**，
    于是"取 x ≤ 门槛"选出的实际比例与 ``level`` 对不上（实测 10 个样本
    level=0.1 时只剩 1 个 = 10%、level=0.05 时也是 1 个 = 10%）。
    改用顺序统计量后：**尾部比例 = k/n ≥ level，且小样本可手算**（防线 5）。
    """
    if not 0 < level < 1:
        fail('tail', 'bad_level', f'level 必须落在 (0,1)，收到 {level}')
    n = int(n)
    return max(1, int(np.ceil(level * n))), n


def value_at_risk(returns, level=DEFAULT_LEVEL):
    """历史 VaR —— 左尾 ``level`` 的**门槛收益**（负数）。

    取第 ``k = max(1, ⌈level·n⌉)`` 小的那个样本（**不插值**）。
    """
    x = _clean(returns)
    if not len(x):
        return np.nan
    k, _ = _tail_size(len(x), level)
    return float(np.sort(x)[k - 1])


def cvar(returns, level=DEFAULT_LEVEL):
    """CVaR / 期望损失 —— 左尾 ``level`` 比例样本的**均值收益**。

    比 VaR 多说一件事：不光看门槛在哪，还看**门槛之外的坑有多深**。
    """
    x = _clean(returns)
    if not len(x):
        return np.nan
    k, _ = _tail_size(len(x), level)
    return float(np.sort(x)[:k].mean())


# 金融文献里的常用名
expected_shortfall = cvar


def downside_deviation(returns, mar=0.0):
    """下行标准差 —— 只对低于 ``mar`` 的部分求标准差。

    与 :func:`~alphalens_cna.inference.multiplicity` 里的波动率类指标互补：
    总波动把上涨也算成风险，下行波动只算亏损那一侧。
    """
    x = _clean(returns)
    if not len(x):
        return np.nan
    d = x[x < mar] - mar
    if len(d) < 2:
        return np.nan
    return float(np.sqrt(np.mean(d ** 2)))


def tail_ratio(returns, level=DEFAULT_LEVEL):
    """右尾 / 左尾 —— 同样大的尾部概率下，赚的和亏的谁更极端。

    ``> 1`` 表示右尾更厚（好）；``< 1`` 表示左尾更厚（尾部风险不对称）。
    """
    x = _clean(returns)
    if not len(x):
        return np.nan
    up = float(np.quantile(x, 1 - level))
    dn = float(np.quantile(x, level))
    if dn == 0:
        return np.nan
    return abs(up / dn)


# --------------------------------------------------------------------------- #
def crash_stats(returns, level=DEFAULT_LEVEL, threshold=None):
    """崩盘统计 —— 命中率 + 崩盘时的平均幅度。

    Parameters
    ----------
    returns : Series | array
        前向收益（一期或多期混在一起都行）。
    level : float
        ``threshold=None`` 时，用**该样本的最差 level 分位**作门槛。
    threshold : float, optional
        绝对门槛（如 ``-0.5`` = 一年腰斩）。给了就用它，不再用分位数。

    Returns
    -------
    dict
        ``n`` / ``n_crash`` / ``hit_rate`` / ``mean_crash`` / ``cvar`` /
        ``threshold`` / ``threshold_mode`` / ``ref``

        ``ref`` = 命中样本的平均收益，与 ``cvar`` 在**分位模式**下相同
        （同一个尾部），在绝对门槛模式下会不同 —— 便于交叉核对。
    """
    x = _clean(returns)
    if not len(x):
        return {'n': 0, 'n_crash': 0, 'hit_rate': np.nan, 'mean_crash': np.nan,
                'cvar': np.nan, 'threshold': np.nan, 'threshold_mode': 'empty'}
    if threshold is None:
        k, _ = _tail_size(len(x), level)
        cut, mode = float(np.sort(x)[k - 1]), 'quantile'
    else:
        cut, mode = float(threshold), 'absolute'
    hit = x <= cut      # 分位模式下恰好 k 个（有并列时略多，如实计数）
    return {
        'n': int(len(x)),
        'n_crash': int(hit.sum()),
        'hit_rate': float(hit.mean()),
        'mean_crash': float(x[hit].mean()) if hit.any() else np.nan,
        'cvar': cvar(x, level),
        'threshold': cut,
        'threshold_mode': mode,
        'ref': float(x[hit].mean()) if hit.any() else np.nan,
    }


# --------------------------------------------------------------------------- #
def tail_by_quantile(data, quantiles=None, horizons=None, level=DEFAULT_LEVEL,
                     threshold=None):
    """**分位 × 持有期**的尾部统计 —— 和 :func:`quantile_stats` 并排看。

    Parameters
    ----------
    data : CleanResult | DataFrame
        需含 ``factor_quantile`` 与 ``forward_return_*``。
    level, threshold
        见 :func:`crash_stats`。``threshold`` 是所有持有期共用一个绝对门槛；
        不给则**每个持有期各自用自己的最差 level 分位**（推荐）。

    Returns
    -------
    DataFrame
        index = ``(h, q)``；columns = ``mean`` / ``cvar`` / ``hit_rate`` /
        ``n_crash`` / ``n`` / ``threshold``。
    """
    df = _with_q(getattr(data, 'data', data))
    cols = return_cols(df, horizons)
    rows = {}
    for c in cols:
        h = horizon_of(c)
        s = df[['factor_quantile', c]].dropna()
        for q, g in s.groupby('factor_quantile'):
            st = crash_stats(g[c], level=level, threshold=threshold)
            rows[(h, int(q))] = {
                'mean': float(g[c].mean()),
                'cvar': st['cvar'], 'hit_rate': st['hit_rate'],
                'n_crash': st['n_crash'], 'n': st['n'],
                'threshold': st['threshold'],
            }
    out = pd.DataFrame(rows).T
    out.index = pd.MultiIndex.from_tuples(out.index, names=['h', 'q'])
    return out.sort_index()


def crash_spread(data, quantiles=None, horizons=None, level=DEFAULT_LEVEL,
                 threshold=None, lags=None):
    """**QN − Q1 的崩盘命中率差**，逐期计算后给 Newey-West t 值。

    这是回答"低分位那组是不是更容易炸"的正确统计量 —— 它不受
    "联合极值簇会抬高秩相关"的影响（命中率是**比例**，不是秩）。

    Returns
    -------
    DataFrame
        index = 持有期；columns = ``hit_hi`` / ``hit_lo`` / ``hit_spread`` /
        ``mean_spread`` / ``t_hit`` / ``t_naive_hit`` / ``n_periods`` / ``lags``。

    Notes
    -----
    ``hit_spread = QN 命中率 − Q1 命中率``。**符号直觉**：若低分位更容易崩，
    这个差是**负的**（Q1 命中率高）。所以看``t_hit``要带着符号读。
    """
    df = _with_q(getattr(data, 'data', data))
    cols = return_cols(df, horizons)
    qs = sorted(pd.unique(df['factor_quantile'].dropna()))
    if len(qs) < 2:
        fail('tail', 'too_few_quantiles',
             f'只有 {len(qs)} 个分位，算不出 QN−Q1')
    lo, hi = int(qs[0]), int(qs[-1])
    rows = {}
    for c in cols:
        h = horizon_of(c)
        s = df[['factor_quantile', c]].dropna()
        per = []
        for d, g in s.groupby(level='date'):
            if threshold is None:
                kk, _ = _tail_size(len(g), level)
                cut = float(np.sort(g[c].to_numpy())[kk - 1])   # 当日截面门槛
            else:
                cut = float(threshold)
            g_lo = g.loc[g['factor_quantile'] == lo, c]
            g_hi = g.loc[g['factor_quantile'] == hi, c]
            if len(g_lo) < 2 or len(g_hi) < 2:
                continue
            per.append((float(g_hi.mean()), float(g_lo.mean()),
                        float((g_hi <= cut).mean()), float((g_lo <= cut).mean())))
        if not per:
            continue
        p = pd.DataFrame(per, columns=['hi', 'lo', 'hit_hi', 'hit_lo'])
        diff = (p['hit_hi'] - p['hit_lo'])
        st = nw_tstat(diff.to_numpy(), lags=lags, horizon=h)
        rows[h] = {
            'mean_hi': float(p['hi'].mean()), 'mean_lo': float(p['lo'].mean()),
            'mean_spread': float((p['hi'] - p['lo']).mean()),
            'hit_hi': float(p['hit_hi'].mean()),
            'hit_lo': float(p['hit_lo'].mean()),
            'hit_spread': float(diff.mean()),
            't_hit': st['t_nw'], 't_naive_hit': st['t_naive'],
            'n_periods': int(len(p)), 'lags': st['lags'], 'vif': st['vif'],
        }
    out = pd.DataFrame(rows).T
    out.index.name = 'h'
    return out
