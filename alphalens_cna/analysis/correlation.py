"""A. 因子相关性 / 冗余度检验 —— 逐期截面相关，再对时间汇总。

补文章第 6 步（"入库前先看与库里因子的相关性"）。

⚠️ 为什么**绝不做整体池化相关**（本模块最重要的一条口径）
-------------------------------------------------------
池化 = 把所有 ``(期, 股票)`` 堆在一起算一个相关系数。它会把
**两因子共同的时间漂移**（这期整体都高、下期整体都低）当成"相关"，
于是**系统性高估**相关性，把互补因子误判成冗余。实测两个**逐期毫无关系**、
只是共享同一份漂移的因子（60 期 × 40 只，漂移 scale=3.0、两个因子都吃它、
各自叠 0.5 的白噪声；构造见 ``tests/test_correlation.py::test_no_pooling_drift_trap``）：

======================  ==============  ==========================================
口径                     相关系数         怎么算的
======================  ==============  ==========================================
**池化（错的）**          0.9541         60 期 × 40 只堆成一份样本
**逐期截面（本模块）**    −0.0057         逐期算、再对 60 期求平均（48.3% 的期为正）
======================  ==============  ==========================================

池化口径会说"**高度冗余**（0.95）"，而真相是**逐期毫无关系**（−0.006）——
这一条就是本模块存在的理由。所以：

1. **逐期**算截面相关 → 再对时间汇总（``mean`` / ``median`` /
   ``positive_rate`` / ``n_periods``）；
2. 判定用 ``mean``（``median`` 一并给出，防止被少数极端期带偏）；
3. **重叠期数不足就给 NaN + 写明原因**，不静默、不硬凑。

骨架复用
--------
"按列分组 → 逐期截面统计 → 时间汇总"这条骨架与 D（分组 IC）**同一个实现**
（:func:`alphalens_cna.analysis.group._by_date_apply` /
:func:`~alphalens_cna.analysis.group._time_summary`）——
同一件事不留两套写法。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..contract.errors import fail
from .group import (
    _MIN_OBS,
    _by_date_apply,
    _check_panel,
    _time_summary,
    _unwrap,
)

__all__ = [
    'factor_correlation', 'redundancy_check',
    'FactorCorrResult', 'RedundancyResult',
]


# --------------------------------------------------------------------------- #
# 入参归一
# --------------------------------------------------------------------------- #
def _factor_series(obj, name, contract):
    """把一个因子（契约对象 / DataFrame / Series）取成 Series。"""
    raw = _unwrap(obj, contract)
    if isinstance(raw, pd.Series):
        return raw
    if isinstance(raw, pd.DataFrame):
        if 'value' in raw.columns:                     # FactorPanel
            return raw['value']
        if name in raw.columns:
            return raw[name]
        if raw.shape[1] == 1:
            return raw.iloc[:, 0]
        fail(contract, 'factor_ambiguous',
             f'因子 `{name}` 的数据框有 {raw.shape[1]} 列 {list(raw.columns)}，'
             f'说不清用哪一列。\n'
             f'  修法：FactorPanel 取 `value` 列；或只传一列 / 传 Series。')
    fail(contract, 'bad_input',
         f'因子 `{name}` 需要 FactorPanel / DataFrame / Series，'
         f'收到 {type(obj).__name__}')


def _as_factor_mapping(factors, contract='correlation'):
    """把入参归一成 ``{名字: Series}``。

    接受两种写法（设计 §1.1）：

    1. ``Mapping[str, FactorPanel | DataFrame | Series]``；
    2. 一个**多列 DataFrame**（每列一个因子）。
    """
    out = {}
    if isinstance(factors, Mapping):
        for name, obj in factors.items():
            out[str(name)] = _factor_series(obj, str(name), contract)
        return out

    raw = _unwrap(factors, contract)
    if isinstance(raw, pd.Series):
        nm = raw.name if isinstance(raw.name, str) else 'factor'
        return {nm: raw}
    if isinstance(raw, pd.DataFrame):
        if 'available_at' in raw.columns and 'value' in raw.columns:
            fail(contract, 'looks_like_panel',
                 '这看起来是 **FactorPanel.df**（列含 value / available_at），'
                 '它只有 1 个因子。\n'
                 f'  修法：多因子请传字典 —— factor_correlation({{"ROE": panel, "PB": p2}})；'
                 f'或传一个每列一个因子的 DataFrame。')
        if raw.columns.duplicated().any():
            dup = list(pd.Index(raw.columns)[raw.columns.duplicated()])
            fail(contract, 'dup_name',
                 f'因子表有重名列 {dup}，无法区分。\n  修法：重命名后再传。')
        for c in raw.columns:
            out[str(c)] = raw[c]
        return out
    fail(contract, 'bad_input',
         f'需要 Mapping[str, 因子] 或多列 DataFrame，收到 {type(factors).__name__}')


def _validate_factors(mapping, contract='correlation'):
    """校验并拼成一张宽表（index = 各因子的**并集**，缺失即 NaN）。

    ⚠️ 取并集而不是交集：两个因子各自的可用期不同时，交集会把
    "其中一个没数据的那几期"整期丢掉，看起来像"重叠很足"。
    真实重叠由 :attr:`FactorCorrResult.n_overlap` 逐对给出。
    """
    if len(mapping) < 2:
        fail(contract, 'too_few',
             f'至少需要 **2** 个因子才能算相关性，收到 {len(mapping)} 个：'
             f'{list(mapping)}\n'
             f'  修法：因子库为空时无从比较；单个因子请直接看它的 IC。')
    series = {}
    for name, s in mapping.items():
        if not isinstance(s, pd.Series):
            s = pd.Series(s)
        if not isinstance(s.index, pd.MultiIndex):
            fail(contract, 'index',
                 f'因子 `{name}` 的索引必须是 MultiIndex(date, asset)，'
                 f'收到 {type(s.index).__name__}。\n'
                 f'  修法：df = df.set_index(["date", "asset"])')
        if list(s.index.names) != ['date', 'asset']:
            fail(contract, 'index_names',
                 f'因子 `{name}` 的索引层级名必须是 ["date", "asset"]，'
                 f'收到 {list(s.index.names)}。\n'
                 f'  修法：df.index = df.index.set_names(["date", "asset"])')
        if not pd.api.types.is_numeric_dtype(s):
            fail(contract, 'not_numeric',
                 f'因子 `{name}` 不是数值（dtype={s.dtype}），算不了相关。\n'
                 f'  修法：先编码成数值（如行业用哑变量，或改用分组 IC）。')
        if s.index.has_duplicates:
            ex = s.index[s.index.duplicated()][0]
            fail(contract, 'index_unique',
                 f'因子 `{name}` 的索引有重复，例如 {ex!r}。\n'
                 f'  含义：同一 (期, 股票) 有两条值，相关会被算重。')
        if s.isna().all():
            fail(contract, 'all_nan', f'因子 `{name}` 全为 NaN，无法比较。')
        series[name] = s
    wide = pd.DataFrame(series)
    _check_panel(wide, contract)
    return wide


def _pairs(names):
    """上三角因子对 ``(a, b)`` 与它们在列里的位置。"""
    pos = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]
    return pos


def _pair_index(obj, axis='columns'):
    """把 ``(a, b)`` 元组索引整成 MultiIndex（``unstack`` / ``xs`` 才认）。"""
    idx = obj.columns if axis == 'columns' else obj.index
    if isinstance(idx, pd.MultiIndex) or not len(idx):
        return obj
    if not all(isinstance(c, tuple) for c in idx):
        return obj
    mi = pd.MultiIndex.from_tuples(idx, names=['factor_a', 'factor_b'])
    obj = obj.copy()
    if axis == 'columns':
        obj.columns = mi
    else:
        obj.index = mi
    return obj


def _symmetrize(v, names, diag):
    """上三角 → 对称矩阵（对角线按 ``diag`` 填）。

    ⚠️ 两处坑（都实测踩过）：

    1. 必须先 ``reindex`` 成**方阵**：``unstack()`` 只长出"出现过"的行/列
       （最小的那个因子只当 b、从不当 a），行列标签对不上就会畸形；
    2. 对称化用 ``combine_first(up.T)``，**不能用 ``up + up.T``** ——
       NaN 参与加法仍是 NaN，上三角加下三角只会在两边都有值处留下数字，
       其余全变 NaN（而且看起来"像是"没数据，特别难查）。
       ``add(fill_value=0)`` 也不行：两边都缺的格子会被填成 0（假数字）。
    """
    up = v.unstack() if isinstance(v, pd.Series) else v
    up = up.reindex(index=names, columns=names)
    m = up.combine_first(up.T)
    arr = m.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, diag)
    out = pd.DataFrame(arr, index=list(names), columns=list(names))
    out.index.name = out.columns.name = 'factor'
    return out


# --------------------------------------------------------------------------- #
# 1. 因子相关性
# --------------------------------------------------------------------------- #
@dataclass
class FactorCorrResult:
    """因子相关性矩阵（**逐期截面相关 → 时间汇总**）。

    Attributes
    ----------
    matrix : DataFrame
        **均值**相关矩阵（``mean``），对称，行/列 = 因子名。
        对角线恒为 1.0（定义，不是算出来的）。
        ⚠️ 重叠期数 < ``min_overlap`` 的**对**是 NaN，
        原因逐对写在 :attr:`insufficient`；原始逐期值仍在 :attr:`per_period`。
    median : DataFrame
        同形状，**中位数** —— 逐期分布被少数极端期带偏时用它复核。
    positive_rate : DataFrame
        同形状，**逐期为正的比例**（分母 = 该对**有值**的期数，
        即 :attr:`n_periods`）。0.5 附近 = 方向随机。
    n_periods : DataFrame
        同形状，**能算出相关系数的期数**（不含因子为常数、共同样本 < 3 的期）。
        对角线 = 该因子自身在多少期"可用"（有效值 ≥ 3 只）。
    n_overlap : DataFrame
        同形状，**两因子都有值且共同样本 ≥ 3 的期数**。
        它 ≥ ``n_periods``：差出来的那几期是"有重叠但算不出相关"
        （因子在期截面里是常数）。
    stats : DataFrame
        逐对明细（**全部因子对**，不截断）：index = ``(factor_a, factor_b)``，
        columns = ``n_periods`` / ``n_overlap`` / ``mean`` / ``median`` /
        ``std`` / ``icir`` / ``positive_rate`` / ``comparable`` / ``short``。
        ⚠️ 被掩码的对**原值仍在这里**（不销毁证据），只是别拿它下结论。
    insufficient : DataFrame
        重叠不足的对：``n_periods`` / ``n_overlap`` / ``need`` / ``short`` /
        ``reason``（人话：哪一对、还差几期）。
    per_period : DataFrame
        index = 日期，columns = ``(factor_a, factor_b)``，值 = 该期截面相关。
        留证据用（想看"这个均值是不是被某一期带出来的"就查它）。
    skipped : DataFrame
        逐期统计阶段被跳过的**期**（该期有效资产 < 3，一行也凑不出相关）：
        columns = ``['date', 'group', 'n', 'reason']``。
    method : str
        ``'spearman'``（默认）/ ``'pearson'``。
    min_overlap : int
        重叠期数下限（默认 20）。
    factors : tuple
        因子名（顺序 = 入参顺序）。
    n_dates : int
        参与统计的期数。
    """

    matrix: pd.DataFrame
    median: pd.DataFrame
    positive_rate: pd.DataFrame
    n_periods: pd.DataFrame
    n_overlap: pd.DataFrame
    stats: pd.DataFrame
    insufficient: pd.DataFrame
    per_period: pd.DataFrame
    skipped: pd.DataFrame
    method: str = 'spearman'
    min_overlap: int = 20
    factors: tuple = ()
    n_dates: int = 0

    @property
    def mean(self):
        """``matrix`` 的别名（判定用的就是它）。"""
        return self.matrix

    @property
    def names(self):
        return list(self.factors)

    def summary(self):
        """逐对明细，按 ``|mean|`` 从大到小 —— 挑冗余时先看这张。"""
        return (self.stats
                .assign(abs_mean=self.stats['mean'].abs())
                .sort_values('abs_mean', ascending=False)
                .drop(columns='abs_mean'))

    def to_frame(self):
        """``stats`` 的别名（贴报告用）。"""
        return self.summary()

    def __str__(self):
        ok = int(self.stats['comparable'].sum()) if len(self.stats) else 0
        top = ''
        if len(self.stats):
            s = self.stats[self.stats['comparable']]
            if len(s):
                i = s['mean'].abs().idxmax()
                top = (f' · 最相关 {i[0]}~{i[1]} ρ={s.loc[i, "mean"]:+.2f}'
                       f'（{int(s.loc[i, "n_periods"])} 期）')
        return (f'<FactorCorr {len(self.factors)} 个因子 · {self.n_dates} 期 · '
                f'{self.method} · 可比 {ok}/{len(self.stats)} 对{top}>')


def factor_correlation(factors, *, method='spearman', min_overlap=20):
    """**因子相关性** —— 逐期截面相关，再对时间汇总（**绝不池化**）。

    Parameters
    ----------
    factors : Mapping[str, FactorPanel | DataFrame | Series] | DataFrame
        因子集合：给字典（名字 → 因子），或给一个**每列一个因子**的 DataFrame。
        ``FactorPanel`` 自动取 ``value`` 列；契约对象已在前视闸门上校验过
        ``available_at ≤ date``，所以这里不再处理"值何时可知"。
    method : {'spearman', 'pearson'}
        ``'spearman'``（默认）= 逐期 RankIC 式的秩相关，对极值稳健，
        也是本库其它地方（IC / 分层）的口径。
    min_overlap : int
        重叠期数下限（默认 20）。低于它的因子对**不给相关数字（NaN）**，
        并在 :attr:`FactorCorrResult.insufficient` 里写明"哪一对、还差几期"。

    Returns
    -------
    FactorCorrResult
        见 :class:`FactorCorrResult`。

    Notes
    -----
    ⚠️ **本函数不做整体池化相关**。池化会把两因子共同的时间漂移算成"相关"，
    系统性高估（实测两个逐期无关的因子：池化 **0.9541**、逐期均值 **−0.0057**）
    —— 详见本模块 docstring 的表。判定请用 ``matrix``（= 逐期均值），
    并对照 ``median`` / ``positive_rate`` / ``n_periods`` 一起看。

    ⚠️ 逐期相关要求该期有 **≥ 3** 只共同样本（与 ``analysis/ic.py`` 同口径）；
    因子在该期是常数 → 该期无相关可言（NaN），**不计入** ``n_periods``，
    但计入 ``n_overlap``。

    Examples
    --------
    >>> r = factor_correlation({'ROE': roe, 'PB': pb, 'MOM': mom})
    >>> r.matrix.round(2)              # 均值相关矩阵
    >>> r.insufficient                 # 哪些对重叠不足、还差几期
    >>> redundancy_check(pb, {'ROE': roe, 'MOM': mom}).reason
    """
    if method not in ('spearman', 'pearson'):
        fail('correlation', 'bad_method',
             f"method 只能是 'spearman'（默认）或 'pearson'，收到 {method!r}")
    if not isinstance(min_overlap, (int, np.integer)) or min_overlap < 1:
        fail('correlation', 'bad_min_overlap',
             f'min_overlap 必须是 ≥ 1 的整数，收到 {min_overlap!r}。\n'
             f'  含义：它是不给相关系数的期数下限（默认 20）。')

    mapping = _as_factor_mapping(factors, 'correlation')
    wide = _validate_factors(mapping, 'correlation')
    names = list(wide.columns)
    pos = _pairs(names)

    def _corr_row(sub):
        """一期截面：逐对相关 + 逐对共同样本数。"""
        x = sub[names]
        m = x.corr(method=method, min_periods=_MIN_OBS)
        ind = x.notna().to_numpy(dtype=float)
        cnt = ind.T @ ind                      # 逐对共同有效样本数
        a = m.to_numpy(dtype=float)
        rec = {}
        for i, j in pos:
            rec[('r', names[i], names[j])] = a[i, j]
            rec[('n', names[i], names[j])] = int(cnt[i, j])
        return rec

    frame, led = _by_date_apply(wide, _corr_row, min_size=_MIN_OBS)
    rc = [('r', names[i], names[j]) for i, j in pos]
    nc = [('n', names[i], names[j]) for i, j in pos]
    for c in rc + nc:                          # 空输入时补列，免得下游 KeyError
        if c not in frame.columns:
            frame[c] = np.nan
    def _take(kind):
        """取出 ``('r', a, b)`` / ``('n', a, b)`` 那一组列，列名降成 ``(a, b)``。"""
        cs = [c for c in frame.columns if isinstance(c, tuple) and c[0] == kind]
        out = frame[cs].copy()
        out.columns = pd.MultiIndex.from_tuples(
            [(c[1], c[2]) for c in cs], names=['factor_a', 'factor_b'])
        return out

    per_period, n_frame = _take('r'), _take('n')

    stats = _pair_index(_time_summary(per_period), axis='index')
    stats.index.names = ['factor_a', 'factor_b']
    # 有重叠但算不出相关的期数（因子在该期是常数）
    stats.insert(1, 'n_overlap',
                 (n_frame >= _MIN_OBS).sum(axis=0).reindex(stats.index).astype(int))
    stats['comparable'] = stats['n_periods'] >= int(min_overlap)
    stats['short'] = (int(min_overlap) - stats['n_periods']).clip(lower=0).astype(int)

    keep = stats['comparable']
    mat = _symmetrize(stats.loc[keep, 'mean'], names, 1.0)
    med = _symmetrize(stats.loc[keep, 'median'], names, 1.0)
    posr = _symmetrize(stats.loc[keep, 'positive_rate'], names, 1.0)
    npd = _symmetrize(stats['n_periods'], names, np.nan)
    nol = _symmetrize(stats['n_overlap'], names, np.nan)
    # 对角线：该因子自身在多少期"可用"（有效值 ≥ 3 只）
    usable = (wide.notna().groupby(level='date').sum() >= _MIN_OBS).sum()
    for nm in names:
        npd.loc[nm, nm] = int(usable.get(nm, 0))
        nol.loc[nm, nm] = int(usable.get(nm, 0))
    npd = npd.astype('Int64')                  # 期数允许缺（对角线以外不会缺）
    nol = nol.astype('Int64')

    bad = stats[~keep].copy()
    if len(bad):
        bad['need'] = int(min_overlap)
        bad['reason'] = [
            f'重叠不足：`{a}` 与 `{b}` 只有 {int(r.n_periods)} 期能算出相关'
            f'（其中 {int(r.n_overlap)} 期两因子都有值），'
            f'阈值 {int(min_overlap)} 期，**还差 {int(r.short)} 期** → '
            f'不给相关系数（NaN），不硬凑。'
            for (a, b), r in bad.iterrows()]
        bad = bad[['n_periods', 'n_overlap', 'need', 'short', 'reason']]
    else:
        # 空表也要**列齐全**：下游（如 redundancy_check）会直接按列取用
        bad = pd.DataFrame(
            columns=['n_periods', 'n_overlap', 'need', 'short', 'reason'],
            index=pd.MultiIndex.from_arrays([[], []],
                                            names=['factor_a', 'factor_b']))

    return FactorCorrResult(
        matrix=mat, median=med, positive_rate=posr, n_periods=npd,
        n_overlap=nol, stats=stats, insufficient=bad, per_period=per_period,
        skipped=led, method=method, min_overlap=int(min_overlap),
        factors=tuple(names),
        n_dates=int(per_period.index.nunique()))


# --------------------------------------------------------------------------- #
# 2. 冗余度判定
# --------------------------------------------------------------------------- #
@dataclass
class RedundancyResult:
    """入库前的冗余度判定（**带依据**，不是光给一个 bool）。

    Attributes
    ----------
    redundant : bool
        是否冗余：库里**存在**某个因子与候选的 ``|均值相关| ≥ threshold``。
        用**绝对值**比较 —— 强负相关（ρ ≈ −0.9）同样是冗余
        （信息重复，只是方向相反）。
    worst : DataFrame
        最相关的几个（默认前 5，按 ``|ρ|`` 降序）：
        ``factor`` / ``rho`` / ``abs_rho`` / ``n_periods`` / ``n_overlap`` /
        ``redundant``（该对是否过阈值）。重叠不足的对**不在这里**
        （它们没有 ρ），而在 :attr:`insufficient`。
    threshold : float
        阈值（默认 0.7）。**绝对值**口径。
    method : str
    reason : str
        **人话判定依据**，形如
        ``非常冗余（|ρ|=0.82 ≥ 阈值 0.7，最相关：ROE_TTM，重叠 58 期）``。
        库为空 / 全部重叠不足时也会明说"无法判定"及原因，不假装不冗余。
    candidate : str
        候选因子的名字（Series 的 ``name``；取不到时为 ``'candidate'``）。
    n_library : int
        库里因子个数（不含候选）。
    insufficient : DataFrame
        候选与库里**重叠不足**的那些对（原样带出，含"还差几期"）。
    stats : DataFrame
        候选 × 每个库因子的逐对明细（与 :attr:`FactorCorrResult.stats` 同列）。
    """

    redundant: bool
    worst: pd.DataFrame
    threshold: float
    method: str
    reason: str
    candidate: str = 'candidate'
    n_library: int = 0
    insufficient: pd.DataFrame = None
    stats: pd.DataFrame = None

    def to_frame(self):
        """一行摘要（汇总多个候选因子时用）。"""
        return pd.DataFrame([{
            'candidate': self.candidate,
            'redundant': self.redundant,
            'threshold': self.threshold,
            'method': self.method,
            'most_correlated': (self.worst.iloc[0]['factor']
                                if len(self.worst) else None),
            'max_abs_rho': (float(self.worst.iloc[0]['abs_rho'])
                            if len(self.worst) else np.nan),
            'n_comparable': len(self.worst),
            'reason': self.reason,
        }])

    def __str__(self):
        flag = '❌ 冗余' if self.redundant else '✅ 不冗余'
        return (f'【冗余判定】{flag}（候选 `{self.candidate}` vs 库 '
                f'{self.n_library} 个）\n  {self.reason}')


def _library_mapping(library):
    """因子库 → ``{名字: 因子}``。

    ⚠️ 裸的 ``FactorPanel`` 会被当成**单因子库**（面板本身没有名字，记为
    ``'factor'``）；要名字好看就传字典 ``{'ROE_TTM': panel}``。
    """
    if isinstance(library, Mapping):
        return _as_factor_mapping(library, 'correlation')
    raw = _unwrap(library, 'correlation')
    if (isinstance(raw, pd.DataFrame) and 'value' in raw.columns
            and 'available_at' in raw.columns):
        return {'factor': raw['value']}
    return _as_factor_mapping(library, 'correlation')


def _candidate_name(candidate):
    """候选因子的名字：Series 的 ``name`` → 单列 DataFrame 的列名 → ``'candidate'``。"""
    raw = _unwrap(candidate, 'correlation')
    if isinstance(raw, pd.Series):
        nm = raw.name
        return nm if isinstance(nm, str) and nm else 'candidate'
    if isinstance(raw, pd.DataFrame):
        if 'value' in raw.columns:                     # FactorPanel.df
            return 'candidate'
        if raw.shape[1] == 1:
            return str(raw.columns[0])
    return 'candidate'


def redundancy_check(candidate, library, *, threshold=0.7, method='spearman',
                     min_overlap=20):
    """**冗余度判定** —— 候选因子与因子库里最相关的那些比。

    Parameters
    ----------
    candidate : FactorPanel | DataFrame | Series
        待入库的因子。
    library : Mapping[str, 因子] | DataFrame | Series | FactorPanel
        因子库（同 :func:`factor_correlation` 的入参）。
    threshold : float
        |ρ| 阈值（默认 0.7）。⚠️ 比较用**绝对值** ——
        强负相关同样是冗余。
    method, min_overlap :
        同 :func:`factor_correlation`。

    Returns
    -------
    RedundancyResult
        见 :class:`RedundancyResult`；判定依据在 ``reason``。

    Notes
    -----
    ⚠️ 判定的数字是**逐期相关的均值**（不是池化相关，理由见模块 docstring）。
    所以它回答的是"**每期都在重复**吗"，而不是"两段长样本整体像不像"。

    ⚠️ 候选若与库中某个因子**重名**，会直接报错而不是悄悄覆盖 ——
    否则你会在"库里的 ROE"还是"我的 ROE"上得出错误结论。

    Examples
    --------
    >>> r = redundancy_check(new_factor, {'ROE_TTM': roe, 'MOM': mom})
    >>> r.redundant, r.reason
    (True, '非常冗余（|ρ|=0.82 ≥ 阈值 0.7，最相关：ROE_TTM，重叠 58 期）')
    """
    if not np.isfinite(threshold) or not (0 < float(threshold) <= 1):
        fail('correlation', 'bad_threshold',
             f'threshold 必须落在 (0, 1] 内，收到 {threshold!r}。\n'
             f'  修法：常用 0.5（松）/ 0.7（默认）/ 0.9（严）。')
    mapping = _library_mapping(library)
    if not mapping:
        fail('correlation', 'empty_library',
             '因子库是空的，无法判定冗余。\n'
             '  修法：给至少 1 个库内因子；库里没有可比对象时，'
             '冗余判定没有意义（不是"不冗余"）。')
    name = _candidate_name(candidate)
    if name in mapping:
        fail('correlation', 'name_collision',
             f'候选因子的名字 `{name}` 与库里某个因子重名。\n'
             f'  含义：合并后无法区分"库里的那个"和"你的候选"，'
             f'最相关因子算出来会指向库里的自己（ρ=1）。\n'
             f'  修法：给候选改名（如 Series.rename("ROE_new")），'
             f'或改库里那个的名字。')
    res = factor_correlation({name: candidate, **mapping}, method=method,
                            min_overlap=min_overlap)
    stats = res.stats
    rel = stats[[(a == name or b == name) for a, b in stats.index]].copy()
    rel['factor'] = [b if a == name else a for a, b in rel.index]
    rel = rel.set_index('factor')
    cmp_ = rel[rel['comparable']]
    worst = (cmp_.assign(abs_rho=cmp_['mean'].abs())
             .sort_values('abs_rho', ascending=False)
             .head(5)
             .reset_index()[['factor', 'mean', 'abs_rho', 'n_periods',
                             'n_overlap', 'comparable']]
             .rename(columns={'mean': 'rho', 'comparable': 'redundant'}))
    bad = res.insufficient
    bad = bad[[(a == name or b == name) for a, b in bad.index]] if len(bad) else bad

    n_lib = len(mapping)
    if not len(worst):
        reason = (f'无法判定（库中 {n_lib} 个因子与候选**全部重叠不足** '
                  f'min_overlap={int(min_overlap)} 期，'
                  f'最少的一对还差 '
                  f'{int(bad["short"].min()) if len(bad) else 0} 期）—— '
                  f'这不是"不冗余"，是"没有可比数据"')
        return RedundancyResult(redundant=False, worst=worst,
                                threshold=float(threshold), method=method,
                                reason=reason, candidate=name, n_library=n_lib,
                                insufficient=bad, stats=rel)
    top = worst.iloc[0]
    n = int(top['n_periods'])
    redundant = bool(top['abs_rho'] >= float(threshold))
    verdict = '非常冗余' if redundant else '不冗余'
    reason = (f'{verdict}（|ρ|={top["abs_rho"]:.2f} '
              f'{"≥" if redundant else "<"} 阈值 {float(threshold):g}，'
              f'最相关：{top["factor"]}，重叠 {n} 期）')
    if len(bad):
        reason += (f'；另有 {len(bad)} 对重叠不足未参与判定'
                   f'（见 insufficient）')
    return RedundancyResult(redundant=redundant, worst=worst,
                            threshold=float(threshold), method=method,
                            reason=reason, candidate=name, n_library=n_lib,
                            insufficient=bad, stats=rel)
