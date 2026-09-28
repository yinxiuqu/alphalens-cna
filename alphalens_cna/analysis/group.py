"""分组骨架 —— 分组 IC / 组内一致性 / 双重排序。

设计 §4.1：A（因子相关性）与 D（分组 IC）共用**同一条骨架**
「**按列分组 → 逐期截面统计 → 时间汇总**」。
本模块把它落成 :func:`_by_date_apply`（阶段一、二）与
:func:`_time_summary` / :func:`_series_stats`（阶段三），
``analysis/correlation.py`` 直接复用 —— **同一件事不留两套写法**
（本库最怕"同一个数字两种口径"）。

分组 IC 回答什么问题
--------------------
全局 IC 是"整条截面平均下来"的一个数，它掩盖了一件事：
**因子可能只在某一类股票里有效**（只在小盘股里、只在某个行业里）。
本模块把它拆开：

=====================  ==========================================================
函数                   回答
=====================  ==========================================================
``grouped_ic``         每期、每组的**组内** IC；附**不分组**的全局 IC 做对照
``group_consistency``  各组 IC 是否同号、离散度多大 —— "只在某一组有效吗"
``double_sort``        ``by`` × ``q`` 的宫格平均收益 + **组内**单调性
=====================  ==========================================================

⚠️ 三条必须先知道的口径（否则数字会被读错）
------------------------------------------
1. **组内 IC 是"组内秩相关"**：Spearman 在**每个组内重新排秩**，
   与"先全局排秩、再分组算相关"不是同一个数。本模块另给
   :attr:`GroupedIC.within_rank_ic`（组内秩池化相关）做对照，推导见
   :meth:`GroupedIC.weighted_ic`。
2. **组内 IC 的样本量加权和 ≠ 朴素全局 IC**，两者都不该被当成"另一个算错了"。
   实测（40 只 × 24 期、每期 5 个各 8 只的等量组、因子与收益真有关系）：
   加权和 **0.3998**，朴素全局 IC **0.4905**，逐期最大差 **0.327**。
   差额 = **组间均值差**（各组因子均值/收益均值 ≠ 全样本均值），不是误差。
   逐位成立的恒等式是「Σ_g (n_g/n)·IC_g = 组内秩池化相关」，
   前提是**逐期**各组样本量相等且无并列值：实测同一份数据若改用
   **全样本**分位分组（逐期组量 2~14 不等），该等式立刻差 **0.325**。
3. **双重排序的"平均收益"＝先逐期求格内均值、再对期求平均** ——
   与 :func:`alphalens_cna.analysis.quantile.quantile_returns` /
   ``quantile_stats`` 同口径，不是把各期池化成一个样本。

⚠️ 缺失一律记在账上
------------------
组内样本不足（``min_group_size``）、组内因子是常数、分组键是 NaN、
双重排序里没样本的格子 —— 一律 **NaN + 记账**
（:attr:`GroupedIC.skipped` / :attr:`DoubleSortResult.missing`），
**不静默填 0、也不静默丢样本**。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..contract.errors import fail
from .ic import horizon_of, information_coefficient, return_cols
from .quantile import _quantile_within, quantile_returns

__all__ = [
    'grouped_ic', 'group_consistency', 'double_sort',
    'GroupedIC', 'DoubleSortResult',
]

#: 一个相关系数至少需要的样本数 —— 与 ``analysis/ic.py::_corr`` 同口径。
#: 少于 3 只算出来的"相关"只有 ±1 两种取值，是噪声不是信号。
_MIN_OBS = 3


# --------------------------------------------------------------------------- #
# 共享工具
# --------------------------------------------------------------------------- #
def _unwrap(obj, contract):
    """接受 DataFrame / Series / 任何带 ``.df`` / ``.data`` 的契约对象。"""
    for attr in ('data', 'df'):
        v = getattr(obj, attr, None)
        if isinstance(v, (pd.DataFrame, pd.Series)):
            return v
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj
    fail(contract, 'bad_input',
         f'需要 DataFrame / Series / 契约对象（FactorPanel、Exposures、Grouping…），'
         f'收到 {type(obj).__name__}')


def _check_panel(df, contract):
    """索引必须是 ``MultiIndex(date, asset)`` —— 全库统一约定。"""
    if not isinstance(df.index, pd.MultiIndex):
        fail(contract, 'index',
             f'索引必须是 MultiIndex(date, asset)，收到 {type(df.index).__name__}。\n'
             f'  修法：df = df.set_index(["date", "asset"])')
    names = list(df.index.names)
    if names != ['date', 'asset']:
        fail(contract, 'index_names',
             f'索引层级名必须是 ["date", "asset"]，收到 {names}。\n'
             f'  修法：df.index = df.index.set_names(["date", "asset"])')
    if df.index.has_duplicates:
        dup = df.index[df.index.duplicated()].unique()
        fail(contract, 'index_unique',
             f'索引 (date, asset) 必须唯一，发现 {len(dup)} 个重复，'
             f'例如 {dup[0]!r}。\n'
             f'  含义：同一 (期, 股票) 有两条记录，截面统计会把它数两次。')


def _key_frame(df, keys, *, contract='group'):
    """把 ``keys`` 归一成与 ``df`` 索引对齐的分组键表。

    ``keys`` 可以是：``None`` / 列名 ``str`` / ``Series`` / ``DataFrame``
    （每列一个键 = 多键分组）/ ``Grouping`` 之类的契约对象 / 上述的 list。

    Returns
    -------
    (key_frame, names)
        ``key_frame``：``DataFrame``，index 与 ``df`` 一致（``keys is None`` 时为 None）；
        ``names``：键名列表。
    """
    if keys is None:
        return None, []

    if isinstance(keys, pd.DataFrame):
        items = [keys[c] for c in keys.columns]
    elif isinstance(keys, (list, tuple)):
        items = list(keys)
    else:
        items = [keys]

    cols, names = {}, []
    for it in items:
        if isinstance(it, str):
            if it not in df.columns:
                fail(contract, 'by_missing',
                     f'by={it!r} 不是 data 的列。可用列：{list(df.columns)}\n'
                     f'  修法：列名写对，或直接把该列作为 Series 传进来。')
            s, nm = df[it], it
        else:
            raw = _unwrap(it, contract)
            if isinstance(raw, pd.DataFrame):
                if 'group' in raw.columns:                        # Grouping 契约
                    s, nm = raw['group'], 'group'
                elif raw.shape[1] == 1:
                    s, nm = raw.iloc[:, 0], str(raw.columns[0])
                else:
                    fail(contract, 'by_ambiguous',
                         f'分组依据的数据框有 {raw.shape[1]} 列 {list(raw.columns)}，'
                         f'说不清用哪一列。\n'
                         f'  修法：只传一列，或传列名（by="行业"）。')
            else:
                s = raw
                nm = getattr(raw, 'name', None)
                nm = 'group' if nm is None else str(nm)
            s = s.reindex(df.index)
            if s.isna().all():
                fail(contract, 'by_align',
                     f'分组键 `{nm}` 与 data 的索引对不上（对齐后全空）。\n'
                     f'  修法：两者的索引都必须是 (date, asset) 且层级名一致。')
        cols[nm] = s
        names.append(nm)
    return pd.DataFrame(cols, index=df.index), names


def _grouper(date_values, key_frame, names, date_level='date'):
    """拼 ``groupby`` 用的分组器：单键给 Index，多键给 MultiIndex。"""
    arrays = [np.asarray(date_values)]
    all_names = [date_level]
    if key_frame is not None and len(names):
        arrays += [np.asarray(key_frame[c]) for c in names]
        all_names += list(names)
    if len(arrays) == 1:
        return pd.Index(arrays[0], name=all_names[0])
    return pd.MultiIndex.from_arrays(arrays, names=all_names)


def _level(names):
    """``groupby(level=...)`` 用：单键给标量、多键给列表。

    ⚠️ 长度 1 的**列表**在 pandas 里是 legacy 写法，未来版本会把组键变成元组
    （pandas 2.3 已发 FutureWarning），所以单键一律走标量。
    """
    return names[0] if len(names) == 1 else list(names)


def _label(key, names):
    """把 ``groupby`` 吐出来的组键整成"人用的"标签（单键去元组）。"""
    if not names:
        return None
    return key[1] if len(names) == 1 else tuple(key[1:])


# --------------------------------------------------------------------------- #
# ★ 共享骨架（阶段一 + 阶段二）
# --------------------------------------------------------------------------- #
def _by_date_apply(data, func, *, keys=None, min_size=1, date_level='date'):
    """**共享骨架**：按列分组 → 逐期截面统计 → 拼回一张表。

    A（因子相关性）与 D（分组 IC）的阶段一、二完全同构：把面板按
    ``(date[, keys])`` 切成一个个**截面**，对每个截面调一次 ``func``，
    再把返回的字典拼成带 ``(date[, keys])`` 索引的表。
    两套写法必然长出两种口径，所以只有这一个实现。

    Parameters
    ----------
    data : DataFrame
        索引 ``(date, asset)`` 的 MultiIndex。
    func : callable
        接收**该格的子表**（索引仍是 ``(date, asset)``，行数 = 该格样本数），
        返回 ``dict``（列名 → 值）。返回 ``None`` 记为"该格跳过"。
        键可以用元组（如 ``('r', 'A', 'B')``），拼出来的列是 MultiIndex。
    keys : None | str | Series | DataFrame | 契约对象, 可选
        ``date`` 之外的**分组键**（如分组标签），按索引对齐到 ``data``。
    min_size : int
        该格行数 < ``min_size`` 时**不调用 func**，直接出 NaN + 记账。
    date_level : str
        日期层级名（默认 ``'date'``）。

    Returns
    -------
    frame : DataFrame
        index = ``(date[, *keys])``；columns = ``func`` 返回的键，外加固定列
        ``n``（该格行数）。**被跳过的格子在表里是 NaN 行**（索引保持完整，
        取数时看得见"这里缺"），原因逐条记在 ``ledger``。
    ledger : DataFrame
        columns = ``['date', 'group', 'n', 'reason']``，每条 = 一个被跳过的格子。
        空时是**同列的空表**（不是 None —— 调用方不必判空）。

    Notes
    -----
    ⚠️ 用 ``groupby(...).indices`` 拿**位置**再 ``take``，不用
    ``groupby.apply`` —— 后者在多键分组时会把索引换成组键，
    下游 ``reindex`` 之后**静默变全 NaN**（``quantile.py`` 踩过同一个坑）。

    ⚠️ 分组键为 NaN 的行**不属于任何组**：它们不进 ``frame``，
    单独记一条台账（原因写明"分组键缺失"），
    避免把"没标签的股票"混成一个隐形分组。
    """
    df = data
    ledger = []
    key_frame, names = (_key_frame(df, keys) if keys is not None
                        else (None, []))
    date = pd.Index(df.index.get_level_values(date_level))

    keep = np.ones(len(df), dtype=bool)
    if key_frame is not None and len(names):
        miss = key_frame.isna().any(axis=1).to_numpy()
        if miss.any():
            # 按日期汇总成一条，避免一期一条把台账撑爆
            grp = pd.Series(miss.astype(int)).groupby(np.asarray(date)).sum()
            for d, n in grp.items():
                if n:
                    ledger.append({
                        'date': d, 'group': None, 'n': int(n),
                        'reason': f'分组键缺失（{"、".join(names)} 有 NaN）：'
                                  f'该期 {int(n)} 行未参与任何组'})
            keep &= ~miss

    pos = np.flatnonzero(keep)
    if len(pos) != len(df):
        df = df.iloc[pos]
        date = date[pos]
        if key_frame is not None:
            key_frame = key_frame.iloc[pos]

    out_index, records = None, []
    if len(df):
        gb = df.groupby(_grouper(date, key_frame, names, date_level),
                        sort=True, dropna=False, observed=True)
        keys_seen = list(gb.indices.keys())
        out_index = (pd.Index(keys_seen, name=date_level) if not names else
                     pd.MultiIndex.from_tuples(keys_seen,
                                               names=[date_level] + list(names)))
        for k, where in gb.indices.items():
            sub = df.take(where)
            n = len(sub)
            if n < min_size:
                records.append({'n': n})
                ledger.append({
                    'date': k[0] if names else k, 'group': _label(k, names), 'n': n,
                    'reason': f'该格样本数 {n} < min_size {min_size}，'
                              f'跳过（不给数字，也不填 0）'})
                continue
            rec = func(sub)
            if rec is None:
                records.append({'n': n})
                ledger.append({
                    'date': k[0] if names else k, 'group': _label(k, names), 'n': n,
                    'reason': 'func 返回 None（该格无法计算）'})
                continue
            rec = dict(rec)
            rec['n'] = n
            records.append(rec)

    frame = (pd.DataFrame.from_records(records, index=out_index)
             if records else pd.DataFrame(index=out_index))
    # 元组键 → MultiIndex 列（from_records 只给一个扁平的元组 Index）
    if len(frame.columns) and all(isinstance(c, tuple) for c in frame.columns):
        frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    if 'n' not in frame.columns:
        frame['n'] = 0
    led = pd.DataFrame(ledger, columns=['date', 'group', 'n', 'reason'])
    return frame, led


# --------------------------------------------------------------------------- #
# ★ 共享骨架（阶段三：时间汇总）
# --------------------------------------------------------------------------- #
def _series_stats(s, *, dropna=True):
    """一条**逐期**序列 → 时间汇总。

    Returns
    -------
    dict
        ``n_periods`` / ``mean`` / ``median`` / ``std`` / ``icir`` /
        ``positive_rate``。

    ⚠️ ``positive_rate`` 的分母是**有值的期数**（``n_periods``），不是总期数 ——
    停牌/缺数据的期不该被当成"不为正"。``n_periods`` 一并给出来：
    期数少时不要过度解读。
    """
    s = pd.to_numeric(pd.Series(s), errors='coerce')
    if dropna:
        s = s.dropna()
    n = len(s)
    mean = float(s.mean()) if n else np.nan
    std = float(s.std(ddof=1)) if n > 1 else np.nan
    return {
        'n_periods': n,
        'mean': mean,
        'median': float(s.median()) if n else np.nan,
        'std': std,
        'icir': (mean / std) if (std and np.isfinite(std) and std > 0) else np.nan,
        'positive_rate': float((s > 0).mean()) if n else np.nan,
    }


_STAT_COLS = ['n_periods', 'mean', 'median', 'std', 'icir', 'positive_rate']


def _time_summary(per_period):
    """阶段三：逐期序列（每列一条）→ 时间汇总表。

    ``per_period``：index = 期，columns = 各个"对象"（如因子对、分组）。
    返回：index 与 ``columns`` 同名，行 = :func:`_series_stats` 的键。
    """
    if not len(per_period.columns):
        return pd.DataFrame(columns=_STAT_COLS)
    return pd.DataFrame({c: _series_stats(per_period[c])
                         for c in per_period.columns}).T[_STAT_COLS]


def _nan_reason(n_valid):
    """相关系数出不来的原因 —— IC 与分组 IC **共用同一句话**。"""
    if n_valid < _MIN_OBS:
        return f'有效样本 {n_valid} < {_MIN_OBS}（因子或收益缺值）'
    return '该截面内因子值或收益为常数（无相关可言）'


def _ic_value(x, y, method):
    """一对数组的相关系数 + 有效样本数 + 出不来的原因。

    ⚠️ 与 ``analysis/ic.py::_corr`` 同口径（至少 3 个共同样本、
    任一侧为常数则无相关可言），保证 IC 与分组 IC **同一把尺子**。

    Returns
    -------
    (value, n_valid, reason)
        ``value`` 为 NaN 时 ``reason`` 是人话说明；正常时为 None。
    """
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < _MIN_OBS:
        return np.nan, n, _nan_reason(n)
    a, b = x[m], y[m]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan, n, _nan_reason(n)
    v = float(pd.Series(a).corr(pd.Series(b), method=method))
    if not np.isfinite(v):
        return np.nan, n, '相关系数无法计算'
    return v, n, None


def _check_method(method, contract):
    if method not in ('spearman', 'pearson'):
        fail(contract, 'bad_method',
             f"method 只能是 'spearman'（默认，RankIC）或 'pearson'，收到 {method!r}")


def _same_group(index, glevels, key):
    """布尔掩码：``index`` 的分组层级等于 ``key`` 的行。"""
    m = np.ones(len(index), dtype=bool)
    for lv, kv in zip(glevels, key):
        m &= (index.get_level_values(lv) == kv)
    return m


# --------------------------------------------------------------------------- #
# 1. 分组 IC
# --------------------------------------------------------------------------- #
@dataclass
class GroupedIC:
    """分组 IC 的结果。

    Attributes
    ----------
    ic : DataFrame
        **逐期、组内** IC。index = ``(date, group)``，columns = 持有期。
        被跳过的格子是 **NaN**（不是 0）—— 原因逐条记在 :attr:`skipped`。
    counts : DataFrame
        与 ``ic`` 同形状，值为该格的**有效样本数**
        （因子与收益同时有值的只数，缺失收益不计入）。
    summary : DataFrame
        index = ``(horizon, group)``，columns = ``n_periods`` / ``mean`` /
        ``median`` / ``std`` / ``icir`` / ``positive_rate`` / ``mean_count``。
        ``n_periods`` 是**能算出 IC 的期数**，不是总期数。
    global_ic : DataFrame
        index = date，columns = 持有期；**不分组**的全局 IC
        （与 :func:`alphalens_cna.analysis.ic.information_coefficient` 同口径）。
        用途只有一个：和组内 IC 对照，看分组拆出了什么。
    within_rank_ic : DataFrame
        与 ``global_ic`` 同形状；**组内秩池化相关** ——
        把每个观测换成"组内秩"后，在**全体**上算 Pearson。
        ⚠️ 它**不是**全局 IC，而是"样本量加权的组内 IC 之和"逐位等于的那个量，
        推导见 :meth:`weighted_ic`。
    skipped : DataFrame
        columns = ``['date', 'group', 'horizon', 'n', 'n_valid', 'reason']``；
        每条 = 一个被跳过的 ``(期, 组, 持有期)`` 格子。
    skip_summary : DataFrame
        index = ``reason``，columns = ``n_cells``（跳了多少格）/
        ``n_periods``（牵涉多少期）—— 回答"跳过了多少、为什么"。
    method : str
    min_group_size : int
        组内样本低于它 → 跳过并记账。
    by_name : str
        分组键的名字（写报告用）。
    horizons : tuple
        本结果涉及的持有期（升序）。
    n_dates, n_groups : int
        期数；组数（各期出现过的组标签并集大小）。
    """

    ic: pd.DataFrame
    counts: pd.DataFrame
    summary: pd.DataFrame
    global_ic: pd.DataFrame
    within_rank_ic: pd.DataFrame
    skipped: pd.DataFrame
    skip_summary: pd.DataFrame
    method: str = 'spearman'
    min_group_size: int = 5
    by_name: str = 'group'
    horizons: tuple = ()
    n_dates: int = 0
    n_groups: int = 0

    @property
    def group_names(self):
        """分组层级名（单键时形如 ``['group']``）。"""
        return [n for n in self.ic.index.names if n != 'date']

    def weighted_ic(self):
        """逐期**按有效样本量加权**的组内 IC 之和（index = date，columns = 持有期）。

        .. code-block:: text

            Σ_g (n_g,h / Σ_g n_g,h) · IC_g,h

        两个用途，别混：

        1. 它**就是**把分组 IC 汇总成一个数的口径（每期一个值），
           ``.mean()`` 即"样本量加权的组内 IC 均值"。
        2. ⚠️ 它**不等于**朴素的全局 IC（:attr:`global_ic`）。实测（40 只 × 24 期、
           每期 5 个各 8 只的等量组）：加权和 0.3998 vs 全局 IC 0.4905
           （逐期最大差 0.327）。差额 = **组间均值差**，是定义差，不是谁算错了。

        逐位成立的恒等式（**逐期**各组样本量相等、无并列值、无缺失时）：

        .. code-block:: text

            Σ_g (n_g/n) · IC_g  ==  within_rank_ic        实测差 1.1e-16

        因为此时每个组的"组内秩"均值都是 (m+1)/2、标准差都是
        sqrt((m²−1)/12)：组间项恰好为 0，分母也恰好相等。
        前提里的"**逐期**"不能省：同一份数据改用**全样本**分位分组
        （逐期组量 2~14 不等）后，该等式立刻差 **0.325**。

        有格子被跳过时，权重在**该期有值的组之间**重新归一
        （已在 :attr:`skipped` 记账），恒等式随之只在近似意义上成立。
        """
        cnt = self.counts.where(self.ic.notna())
        tot = cnt.groupby(level='date').transform('sum')
        with np.errstate(invalid='ignore', divide='ignore'):
            w = cnt / tot
        return (self.ic * w).groupby(level='date').sum(min_count=1)

    def to_frame(self):
        """组 × 持有期的**均值 IC** 宽表（贴报告用）。"""
        return self.summary['mean'].unstack('horizon')

    def __str__(self):
        hd = '、'.join(f'h={h}' for h in self.horizons) or '—'
        n_skip = 0 if self.skipped is None or not len(self.skipped) else len(self.skipped)
        return (f'<GroupedIC {self.n_dates} 期 × {self.n_groups} 组 · 持有期 {hd} · '
                f'{self.method} · 组内样本下限 {self.min_group_size} · '
                f'跳过 {n_skip} 格>')


def grouped_ic(data, *, by, horizons=None, method='spearman',
               min_group_size=5):
    """**分组 IC** —— 逐期、**组内**截面 IC，再对时间汇总。

    Parameters
    ----------
    data : DataFrame | CleanResult
        索引 ``(date, asset)``，含 ``factor`` 与 ``forward_return_*``
        （用 :func:`alphalens_cna.analysis.ic.return_cols` 取列）。
    by : str | Series | DataFrame | Grouping
        分组依据（行业、规模组、自定义标签…）：

        * ``str`` —— ``data`` 的列名；
        * ``Series`` —— 按 ``(date, asset)`` 对齐；
        * ``DataFrame`` —— 每列一个键（**多键分组**，输出索引随之多一层）；
        * ``Grouping`` 契约对象 —— 取 ``group`` 列。

        ⚠️ 分组键**必须与 ``date`` 对齐**：本函数**不做 as-of 回填**。
        行业/规模这类变量若按"最新所属"给，**等于前视**（今天的行业分类
        被用到了三年前）—— 请在入口处按 ``available_at`` 取 as-of 值。
    horizons : list[int], 可选
        只要这些持有期；默认全部 ``forward_return_*``。
    method : {'spearman', 'pearson'}
        ``'spearman'``（默认）= 组内 RankIC。
    min_group_size : int
        组内样本 < 它 → **该期该组跳过**（NaN + 记账），默认 5。
        要求 ≥ 3：2 只股票的相关只有 ±1 两种取值，是噪声。

    Returns
    -------
    GroupedIC
        见 :class:`GroupedIC`。核心三张表：``ic``（逐期明细）、
        ``summary``（时间汇总）、``skipped``（跳过台账）。

    Notes
    -----
    ⚠️ **不要**把 ``weighted_ic()`` 的结果当成"全局 IC"用 —— 两者定义不同，
    差额是组间均值差（实测可达 0.3 量级）。要全局 IC 就用 ``global_ic``
    （= :func:`~alphalens_cna.analysis.ic.information_coefficient`）。
    推导见 :meth:`GroupedIC.weighted_ic`。

    ⚠️ 组内 IC 的**期数可能低于总期数**（小样本组被跳过）——
    ``summary['n_periods']`` 一并给出，期数少时不要过度解读。

    Examples
    --------
    >>> g = grouped_ic(data, by='行业')
    >>> g.summary.loc[21]                       # 各行业在 21 日持有期上的 IC
    >>> group_consistency(g).attrs['across_groups']   # 是不是只有某个行业有效
    """
    df = _unwrap(data, 'group')
    if not isinstance(df, pd.DataFrame):
        fail('group', 'bad_input', f'需要 DataFrame，收到 {type(df).__name__}')
    _check_panel(df, 'group')
    _check_method(method, 'group')
    if 'factor' not in df.columns:
        fail('group', 'no_factor',
             f'缺 `factor` 列；实际列：{list(df.columns)[:12]}\n'
             f'  修法：清洗后的面板（clean() 的 data）都有这一列。')
    if not len(df):
        fail('group', 'no_obs',
             '一条有效观测都没有，算不出分组 IC。\n'
             '  常见原因：持有期超过样本跨度，或全部样本被成交规则剔除。')
    if not isinstance(min_group_size, (int, np.integer)) or min_group_size < _MIN_OBS:
        fail('group', 'bad_min_group_size',
             f'min_group_size 必须是 ≥ {_MIN_OBS} 的整数，收到 {min_group_size!r}。\n'
             f'  原因：{_MIN_OBS} 只以下算出的相关系数只有 ±1 两种取值（噪声）。')
    cols = return_cols(df, horizons)                 # 缺列时以 'ic' 契约报错
    h_of = {c: horizon_of(c) for c in cols}
    hcols = sorted(h_of.values())

    key_frame, names = _key_frame(df, by, contract='group')
    by_name = '、'.join(names)
    work = df[cols].copy()
    work['__factor'] = pd.to_numeric(df['factor'], errors='coerce').to_numpy()

    def _cell(sub):
        x = sub['__factor'].to_numpy(dtype=float)
        rec = {}
        for c in cols:
            v, n, _why = _ic_value(x, sub[c].to_numpy(dtype=float), method)
            rec[f'ic_{h_of[c]}'] = v
            rec[f'n_{h_of[c]}'] = n
        return rec

    frame, led = _by_date_apply(work, _cell, keys=key_frame,
                                min_size=min_group_size)
    for c in [f'ic_{h}' for h in hcols] + [f'n_{h}' for h in hcols]:
        if c not in frame.columns:              # 所有格子都被跳过时补列
            frame[c] = np.nan
    ic = frame[[f'ic_{h}' for h in hcols]].copy()
    ic.columns = hcols
    counts = frame[[f'n_{h}' for h in hcols]].copy()
    counts.columns = hcols
    ic.index.names = ['date'] + list(names)
    counts.index.names = ic.index.names

    # -- 记账：① 骨架里"样本不足"的格；② 样本够但 IC 仍为 NaN 的格 ------------ #
    recs = []
    if len(led):
        for _, r in led.iterrows():
            for h in hcols:
                recs.append({'date': r['date'], 'group': r['group'], 'horizon': h,
                             'n': r['n'], 'n_valid': np.nan, 'reason': r['reason']})
    cell_n = frame['n']
    for h in hcols:
        bad = ic[h].isna().to_numpy() & (cell_n.to_numpy() >= min_group_size)
        for k in ic.index[bad]:
            nv = int(counts.loc[k, h])
            recs.append({
                'date': k[0],
                'group': (k[1] if len(k) == 2 else tuple(k[1:])),
                'horizon': h, 'n': int(cell_n.loc[k]), 'n_valid': nv,
                'reason': f'组内{_nan_reason(nv)}'})
    skipped = pd.DataFrame(recs, columns=['date', 'group', 'horizon', 'n',
                                          'n_valid', 'reason'])
    if len(skipped):
        sk = skipped.groupby('reason', sort=False)
        skip_summary = pd.DataFrame({'n_cells': sk.size(),
                                     'n_periods': sk['date'].nunique()})
    else:
        skip_summary = pd.DataFrame(columns=['n_cells', 'n_periods'])

    # -- 时间汇总：逐 (持有期, 组) --------------------------------------------- #
    rows = []
    for h in hcols:
        for gkey, s in ic[h].groupby(level=_level(names), sort=True):
            key = gkey if isinstance(gkey, tuple) else (gkey,)
            mask = _same_group(ic.index, names, key)
            rows.append({'horizon': h, **dict(zip(names, key)),
                         **_series_stats(s),
                         'mean_count': float(counts.loc[mask, h].mean())})
    if rows:
        summary = (pd.DataFrame(rows).set_index(['horizon'] + list(names))
                   [['n_periods'] + _STAT_COLS[1:] + ['mean_count']])
    else:
        cols_empty = _STAT_COLS + ['mean_count']
        summary = pd.DataFrame(columns=['horizon'] + list(names) + cols_empty)

    # -- 对照：全局 IC 与"组内秩池化相关" -------------------------------------- #
    # 全局 IC 交给库里的 information_coefficient（**同一把尺子**，测试逐位对拍）
    gi_src = df[cols].copy()
    gi_src['factor'] = work['__factor'].to_numpy()
    global_ic = information_coefficient(gi_src, horizons=hcols, method=method)
    global_ic.columns.name = 'horizon'

    # 组内秩（一次算完，别算两遍）
    rank_src = df[cols].copy()
    rank_src['factor'] = work['__factor'].to_numpy()
    ranked = _rank_within(rank_src, ['factor'] + cols, key_frame, names)

    def _pooled(sub):
        x = sub['factor'].to_numpy(dtype=float)
        return {f'ic_{h_of[c]}': _ic_value(x, sub[c].to_numpy(dtype=float),
                                           'pearson')[0] for c in cols}

    pooled, _led2 = _by_date_apply(ranked, _pooled, min_size=_MIN_OBS)
    within_rank_ic = pooled[[f'ic_{h}' for h in hcols]].copy()
    within_rank_ic.columns = hcols
    within_rank_ic.index.name = 'date'
    within_rank_ic.columns.name = 'horizon'

    n_groups = (int(len(summary.index.droplevel('horizon').unique()))
                if rows else 0)
    return GroupedIC(
        ic=ic, counts=counts, summary=summary, global_ic=global_ic,
        within_rank_ic=within_rank_ic, skipped=skipped,
        skip_summary=skip_summary, method=method,
        min_group_size=int(min_group_size), by_name=by_name,
        horizons=tuple(hcols), n_dates=int(ic.index.get_level_values('date').nunique()),
        n_groups=n_groups)


def _rank_within(df, cols, key_frame, names):
    """把每列换成**组内秩**（组 = ``(date, keys)``）。

    ⚠️ 分组键缺失的行**秩为 NaN**（它们不属于任何组），
    否则会被"凑"进某个隐形组，把池化相关算脏。
    """
    if not len(names):
        return df[cols].astype(float)
    out = pd.DataFrame(np.nan, index=df.index, columns=cols, dtype=float)
    valid = ~key_frame[names].isna().any(axis=1).to_numpy()
    if valid.any():
        sub = df.loc[valid, cols]
        grouper = _grouper(pd.Index(df.index.get_level_values('date')[valid]),
                           key_frame.loc[valid], names)
        out.loc[valid, :] = sub.groupby(grouper).rank().to_numpy()
    return out


# --------------------------------------------------------------------------- #
# 2. 组内一致性
# --------------------------------------------------------------------------- #
def group_consistency(grouped, *, horizons=None):
    """**组内一致性** —— 各组 IC 的同号比例与离散度。

    回答分组 IC 最关键的一句话：**"因子是不是只在某一组里有效"**。

    Parameters
    ----------
    grouped : GroupedIC
        :func:`grouped_ic` 的返回。
    horizons : list[int], 可选
        只看这些持有期。

    Returns
    -------
    DataFrame
        index = ``(horizon, group)``，columns：

        ``n_periods``
            能算出 IC 的期数（下面这些比例的分母）。
        ``mean`` / ``median`` / ``std`` / ``icir``
            组内 IC 的位置与**离散度**；``std`` 越大越不稳定。
        ``positive_rate``
            IC > 0 的期数占比。
        ``sign``
            主号（``mean`` 的符号）：``+1`` / ``-1`` / ``0``。
        ``same_sign_rate``
            与该组**主号**同号的期数占比 —— 0.5 附近 = 方向随机，
            接近 1 才是稳定有效。**读"只在某一组有效"先看这一列。**
        ``sign_vs_global_rate``
            该组 IC 与**当日全局 IC**（``grouped.global_ic``）同号的期数占比 ——
            低，说明这组的因子行为跟整体不一致。
        ``mean_count``
            该组每期平均有效样本数。

        ``df.attrs['across_groups']``：按持有期**跨组**汇总，columns =
        ``n_groups`` / ``mean_of_means`` / ``std_of_means``（组间离散度）/
        ``min_group_mean`` / ``max_group_mean`` / ``majority_sign`` /
        ``sign_agreement_rate``（多数号一致的组占比：5 组里只有 1 组为正 → 0.2，
        一眼看出"只在某一组有效"）。

    Notes
    -----
    ⚠️ 比例为 1.0 不代表"稳"：两期也能凑出 1.0。
    请与 ``n_periods`` 一起读。
    """
    if not isinstance(grouped, GroupedIC):
        fail('group', 'bad_grouped',
             f'group_consistency 需要 grouped_ic(...) 的返回（GroupedIC），'
             f'收到 {type(grouped).__name__}。\n'
             f'  修法：g = grouped_ic(data, by=...); group_consistency(g)')
    glevels = grouped.group_names
    if horizons is None:
        hcols = list(grouped.horizons)
    else:
        hcols = [h for h in grouped.horizons if h in set(horizons)]
        if not hcols:
            fail('group', 'no_horizon',
                 f'horizons={horizons} 在本结果里一个都没有；'
                 f'本结果是 {list(grouped.horizons)}')
    ic, cnt = grouped.ic[hcols], grouped.counts[hcols]
    gic = grouped.global_ic[hcols]

    rows = []
    for h in hcols:
        s_h = ic[h]
        date = pd.Index(s_h.index.get_level_values('date'))
        sign_global = np.sign(gic[h].reindex(date).to_numpy())
        for gkey, s in s_h.groupby(level=_level(glevels), sort=True):
            key = gkey if isinstance(gkey, tuple) else (gkey,)
            mask = _same_group(ic.index, glevels, key)
            v = s.dropna()
            n = len(v)
            sg = np.sign(v.to_numpy())
            main = int(np.sign(v.mean())) if n else 0
            agree = float((sg == main).mean()) if (n and main) else np.nan
            gv = sign_global[mask]
            ok = np.isfinite(gv) & (gv != 0)
            vs_global = float((sg[ok] == gv[ok]).mean()) if (n and ok.sum()) else np.nan
            st = _series_stats(v, dropna=False)
            rows.append({'horizon': h, **dict(zip(glevels, key)),
                         **{k: st[k] for k in _STAT_COLS},
                         'sign': main, 'same_sign_rate': agree,
                         'sign_vs_global_rate': vs_global,
                         'mean_count': (float(cnt.loc[mask, h].mean())
                                        if mask.any() else np.nan)})
    out = (pd.DataFrame(rows).set_index(['horizon'] + list(glevels))
           if rows else pd.DataFrame(
               columns=['horizon'] + list(glevels) + _STAT_COLS
               + ['sign', 'same_sign_rate', 'sign_vs_global_rate', 'mean_count']))

    # -- 跨组汇总：组间离散度 + 多数号一致的组占比 ------------------------------- #
    across = []
    for h in hcols:
        sub = out.xs(h, level='horizon') if len(out) else out
        m = sub['mean'].dropna()
        signs = sub['sign'].dropna()
        majority, agree_rate = 0, np.nan
        if len(signs):
            vc = signs.value_counts()
            majority = int(vc.index[0])
            agree_rate = float(vc.iloc[0] / len(signs))
        across.append({'horizon': h, 'n_groups': int(len(sub)),
                       'mean_of_means': float(m.mean()) if len(m) else np.nan,
                       'std_of_means': float(m.std(ddof=1)) if len(m) > 1 else np.nan,
                       'min_group_mean': float(m.min()) if len(m) else np.nan,
                       'max_group_mean': float(m.max()) if len(m) else np.nan,
                       'majority_sign': majority,
                       'sign_agreement_rate': agree_rate})
    out.attrs['across_groups'] = (pd.DataFrame(across).set_index('horizon')
                                 if across else pd.DataFrame())
    out.attrs['by_name'] = grouped.by_name
    out.attrs['method'] = grouped.method
    return out


# --------------------------------------------------------------------------- #
# 3. 双重排序（by × q 网格）
# --------------------------------------------------------------------------- #
@dataclass
class DoubleSortResult:
    """双重排序（``by`` × ``q`` 网格）的结果。

    Attributes
    ----------
    mean : DataFrame
        **宫格平均收益**：index = ``(q_by, q)``，columns = 持有期，
        形状恒为 ``n_by × n``（没有样本的格子是 NaN）。
        ⚠️ 口径 = **先逐期求格内均值，再对期求平均**（与 ``quantile_returns`` /
        ``quantile_stats`` 一致），不是把各期池化成一个样本。
    count : Series
        index = ``(q_by, q)``：该格**平均每期样本数**（与持有期无关，
        按第一个持有期的有效收益计）。样本只有一两只的格子不要当结论用。
    n_periods : Series
        index = ``(q_by, q)``：该格**出现过的期数**（0 = 从未出现）。
    monotonicity : DataFrame
        index = ``(q_by, horizon)``，值 = **该 by 组内**层序 ``q`` 与平均收益的
        Spearman（与 ``quantile_stats.monotonicity`` 同一算法、同一"先池化"口径）。
        层数 < 3 时为 NaN（三层的相关才有意义）。
    pooled_monotonicity : Series
        index = horizon；把 ``q_by`` 也池掉之后的单调性 ——
        与 ``quantile_stats`` 的 ``monotonicity`` **逐位同口径**
        （``n_by == n`` 时可直接对拍）。
    missing : DataFrame
        columns = ``['q_by', 'q', 'kind', 'n_periods', 'n_dates', 'reason']``。
        ``kind='全缺'``：任何一期都没有样本（``mean`` 为 NaN）；
        ``kind='部分缺'``：只在部分期有样本（读数前先看 ``n_periods``）。
        排序上「全缺」在前 —— 那才是真正需要盯的格。
    dropped : DataFrame
        columns = ``['n_rows', 'reason']``：没进网格的观测数及原因
        （``by`` 缺失 / ``factor`` 缺失 / 该期样本不足分不出层 / 收益缺值）。
        **恒有** ``n_obs + dropped.n_rows.sum() == len(data)``（不静默丢样本）。
    grid : DataFrame
        原始明细：index = ``(date, q_by, q)``，columns = 各持有期 + ``count``。
        留证据用（想自己复核口径就查它）。
    by_name, n_by, n, method, n_obs, n_dates
        ``n_obs`` = 进入网格的观测数。
    """

    mean: pd.DataFrame
    count: pd.Series
    n_periods: pd.Series
    monotonicity: pd.DataFrame
    pooled_monotonicity: pd.Series
    missing: pd.DataFrame
    dropped: pd.DataFrame
    grid: pd.DataFrame
    by_name: str = 'by'
    n_by: int = 5
    n: int = 5
    method: str = 'independent'
    n_obs: int = 0
    n_dates: int = 0

    def to_frame(self, horizon=None):
        """``q_by × q`` 的**宽表**（平均收益），贴报告 / 画热力图用。"""
        cols = list(self.mean.columns)
        if not cols:
            return pd.DataFrame()
        h = cols[0] if horizon is None else horizon
        if h not in cols:
            fail('group', 'no_horizon',
                 f'horizon={h} 不在结果里；可用：{cols}')
        return self.mean[h].unstack('q')

    def __str__(self):
        miss = self.missing if self.missing is not None else pd.DataFrame()
        if len(miss) and 'kind' in miss.columns:
            vc = miss['kind'].value_counts()
            txt = (f"缺格 全缺 {int(vc.get('全缺', 0))} / "
                   f"部分缺 {int(vc.get('部分缺', 0))}")
        else:
            txt = '缺格 0'
        return (f'<DoubleSort {self.n_by}×{self.n} 宫格 · by={self.by_name} · '
                f'{self.method} · {self.n_dates} 期 · {self.n_obs:,} 条观测 · {txt}>')


def double_sort(data, *, by, n_by=5, n=5, method='independent',
                horizons=None):
    """**双重排序** —— ``by`` × ``q`` 的网格平均收益 + **组内**单调性。

    设计 §4.1 的可证伪预测：

    > 若"因子负向 = 规模混淆"成立，则**同一规模组内**因子的单调性应当消失。

    所以本函数的核心数字不是整体单调性，而是
    :attr:`DoubleSortResult.monotonicity` —— **每个 ``by`` 组内**、层序与收益的
    Spearman。整体单调性另给 ``pooled_monotonicity``（与 ``quantile_stats`` 同口径）。

    Parameters
    ----------
    data : DataFrame | CleanResult
        索引 ``(date, asset)``，含 ``factor`` 与 ``forward_return_*``。
    by : str | Series | DataFrame | Grouping
        控制变量（规模、行业…）。``str`` = 列名；**必须是数值**
        （要按它分 ``n_by`` 层）。⚠️ 同 :func:`grouped_ic`：**必须 as-of 对齐**，
        本函数不做回填。
    n_by : int
        ``by`` 的层数（默认 5，即先按规模分 5 组）。
    n : int
        因子的层数（默认 5 分位）。
    method : {'independent', 'conditional'}
        * ``'independent'``（默认）：因子分位点取自**每期全样本** ——
          两种排序相互独立，但各格样本数可能很不均；
        * ``'conditional'``：因子分位点取自 **``by`` 组内** ——
          每格样本均衡，干净回答"控制掉 by 之后因子还有没有区分度"。

        与 :func:`alphalens_cna.analysis.quantile.quantize` 的 ``method`` 同义，
        分位点也走**同一个** :func:`~alphalens_cna.analysis.quantile._quantile_within`
        （不另写一份 qcut，免得两种口径）。
    horizons : list[int], 可选

    Returns
    -------
    DoubleSortResult
        见 :class:`DoubleSortResult`。

    Notes
    -----
    ⚠️ **双重排序只能去掉"组间"那一部分混淆**。实测（200 只 × 40 期）：
    因子 = 0.9·规模 + 噪声、收益 = 0.9·规模 + 噪声（规模效应是**连续**的）时，
    池化单调性 1.000，而**组内**单调性最高仍到 1.000 —— 因为组内的规模差异还在，
    因子依旧是它的代理。要得到"组内单调性消失"，前提是控制变量的效应被分组
    **完全吸收**（例如收益只看规模**组**的阶梯值）。读组内单调性时先想清楚这一点。

    ⚠️ **缺格不填 0**：某格若从未出现过（如 ``conditional`` 下该 ``by`` 组每期
    样本不足 ``n`` 只），``mean`` 里是 NaN，并逐格记进 ``missing``（含原因）；
    只在部分期出现的格子记 ``kind='部分缺'``。
    另外 ``n_obs + dropped.n_rows.sum() == len(data)`` 恒成立 —— 一条样本的去向
    都能对上账。

    ⚠️ **0.4.0 起 ``double_sort`` 就是本函数**（原先 ``analysis/quantile.py`` 里那个
    返回 ``(date, q_by, q)`` 逐期立方的薄封装已删除）。与它不同的三处，迁移时注意：

    ==================  ==========================  ==============================
    维度                旧薄封装                     本函数
    ==================  ==========================  ==============================
    返回                ``(date, q_by, q)`` 立方      ``DoubleSortResult``
    ``method`` 默认      ``'conditional'``           ``'independent'``
    ``by``              只能 Series                  ``str`` 列名 / Series / 契约对象
    ==================  ==========================  ==============================

    要旧的逐期立方：取 :attr:`DoubleSortResult.grid`
    （index 就是 ``(date, q_by, q)``，columns = 各持有期 + ``count``）——
    但**默认 method 变了**，要 ``conditional`` 就得显式写出来。

    Examples
    --------
    >>> r = double_sort(data, by='ln_mv', n_by=5, n=5)
    >>> r.to_frame(21)                     # 5×5 宫格平均收益（21 日持有期）
    >>> r.monotonicity                     # 每个规模组内的单调性：混淆一验便知
    """
    df = _unwrap(data, 'group')
    if not isinstance(df, pd.DataFrame):
        fail('group', 'bad_input', f'需要 DataFrame，收到 {type(df).__name__}')
    _check_panel(df, 'group')
    if 'factor' not in df.columns:
        fail('group', 'no_factor',
             f'缺 `factor` 列；实际列：{list(df.columns)[:12]}')
    if not len(df):
        fail('group', 'no_obs', '一条有效观测都没有，做不了双重排序。')
    for nm, v in (('n_by', n_by), ('n', n)):
        if not isinstance(v, (int, np.integer)) or v < 2:
            fail('group', f'bad_{nm}',
                 f'{nm} 必须是 ≥ 2 的整数，收到 {v!r}。\n'
                 f'  原因：1 层等于没分层 —— "每层均值"退化成全样本均值，'
                 f'单调性也无从谈起。')
    if method not in ('independent', 'conditional'):
        fail('group', 'bad_method',
             f"method 只能是 'independent' / 'conditional'，收到 {method!r}")
    cols = return_cols(df, horizons)                 # 缺列时以 'ic' 契约报错

    key_frame, names = _key_frame(df, by, contract='group')
    if len(names) != 1:
        fail('group', 'by_ambiguous',
             f'双重排序的 by 只能有 **1 个**分组变量（这是二维网格），'
             f'收到 {len(names)} 个：{names}。\n'
             f'  修法：合成一个变量（如"行业×规模"标签），或分两次做。')
    by_name = names[0]
    b = key_frame[by_name]
    b_num = pd.to_numeric(b, errors='coerce')
    bad_num = b.notna() & b_num.isna()          # 缺失(NaN)是合法的，非数字不是
    if bad_num.any():
        fail('group', 'by_not_numeric',
             f'双重排序的 by=`{by_name}` 必须是**数值**（要按它分 {n_by} 层），'
             f'发现 {int(bad_num.sum())} 行不是数字（如 {b[bad_num].iloc[0]!r}）。\n'
             f'  修法：行业这类分类变量请先编码，或改用 grouped_ic(data, by=...)')
    if not b_num.notna().any():
        fail('group', 'by_all_nan', f'by=`{by_name}` 全是 NaN，分不了层。')

    # -- 打标签：q_by（by 的每期 n_by 层）与 q（因子每期 n 层）------------------ #
    # ⚠️ 直接复用 quantile 的 `_quantile_within`（同一套 qcut / 重复值 / 组内样本
    #    不足的口径）。**不能**用 quantize(..., by=q_by, method='conditional')：
    #    它内部会拿 n 把已经分好 n_by 层的标签**再分一次**，n_by ≠ n 时分层就错了。
    dates = np.asarray(pd.Index(df.index.get_level_values('date')))
    f = pd.to_numeric(df['factor'], errors='coerce')
    qb = _quantile_within(df.index, b_num, [dates], n_by) + 1
    if method == 'independent':
        q = _quantile_within(df.index, f, [dates], n) + 1
    else:
        q = _quantile_within(df.index, f, [dates, np.asarray(qb)], n) + 1
    qb[b_num.isna()] = np.nan               # 控制变量缺失 → 不参与分层
    q[qb.isna()] = np.nan                   # 没有 by 组，就谈不上"组内分位"
    labels = pd.DataFrame({'q_by': qb, 'q': q})
    grid = quantile_returns(df, quantiles=labels,
                            horizons=[horizon_of(c) for c in cols])
    gcols = [c for c in grid.columns if c != 'count']
    hcols = [horizon_of(c) for c in gcols]          # 对外一律用持有期（int）当列名

    # -- 宫格汇总（形状恒为 n_by × n，缺格 NaN）-------------------------------- #
    full = pd.MultiIndex.from_product([range(1, int(n_by) + 1),
                                       range(1, int(n) + 1)],
                                      names=['q_by', 'q'])
    gb = grid.groupby(level=['q_by', 'q'], sort=True)
    mean = gb[gcols].mean().reindex(full)
    mean.columns = hcols
    count = gb['count'].mean().reindex(full)
    nper = gb.size().reindex(full).fillna(0).astype(int)
    n_dates = int(grid.index.get_level_values('date').nunique())

    # -- 样本去向 + 缺格台账 --------------------------------------------------- #
    # 每行观测的 (date, q_by, q)；"在 grid 里"= 该格该期真的有有效收益。
    row_key = pd.MultiIndex.from_arrays(
        [np.asarray(pd.Index(df.index.get_level_values('date'))),
         np.asarray(qb), np.asarray(q)], names=['date', 'q_by', 'q'])
    in_grid = (np.asarray(row_key.isin(grid.index)) if len(grid)
               else np.zeros(len(df), dtype=bool))
    n_obs = int(in_grid.sum())
    avg_size = (float(pd.Series(1.0, index=df.index)
                      .groupby([dates, np.asarray(qb)]).size().mean())
                if qb.notna().any() else np.nan)

    present_qb = set(pd.Series(qb).dropna().unique())
    present_q = set(pd.Series(q).dropna().unique())
    recs = []
    for qbv, qv in full:
        npd = int(nper.loc[(qbv, qv)])
        if npd == n_dates and npd:
            continue
        if npd == 0:
            kind = '全缺'
            if qbv not in present_qb:
                why = (f'by 的第 {qbv} 层在数据里根本不存在 —— '
                       f'`_qcut` 因控制变量取值重复/样本不足，把 {n_by} 层合并成了 '
                       f'{len(present_qb)} 层（实际出现：{sorted(present_qb)}）。\n'
                       f'  修法：看 by 的取值分布，或减小 n_by。')
            elif not present_q:
                why = (f'一层都没分出来：{method} 分层要求每个 (期, by 组) 至少 '
                       f'{n} 只，实测平均只有 {avg_size:.1f} 只。\n'
                       f'  修法：减小 n，或看 dropped 台账里这些行的去向。')
            elif qv not in present_q:
                why = (f'因子的第 {qv} 层在数据里不存在 —— 因子取值重复过多，'
                       f'{n} 层被合并成了 {len(present_q)} 层'
                       f'（实际出现：{sorted(present_q)}）。')
            elif method == 'conditional':
                why = (f'({qbv}, {qv}) 在任何一期都没有样本：conditional 分层要求该 '
                       f'by 组内每期 ≥ {n} 只才能分出 {n} 层，'
                       f'该数据里每个 (期, by 组) 平均只有 {avg_size:.1f} 只。')
            else:
                why = (f'({qbv}, {qv}) 在任何一期都没有样本：'
                       f'该 (by 组, 因子层) 组合在这份数据的分布下不会出现'
                       f'（independent 允许各格样本不均）。')
        else:
            kind = '部分缺'
            why = (f'({qbv}, {qv}) 只在 {npd}/{n_dates} 期有样本：其余期该格为空'
                   f'（多为样本不足）。该格均值只用这 {npd} 期，读数前先看 n_periods')
        recs.append({'q_by': qbv, 'q': qv, 'kind': kind, 'n_periods': npd,
                     'n_dates': n_dates, 'reason': why})
    missing = pd.DataFrame(recs, columns=['q_by', 'q', 'kind', 'n_periods',
                                          'n_dates', 'reason'])
    if len(missing):
        # 「全缺」（均值是 NaN）排在前面 —— 那才是真正需要盯的格
        missing = (missing
                   .assign(_o=(missing['kind'] == '部分缺').astype(int))
                   .sort_values(['_o', 'q_by', 'q'])
                   .drop(columns='_o').reset_index(drop=True))

    # -- 没进网格的样本也要有去向 --------------------------------------------- #
    lab_ok = labels.notna().all(axis=1).to_numpy()
    m_by = b.isna().to_numpy()
    m_fac = (df['factor'].isna().to_numpy() & ~m_by)
    m_lab = (~lab_ok & ~m_by & ~m_fac)
    m_ret = (~m_by & ~m_fac & lab_ok & ~in_grid)
    dropped = pd.DataFrame([
        {'n_rows': int(m_by.sum()),
         'reason': f'by=`{by_name}` 缺失：没有控制变量就分不了 n_by 层'},
        {'n_rows': int(m_fac.sum()), 'reason': '`factor` 缺失'},
        {'n_rows': int(m_lab.sum()),
         'reason': f'该期该 by 组样本不足、或取值没有变化，分不出 {n} 层'
                   f'（分位点算不出来）'},
        {'n_rows': int(m_ret.sum()),
         'reason': '该格没有任何持有期的有效收益（未进入网格）'},
    ])

    # -- 组内单调性（与 quantile_stats.monotonicity 同算法、同池化口径）--------- #
    def _mono(y_by_q):
        """层序（q 标签）与平均收益的 Spearman；< 3 层 → NaN。"""
        ok = np.isfinite(y_by_q)
        if ok.sum() < 3:
            return np.nan
        x = np.asarray(y_by_q.index, dtype=float)[ok]
        y = np.asarray(y_by_q, dtype=float)[ok]
        return float(pd.Series(x).corr(pd.Series(y), method='spearman'))

    qs = list(full.get_level_values('q').unique())
    mrecs = []
    for c in gcols:
        # ALL 行 = 把 q_by 也池掉：与 quantile_stats 里的算法逐字一致
        pooled_layer = grid[c].groupby('q').mean().reindex(qs)
        mrecs.append({'q_by': 'ALL', 'horizon': horizon_of(c),
                      'spearman': _mono(pooled_layer)})
        by_layer = grid[c].groupby(level=['q_by', 'q']).mean().reindex(full)
        for qbv in full.get_level_values('q_by').unique():
            mrecs.append({'q_by': qbv, 'horizon': horizon_of(c),
                          'spearman': _mono(by_layer.loc[qbv].reindex(qs))})
    mono_all = pd.DataFrame(mrecs).set_index(['q_by', 'horizon'])
    pooled = mono_all.xs('ALL', level='q_by')['spearman']
    pooled.name = 'pooled_monotonicity'
    mono = mono_all.drop(index='ALL', level='q_by')

    return DoubleSortResult(
        mean=mean, count=count, n_periods=nper, monotonicity=mono,
        pooled_monotonicity=pooled, missing=missing, dropped=dropped,
        grid=grid, by_name=by_name, n_by=int(n_by), n=int(n), method=method,
        n_obs=n_obs, n_dates=n_dates)
