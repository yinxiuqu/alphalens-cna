"""L3 预处理层 —— 去极值 / 标准化 / 中性化 / 正交化 / 合成。

**本层只改因子值，不动样本。** 行数进出必须一致 ——
任何"顺手删几行"的行为都是清洗层（``engine.clean``）该干的事，
在这里偷偷删等于把样本账做花。

三条铁律
--------
1. **逐期截面做**（``by_date=True`` 默认）。因子分析全部结论都以"截面"为单位，
   混着全样本做标准化会把时间趋势也一起抹掉。
2. **NaN 进、NaN 出**。缺失不动、不填。要填是使用者的决定，不是预处理偷偷干的。
3. **每一处修改都留痕**（``.attrs['preprocess']``）：改了多少行、跳过几个截面。

顺序（经验默认，不是定理）
--------------------------
``winsorize → standardize → neutralize/orthogonalize → combine``

* 先winsorize：极值会把 OLS 中性化的斜率整个拽偏
* 再标准化：让不同量纲的因子可比
* 中性化放在标准化之后：残差的尺度才稳定
* 合成放最后：合成后的分布还要再标准化一次（``combine`` 内部会做）

用法
----
>>> f = acna.winsorize(factor, method='mad', n=3)
>>> f = acna.standardize(f, method='zscore')
>>> f = acna.neutralize(f, exposures=exposures, groups=groups)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail

__all__ = ['winsorize', 'standardize', 'neutralize', 'orthogonalize', 'combine',
           'preprocess_log']

WINSOR_METHODS = ('mad', 'sigma', 'quantile')
STD_METHODS = ('zscore', 'rank', 'demean')

MAD_TO_SIGMA = 1.4826          # 正态下 MAD → σ 的换算常数


# --------------------------------------------------------------------------- #
def _series(obj, name='value'):
    """从 Series / DataFrame / 清洗结果里取出因子值，并记住怎么装回去。"""
    if isinstance(obj, pd.Series):
        return obj, None
    if isinstance(obj, pd.DataFrame):
        col = name if name in obj.columns else (
            'factor' if 'factor' in obj.columns else obj.columns[0])
        return obj[col], col
    df = getattr(obj, 'data', None)
    if isinstance(df, pd.DataFrame):
        col = 'factor' if 'factor' in df.columns else df.columns[0]
        return df[col], col
    fail('preprocess', 'bad_input',
         f'需要 Series / DataFrame / 带 .data 的结果对象，收到 {type(obj).__name__}')


def _put_back(obj, s, col):
    """把处理后的列装回原对象，并**继承上一步的痕迹**。

    ⚠️ pandas 的 ``attrs`` 不会穿过 ``groupby.apply`` ——
    不显式继承的话，第二步会把第一步的记录冲掉（实测踩过）。
    """
    if col is None:
        out = s
    elif isinstance(obj, pd.DataFrame):
        out = obj.copy()
        out[col] = s
    else:
        out = getattr(obj, 'data').copy()
        out[col] = s
    prev = getattr(obj, 'attrs', None) or {}
    out.attrs.update(prev)
    return out


def _check_index(s):
    if not isinstance(s.index, pd.MultiIndex) or \
            list(s.index.names[:2]) != ['date', 'asset']:
        fail('preprocess', 'index',
             '需要 MultiIndex(date, asset)；提示：df.set_index(["date","asset"])')


def _log(obj, **kv):
    """把这一步的痕迹记到 ``attrs`` 上（原地累加）。"""
    a = getattr(obj, 'attrs', None)
    if a is None:
        return
    a.setdefault('preprocess', []).append(kv)


def preprocess_log(obj):
    """读回预处理痕迹。"""
    a = getattr(obj, 'attrs', {}) or {}
    rows = a.get('preprocess', [])
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['step', 'n_rows', 'n_changed', 'n_skipped_sections'])


# --------------------------------------------------------------------------- #
def winsorize(factor, method='mad', n=3.0, limits=(0.01, 0.99), by_date=True):
    """去极值（截面）。

    Parameters
    ----------
    method : {'mad', 'sigma', 'quantile'}
        ``'mad'``（默认）—— 中位数 ± ``n`` × 1.4826 × MAD。**对极值本身不敏感**，
        这是它比 ``sigma`` 好的地方（σ 会被要处理的极值撑大）。
        ``'sigma'`` —— 均值 ± ``n`` × 标准差。
        ``'quantile'`` —— 直接截到 ``limits`` 分位。
    n : float
        ``mad`` / ``sigma`` 的倍数。
    limits : (float, float)
        分位法用的上下限。

    Returns
    -------
    与输入同型（Series 进 Series 出）。**行数不变**，只把越界的值拉回边界。
    """
    if method not in WINSOR_METHODS:
        fail('preprocess', 'bad_method', f'method 只能是 {WINSOR_METHODS}')
    s, col = _series(factor)
    _check_index(s)

    def _one(g):
        x = g.to_numpy(dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 2:
            return g
        v = x[ok]
        if method == 'quantile':
            lo, hi = np.quantile(v, limits[0]), np.quantile(v, limits[1])
        elif method == 'sigma':
            m, sd = v.mean(), v.std(ddof=1)
            lo, hi = m - n * sd, m + n * sd
        else:
            med = np.median(v)
            mad = np.median(np.abs(v - med)) * MAD_TO_SIGMA
            if mad == 0:
                return g                    # 半数列相同 → 不动，别把因子压成常数
            lo, hi = med - n * mad, med + n * mad
        out = x.copy()
        out[ok] = np.clip(v, lo, hi)
        return pd.Series(out, index=g.index)

    res = (s.groupby(level='date', group_keys=False).apply(_one)
           if by_date else _one(s))
    res = res.reindex(s.index) if by_date else res
    n_changed = int((np.asarray(res, dtype=float) != np.asarray(s, dtype=float)
                     ).sum())
    out = _put_back(factor, res, col)
    _log(out, step=f'winsorize({method},n={n})', n_rows=len(s),
         n_changed=n_changed, n_skipped_sections=0)
    return out


# --------------------------------------------------------------------------- #
def standardize(factor, method='zscore', by_date=True):
    """标准化（截面）。

    ``'zscore'`` 减均值除标准差；``'rank'`` 转成 ``[0,1]`` 均匀分数
    （**对极值免疫**，稳健性最好）；``'demean'`` 只减均值。
    """
    if method not in STD_METHODS:
        fail('preprocess', 'bad_method', f'method 只能是 {STD_METHODS}')
    s, col = _series(factor)
    _check_index(s)

    def _one(g):
        x = g.to_numpy(dtype=float)
        ok = np.isfinite(x)
        if ok.sum() < 2:
            return g
        out = x.copy()
        if method == 'zscore':
            sd = x[ok].std(ddof=1)
            if sd == 0:
                return g
            out[ok] = (x[ok] - x[ok].mean()) / sd
        elif method == 'demean':
            out[ok] = x[ok] - x[ok].mean()
        else:
            r = pd.Series(x[ok]).rank(method='average').to_numpy()
            out[ok] = (r - 0.5) / len(r)
        return pd.Series(out, index=g.index)

    res = (s.groupby(level='date', group_keys=False).apply(_one)
           if by_date else _one(s))
    res = res.reindex(s.index) if by_date else res
    out = _put_back(factor, res, col)
    _log(out, step=f'standardize({method})', n_rows=len(s),
         n_changed=int(np.isfinite(np.asarray(res, dtype=float)).sum()),
         n_skipped_sections=0)
    return out


# --------------------------------------------------------------------------- #
def _residualize(s, X, min_obs=10):
    """逐期对 ``X`` 回归取残差。``X`` 是 DataFrame（index 与 s 相同）。"""
    rows, skipped = [], 0
    dates = s.index.get_level_values('date')
    Xv = np.asarray(X, dtype=float)
    sv = np.asarray(s, dtype=float)
    for d, pos in pd.Series(np.arange(len(s)), index=dates).groupby(level=0,
                                                                   sort=True):
        p = np.asarray(pos)
        y, A = sv[p], Xv[p]
        ok = np.isfinite(y) & np.isfinite(A).all(axis=1)
        if ok.sum() < max(min_obs, A.shape[1] + 1):
            rows.append(np.full(len(p), np.nan))
            skipped += 1
            continue
        M = np.column_stack([np.ones(ok.sum()), A[ok]])
        beta, *_ = np.linalg.lstsq(M, y[ok], rcond=None)
        r = np.full(len(p), np.nan)
        r[ok] = y[ok] - M @ beta
        rows.append(r)
    return pd.Series(np.concatenate(rows), index=s.index), skipped


def neutralize(factor, exposures=None, groups=None, min_obs=10):
    """中性化 —— 剥掉风格暴露与行业分组，取残差。

    Parameters
    ----------
    exposures : DataFrame | list[DataFrame], optional
        风格暴露（市值对数、Beta、动量…），索引同 ``factor``。
    groups : Series | DataFrame, optional
        分组标签（行业）。给了就走**组内去均值**（简单行业中性化）；
        否则走对 ``exposures`` 的 OLS 残差。

    Returns
    -------
    与输入同型。**行数不变**；样本不足的截面整段置 NaN 并在痕迹里记数。
    """
    s, col = _series(factor)
    _check_index(s)
    if groups is not None:
        g = (groups['group'] if isinstance(groups, pd.DataFrame) else groups)
        g = g.reindex(s.index)
        if g.isna().all():
            fail('preprocess', 'bad_groups',
                 'groups 与 factor 的索引对不上（reindex 后全空）')
        gm = s.groupby([s.index.get_level_values('date'), g]).transform('mean')
        res = s - gm
        n_skip = int(res.isna().sum() > 0 and 0 or 0)
        out = _put_back(factor, res, col)
        _log(out, step='neutralize(group)', n_rows=len(s),
             n_changed=int(res.notna().sum()), n_skipped_sections=n_skip)
        return out

    if exposures is None:
        fail('preprocess', 'no_exposure',
             '至少要给 `exposures`（风格）或 `groups`（行业）之一')
    X = exposures if isinstance(exposures, pd.DataFrame) else pd.concat(
        [e if isinstance(e, pd.DataFrame) else e.to_frame() for e in exposures],
        axis=1)
    X = X.reindex(s.index)
    res, skipped = _residualize(s, X, min_obs=min_obs)
    out = _put_back(factor, res, col)
    _log(out, step='neutralize(ols)', n_rows=len(s),
         n_changed=int(res.notna().sum()), n_skipped_sections=skipped)
    return out


def orthogonalize(factor, others, min_obs=10):
    """正交化 —— 把 ``others`` 里已被解释掉的部分从因子里剔除。

    与 :func:`neutralize` 的区别只是**语义**：中性化剥的是"风险暴露"，
    正交化剥的是"另一个因子"。实现相同（逐期 OLS 取残差），
    分开命名是为了让调用处读起来就是研究意图。
    """
    return neutralize(factor, exposures=others, min_obs=min_obs)


# --------------------------------------------------------------------------- #
def combine(factors, weights=None, standardize_first=True, min_obs=10):
    """多因子合成 —— 先各自截面标准化，再按权重相加，最后再标准化一次。

    Parameters
    ----------
    factors : list[Series | DataFrame]
    weights : list[float], optional
        默认等权。**会自动归一化到和=1**（不要求使用者自己算）。
    standardize_first : bool
        默认 ``True``：量纲不同的因子直接相加是可笑的（一个以"元"为单位、
        一个以"%"为单位），必须先各自标准化。

    Returns
    -------
    Series
        合成后的因子（已标准化），索引与输入一致。
    """
    if not factors:
        fail('preprocess', 'no_factors', 'factors 不能为空')
    if weights is None:
        w = np.full(len(factors), 1.0 / len(factors))
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(factors):
            fail('preprocess', 'bad_weights',
                 f'weights 有 {len(w)} 个，factors 有 {len(factors)} 个')
        if not np.isfinite(w).all() or w.sum() == 0:
            fail('preprocess', 'bad_weights', 'weights 必须有限且和不为 0')
        w = w / w.sum()

    parts = []
    for f in factors:
        s, _ = _series(f)
        _check_index(s)
        parts.append(standardize(s, 'zscore') if standardize_first else s)
    base = parts[0].index
    M = pd.concat([p.reindex(base) for p in parts], axis=1)
    comp = (M * w).sum(axis=1, min_count=len(parts) - 1)   # 允许个别因子缺失
    out = standardize(comp, 'zscore')
    out.name = 'value'
    _log(out, step='combine', n_rows=len(out),
         n_changed=int(out.notna().sum()), n_skipped_sections=0)
    return out
