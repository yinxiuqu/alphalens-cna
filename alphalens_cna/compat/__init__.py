"""等价性回归（**六道防线 · 第 4 条**）—— 与 alphalens 对拍。

为什么只对拍**三个量**
----------------------
alphalens 有已知缺陷，**我们明确不复刻**：

* **D1** 非日频因子索引直接崩（``freq`` 校验）
* **D2** 月度面板把列名标成 ``1D``
* **D9** 因子日期不落在价格索引的频率网格上时，``get_clean_factor_and_forward_returns``
  **在清洗阶段就抛 ValueError**（``alphalens/utils.py:358`` 硬写
  ``df.index.levels[0].freq = freq``，pandas 拒绝给不规则索引设频率）。
  真实 A 股月频面板（每月最后一个交易日，``freq is None``）**根本跑不起来** ——
  比"返回 NaN"严重得多。详见 ``outputs/换手口径与D9核实.md``。

所以"全链路 1e-10 一致"既不可能也不该追求 —— 那等于把 bug 一起复刻。
本模块只在**退化场景**下对拍**三个核心量**：

1. **IC** —— ``factor_information_coefficient``
2. **分层均值** —— ``mean_return_by_quantile``
3. **换手** —— ``quantile_turnover``

退化场景 = **日频 + 收盘成交 + 关闭全部 A 股规则**。
这正是 alphalens 唯一能算对的场景，也正是"证明引擎没写错"的基准。

**先把新库调成和 alphalens 一模一样，再逐步打开 A 股规则，
每一处差异都能被逐条归因。解释不了的差异就是 bug。**
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contract.errors import ContractError, fail

__all__ = ['ParityCheck', 'ParityReport', 'check_parity', 'alphalens_available']

DEFAULT_TOL = 1e-10


def alphalens_available():
    """alphalens 是否装了（对拍是**可选**依赖）。"""
    try:
        import alphalens  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class ParityCheck:
    """一个量的对拍结果。"""

    name: str
    ok: bool
    max_diff: float = np.nan
    tol: float = DEFAULT_TOL
    n_compared: int = 0
    detail: str = ''

    def __str__(self):
        flag = '✅' if self.ok else '❌'
        d = f'{self.max_diff:.3e}' if np.isfinite(self.max_diff) else '—'
        return (f'{flag} {self.name:<14} 最大差 {d:<12} '
                f'(容差 {self.tol:.0e}, 比对 {self.n_compared:,} 点)'
                + (f'\n     {self.detail}' if self.detail else ''))


@dataclass
class ParityReport:
    """对拍总报告。"""

    checks: list = field(default_factory=list)
    n_ours: int = 0
    n_alphalens: int = 0
    freq: str = '日频'
    notes: list = field(default_factory=list)

    @property
    def ok(self):
        return all(c.ok for c in self.checks) if self.checks else False

    def __str__(self):
        lines = [f'与 alphalens 等价性回归（退化场景：{self.freq} + 收盘成交 + 关 A 股规则）',
                 f'  样本：本库 {self.n_ours:,} / alphalens {self.n_alphalens:,}']
        lines += ['  ' + str(c) for c in self.checks]
        lines.append(f'  结论：{"✅ 通过" if self.ok else "❌ 未通过"}')
        for n in self.notes:
            lines.append(f'  注：{n}')
        return '\n'.join(lines)


# --------------------------------------------------------------------------- #
def check_parity(factor, prices, calendar, horizons=(1, 5, 21), quantiles=5,
                 tol=DEFAULT_TOL, checks=('ic', 'quantile', 'turnover')):
    """与 alphalens 对拍三个核心量。

    Parameters
    ----------
    factor : FactorPanel | DataFrame | Series
        ``MultiIndex(date, asset)`` 的因子值。**需要是日频** ——
        非日频下 alphalens 会崩或返 NaN，那不是我们该复刻的行为。
    prices : PricePanel | DataFrame
        需要 ``adj_close``（退化场景一律用复权收盘价）。
    calendar : Calendar
    horizons : tuple[int]
    quantiles : int
    tol : float
        数值容差，默认 ``1e-10``。

    Returns
    -------
    ParityReport
    """
    if not alphalens_available():
        fail('compat', 'alphalens_missing',
             '没装 alphalens，无法对拍。\n'
             '  pip install alphalens-reloaded —— 它是**可选**依赖，'
             '只在跑对拍时需要，运行本库本身不需要。')

    from alphalens import performance as aperf
    from alphalens import utils as autils

    from ..analysis import information_coefficient, quantile_returns, quantize
    from ..engine.clean import clean
    from ..engine.returns import ReturnModel, forward_returns

    rep = ParityReport()
    px = getattr(prices, 'df', prices)
    f = _factor_series(factor)
    rep.freq = _freq_label(calendar)

    # ── 我们的：退化配置 ────────────────────────────────────────────
    model = ReturnModel(entry='close', exit_price='close')   # = alphalens 口径
    r = forward_returns(px, calendar, list(horizons), model=model)
    cr = clean(f, r, name='factor')
    rep.n_ours = len(cr.data)
    our_q = quantize(cr, n=quantiles)

    # ── alphalens 的 ────────────────────────────────────────────────
    # 它自己会崩/返 NaN 的地方（D1/D9）不该让本库漏出第三方堆栈：
    # 捕获后如实记录，并把三个检查标成"无法对拍"。
    wide = px['adj_close'].unstack('asset')
    try:
        af = autils.get_clean_factor_and_forward_returns(
            f, wide, quantiles=quantiles, periods=tuple(horizons), max_loss=1.0)
    except Exception as e:                                   # noqa: BLE001
        rep.notes.append(
            f'alphalens 自己抛异常，无法对拍：{type(e).__name__}: {e}\n'
            f'  这通常是 D1（非日频索引）。本库在同输入下正常返回 '
            f'{rep.n_ours:,} 行。')
        rep.n_alphalens = 0
        for nm in checks:
            rep.checks.append(ParityCheck(
                _CHECK_NAME.get(nm, nm), False, np.nan, tol, 0,
                'alphalens 崩溃，无从比对'))
        return rep
    rep.n_alphalens = len(af)

    if rep.n_ours != rep.n_alphalens:
        rep.notes.append(
            f'清洗后样本数不同（本库 {rep.n_ours:,} vs alphalens {rep.n_alphalens:,}，'
            f'差 {rep.n_ours - rep.n_alphalens:+,}）—— '
            f'下面只在**共同样本**上比对数值')

    # 共同样本不必在这里切：`_cmp` 内部就是 join='inner'（见下方各检查）。

    # ── ① IC ────────────────────────────────────────────────────────
    if 'ic' in checks:
        rep.checks.append(_guard(
            'IC', tol,
            lambda: _cmp('IC', _norm_cols(information_coefficient(cr)),
                         _norm_cols(aperf.factor_information_coefficient(af)), tol)))

    # ── ② 分层均值 ──────────────────────────────────────────────────
    if 'quantile' in checks:

        def _q():
            ours = quantile_returns(cr, quantiles=our_q)
            ours = _norm_cols(ours[[c for c in ours.columns if c != 'count']])
            theirs, _ = aperf.mean_return_by_quantile(af, by_date=True, demeaned=False)
            theirs = _norm_cols(theirs)
            theirs = theirs.reorder_levels(['date', 'factor_quantile']).sort_index()
            theirs.index = theirs.index.set_names(['date', 'q'])
            ours.index = ours.index.set_names(['date', 'q'])
            return _cmp('分层均值', ours, theirs, tol)

        rep.checks.append(_guard('分层均值', tol, _q))

    # ── ③ 换手 ──────────────────────────────────────────────────────
    if 'turnover' in checks:

        def _t():
            ours = _our_turnover(our_q, quantiles)
            theirs = {q: aperf.quantile_turnover(af['factor_quantile'], q, period=1)
                      for q in range(1, quantiles + 1)}
            theirs = pd.DataFrame(theirs)
            theirs.columns.name = None
            return _cmp('换手', ours, theirs, tol)

        rep.checks.append(_guard('换手', tol, _t))

    return rep


# --------------------------------------------------------------------------- #
def _norm_cols(df):
    """列名 → 整数持有期。

    alphalens 的列名形如 ``'1D'`` / ``'5D'`` / ``'21D'`` —— 注意这是 D2：
    **月度面板它也会标成 ``1D``**（因为它拿不到真实频率）。
    这里只做数值解析，不改语义。
    """
    if df is None or not len(df.columns):
        return df
    out = df.copy()
    out.columns = [_digits(c) for c in out.columns]
    return out


def _digits(c):
    s = ''.join(ch for ch in str(c) if ch.isdigit())
    return int(s) if s else str(c)


_CHECK_NAME = {'ic': 'IC', 'quantile': '分层均值', 'turnover': '换手'}


def _freq_label(calendar):
    """按日历的实际间隔给人话频率名（报告标题用）。"""
    try:
        idx = getattr(calendar, 'index', calendar)
        d = pd.DatetimeIndex(idx)
        if len(d) < 3:
            return '日频'
        days = np.diff(d.values).astype('timedelta64[D]').astype(float)
        gap = float(np.median(days))
    except Exception:                                        # noqa: BLE001
        return '日频'
    if gap <= 4:
        return '日频'
    if gap <= 9:
        return '周频'
    if gap <= 45:
        return '月频'
    return f'约 {gap:.0f} 天/期'


def _guard(name, tol, fn):
    """跑一个对拍检查，但别把异常变成堆栈。

    本库自己的契约违规（``ContractError``）照常抛出 —— 那是我们的 bug，
    必须炸；只有 alphalens 一侧的问题才降级成"无法对拍"。
    """
    try:
        return fn()
    except ContractError:
        raise
    except Exception as e:                                   # noqa: BLE001
        return ParityCheck(name, False, np.nan, tol, 0,
                           f'alphalens 一侧抛异常，无从比对：'
                           f'{type(e).__name__}: {e}')


def _our_turnover(q, quantiles):
    """本库的逐分位换手，整理成与 alphalens 同形（index=date, columns=1..n）。"""
    from ..analysis import quantile_turnover
    to = quantile_turnover(q['q'])
    to.columns = [int(c) for c in to.columns]
    return to


def _cmp(name, ours, theirs, tol):
    """比两个 DataFrame/Series。"""
    if ours is None or theirs is None or not len(ours) or not len(theirs):
        return ParityCheck(name, False, np.nan, tol, 0, '一方为空，无法比对')
    a, b = ours.align(theirs, join='inner')
    if not len(a):
        return ParityCheck(name, False, np.nan, tol, 0, '索引无交集')
    av = a.to_numpy(dtype=float)
    bv = b.to_numpy(dtype=float)
    m = np.isfinite(av) & np.isfinite(bv)
    if not m.any():
        return ParityCheck(name, False, np.nan, tol, 0,
                           '共同有效值全为 NaN —— alphalens 在非日频下会这样（D9）')
    diff = np.abs(av[m] - bv[m])
    md = float(diff.max())
    ok = md <= tol
    detail = ''
    if not ok:
        i = int(np.argmax(diff))
        detail = (f'最大差出现在第 {i} 个点：本库 {av[m][i]:.12g} '
                  f'vs alphalens {bv[m][i]:.12g}')
    nan_a = int((~np.isfinite(av)).sum())
    nan_b = int((~np.isfinite(bv)).sum())
    if nan_a != nan_b:
        detail += (f'；NaN 数不同（本库 {nan_a} vs alphalens {nan_b}）'
                   f'—— 这本身可能正是 D9')
    return ParityCheck(name, ok, md, tol, int(m.sum()), detail)


def _factor_series(factor):
    """因子 → ``MultiIndex(date, asset)`` 的 Series（列名 ``factor``）。"""
    if isinstance(factor, pd.Series):
        s = factor
    else:
        df = getattr(factor, 'df', factor)
        for c in ('factor', 'value'):
            if c in df.columns:
                s = df[c]
                break
        else:
            cand = [c for c in df.columns
                    if c not in ('available_at', 'date', 'asset')]
            if len(cand) != 1:
                fail('compat', 'ambiguous_factor',
                     f'看不出哪列是因子值（候选 {cand}）；命名为 value 或 factor')
            s = df[cand[0]]
    s = s.copy()
    s.name = 'factor'
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    return s
