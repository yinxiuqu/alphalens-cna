"""事件研究 —— 事件日前后对齐的收益路径与漂移检验。

和因子分析的关系
----------------
因子分析问"因子值高的票后面涨不涨"；事件研究问
"**某件事发生之后，股价怎么走**"（PEAD、定增、解禁、指数纳入…）。
两者共用同一套价格与日历，区别只是**对齐方式**：按事件日对齐，而不是按调仓日。

核心量
------
* ``ret``     —— 相对第 j 日的当日收益（收盘对收盘）
* ``cum_ret`` —— 从窗口起点累乘到 j 的原始累计收益
* ``car``     —— 相对 ``base_day`` 的累计收益（``car[base_day] = 0``）
* ``abn_ret`` —— 减掉基准同窗收益后的**异常收益**（给了 ``benchmark`` 才有）

``base_day`` 默认 **0**（事件日收盘为基准）→ ``car[+1..+N]`` 就是**公告后的漂移**。
想连公告当天一起算就传 ``base_day=-1``。

⚠️ 事件聚集时朴素 t 会虚高（本库的立场）
----------------------------------------
同一段时间里密集发生的事件（比如财报季），其窗口互相重叠，
**横截面 t 检验会严重高估显著性** —— 这正是 alphalens 一类工具的常见隐患。

所以 :func:`event_summary` 同时给两个：

* ``t_naive`` —— 横截面对 CAR 做 t 检验。假设事件独立，**财报季下不成立**。
* ``t_ct``    —— **日历时间组合法**：把"当天所有事件的异常收益均值"当成一条
  时间序列（每个事件每天最多贡献一个观测），再用
  :func:`~alphalens_cna.inference.newey_west.nw_tstat` 给 t。
  这是事件聚集下的标准做法，**结论请看这个**。

用法
----
>>> ev = acna.Events(df)                      # index=(date, asset)，列含 event_type
>>> w = align_event_windows(prices, ev, calendar, window=(-5, 20))
>>> print(w.summary())
>>> w.df                                      # tidy：index=(event_id, rel_day)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contract.errors import fail
from ..inference.newey_west import nw_tstat

__all__ = ['EventWindows', 'align_event_windows', 'event_summary',
           'event_path_by_group']

DEFAULT_WINDOW = (-5, 20)


@dataclass
class EventWindows:
    """对齐后的事件窗口。

    Attributes
    ----------
    df : DataFrame
        tidy 长表，``index=(event_id, rel_day)``；列见模块文档。
    ledger : dict
        带原因的账：``n_events`` / ``kept`` / ``not_on_calendar`` /
        ``missing_price`` / ``no_benchmark``。
    window : tuple[int, int]
    base_day : int
    """

    df: pd.DataFrame
    ledger: dict = field(default_factory=dict)
    window: tuple = DEFAULT_WINDOW
    base_day: int = 0
    benchmark: str | None = None

    def __len__(self):
        if not len(self.df):
            return 0
        return int(self.df.index.get_level_values('event_id').nunique())

    def summary(self, lags=None):
        return event_summary(self, lags=lags)

    def __str__(self):
        led = self.ledger
        return (f'事件窗口（{led.get("kept", 0):,}/{led.get("n_events", 0):,} 个事件'
                f'被保留，相对日 {self.window[0]} ~ {self.window[1]}，'
                f'基准日 {self.base_day}）\n'
                f'  剔除账：' + '、'.join(f'{k} {v:,}' for k, v in led.items()
                                         if k not in ('n_events', 'kept') and v)
                or '  剔除账：（无）')


def _as_df(obj):
    return getattr(obj, 'df', obj)


# --------------------------------------------------------------------------- #
def align_event_windows(prices, events, calendar, *, window=DEFAULT_WINDOW,
                        base_day=0, benchmark=None, price_col='adj_close'):
    """把每个事件日前后的收益窗口**对齐**到共同相对日。

    Parameters
    ----------
    prices : PricePanel | DataFrame
        ``(date, asset)`` 索引，需含 ``price_col``（默认 ``adj_close``）。
    events : Events | DataFrame
        ``(date, asset)`` 索引，事件日当天一行。其它列（如 ``surprise``）
        会被原样带到输出里，便于分组。
    calendar : Calendar
        交易日历。相对日走它，不用 pandas 的 freq。
    window : (int, int)
        ``(before, after)``，相对事件日的交易日区间（含两端）。
    base_day : int
        ``car`` 的基准相对日，必须落在 ``window`` 内。默认 0。
    benchmark : str | DataFrame, optional
        基准：给列名（``prices`` 里的某个 asset）或独立的 Series/DataFrame。
        给了就算 ``abn_ret`` 与基于异常的 ``car``。
    price_col : str

    Returns
    -------
    EventWindows
    """
    px = _as_df(prices)
    ev = _as_df(events)
    if price_col not in px.columns:
        fail('event', 'missing_price',
             f'价格面板缺 `{price_col}`。事件研究一律用**复权价** —— '
             f'原始价在除权日会有假跳变，会把事件效应整个盖掉。')
    lo, hi = int(window[0]), int(window[1])
    if lo > hi:
        fail('event', 'bad_window', f'window 必须 (前, 后)，收到 {window}')
    if not (lo <= base_day <= hi):
        fail('event', 'bad_base_day',
             f'base_day={base_day} 必须落在 window={window} 内')

    cal = pd.DatetimeIndex(getattr(calendar, 'index', calendar))
    # ⚠️ 必须显式 ``fill_method=None``：pandas 的 pct_change 默认**前值填充**，
    #    会把停牌日伪装成"0 收益"，于是窗口看着完整、其实中间停了两周。
    #    置 None 后停牌 → NaN → 该事件按 `missing_price` 剔除，如实记账。
    ret = px[price_col].groupby(level='asset').pct_change(fill_method=None)
    pos_of = pd.Series(np.arange(len(cal)), index=cal)

    bret = None
    bname = None
    if benchmark is not None:
        if isinstance(benchmark, str):
            bname = benchmark
            if benchmark not in px.index.get_level_values('asset'):
                fail('event', 'bad_benchmark',
                     f'基准 {benchmark!r} 不在价格面板里')
            bser = px[price_col].xs(benchmark, level='asset')
        else:
            bser = pd.Series(np.asarray(benchmark).ravel(),
                             index=pd.DatetimeIndex(
                                 getattr(benchmark, 'index', cal)))
        bret = bser.sort_index().pct_change().reindex(cal)

    rel_days = np.arange(lo, hi + 1)
    rows, keep, dropped = [], 0, {'not_on_calendar': 0, 'missing_price': 0,
                                  'no_benchmark': 0}
    ev_frame = ev.reset_index()
    date_col = 'date' if 'date' in ev_frame.columns else ev_frame.columns[0]
    asset_col = 'asset' if 'asset' in ev_frame.columns else ev_frame.columns[1]
    carry = [c for c in ev_frame.columns if c not in (date_col, asset_col)]
    idx_ret = ret.index

    for i, r in enumerate(ev_frame.itertuples(index=False)):
        d, a = getattr(r, date_col), getattr(r, asset_col)
        p = pos_of.get(pd.Timestamp(d), np.nan)
        if not np.isfinite(p):
            dropped['not_on_calendar'] += 1
            continue
        p = int(p)
        if p + lo < 0 or p + hi >= len(cal):
            dropped['not_on_calendar'] += 1
            continue
        keys = pd.MultiIndex.from_arrays(
            [cal[p + lo:p + hi + 1], [a] * len(rel_days)], names=['date', 'asset'])
        rr = ret.reindex(keys).to_numpy(dtype=float)
        if not np.isfinite(rr).all():
            dropped['missing_price'] += 1
            continue
        br = (bret.reindex(cal[p + lo:p + hi + 1]).to_numpy(dtype=float)
              if bret is not None else None)
        if br is not None and not np.isfinite(br).all():
            dropped['no_benchmark'] += 1
            continue
        abn = rr - br if br is not None else np.full(len(rr), np.nan)
        base_pos = int(base_day - lo)          # base_day 在窗口数组里的下标
        cum = np.cumprod(1 + rr) - 1
        car = cum - cum[base_pos]
        if br is not None:
            cum_a = np.cumprod(1 + abn) - 1
            car = cum_a - cum_a[base_pos]
        extra = {c: getattr(r, c) for c in carry}
        for j, rd in enumerate(rel_days):
            rows.append({
                'event_id': i, 'rel_day': rd, 'asset': a,
                'event_date': pd.Timestamp(d), 'rel_date': cal[p + rd],
                'ret': rr[j], 'abn_ret': abn[j], 'cum_ret': cum[j],
                'car': car[j], **extra,
            })
        keep += 1

    if not rows:
        fail('event', 'no_events',
             f'一个事件都没对齐上。剔除账：{dropped}。'
             f'常见原因：事件日不在日历上，或窗口越界/该股窗口内停牌。')
    df = pd.DataFrame(rows).set_index(['event_id', 'rel_day']).sort_index()
    led = {'n_events': int(len(ev_frame)), 'kept': keep, **dropped}
    return EventWindows(df=df, ledger=led, window=(lo, hi), base_day=base_day,
                        benchmark=bname)


# --------------------------------------------------------------------------- #
def event_summary(windows, *, lags=None, value='car'):
    """逐相对日的汇总：均值、中位、胜率、朴素 t、**日历时间 t**。

    Parameters
    ----------
    windows : EventWindows | DataFrame
    lags : int, optional
        日历时间序列的 Newey-West 滞后阶数；不给按 ``auto_lags`` 自动定。
    value : {'car', 'abn_ret', 'ret'}
        汇总哪个量。默认 ``car``（累计异常/绝对收益）。

    Returns
    -------
    DataFrame
        index = ``rel_day``；columns = ``n_events`` / ``mean`` / ``median`` /
        ``hit_rate`` / ``std`` / ``t_naive`` / ``t_ct`` / ``n_days``。
    """
    df = windows.df if isinstance(windows, EventWindows) else windows
    if value not in df.columns:
        fail('event', 'bad_value',
             f'`{value}` 不在窗口表里；可选 {sorted(set(df.columns))}')
    rows = {}
    for rd, g in df.groupby(level='rel_day'):
        x = g[value].dropna().to_numpy(dtype=float)
        if not len(x):
            continue
        sd = float(x.std(ddof=1)) if len(x) > 1 else np.nan
        t_naive = (float(x.mean() / (sd / np.sqrt(len(x))))
                   if np.isfinite(sd) and sd > 0 else np.nan)
        # ── 日历时间组合法：把"当天所有事件的平均值"当成一条时间序列 ──
        dser = g.reset_index().groupby('rel_date')[value].mean().sort_index()
        st = nw_tstat(dser.to_numpy(dtype=float), lags=lags) if len(dser) >= 3 \
            else {'t_nw': np.nan, 'lags': 0}
        rows[int(rd)] = {
            'n_events': int(len(x)), 'mean': float(x.mean()),
            'median': float(np.median(x)), 'hit_rate': float((x > 0).mean()),
            'std': sd, 't_naive': t_naive,
            't_ct': st['t_nw'], 'n_days': int(len(dser)),
            'ct_lags': int(st.get('lags', 0)),
        }
    out = pd.DataFrame(rows).T
    out.index.name = 'rel_day'
    return out


def event_path_by_group(windows, group_col, *, value='car', quantiles=None,
                        bins=5):
    """按事件属性分组，画出各自的累计路径（PEAD 的标准做法）。

    Parameters
    ----------
    windows : EventWindows | DataFrame
    group_col : str
        事件表里带过来的列（如 ``surprise``）。
    quantiles : int, optional
        按该列分位分组；给了就忽略 ``bins``，用等频分位。
    bins : int
        等宽分箱数（``quantiles`` 为空时用）。

    Returns
    -------
    DataFrame
        index = ``rel_day``，columns = 各组的 ``car`` 均值；另附
        ``n_<组>`` 计数列与末尾的 ``spread_<高>−<低>``。
    """
    df = windows.df if isinstance(windows, EventWindows) else windows
    if group_col not in df.columns:
        fail('event', 'bad_group',
             f'`{group_col}` 不在窗口表里；可选 {sorted(set(df.columns))}')
    ev = df.reset_index().groupby('event_id')[group_col].first().dropna()
    if quantiles:
        lab = pd.qcut(ev, quantiles, labels=False, duplicates='drop') + 1
    else:
        lab = pd.cut(ev, bins, labels=False) + 1
    lab = lab.astype('Int64')
    d = df.reset_index()
    d['grp'] = d['event_id'].map(lab)
    d = d[d['grp'].notna()]
    piv = d.pivot_table(index='rel_day', columns='grp', values=value,
                        aggfunc='mean')
    cnt = d.groupby('grp')['event_id'].nunique()
    piv.columns = [f'G{int(c)}' for c in piv.columns]
    out = piv.copy()
    for c in piv.columns:
        out[f'n_{c}'] = int(cnt.get(int(c[1:]), 0))
    lo, hi = piv.columns[0], piv.columns[-1]
    out[f'spread_{hi}-{lo}'] = piv[hi] - piv[lo]
    return out
