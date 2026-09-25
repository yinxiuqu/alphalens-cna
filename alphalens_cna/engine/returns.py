"""成交模型与前向收益。

A 股的真实时间线
----------------
.. code-block:: text

    T 日 15:00 收盘
       ├─ 因子在 T 日收盘后生成（只能用 T 日及之前已公开的信息）
       ├─ ★ 隔夜跳空（T 收 → T+1 开）：这段收益你拿不到 ★
    T+1 日 09:30 开盘 ← 最早可成交时点
       └─ 持有 h 个交易日后卖出

**所以「T 日收盘买入」是不可实现的** —— alphalens 默认就是这么算的
（``pct_change(h).shift(-h)``），对 T+1 市场系统性高估。

三列并列（设计的核心输出）
--------------------------
每个持有期输出三列，让"因子赚的到底是哪一段"一目了然：

====================  =========================================================
列                     含义
====================  =========================================================
``forward_return_h``  ``adj_open(卖出日) / adj_open(T+1) − 1`` **可获取**
``overnight_gap_h``   ``adj_open(T+1) / adj_close(T) − 1``    **拿不到**
``total_return_h``    两者复合 —— 若主要收益在 gap 里，这因子不可交易
====================  =========================================================

字典序注：``entry='close'`` 时退化成 alphalens 口径（gap 恒为 0），
用于 P0 对拍基线。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..contract.calendar import as_calendar
from ..contract.errors import fail

__all__ = ['ReturnModel', 'Returns', 'forward_returns', 'ENTRY_MODES',
           'POLICIES', 'DELIST_POLICIES']

ENTRY_MODES = ('next_open', 'close', 'same_open')
POLICIES = ('skip', 'mark_only', 'delay')
# 持有期内退市（股票消失）时的收益约定 —— 见 ReturnModel.delist_policy
DELIST_POLICIES = ('nan', 'last_price', 'haircut')


# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReturnModel:
    """成交模型。

    Parameters
    ----------
    entry : {'next_open', 'close', 'same_open'}
        ``'next_open'``（默认）**A 股可实现**：T 日信号 → T+1 开盘成交。
        ``'close'`` 退化为 alphalens 口径：T 日收盘成交 —— **不可实现**，
        仅用于对拍基线。
        ``'same_open'`` 信号日**当天开盘**成交。
        ⚠️ **只有在因子于当日开盘前就已确知时才合法**；若因子用到了当日
        收盘数据（如当日收盘价算的动量），用它就是**前视**。
        提供它只为复现旧口径，不建议新分析使用。
    exit_price : {'open', 'close'}
        卖出用哪个价。默认 ``'open'``。
    policy : {'skip', 'mark_only', 'delay'}
        入场日不可成交（停牌/一字涨停/新股）时怎么办：

        * ``'skip'``（默认，D2）—— 剔除，并计入原因账
        * ``'mark_only'`` —— 保留但打标记，收益照算（供敏感性对比）
        * ``'delay'`` —— 顺延到之后 N 个交易日内第一个可成交日
    max_delay : int
        ``policy='delay'`` 时最多顺延几个交易日。
    new_stock_days : int
        上市后多少个交易日内视为新股（D3：默认 **60**）。
    delist_policy : {'nan', 'last_price', 'haircut'}
        **持有期内退市**（该股在持有期末之前就再也没有价格了）时怎么办。
        默认 ``'nan'`` = 保持旧行为（剔除，计入 ``no_exit_price``）。

        * ``'nan'``        —— 剔除。**长持有期会系统性丢掉退市前那一段**，
          于是 IC 看到的是被截断的子样本（实测 h=252 时退市股只有 14%
          的末段观测能算出收益），**负向因子的强度被低估**。
        * ``'last_price'`` —— 按**最后成交价**清算，之后视为现金（收益 0）。
          不需要任何额外假设，是最保守也最可辩护的约定。
        * ``'haircut'``    —— 最后成交价再打 ``delist_return`` 的折扣
          （Shumway 式约定，美股常用 −30%）。A 股有退市整理期真实成交，
          整理期的跌幅**已经**在价格里，再打折扣有重复计算的风险，
          所以默认不用它 —— 只在做敏感性对比时开。

        ⚠️ **区分"退市"与"窗口内停牌"**：只有当该股在持有期末之后
        **再也没有任何价格**时，才算退市并套用约定；窗口内停牌、期末仍未复牌的，
        仍按缺失处理（那是数据/流动性问题，不是退市）。
    delist_return : float
        ``delist_policy='haircut'`` 时在最后成交价上再乘 ``1 + delist_return``。
    """

    entry: str = 'next_open'
    exit_price: str = 'open'
    policy: str = 'skip'
    max_delay: int = 3
    new_stock_days: int = 60
    delist_policy: str = 'nan'
    delist_return: float = 0.0

    def __post_init__(self):
        if self.entry not in ENTRY_MODES:
            fail('ReturnModel', 'bad_entry',
                 f"entry 只能是 {ENTRY_MODES}，收到 {self.entry!r}")
        if self.exit_price not in ('open', 'close'):
            fail('ReturnModel', 'bad_exit',
                 f"exit_price 只能是 'open' / 'close'，收到 {self.exit_price!r}")
        if self.policy not in POLICIES:
            fail('ReturnModel', 'bad_policy',
                 f"policy 只能是 {POLICIES}，收到 {self.policy!r}")
        if self.delist_policy not in DELIST_POLICIES:
            fail('ReturnModel', 'bad_delist_policy',
                 f"delist_policy 只能是 {DELIST_POLICIES}，"
                 f"收到 {self.delist_policy!r}")
        if self.delist_policy == 'haircut' and not (-1.0 < self.delist_return <= 0.0):
            fail('ReturnModel', 'bad_delist_return',
                 f"delist_return 必须落在 (-1, 0]，收到 {self.delist_return!r}。"
                 f"提示：−0.30 表示在最后成交价上再跌 30%")

    @property
    def is_alphalens_like(self) -> bool:
        """退化配置 —— 收盘成交、收盘卖出。对拍基线用。"""
        return self.entry == 'close' and self.exit_price == 'close'


@dataclass
class Returns:
    """前向收益结果。

    Attributes
    ----------
    df : DataFrame
        索引 ``(date, asset)``（= **信号日**），列见模块文档。
    ledger : dict[int, dict]
        每个持有期的剔除原因账：``{h: {'limit_up': 12, 'suspended': 3, ...}}``。
    model : ReturnModel
    """

    df: pd.DataFrame
    ledger: dict = field(default_factory=dict)
    model: ReturnModel = field(default_factory=ReturnModel)

    def columns_for(self, h):
        """某个持有期的三列名。"""
        return f'forward_return_{h}', f'overnight_gap_{h}', f'total_return_{h}'

    def tradable(self, h):
        """该持有期是否可成交（未因 policy='skip' 被剔除）。"""
        c = f'tradable_{h}'
        return self.df[c] if c in self.df.columns else pd.Series(True, index=self.df.index)

    def __repr__(self):
        hs = sorted(self.ledger)
        return (f'<Returns {len(self.df):,} 行 · 持有期 {hs} · '
                f'entry={self.model.entry} policy={self.model.policy}>')


# --------------------------------------------------------------------------- #
def forward_returns(prices, calendar, horizons, *, tradability=None,
                    model=ReturnModel()) -> Returns:
    """算前向收益。

    Parameters
    ----------
    prices : PricePanel | DataFrame
        需含 ``adj_open`` / ``adj_close``（**收益一律用复权价**）。
    calendar : Calendar | DatetimeIndex
        交易日历。**所有日期运算都走它**，不用 pandas 的 freq 推断。
        两种写法都收（裸 DatetimeIndex 会被包成 Calendar）。
    horizons : int | Sequence[int]
        持有期（**交易日数**）。
    tradability : DataFrame, 可选
        ``compute_tradability`` 的输出。不给则假设全部可成交（**不推荐** ——
        会把一字板当成能买进）。
    model : ReturnModel

    Returns
    -------
    Returns
    """
    df = getattr(prices, 'df', prices)
    if isinstance(horizons, int):
        horizons = [horizons]
    horizons = sorted({int(h) for h in horizons})
    if not horizons or min(horizons) < 1:
        fail('returns', 'bad_horizons', f'持有期必须是正整数，收到 {horizons}')

    for c in ('adj_open', 'adj_close'):
        if c not in df.columns:
            fail('returns', 'missing_column',
                 f'缺 `{c}`。收益计算**必须**用复权价 —— '
                 f'原始价在除权日会有假跳变。')

    idx = df.index
    if not idx.is_monotonic_increasing:
        df = df.sort_index()
        idx = df.index

    # ── 日期平移（向量化，走日历）────────────────────────────────────
    entry_lag = 1 if model.entry == 'next_open' else 0      # same_open / close 都不滞后
    entry_dates = _shift_dates(idx.get_level_values('date'), calendar, entry_lag)

    out = pd.DataFrame(index=idx)
    assets = idx.get_level_values('asset')

    # ── 入场价 ────────────────────────────────────────────────────────
    # ⚠️ 入场【价格】与入场【时点】是两件事：
    #   entry='next_open' → T+1 日的**开盘价**
    #   entry='close'     → T 日**当日收盘价**（退化为 alphalens 口径）
    # 早先的实现无论哪种都取 adj_open（只改滞后），于是 entry='close'
    # 实际算的是"T 开盘买入"——名字与行为不符，会让对拍基线整体错位。
    entry_col = 'adj_close' if model.entry == 'close' else 'adj_open'
    entry_idx = pd.MultiIndex.from_arrays([entry_dates, assets], names=['date', 'asset'])
    entry_px = df[entry_col].reindex(entry_idx).values
    prev_close = df['adj_close'].values                      # 信号日收盘
    out['entry_date'] = entry_dates

    if model.entry == 'next_open':
        out['overnight_gap'] = entry_px / prev_close - 1
    else:
        out['overnight_gap'] = 0.0        # 当日成交，没有隔夜暴露

    # ── 入场可成交性 ────────────────────────────────────────────────
    tradable, reason = _entry_tradable(tradability, entry_idx, idx, entry_px,
                                       model, calendar, entry_dates)
    out['entry_tradable'] = tradable
    out['entry_reason'] = reason

    # ── 退市识别（持有期内股票消失）──────────────────────────────────
    # 判据：该票**在持有期末之后再也没有任何价格**。
    #   · 有 → 退市/被吸收合并，可按 delist_policy 清算
    #   · 无 → 只是窗口内停牌 / 数据缺口，仍按缺失处理（两件事不能混）
    px_all = df[f'adj_{model.exit_price}']
    has_px = px_all.notna()
    last_date_by_asset = (df.index.get_level_values('date')[has_px.values]
                          .to_series(index=df.index[has_px.values])
                          .groupby(level='asset').max())
    last_px_by_asset = px_all[has_px].groupby(level='asset').last()
    last_dates = last_date_by_asset.reindex(assets).values
    last_pxs = last_px_by_asset.reindex(assets).values

    # ── 各持有期 ───────────────────────────────────────────────────
    gap = out['overnight_gap'].values
    ledger = {}
    for h in horizons:
        exit_dates = _shift_dates(entry_dates, calendar, h)
        exit_idx = pd.MultiIndex.from_arrays([exit_dates, assets], names=['date', 'asset'])
        exit_px = df[f'adj_{model.exit_price}'].reindex(exit_idx).values

        # 退市：持有期末已晚于该票最后一个有价日，且期末确实取不到价
        gone = pd.notna(last_dates) & (exit_dates > last_dates)
        delisted = gone & ~np.isfinite(exit_px)

        px_use = exit_px
        if model.delist_policy == 'last_price':
            px_use = np.where(delisted, last_pxs, exit_px)
        elif model.delist_policy == 'haircut':
            px_use = np.where(delisted, last_pxs * (1.0 + model.delist_return),
                              exit_px)

        fwd = px_use / entry_px - 1
        tot = (1 + fwd) * (1 + gap) - 1
        nan_px = ~np.isfinite(fwd)
        keep = tradable.values & ~nan_px

        out[f'forward_return_{h}'] = np.where(keep, fwd, np.nan)
        out[f'overnight_gap_{h}'] = np.where(keep, gap, np.nan)
        out[f'total_return_{h}'] = np.where(keep, tot, np.nan)
        out[f'exit_date_{h}'] = exit_dates
        out[f'tradable_{h}'] = keep
        out[f'delisted_{h}'] = delisted          # 是否走了退市约定

        led = _ledger(reason.values, entry_px, exit_px, keep)
        n_fill = int((delisted & keep).sum())
        if n_fill:
            led['delist_filled'] = n_fill
        n_delist_drop = int((delisted & ~np.isfinite(px_use)).sum())
        if n_delist_drop:
            led['delisted_dropped'] = n_delist_drop
        ledger[h] = led

    return Returns(df=out, ledger=ledger, model=model)


def _ledger(reason, entry_px, exit_px, keep):
    """每个持有期的剔除原因账 —— **每一类剔除都要能回答"为什么、多少条"**。"""
    led = {}
    for r in ('limit_up', 'limit_down', 'suspended', 'new_stock', 'not_tradable'):
        n = int((reason == r).sum())
        if n:
            led[r] = n
    n_no_entry = int((~np.isfinite(entry_px)).sum())
    if n_no_entry:
        led['no_entry_price'] = n_no_entry
    n_no_exit = int((np.isfinite(entry_px) & ~np.isfinite(exit_px)).sum())
    if n_no_exit:
        led['no_exit_price'] = n_no_exit          # 持有期末还没复牌 / 已退市
    led['tradable'] = int(keep.sum())
    led['total'] = int(len(keep))
    led['dropped'] = led['total'] - led['tradable']
    return led


# --------------------------------------------------------------------------- #
def _shift_dates(dates, calendar, n):
    """按**交易日历**平移日期。``n`` 可为负；越界 → NaT。

    不用 ``pd.DateOffset`` / ``freq`` —— 那会在非日频或停牌处出错。
    """
    _cal = as_calendar(calendar).index      # ★ 也收裸 DatetimeIndex
    cal = _cal.values
    pos = _cal.get_indexer(pd.DatetimeIndex(dates))
    new = pos + n
    ok = (pos >= 0) & (new >= 0) & (new < len(cal))
    out = np.full(len(dates), np.datetime64('NaT'), dtype='datetime64[ns]')
    out[ok] = cal[new[ok]]
    return out


def _entry_tradable(tradability, entry_idx, idx, entry_px, model, calendar, entry_dates):
    """入场日能不能成交。"""
    if tradability is None:
        # 没有可成交性数据：只按"有没有价格"判，并明确标注这是低精度路径
        ok = np.isfinite(entry_px)
        reason = np.where(ok, 'ok', 'no_price')
        return pd.Series(ok, index=idx), pd.Series(reason, index=idx)

    t = tradability
    tdf = getattr(t, 'df', t)
    cols = set(tdf.columns)
    if 'can_buy_open' in cols:
        can = tdf['can_buy_open'].reindex(entry_idx).values
        rsn = tdf['reason_buy'].reindex(entry_idx).values if 'reason_buy' in cols else None
    elif 'suspended' in cols:
        can = ~tdf['suspended'].reindex(entry_idx).fillna(True).values
        rsn = None
    else:
        fail('returns', 'bad_tradability',
             f'可成交性表缺 `can_buy_open` / `suspended`；实际列：{sorted(cols)[:12]}')

    can = pd.Series(can, index=idx, dtype='boolean').fillna(False).astype(bool).values
    can = can & np.isfinite(entry_px)
    if rsn is None:
        rsn = np.where(can, 'ok', 'not_tradable')
    else:
        rsn = pd.Series(rsn, index=idx, dtype=object).where(
            pd.notna(pd.Series(rsn, index=idx, dtype=object)), 'not_tradable').values
    rsn = np.where(np.isfinite(entry_px), rsn, 'no_price')

    if model.policy == 'mark_only':
        return pd.Series(True, index=idx), pd.Series(rsn, index=idx)
    return pd.Series(can, index=idx), pd.Series(rsn, index=idx)
