"""体检 · 行情类 —— OHLC 不变量 / 极端涨跌 / 复权连续性 / 停牌率。

这一组专治**"结构完全合法、数字完全错误"**的行情数据。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import Finding, fmt_pct

__all__ = [
    'check_ohlc', 'check_extreme_moves', 'check_adjust_continuity',
    'check_suspension',
]

PRICE_TOL = 1e-6


# --------------------------------------------------------------------------- #
def check_ohlc(prices):
    """OHLC 结构不变量 —— 用**原始价**判（复权不该改变高低关系）。

    四条：``high ≥ max(open, close)``、``low ≤ min(open, close)``、
    ``high ≥ low``、``low > 0``。
    """
    need = [c for c in ('raw_open', 'raw_high', 'raw_low', 'raw_close')
            if c in prices.columns]
    if len(need) < 4:
        missing = sorted({'raw_open', 'raw_high', 'raw_low', 'raw_close'}
                         - set(need))
        return Finding('OHLC 结构', 'skip', f'缺列 {missing}，跳过')
    o, h, l, c = (prices[x] for x in
                  ('raw_open', 'raw_high', 'raw_low', 'raw_close'))
    bad = {
        'high<max(open,close)': h < np.maximum(o, c) - PRICE_TOL,
        'low>min(open,close)': l > np.minimum(o, c) + PRICE_TOL,
        'high<low': h < l - PRICE_TOL,
        'low<=0': l <= 0,
    }
    counts = {k: int(m.sum()) for k, m in bad.items()}
    any_bad = np.logical_or.reduce([np.asarray(m) for m in bad.values()])
    n_rows = int(any_bad.sum())
    if n_rows:
        rows = []
        for k, m in bad.items():
            if m.any():
                rows.append((k, counts[k], prices.index[np.asarray(m)][0]))
        detail = pd.DataFrame(rows, columns=['问题', '行数', '首个例子'])
        return Finding('OHLC 结构', 'fail',
                       f'{n_rows:,} 行违反 OHLC 关系：'
                       + '、'.join(f'{k} {v:,}' for k, v in counts.items() if v),
                       float(n_rows), detail)
    return Finding('OHLC 结构', 'pass', f'{len(prices):,} 行全部满足 OHLC 关系')


# --------------------------------------------------------------------------- #
def check_extreme_moves(prices, th):
    """极端涨跌 —— **用原始价算收益**，专抓"复权没算对"。

    A 股有涨跌停制度（主板 10% / 创业科创 20% / 北交所 30% / ST 5%），
    所以单日 |收益| 超过 50% 几乎不可能是真行情。除权日若 ``adj_factor``
    没跟上，原始价就会"凭空"跌一大截 —— 这条能把它抓出来。

    每条都做**归因**：如果复权后收益正常，就是除权没算对（数据问题）；
    如果复权后也大，那可能是真的（停牌复牌 / 重组），只提示不下结论。
    """
    if 'raw_close' not in prices.columns:
        return Finding('极端涨跌', 'skip', '无 `raw_close`，跳过')
    # ★ 这里**故意**用"上一个有效价"当分母，而不是前一日价格：
    #   停牌三周后复牌暴涨 100% 是真实跳变，正是这条检查该抓的东西。
    #   但只在**有实际成交价**的日子计数 —— 所以不会像 ``pct_change()``
    #   的默认前值填充那样，在停牌期间凭空造出一串"0 收益"。
    #   （同一个坑在 ``analysis/event.py`` 里也踩过：那里要的是"窗口内不许有停牌"，
    #     所以用的是 ``fill_method=None`` —— 两处语义不同，别照抄。）
    c = prices['raw_close']
    last_valid = c.groupby(level='asset').ffill()
    prev = last_valid.groupby(level='asset').shift(1)
    r = (c / prev - 1).where(c.notna() & prev.notna())
    # 每只票的第一行没有前收，不算
    first = prices.groupby(level='asset').cumcount() == 0
    flag = (r.abs() > th['extreme_move']) & ~first
    n = int(flag.sum())
    if not n:
        return Finding('极端涨跌', 'pass',
                       f'无 |日收益| > {th["extreme_move"]:.0%} 的记录',
                       0.0, None, th['extreme_move'])

    idx = prices.index[np.asarray(flag)]
    raw_r = r[flag]
    if 'adj_close' in prices.columns:
        ar = prices['adj_close'].groupby(level='asset').pct_change()[flag]
    else:
        ar = pd.Series(np.nan, index=idx)
    detail = pd.DataFrame({
        'date': idx.get_level_values('date'),
        'asset': idx.get_level_values('asset'),
        'raw_return': raw_r.values,
        'adj_return': ar.values,
    })
    # 复权后正常 → 基本可以断定是复权因子的问题
    explained = detail['adj_return'].notna() & \
        (detail['adj_return'].abs() <= th['extreme_move'])
    n_expl = int(explained.sum())
    if n_expl > 0.8 * n:
        return Finding(
            '极端涨跌', 'fail',
            f'{n:,} 条 |原始收益| > {th["extreme_move"]:.0%}，其中 {n_expl:,} 条'
            f'复权后正常 —— 几乎可以断定 `adj_factor` 没盖住这些除权日',
            float(n), detail, th['extreme_move'])
    return Finding(
        '极端涨跌', 'warn',
        f'{n:,} 条 |原始收益| > {th["extreme_move"]:.0%}'
        f'（其中 {n - n_expl:,} 条复权后仍大，可能是停牌复牌/重组）',
        float(n), detail, th['extreme_move'])


# --------------------------------------------------------------------------- #
def check_adjust_continuity(prices, th):
    """复权连续性 —— 因子该**单调不减**，且不该一天翻几倍。

    后复权因子首日 = 1、随除权递增；前复权是它除以末值，同样递增。
    所以**任何一次下降都是错误**（等于"负分红"）。跳变过大（默认 2 倍）
    也可疑，但要人工看一眼是不是送转股。
    """
    if 'adj_factor' not in prices.columns:
        return Finding('复权连续性', 'skip', '无 `adj_factor`，跳过')
    f = prices['adj_factor']
    if (f <= 0).any():
        return Finding('复权连续性', 'fail',
                       f'{int((f <= 0).sum()):,} 行 `adj_factor` ≤ 0',
                       float((f <= 0).sum()))
    d = f.groupby(level='asset').diff()
    down = d < -1e-12
    n_down = int(down.sum())
    rel = f / f.groupby(level='asset').shift(1)
    jump = rel > th['adj_jump']
    n_jump = int(jump.sum())

    rows = []
    for label, m in (('因子倒退', down), ('跳变过大', jump)):
        if m.any():
            idx = prices.index[np.asarray(m)]
            rows.append(pd.DataFrame({
                'date': idx.get_level_values('date'),
                'asset': idx.get_level_values('asset'),
                'adj_factor': f[m].values,
                '倍数': rel[m].values,
                '问题': label,
            }))
    detail = pd.concat(rows) if rows else None

    if n_down:
        return Finding('复权连续性', 'fail',
                       f'{n_down:,} 处复权因子**倒退**（复权因子必须单调不减）',
                       float(n_down), detail)
    if n_jump:
        return Finding('复权连续性', 'warn',
                       f'{n_jump:,} 处单日因子跳变 > {th["adj_jump"]:.0f} 倍'
                       f'（看一眼是不是送转股）',
                       float(n_jump), detail, th['adj_jump'])
    return Finding('复权连续性', 'pass',
                   f'因子单调不减，最大单日倍数 '
                   f'{float(rel.max()):.4f}', float(rel.max()), None,
                   th['adj_jump'])


# --------------------------------------------------------------------------- #
def check_suspension(prices, th):
    """停牌率 —— 用成交量判（``volume == 0`` 或缺失）。

    没有 volume 就跳过，**不猜**。
    """
    if 'volume' not in prices.columns:
        return Finding('停牌率', 'skip', '无 `volume` 列，无法判停牌')
    vol = pd.to_numeric(prices['volume'], errors='coerce')
    susp = (vol.fillna(0) <= 0)
    by_date = susp.groupby(level='date').mean()
    overall = float(susp.mean())
    bad = by_date[by_date > th['max_suspension_rate']]
    if len(bad):
        detail = pd.DataFrame({'date': bad.index, 'suspended_rate': bad.values})
        return Finding('停牌率', 'warn',
                       f'{len(bad)} 个交易日停牌占比 > '
                       f'{th["max_suspension_rate"]:.0%}（疑似整段缺失）',
                       float(by_date.max()), detail,
                       th['max_suspension_rate'])
    return Finding('停牌率', 'info',
                   f'整体 {fmt_pct(overall)}，单日最高 {fmt_pct(by_date.max())}',
                   overall, None, th['max_suspension_rate'])
