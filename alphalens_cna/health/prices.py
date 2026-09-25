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
# 复权因子的相对容差：10 位小数的存储舍入在价格量级下约 1e-10，留两个数量级余量
ADJ_REL_TOL = 1e-8


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
    所以单日 |收益| 超过 50% 几乎不可能是真行情。它先用原始价把所有超过阈值的
    日子挑出来，再用**复权价**给每一条做归因 —— 复权价有没有把跳变抹平，
    就是"复权因子有没有跟上"的直接判据。

    归因三分（`adj = raw × adj_factor`）：

    ==================  ==========================================  ==========
    复权后            含义                                          判定
    ==================  ==========================================  ==========
    正常（跳变被抹平）  除权日且因子已覆盖 —— **这才是对的状态**     计入"已覆盖"
    也一样大          因子没跟上，或停牌复牌/重组这类真实跳变      需人看
    缺失              没有复权价可判断                             需人看
    ==================  ==========================================  ==========

    「已覆盖」占比 > 80% 且**没有任何缺失** → ``pass``；否则 ``warn``。
    两种情况都会在文案里交代未被覆盖的条数 —— 不靠读者自己去明细里数。

    ⚠️ 2026-09-25 修正：本函数此前把前两支**判反了** —— 原文写成
    "如果复权后收益正常，就是除权没算对（数据问题）"，于是把正确处理了除权日的
    面板报成硬错误，反而把真正可疑的"复权后也大"只记为告警。
    数学上 ``adj = raw × factor``：因子在除权日抬升，复权价才会连续 ——
    "复权后正常"恰恰是**因子已覆盖**的签名。
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
        # 这里**要的就是前值填充**（跨停牌区间比），与上面原始收益的口径对齐 ——
        # 只有同口径才谈得上"复权后是不是也涨这么多"。
        # 自己 ffill 再自己算，不调 pct_change：它的默认 fill_method 在 2.x 会告警，
        # 3.0 起默认改成**不填充**（语义会悄悄反过来），关键字本身也会被移除。
        # 手写 `s/s.shift(1)-1` 与旧默认逐位相同（已用 8000 点含大量 NaN 的面板核对）。
        _adj = prices['adj_close'].groupby(level='asset').ffill()
        ar = (_adj / _adj.groupby(level='asset').shift(1) - 1)[flag]
        # ★ 缺失必须按**原始数据**判：上面的 ffill 会把缺失行的复权价填成前值，
        #   于是那一天算出的收益是 0% —— 从 ar 上**已经看不出**"这行没有复权价"。
        #   早前就是在这里把"缺失"读成了"复权后正常（已覆盖）"，给出假 all-clear。
        adj_missing = prices['adj_close'][flag].isna()
    else:
        ar = pd.Series(np.nan, index=idx)
        adj_missing = pd.Series(True, index=idx)      # 整个复权价都没有 → 全部无从判断
    _ar = np.asarray(ar.values, dtype=float)
    _na = np.asarray(adj_missing.values, dtype=bool)
    # 三档：缺失（无从判断）／已覆盖（被复权抹平）／复权后仍大
    unknown = _na | ~np.isfinite(_ar)
    explained = ~unknown & (np.abs(_ar) <= th['extreme_move'])
    big = ~unknown & ~explained
    detail = pd.DataFrame({
        'date': idx.get_level_values('date'),
        'asset': idx.get_level_values('asset'),
        'raw_return': raw_r.values,
        'adj_return': ar.values,
        # 明细自带标签：读者不必自己回去数哪几条是"没被覆盖"的
        '问题': np.where(unknown, '复权价缺失',
                         np.where(explained, '已覆盖（因子已跟上）', '复权后仍大')),
    })
    # ── 归因：**复权后正常 = 复权是对的** ───────────────────────────────
    # 除权日（尤其高送转 10送10/15）原始价必然"凭空"腰斩，这是制度性的。
    # 关键在于复权价有没有把这个跳变抹平：
    #   · adj 正常  → `adj_factor` 已覆盖该除权日 —— **这是正确状态**，不是缺陷
    #   · adj 也大  → 因子没跟上（adj = raw × factor，factor 不动则 adj 同跌），
    #                 或停牌复牌/重组这类真实跳变 —— 需要人看
    #   · adj 缺失  → 复权价都没有，无从判断 —— **不能算"已覆盖"**，也要人看
    #
    # ⚠️ 2026-09-25 修正：原实现把这两支**判反了** ——
    #    把"复权后正常"报成硬错误（"adj_factor 没盖住除权日"），
    #    却把真正可疑的"复权后仍大"降级成告警。
    #    之所以长期没暴露：2020-2024 高送转稀少，触发不到 fail 分支；
    #    而在 2016-2022（高送转密集期）上，400 只样本一次就报出 38 条，
    #    其中 31 条经 xdxr 核对全部是 category==1 的正常除权除息日。
    #
    # ⚠️ 2026-09-25 补：**缺失 ≠ 正常**。adj 为 NaN 时，上面的 `ar` 是**前值填充**
    #    后算的（跨停牌区间比，与 raw 口径对齐，见 line 90 附近），于是缺失日算出来
    #    的收益是 0% —— 看着"正常"，其实"没有复权价可判断"。此前它被并进
    #    `explained`，一旦占比超过 80% 就输出 `pass`＋"复权价连续"，是**假 all-clear**。
    #    现在单列一档，且**缺失不为 0 就不给 pass**："没检验"与"检验通过"必须分开。
    n_expl = int(explained.sum())
    n_missing = int(unknown.sum())
    n_big = int(big.sum())                  # 复权后确实还是大跳变

    def _rest():
        """pass 时也要交代没被覆盖的那部分 —— 不然汇总行等于把它们藏起来。"""
        bits = []
        if n_big:
            bits.append(f'{n_big:,} 条复权后仍大')
        if n_missing:
            bits.append(f'{n_missing:,} 条复权价缺失（无从判断）')
        return ('；另有 ' + '、'.join(bits) + ' —— 见下方明细') if bits else ''

    if n_missing == 0 and n_expl > 0.8 * n:
        return Finding(
            '极端涨跌', 'pass',
            f'{n:,} 条 |原始收益| > {th["extreme_move"]:.0%}，其中 {n_expl:,} 条'
            f'复权后正常 —— 是除权日且 `adj_factor` **已覆盖**（复权价连续）'
            f'{_rest()}',
            float(n), detail, th['extreme_move'])
    # warn 文案只念**非 0** 的档 —— "0 条复权后仍然很大"这种句子只会让人分心。
    # （n_big 与 n_missing 至少有一个非 0：全被抹平时走的是上面的 pass。）
    parts = []
    if n_big:
        parts.append(f'{n_big:,} 条复权后**仍然**很大（要么 `adj_factor` 没盖住除权日，'
                     f'要么是停牌复牌/重组这类真实跳变）')
    if n_missing:
        parts.append(f'{n_missing:,} 条复权价缺失（无从判断）')
    if n_expl:
        parts.append(f'{n_expl:,} 条已被复权抹平')
    return Finding(
        '极端涨跌', 'warn',
        f'{n:,} 条 |原始收益| > {th["extreme_move"]:.0%}：' + '、'.join(parts) + ' —— 需人看',
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
    # ⚠️ 必须用**相对**容差：adj_factor 常以 10 位小数存储（round(x,10)），
    #    价格量级下这个舍入本身就有 ~1e-10。用绝对容差 -1e-12 会把纯舍入
    #    判成"因子倒退"（硬错误），让整个体检结论不可信。
    #    （同一个病早前在 engine/adjust.py 的 check_adjust_agreement 修过，
    #      这里漏了一处 —— 两处判据必须同源。）
    down = d < -ADJ_REL_TOL * f.abs()
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
        # ⚠️ fail 的优先级高于跳变，但**不能因此把跳变藏起来**。
        #    全量回归里就撞上过：3 处存储舍入造成的假"倒退"把 3 处真实的
        #    2 倍跳变盖成了 fail 里的沉默项 —— 报告读起来像"只有复权问题"。
        #    detail 里本来就带着两批行，汇总文案也得说全。
        #    （metric 仍是 n_down：硬错误的数量不该被跳变稀释。）
        extra = (f'；另有 {n_jump:,} 处单日跳变 > {th["adj_jump"]:.0f} 倍'
                 f'（也看一眼是不是送转股）') if n_jump else ''
        return Finding('复权连续性', 'fail',
                       f'{n_down:,} 处复权因子**倒退**（复权因子必须单调不减）{extra}',
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
