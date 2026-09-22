"""报告层 —— **零绘图依赖**。

核心只产出两类东西：

1. **tidy DataFrame** —— 任何绘图工具（matplotlib / plotly / 你自己的）都能直接消费
2. **Markdown 文本** —— 人能直接读

**不引入任何绘图库，也不内置画图函数**（设计 §9.1）。
alphalens 的 ``import alphalens`` 会连带拉起 seaborn/matplotlib（D3），本库不这样。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ['Report', 'build_report']


@dataclass
class Report:
    """一次因子分析的完整产物。

    Attributes
    ----------
    factor_name : str
    clean : CleanResult
    ic : DataFrame
        逐期 IC（index=date, columns=持有期）。
    ic_summary, nw : DataFrame
    quantiles : DataFrame
        各分位平均收益（index=(date, q)）。
    quantile_stats : DataFrame
    turnover : DataFrame
    verdict : Verdict
    tail : DataFrame, 可选
        分位 × 持有期的**尾部统计**（均值 / CVaR / 崩盘命中率）。
        与 ``quantiles`` 并排看：均值差可能很小，尾部差却很大。
    crash : DataFrame, 可选
        ``crash_spread`` 的输出：QN−Q1 的崩盘命中率差 + Newey-West t。
    health : HealthReport, 可选
        数据体检结果（**六道防线 · 第 2 条**）。有它才谈得上"这份数字可不可信"。
    extra : dict[str, DataFrame]
        其它要一起带出去的 tidy 表。
    """

    factor_name: str = 'factor'
    clean: object = None
    ic: pd.DataFrame = None
    ic_summary: pd.DataFrame = None
    nw: pd.DataFrame = None
    quantiles: pd.DataFrame = None
    quantile_stats: pd.DataFrame = None
    turnover: pd.DataFrame = None
    verdict: object = None
    tail: pd.DataFrame = None
    crash: pd.DataFrame = None
    health: object = None
    extra: dict = field(default_factory=dict)

    # -- tidy 输出 ----------------------------------------------------------
    def frames(self):
        """所有表 → ``dict[str, DataFrame]``（**绘图工具的入口**）。"""
        out = {
            'ic': self.ic,
            'ic_summary': self.ic_summary,
            'newey_west': self.nw,
            'quantile_returns': self.quantiles,
            'quantile_stats': self.quantile_stats,
            'turnover': self.turnover,
            'ledger': (self.clean.ledger.to_frame()
                       if self.clean is not None else None),
            'verdict': (self.verdict.to_frame()
                        if self.verdict is not None else None),
            'tail': self.tail,
            'crash': self.crash,
            'health': (pd.DataFrame([{'项目': f.name, '严重度': f.severity,
                                      '结论': f.summary, '指标': f.metric}
                                     for f in self.health.findings])
                       if self.health is not None else None),
        }
        out.update(self.extra)
        return {k: v for k, v in out.items() if v is not None}

    def equity_curves(self):
        """累计净值曲线（**图表就绪**）—— index=日期，columns=各持有期。"""
        fr = self.extra.get('long_short')
        if fr is None or not len(fr):
            return None
        return (1 + fr.fillna(0)).cumprod()

    # -- Markdown ----------------------------------------------------------
    def to_markdown(self, title=None):
        """渲染成 Markdown。"""
        L = []
        L.append(f'# {title or self.factor_name + " 因子分析报告"}')
        L.append('')
        L.append('> 由 alphalens-cna 生成。**所有数字都带不确定性标注** —— '
                 '未校正的 p 值不作为结论依据。')
        L.append('')

        if self.verdict is not None:
            L.append('## 一、结论')
            L.append('')
            L.append('```')
            L.append(str(self.verdict))
            L.append('```')
            L.append('')

        if self.health is not None:
            L.append('## 二、数据体检')
            L.append('')
            L.append(f'已查 **{self.health.n_checked}** 项，'
                     f'硬错误 {len(self.health.fails)}，'
                     f'告警 {len(self.health.warns)}，'
                     f'跳过 {len(self.health.skips)}。')
            L.append('')
            L.append('```')
            L.append(str(self.health))
            L.append('```')
            L.append('')

        if self.clean is not None and getattr(self.clean, 'ledger', None):
            L.append('## 三、样本账')
            L.append('')
            L.append('> **每一步剔除都可归因。** 输入 = 输出 + 各类剔除之和。')
            L.append('')
            L.append('```')
            L.append(str(self.clean.ledger))
            L.append('```')
            L.append('')

        for sec, (df, note) in enumerate([
            (self.ic_summary,
             '`t_naive` 是朴素 t，重叠观测下**会虚高**。'
             '下一节的 Newey-West 才是可信的那个。'),
            (self.nw,
             '重叠观测使相邻样本不独立，朴素 t 被高估。'
             '`t_inflation` 就是虚高倍数。'),
        ]):
            if df is None or not len(df):
                continue
            name = '四、IC' if sec == 0 else '五、Newey-West 修正'
            L.append(f'## {name}')
            L.append('')
            L.append(_md_table(df))
            L.append('')
            L.append(f'> {note}')
            L.append('')

        if self.quantile_stats is not None and len(self.quantile_stats):
            L.append('## 六、分层')
            L.append('')
            L.append(_md_table(self.quantile_stats))
            L.append('')
            L.append('> `monotonicity` 是层序与收益的 Spearman。'
                     '接近 ±1 才算"单调"；两端清晰、中间乱序说明因子定义可能有问题。')
            L.append('')

        if self.crash is not None and len(self.crash):
            L.append('## 七、尾部风险')
            L.append('')
            L.append('> **均值与 IC 看不见的那一块。** 秩相关度量的是全样本成对单调性，'
                     '一团挤在"双低角"的样本彼此同序，会把 ρ 往正方向拉 —— '
                     '效应全在左尾时，IC 反而可能变弱。')
            L.append('')
            _cols = [c for c in ('hit_lo', 'hit_hi', 'hit_spread', 'mean_hi',
                                 'mean_lo', 'mean_spread', 't_hit', 't_naive_hit',
                                 'n_periods') if c in self.crash.columns]
            L.append(_md_table(_int_cols(
                self.crash.reset_index()[['h'] + _cols], 'h', 'n_periods')))
            L.append('')
            L.append('> `hit_spread = QN − Q1`。**符号**：低分位更容易崩时它为**负**。')
            L.append('> 均值差与崩盘率差**可以结论相反**（实测 ROE 就是：'
                     '均值差变弱，崩盘率差变显著），因为它们回答的不是同一个问题。')
            L.append('')
            if self.tail is not None and len(self.tail):
                _tcols = [c for c in ('mean', 'cvar', 'hit_rate', 'n_crash',
                                      'n', 'threshold')
                          if c in self.tail.columns]
                tb = _int_cols(self.tail.reset_index()[['h', 'q'] + _tcols],
                               'h', 'q', 'n_crash', 'n')
                L.append(_md_table(tb))
                L.append('')
            L.append(_tail_mode_note(self.crash))
            L.append('')

        if self.turnover is not None and len(self.turnover):
            L.append('## 八、换手与成本')
            L.append('')
            L.append(_md_table(self.turnover))
            L.append('')

        if self.clean is not None and getattr(self.clean, 'ledger', None):
            L.append('## 九、剔除明细')
            L.append('')
            L.append(_md_table(self.clean.ledger.to_frame().set_index('reason'),
                               index=True))
            L.append('')
            L.append('> 每个原因都带样本索引，可直接回原始数据核对。')
            L.append('')

        L += [
            '---', '',
            '## 怎么读这份报告', '',
            '| 字段 | 含义 |', '|---|---|',
            '| `p_adj` | **校正后** p 值 —— 只有它该拿去做决策 |',
            '| `n_trials` | 你一共测过多少个假设。**它决定门槛多高** |',
            '| `t_inflation` | 朴素 t 相对 NW t 虚高了几倍 |',
            '| `n_eff` | 有效样本量（启发式）。名义 344 可能只剩 40 |',
            '| `tradable_ratio` | 结论建立在多少可交易样本上 |',
            '',
        ]
        return '\n'.join(L)

    def save(self, path, kind='markdown', title=None):
        """存盘。``kind='markdown'`` 或 ``'frames'``（后者存成 parquet 目录）。"""
        import os
        if kind == 'markdown':
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.to_markdown(title))
            return path
        os.makedirs(path, exist_ok=True)
        for nm, df in self.frames().items():
            try:
                df.to_parquet(os.path.join(path, f'{nm}.parquet'))
            except Exception:
                df.to_csv(os.path.join(path, f'{nm}.csv'))
        return path

    def __repr__(self):
        v = '显著' if (self.verdict is not None and self.verdict.significant) else '不显著'
        n = len(self.ic) if self.ic is not None else 0
        return f'<Report {self.factor_name} · {n} 期 · {v}>'


def _int_cols(df, *cols):
    """把持有期/分位列转成整数 —— 否则会被渲染成 ``1.0000``。"""
    d = df.copy()
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce').astype('Int64')
    return d


def _tail_mode_note(crash):
    """尾部门槛模式的提醒 —— 别把"分位门槛"当成"某组更容易炸"的证据。"""
    thr = crash.get('threshold')
    if thr is None or not len(thr):
        return ''
    frac = crash['hit_lo'].mean() + 1e-12
    if abs(frac - 0.10) < 0.02 and 'threshold' in crash.columns:
        return ('> ⚠️ 当前用**分位门槛**（当日截面最差 10%）：每个分位的命中率'
                '必然 ≈10%，它只能发现尾部**集中度**，'
                '发现不了"某组更容易炸"。要判断后者请传 `crash_threshold=-0.5`。')
    return (f'> 门槛 = 绝对值 {thr.iloc[-1]:.1%}（`crash_threshold`）—— '
            f'这才适合回答"某组是不是更容易炸"。')


def _md_table(df, index=True):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, row in d.iterrows():
        vals = list(row.values)
        for i, v in enumerate(vals):
            # 索引列（持有期 / 分位标签）是整数语义，别渲染成 1.0000
            if i == 0 and index and isinstance(v, (float, np.floating)) \
                    and np.isfinite(v) and float(v).is_integer():
                vals[i] = int(v)
        lines.append('| ' + ' | '.join(_fmt(v) for v in vals) + ' |')
    return '\n'.join(lines)


def _fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return '是' if v else '否'
    if isinstance(v, (int, np.integer)):
        return f'{v:,}'
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return '—'
        if v != 0 and abs(v) < 1e-4:
            return f'{v:.3e}'
        return f'{v:,.4f}'
    return str(v)


# --------------------------------------------------------------------------- #
def build_report(factor, prices, calendar, *, horizons=(1, 5, 21), quantiles=5,
                 universe=None, names=None, list_dates=None, limit_status=None,
                 exposures=None, groupby=None, by=None, sort_method='conditional',
                 model=None, n_trials=None, method='bhy', cost_bps=15.0,
                 name='factor', new_stock_days=60, prepare_tradability=True,
                 health=True, crash_threshold=None):
    """**一条龙**：数据体检 → 可成交性 → 前向收益 → 清洗 → 分层 → IC → NW → Verdict。

    Parameters
    ----------
    factor : FactorPanel | DataFrame
        需含 ``value``（会归一成 ``factor``）。
    prices : PricePanel | DataFrame
        需三价并存（见 ``docs/输入数据规格.md``）。
    calendar : Calendar
    horizons : tuple[int]
    n_trials : int, 可选
        **一共测过多少个假设。** 不给就不做多重检验校正，并在结论里**明说**。
    prepare_tradability : bool
        默认 ``True``，自动算可成交性。若 ``prices`` 已带 ``can_buy_open`` 则跳过。
    health : bool
        默认 ``True``，跑一遍数据体检（防线 2）并把结论带进报告。
        **"数字对不对"之前先回答"数据能不能用"。**
    crash_threshold : float, optional
        "崩盘"的**绝对**门槛（如 ``-0.5`` = 一年腰斩）。
        不给则用当日截面最差 10% 分位作门槛 —— 但**该模式对等规模分组不敏感**
        （实测每个分位的命中率都恰好 10%），只能发现尾部**集中度**。
        要判断"某组是不是更容易炸"，**必须传绝对门槛**。

    Returns
    -------
    Report
    """
    from ..analysis import (factor_returns, ic_summary, information_coefficient,
                            quantile_returns, quantile_stats, quantile_turnover,
                            quantize)
    from ..engine.clean import clean
    from ..engine.returns import ReturnModel, forward_returns
    from ..engine.tradability import compute_tradability
    from ..inference.newey_west import newey_west_summary
    from ..inference.verdict import assess

    px = getattr(prices, 'df', prices)
    model = model or ReturnModel(entry='next_open', policy='skip',
                                 new_stock_days=new_stock_days)

    trad = None
    if prepare_tradability or 'can_buy_open' not in px.columns:
        trad = compute_tradability(px, calendar=calendar, names=names,
                                   list_dates=list_dates,
                                   limit_status=limit_status,
                                   new_stock_days=new_stock_days)
    ret = forward_returns(px, calendar, list(horizons), tradability=trad,
                          model=model)
    cr = clean(factor, ret, universe=universe, exposures=exposures,
               groupby=groupby, name=name)

    ic = information_coefficient(cr)
    q = quantize(cr, n=quantiles, by=by, method=sort_method)
    to = quantile_turnover(q['q'])
    ts = pd.DataFrame({'turnover': to.mean(),
                       'cost_per_period': to.mean() * 2 * (cost_bps / 10000.0),
                       'n_periods': to.notna().sum()})
    ts.index.name = 'q'

    from ..analysis import crash_spread, tail_by_quantile

    qdf = q.rename(columns={'q': 'factor_quantile'}) \
        if 'q' in q.columns else q
    tail_tbl = tail_by_quantile(cr.data.join(qdf[['factor_quantile']]),
                                horizons=list(horizons),
                                threshold=crash_threshold)
    crash_tbl = crash_spread(cr.data.join(qdf[['factor_quantile']]),
                             horizons=list(horizons),
                             threshold=crash_threshold)

    hp = None
    if health:
        from ..health import check as _health_check
        hp = _health_check(prices=px, factor=getattr(factor, 'df', factor),
                           tradability=trad, calendar=calendar, name=name)

    fr = factor_returns(cr, quantiles=q)
    v = assess(ic=ic, n_trials=n_trials, method=method, ledger=cr.ledger,
               returns=(fr[fr.columns[0]] if len(fr.columns) else None),
               turnover=to.mean(axis=1), cost_bps=cost_bps,
               crash=crash_tbl)

    return Report(
        factor_name=name, clean=cr, ic=ic,
        ic_summary=ic_summary(ic), nw=newey_west_summary(ic),
        quantiles=quantile_returns(cr, quantiles=q),
        quantile_stats=quantile_stats(cr, quantiles=q),
        turnover=ts, verdict=v, tail=tail_tbl, crash=crash_tbl,
        health=hp, extra={'long_short': fr})
