"""清洗 —— **带原因的账** + 不变量对账（防线 3）。

设计原则
--------
> **每一步剔除都可归因。** 任何被丢掉的样本，都必须能回答
> "为什么丢的、丢了多少、丢的是谁"。
> 损失不可见 = bug 不可见。

alphalens 的做法是一句 ``dropna()`` —— 丢了多少不知道，为什么丢不知道。
本模块把每次剔除都记进 :class:`DropLedger`，并且在构造时**断言账要平**：

.. code-block:: text

    输入条数 == 输出条数 + 各类剔除之和

对不上就抛异常（而不是默默少几行）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..contract.errors import fail

__all__ = ['DropLedger', 'CleanResult', 'clean']

#: 剔除原因 → 人话解释。错误信息和报告都用它。
REASON_TEXT = {
    'no_factor': '因子值缺失',
    'no_return': '前向收益缺失（入场或出场无价：停牌 / 退市 / 区间末尾）',
    'not_in_universe': '不在 as-of 股票池内（生存者偏差护栏）',
    'limit_up': '入场日一字涨停，买不进',
    'limit_down': '出场日一字跌停，卖不出',
    'suspended': '入场日停牌',
    'new_stock': '上市未满 N 个交易日（新股）',
    'not_tradable': '入场日不可成交（其他）',
    'no_group': '分组标签缺失',
    'no_exposure': '控制变量缺失',
    'duplicate': '重复的 (date, asset)',
}


def _fmt_sample(v, max_items=3):
    """把剔除样本的 ``[(日期, 代码), ...]`` 排成人话。

    修前是 ``str(list)`` —— 公众号报告里直接出现
    ``[(Timestamp('2023-01-31 00:00:00'), '000000'), ...]`` 这种 Python repr，
    能看但对读者不友好。

    >>> _fmt_sample([(pd.Timestamp('2023-01-31'), '000000'), (pd.Timestamp('2023-02-28'), '000001')])
    '2023-01-31 000000; 2023-02-28 000001'
    """
    if v is None:
        return ''
    if not isinstance(v, (list, tuple, set)):
        return str(v)
    items = list(v)
    if not items:
        return ''
    parts = []
    for it in items[:max_items]:
        if isinstance(it, (tuple, list)) and len(it) == 2:
            d, a = it
            try:
                ds = '—' if pd.isna(d) else pd.Timestamp(d).strftime('%Y-%m-%d')
            except Exception:                                    # noqa: BLE001
                ds = str(d)
            parts.append(f'{ds} {a}')
        else:
            parts.append(str(it))
    out = '; '.join(parts)
    if len(items) > max_items:
        out += f' …（共 {len(items)} 条）'
    return out



# --------------------------------------------------------------------------- #
@dataclass
class DropLedger:
    """剔除明细账。

    Attributes
    ----------
    counts : dict[str, int]
        原因 → 条数。**顺序即剔除顺序**，这样"同一个样本被两个原因命中时
        算哪个"是确定的。
    examples : dict[str, list]
        每个原因抽 3 个索引，便于直接去查原始数据。
    """

    counts: dict = field(default_factory=dict)
    examples: dict = field(default_factory=dict)
    n_input: int = 0
    n_output: int = 0

    def record(self, reason, index, n=None):
        """记一笔剔除。``index`` 是**被剔除样本**的索引（用于抽样例）。"""
        n = int(len(index) if n is None else n)
        if n <= 0:
            return
        self.counts[reason] = self.counts.get(reason, 0) + n
        if reason not in self.examples:
            self.examples[reason] = list(index[:3])

    @property
    def total_dropped(self) -> int:
        return sum(self.counts.values())

    def check(self, n_input=None, n_output=None):
        """不变量：输入 = 输出 + 剔除之和。对不上直接抛。"""
        ni = self.n_input if n_input is None else n_input
        no = self.n_output if n_output is None else n_output
        if ni != no + self.total_dropped:
            fail('clean', 'ledger_not_balanced',
                 f'账不平：输入 {ni:,} ≠ 输出 {no:,} + 剔除 {self.total_dropped:,} '
                 f'（差 {ni - no - self.total_dropped:+,}）。\n'
                 f'  含义：有条样本凭空消失或凭空出现 —— 这是 bug，不是数据问题。\n'
                 f'  明细：{self.counts}')
        return True

    def to_frame(self) -> pd.DataFrame:
        """明细账转成表，便于写进报告。"""
        rows = [{'reason': r, 'count': n,
                 'pct': (n / self.n_input * 100) if self.n_input else np.nan,
                 'meaning': REASON_TEXT.get(r, r),
                 'sample': self.examples.get(r, [])}
                for r, n in self.counts.items()]
        df = pd.DataFrame(rows)
        if len(df):
            df.loc[len(df)] = {'reason': '—— 保留 ——', 'count': self.n_output,
                               'pct': (self.n_output / self.n_input * 100)
                                      if self.n_input else np.nan,
                               'meaning': '进入分析', 'sample': ''}
        if len(df):
            # ★ `sample` 是给人看的（形如 "[(Timestamp(...), '301277'), ...]"）。
            #   保持 object 列会让整张表**无法写成 parquet**
            #   （ArrowTypeError），于是存盘目录里出现"14 张 parquet + 1 张 csv"
            #   的格式混杂。转成字符串后 15 张表格式统一，报告渲染不变。
            df['sample'] = df['sample'].map(_fmt_sample)
        return df

    def __str__(self):
        if not self.counts and not self.n_input:
            return '<DropLedger 空>'
        lines = [f'  输入 {self.n_input:,} 行']
        for r, n in self.counts.items():
            pct = n / self.n_input * 100 if self.n_input else 0
            lines.append(f'  − {n:>8,} ({pct:5.1f}%)  {r:<16} {REASON_TEXT.get(r, "")}')
        pct = self.n_output / self.n_input * 100 if self.n_input else 0
        lines.append(f'  = {self.n_output:>8,} ({pct:5.1f}%)  保留')
        return '\n'.join(lines)


# --------------------------------------------------------------------------- #
@dataclass
class CleanResult:
    """清洗结果。

    Attributes
    ----------
    data : DataFrame
        索引 ``(date, asset)``，列含 ``factor`` 与 ``forward_return_{h}`` 等。
    ledger : DropLedger
        **账。** 输入 = 输出 + 各类剔除。
    horizons : list[int]
    factor_name : str
    """

    data: pd.DataFrame
    ledger: DropLedger
    horizons: list = field(default_factory=list)
    factor_name: str = 'factor'

    def __post_init__(self):
        self.ledger.n_output = len(self.data)
        self.ledger.check()

    def return_cols(self):
        return [f'forward_return_{h}' for h in self.horizons
                if f'forward_return_{h}' in self.data.columns]

    def __repr__(self):
        drop = self.ledger.total_dropped
        rate = drop / self.ledger.n_input * 100 if self.ledger.n_input else 0
        return (f'<CleanResult {self.ledger.n_input:,} → {len(self.data):,} 行 '
                f'（剔除 {drop:,}, {rate:.1f}%）· 持有期 {self.horizons}>')


# --------------------------------------------------------------------------- #
def clean(factor, returns, *, tradability=None, universe=None,
          exposures=None, groupby=None, horizons=None, name='factor') -> CleanResult:
    """把因子与前向收益合成分析面板，**每一步剔除都记账**。

    Parameters
    ----------
    factor : FactorPanel | DataFrame
        需含 ``value``（或单列）。
    returns : Returns | DataFrame
        :func:`~alphalens_cna.engine.returns.forward_returns` 的输出。
    tradability : DataFrame, 可选
        若 ``returns`` 已按策略剔除过，这里不必再给（给了会二次过滤）。
    universe : Universe | DataFrame, 可选
        as-of 股票池。**强烈建议给** —— 不给就挡不住生存者偏差。
    exposures, groupby : DataFrame, 可选
        给了就会要求它们非缺失（缺的样本计入剔除）。
    horizons : list[int], 可选
        不传则从 ``returns`` 的列名推断。
    name : str
        因子名，进结果与账。

    Returns
    -------
    CleanResult
    """
    r = _unwrap(returns)
    if horizons is None:
        horizons = sorted({int(c.rsplit('_', 1)[1])
                           for c in r.columns if c.startswith('forward_return_')})
    if not horizons:
        fail('clean', 'no_horizons',
             'returns 里找不到 `forward_return_*` 列；'
             '是不是没跑 forward_returns()？')

    f = _factor_frame(factor)
    led = DropLedger()

    # ── 起点：因子与收益的**内连接** ──────────────────────────────
    # 用 left join 再记 "收益侧缺失"，这样账才能对上（inner 会静默少行）。
    base = f.join(r, how='left')
    led.n_input = len(base)

    keep = pd.Series(True, index=base.index)

    def drop(reason, mask):
        """按顺序剔除；已剔除的不再重复计入后面的原因。"""
        nonlocal keep
        m = keep.values & np.asarray(mask, dtype=bool)
        led.record(reason, base.index[m])
        keep = pd.Series(keep.values & ~m, index=base.index)

    # ① 重复索引
    if base.index.has_duplicates:
        dup = base.index.duplicated(keep='first')
        drop('duplicate', dup)
        base = base[keep.values]

    # ② 因子值缺失
    drop('no_factor', base['factor'].isna().values)

    # ③ 入场不可成交（**必须在 no_return 之前**）
    # returns 模块对这些样本也置了 NaN，若先跑 no_return 就会把
    # "一字涨停买不进" 笼统记成 "收益缺失"，账就失去了诊断价值。
    if 'entry_reason' in base.columns:
        for reason in ('limit_up', 'limit_down', 'suspended', 'new_stock',
                       'not_tradable'):
            m = (base['entry_reason'].values == reason) & keep.values
            if m.any():
                led.record(reason, base.index[m])
                keep = pd.Series(keep.values & ~m, index=base.index)

    # ④ 前向收益缺失（**逐个持有期都要有** —— 缺任何一个该样本都用不了）
    ret_cols = [f'forward_return_{h}' for h in horizons]
    missing_ret = base[ret_cols].isna().any(axis=1)
    drop('no_return', missing_ret.values)

    # ⑤ 股票池（生存者偏差护栏）
    if universe is not None:
        u = _unwrap(universe)
        col = 'in_universe' if 'in_universe' in u.columns else u.columns[0]
        # ⚠️ 不要写 ``.fillna(False)``：对象列上的 fillna 在 pandas 2.2+ 会告警
        #    （"Downcasting object dtype arrays on .fillna is deprecated"），
        #    而**追加 `.infer_objects(copy=False)` 并不能消掉它** —— 告警由 fillna
        #    自己发出，它不知道后面跟了什么（实测过）。
        #    `.where(notna(), False)` 值完全相同、零告警，再 astype(bool) 拿回 bool dtype。
        _u = u[col].reindex(base.index)
        inside = _u.where(_u.notna(), False).astype(bool)
        drop('not_in_universe', ~inside.values)

    # ⑥ 分组 / 控制变量缺失
    if groupby is not None:
        g = _unwrap(groupby)
        drop('no_group', g.iloc[:, 0].reindex(base.index).isna().values)
    if exposures is not None:
        e = _unwrap(exposures)
        # reindex 一引入缺失，bool 就会被提升成 object —— 同一条 fillna 告警的另一个入口
        # （测试者报的 274/283 就是这两处）。同样用 where 绕开，值不变。
        _ex = e.isna().any(axis=1).reindex(base.index)
        drop('no_exposure', _ex.where(_ex.notna(), True).astype(bool).values)

    # ── 组装输出 ─────────────────────────────────────────────────
    cols = ['factor'] + ret_cols
    cols += [f'overnight_gap_{h}' for h in horizons if f'overnight_gap_{h}' in base.columns]
    cols += [f'total_return_{h}' for h in horizons if f'total_return_{h}' in base.columns]
    cols += [c for c in ('entry_date', 'exit_date', 'calculated', 'factor_quantile')
             if c in base.columns]
    if groupby is not None:
        cols += list(_unwrap(groupby).columns)
    if exposures is not None:
        cols += list(_unwrap(exposures).columns)

    out = base.loc[keep.values, [c for c in dict.fromkeys(cols) if c in base.columns]]
    out = out.sort_index()

    led.n_output = len(out)
    return CleanResult(data=out, ledger=led, horizons=list(horizons),
                       factor_name=name)


# --------------------------------------------------------------------------- #
def _unwrap(obj):
    return getattr(obj, 'df', obj)


def _factor_frame(factor):
    """因子 → 单列 DataFrame，列名统一为 ``factor``。"""
    df = _unwrap(factor)
    if isinstance(df, pd.Series):
        df = df.to_frame('factor')
    if 'factor' in df.columns:
        return df[['factor']]
    if 'value' in df.columns:
        return df[['value']].rename(columns={'value': 'factor'})
    cand = [c for c in df.columns if c not in ('available_at', 'date', 'asset')]
    if len(cand) == 1:
        return df[[cand[0]]].rename(columns={cand[0]: 'factor'})
    fail('clean', 'ambiguous_factor',
         f'因子表看不出哪列是因子值（候选 {cand}）。\n'
         f'  修法：把列命名为 `value` 或 `factor`。')
