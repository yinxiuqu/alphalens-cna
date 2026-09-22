"""体检核心类型 —— ``Finding`` / ``HealthReport`` / 阈值默认值。

单独一层是为了避免 ``health/__init__`` 与各检查模块循环导入。

严重度（五档）
--------------
========  ==========================================================
``pass``  查过了，没问题
``info``  查过了，有值得知道的量，但不构成问题（如停牌率 0.8%）
``warn``  可疑，**看数的人需要知道**（如中途退市股为 0 → 存活偏差）
``fail``  数据有硬错误（如 ``high < low``、复权因子倒退）
``skip``  没查（缺输入）—— **绝不省略**，没查 ≠ 没问题
========  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..report import _md_table

__all__ = ['Finding', 'HealthReport', 'DEFAULTS', 'fmt_pct']

# 严重度排序（报告里按重要度展示）
_ORDER = {'fail': 0, 'warn': 1, 'info': 2, 'pass': 3, 'skip': 4}
_ICON = {'fail': '❌', 'warn': '⚠️', 'info': '·', 'pass': '✅', 'skip': '⏭'}

# 阈值默认值。每个都有依据，写在注释里；调用方可逐个覆盖。
DEFAULTS = {
    # 单日股票数 < 中位数 × 该比例 → 告警（数据缺口）
    'min_coverage_ratio': 0.5,
    # |日收益| 超过该值 → 告警。A 股制度上限：主板 10% / 创业科创 20% /
    # 北交所 30% / ST 5%，留足余量后 50% 几乎只可能是复权错误。
    'extreme_move': 0.5,
    # 截面唯一值少于该数 → 当天算不了 IC
    'min_cross_section': 2,
    # 因子值连续这么多期不变 → 疑似冻结/未更新
    'freeze_days': 20,
    # 单日停牌占比超过该值 → 疑似整段缺失而非真停牌
    'max_suspension_rate': 0.5,
    # 样本末尾这么多交易日内结束，不算"中途消失"
    'survivorship_grace': 20,
    # 单日复权因子跳变倍数（2.0 = 一天翻倍）→ 可疑
    'adj_jump': 2.0,
    # 因子陈旧度中位数超过这么多天 → 告警（年频财报 ≈ 365 天）
    'stale_days': 400,
    # 平均可买比例低于该值 → 告警
    'min_buyable_rate': 0.90,
}


@dataclass
class Finding:
    """一条体检结论。"""

    name: str
    severity: str
    summary: str
    metric: float | None = None
    detail: pd.DataFrame | None = None
    threshold: float | None = None

    def __post_init__(self):
        if self.severity not in _ORDER:
            raise ValueError(f'未知严重度 {self.severity!r}，'
                             f'只能是 {sorted(_ORDER)}')

    def __str__(self):
        m = '' if self.metric is None else f'  [{self.metric:.4g}]'
        return f'{_ICON[self.severity]} {self.name:<14} {self.summary}{m}'


@dataclass
class HealthReport:
    """体检总报告。"""

    findings: list = field(default_factory=list)

    # -- 分类 ---------------------------------------------------------------
    def _sev(self, *sev):
        return [f for f in self.findings if f.severity in sev]

    @property
    def fails(self):
        return self._sev('fail')

    @property
    def warns(self):
        return self._sev('warn')

    @property
    def passes(self):
        return self._sev('pass')

    @property
    def skips(self):
        return self._sev('skip')

    @property
    def n_checked(self):
        """真正查过的条数（不含 skip）。"""
        return len(self._sev('pass', 'info', 'warn', 'fail'))

    @property
    def warnings(self):
        """人话告警列表 —— 直接可塞进报告头部。"""
        return [f'{f.name}：{f.summary}' for f in self.fails + self.warns]

    @property
    def ok(self):
        """无 fail，**且**至少跑了一个检查。

        空报告不算通过 —— "没查" 和 "没问题" 是两件事。
        """
        return not self.fails and self.n_checked > 0

    # -- 输出 ---------------------------------------------------------------
    def __str__(self):
        lines = [f'数据体检（{self.n_checked} 项已查，'
                 f'{len(self.skips)} 项跳过）']
        for f in sorted(self.findings, key=lambda x: _ORDER[x.severity]):
            lines.append('  ' + str(f))
        lines.append(f'  结论：{"✅ 通过" if self.ok else "❌ 未通过"}'
                     + (f'（{len(self.fails)} 项硬错误）' if self.fails else ''))
        return '\n'.join(lines)

    def to_markdown(self, title='数据体检报告', max_detail=10):
        L = [f'# {title}', '']
        L.append(f'- 已查 **{self.n_checked}** 项，跳过 {len(self.skips)} 项')
        L.append(f'- 硬错误 **{len(self.fails)}**，告警 **{len(self.warns)}**')
        L.append(f'- 结论：**{"通过" if self.ok else "未通过"}**')
        L.append('')
        L.append('| 严重度 | 项目 | 结论 | 指标 | 阈值 |')
        L.append('|---|---|---|---|---|')
        for f in sorted(self.findings, key=lambda x: _ORDER[x.severity]):
            m = '—' if f.metric is None else f'{f.metric:.4g}'
            th = '—' if f.threshold is None else f'{f.threshold:.4g}'
            L.append(f'| {_ICON[f.severity]} {f.severity} | {f.name} | '
                     f'{f.summary} | {m} | {th} |')
        L.append('')
        for f in sorted(self.findings, key=lambda x: _ORDER[x.severity]):
            if f.detail is None or not len(f.detail):
                continue
            L.append(f'### {f.name} 明细')
            L.append('')
            L.append(_md_table(f.detail.head(max_detail)))
            if len(f.detail) > max_detail:
                L.append('')
                L.append(f'…共 {len(f.detail):,} 行，只列前 {max_detail} 行。')
            L.append('')
        return '\n'.join(L)


def fmt_pct(x):
    """百分比格式化（体检文案里到处在用）。"""
    return '—' if x is None or not np.isfinite(x) else f'{x:.2%}'
