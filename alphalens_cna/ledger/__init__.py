"""研究台账 —— 让 ``n_trials`` 从"凭记忆填"变成"自动记账"。

为什么这是本库的**灵魂**而不是附属品
------------------------------------
实测（`outputs/实施…`，本质检项目的老账）：

| 若共测试 | Bonferroni 阈值 | 换手因子 (t = −3.257) |
|---|---|---|
| 4 个 | 2.498 | 存活 |
| 20 个 | 3.023 | 存活 |
| **100 个** | **3.481** | **掉出去** |

**同一因子、同一数据，只因"你一共测了多少个"，结论就翻转。**

而现实中这个数字是**没人记得住的** —— 你不会记得自己下午三点还试过
"把窗口从 20 改成 25 看看"。于是最常见的学术不端不是造假，
而是**无意识地把自己搜过的上百个组合忘掉**，只报那个最好看的。

本模块做的事很小：**每次假设检验都记一笔，校正时回来数一数。**

用法
----
>>> led = ResearchLedger('roe_study.jsonl')
>>> led.record(goal='roe2026', factor='roe', horizon=21, t=-1.42)
>>> led.record(goal='roe2026', factor='roe', horizon=63, t=-1.71)
>>> led.n_trials('roe2026')          # → 2（自动数出来的，不是猜的）
>>> v = acna.assess(ic=ic, n_trials=led.n_trials('roe2026'))

设计取舍
--------
* **append-only JSONL**，不引数据库。可读、可 diff、可 git 管。
* ``n_trials`` 数的是**不同假设**（factor × horizon 去重），
  不是"跑了几次" —— 同一个检验重跑十遍不算十个假设。
* 记录里带 **timestamp**：什么时候想到的、什么时候改的口径，都留痕。
* **不联网、不上报**。台账是使用者自己的东西。
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

__all__ = ['Entry', 'ResearchLedger', 'default_path']

DEFAULT_FILE = '.alphalens_ledger.jsonl'


def default_path():
    """默认台账路径（可用环境变量 ``ALPHALENS_LEDGER`` 覆盖）。"""
    return os.environ.get('ALPHALENS_LEDGER', DEFAULT_FILE)


@dataclass
class Entry:
    """一次假设检验的记录。"""

    goal: str
    factor: str
    horizon: int | None = None
    t: float | None = None
    p: float | None = None
    n: int | None = None
    method: str = 'bhy'
    note: str = ''
    ts: float = field(default_factory=time.time)

    @property
    def key(self):
        """假设的唯一键 —— 校正基数按它去重。"""
        return (self.goal, self.factor, self.horizon)

    @property
    def when(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.ts))


class ResearchLedger:
    """研究台账。

    Parameters
    ----------
    path : str, optional
        JSONL 文件路径。默认 ``.alphalens_ledger.jsonl``（可用环境变量
        ``ALPHALENS_LEDGER`` 指定）。传 ``None`` 则该实例**只在内存里**
        （测试或一次性分析用）。
    """

    def __init__(self, path=None):
        self.path = default_path() if path is None and path is not False else path
        self._mem = []

    # -- 写 ---------------------------------------------------------------
    def record(self, *, goal, factor, horizon=None, t=None, p=None, n=None,
               method='bhy', note=''):
        """记一笔假设检验。返回 :class:`Entry`。"""
        if not goal or not factor:
            raise ValueError('`goal` 与 `factor` 都不能为空 —— '
                             '台账没名字就等于没记')
        e = Entry(goal=str(goal), factor=str(factor),
                  horizon=None if horizon is None else int(horizon),
                  t=None if t is None else float(t),
                  p=None if p is None else float(p),
                  n=None if n is None else int(n),
                  method=str(method), note=str(note))
        self._append(e)
        return e

    def _append(self, e):
        line = json.dumps(asdict(e), ensure_ascii=False)
        if self.path:
            d = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(d, exist_ok=True)
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        else:
            self._mem.append(e)

    # -- 读 ---------------------------------------------------------------
    def entries(self, goal=None):
        """全部记录（可按 ``goal`` 过滤）。"""
        out = list(self._mem)
        if self.path and os.path.exists(self.path):
            for line in open(self.path, encoding='utf-8'):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Entry(**json.loads(line)))
                except Exception:                                 # noqa: BLE001
                    continue        # 坏行跳过，但绝不因为一行坏掉整本台账
        if goal is not None:
            out = [e for e in out if e.goal == goal]
        return out

    def n_trials(self, goal=None, family=None):
        """**校正基数** —— 不同假设的个数（按 goal/factor/horizon 去重）。

        Parameters
        ----------
        goal : str, optional
            限定某个研究目标。
        family : str, optional
            ``'*'`` 表示**跨 goal 汇总**（"我这辈子一共测过多少"），
            默认按 goal 分开数 —— 因为校正应该在**同一个数据 + 同一批假设**
            的家族内做。
        """
        es = self.entries(None if family == '*' else goal)
        return len({e.key for e in es})

    def hypotheses(self, goal=None):
        """去重后的假设列表。"""
        seen = {}
        for e in self.entries(goal):
            seen.setdefault(e.key, e)
        return list(seen.values())

    def threshold(self, goal=None, alpha=0.05):
        """按台账记的假设数算 **Bonferroni |t| 门槛**（家族错误率控制）。

        用 Bonferroni 而不是 BHY：台账的用途是**给校正基数**，
        基数定下来之后用哪种校正由 ``assess(method=...)`` 决定；
        这里给最保守的那条线，方便一眼看出"要多大才算数"。
        台账为空 → NaN（**没记录就没有门槛**，不要假装是 1.96）。
        """
        from ..inference.multiplicity import t_threshold
        n = self.n_trials(goal)
        if n < 1:
            return np.nan
        return t_threshold(n, alpha=alpha)

    def frame(self, goal=None):
        """全部记录 → DataFrame。"""
        es = self.entries(goal)
        if not es:
            return pd.DataFrame(columns=['goal', 'factor', 'horizon', 't', 'p',
                                         'n', 'method', 'note', 'when'])
        d = pd.DataFrame([asdict(e) for e in es])
        d['when'] = [e.when for e in es]
        return d.drop(columns=['ts'])

    def summary(self, goal=None):
        """人话摘要。"""
        es = self.entries(goal)
        if not es:
            return f'台账为空（{self.path or "内存"}）'
        n = self.n_trials(goal)
        L = [f'研究台账：{len(es)} 条记录 / {n} 个不同假设'
             + (f'（goal={goal}）' if goal else '')]
        for g in sorted({e.goal for e in es}):
            sub = self.entries(g)
            L.append(f'  · {g}: {len(sub)} 条记录，'
                     f'{self.n_trials(g)} 个假设 → '
                     f'Bonferroni |t| 门槛 {self.threshold(g):.3f}')
        return '\n'.join(L)

    def __len__(self):
        return len(self.entries())

    def __repr__(self):
        return (f'ResearchLedger({self.path!r}, {len(self)} 条记录, '
                f'{self.n_trials()} 个假设)')

    def clear(self, goal=None):
        """清空（可按 goal）。内存实例直接清；文件实例重写剩余记录。"""
        if goal is None:
            self._mem = []
            if self.path and os.path.exists(self.path):
                os.remove(self.path)
            return
        rest = [e for e in self.entries() if e.goal != goal]
        self._mem = [e for e in rest if self.path is None]
        if self.path:
            with open(self.path, 'w', encoding='utf-8') as f:
                for e in rest:
                    f.write(json.dumps(asdict(e), ensure_ascii=False) + '\n')

    # -- 与 assess 对接 ---------------------------------------------------
    def check_n_trials(self, claimed, goal=None):
        """**核对**使用者报的 ``n_trials`` 与台账实际记录。

        Returns
        -------
        (ok, message)
            报了比台账少的数 → 不通过（那就是选择性汇报）。
        """
        actual = self.n_trials(goal)
        if claimed is None:
            return False, f'没报 n_trials；台账里已记 {actual} 个假设'
        if claimed < actual:
            return False, (f'报的 n_trials={claimed} **少于**台账记录的 {actual} —— '
                           f'校正基数被低报，结论会偏乐观')
        if claimed > actual:
            return True, f'报的 {claimed} 多于台账的 {actual}（更保守，可接受）'
        return True, f'与台账一致（{actual}）'

    def fingerprint(self):
        """台账指纹 —— 便于把"结论是基于哪本台账"写进报告。"""
        h = hashlib.md5()
        for e in sorted(self.entries(), key=lambda x: (x.goal, x.factor,
                                                       x.horizon or -1)):
            h.update(f'{e.goal}|{e.factor}|{e.horizon}'.encode())
        return h.hexdigest()[:12]
