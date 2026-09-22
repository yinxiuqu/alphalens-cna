"""生成 README 上的示范图 —— **只用合成数据，任何人都能复现**。

    python examples/make_charts.py        # → docs/images/demo.png

为什么不用真实数据出的图
------------------------
本库的卖点是"证据可复现"。用私有数据生成的图，读者只能看、不能验 ——
那正好违背它自己的原则。所以示范图一律来自 ``examples/quickstart.py``
同一份合成数据；真实数据的图表放在 ``docs/cases/``，并明确标注不可复现。

绘图库只在**这个脚本**里用，核心包零绘图依赖（见 CI 的 no-plotting-dep job）。
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alphalens_cna as acna                      # noqa: E402
from quickstart import make_data                  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'docs', 'images', 'demo.png')


def main():
    px, f, cal = make_data(n_days=400, n_assets=120, seed=11)
    rep = acna.build_report(f, px, cal, horizons=(21, 63, 126),
                            quantiles=5, n_trials=3, crash_threshold=-0.30,
                            name='demo')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle('alphalens-cna — reproducible demo (synthetic data, '
                 'python examples/make_charts.py)', fontsize=11)

    # ① 逐期 IC 时序
    ax = axes[0, 0]
    ic = rep.ic.dropna(how='all')
    for c in ic.columns:
        ax.plot(ic.index, ic[c].rolling(20, min_periods=5).mean(),
                lw=1.6, label=f'h={c}')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('Rolling mean IC (20 periods)')
    ax.set_ylabel('Rank IC'); ax.legend(fontsize=8)

    # ② IC 汇总 + Newey-West 修正
    ax = axes[0, 1]
    s = rep.ic_summary
    x = np.arange(len(s.index)); w = 0.38
    ax.bar(x - w / 2, s['mean'].values, w, label='mean IC', color='#4C72B0')
    t = rep.nw['t_nw'].reindex(s.index).values if 't_nw' in rep.nw else None
    if t is not None:
        ax2 = ax.twinx()
        ax2.plot(x, t, 'o-', color='#C44E52', label='t (Newey-West)')
        ax2.axhline(0, color='k', lw=0.6); ax2.set_ylabel('t')
        ax2.legend(fontsize=8, loc='lower left')
    ax.set_xticks(x); ax.set_xticklabels([f'h={i}' for i in s.index])
    ax.set_title('IC by horizon'); ax.set_ylabel('IC')
    ax.legend(fontsize=8)

    # ③ 分位平均收益
    ax = axes[1, 0]
    q = rep.quantile_stats
    cols = [c for c in q.columns if c.startswith('q') and c.endswith('_mean')]
    if cols:
        qq = q[cols].T
        qq.index = [c.split('_')[0].upper() for c in cols]
        qq.plot(kind='bar', ax=ax, legend=True, width=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('Mean forward return by quantile')
    ax.set_ylabel('return'); ax.legend(fontsize=8)

    # ④ 尾部风险 —— IC 看不见的那一块
    ax = axes[1, 1]
    cr = rep.tail['hit_rate'].unstack('q')
    cr.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Crash hit rate by quantile (threshold −30%)')
    ax.set_ylabel('hit rate'); ax.set_xlabel('horizon')
    ax.legend(fontsize=8, title='quantile')

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f'✓ 已写出 {os.path.relpath(OUT)}  ({os.path.getsize(OUT) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
