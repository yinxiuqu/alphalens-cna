"""尾部信号的**样本外验证** —— 把探索性发现变成结论的必要一步。

上一轮发现（全样本）：低 ROE 组腰斩率显著更高，且呈 U 型。
但那是在同一份数据上用新统计量做的**探索性**发现，必须分段复核。

分段：2019-01 ~ 2022-12（前半） / 2023-01 ~ 2026-07（后半）
门槛：绝对门槛 −50%（腰斩）—— 分位门槛对等规模分组不敏感，已弃用。
"""
from __future__ import annotations
import json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('PIT_MODE', 'delisted+zerofix')
import accept_m0 as A, alphalens_cna as acna
from alphalens_cna.engine.clean import clean
from alphalens_cna.engine.returns import ReturnModel, forward_returns

ABS = -0.50
SPLIT = pd.Timestamp('2023-01-01')


def main():
    px, op = A.load_panels()
    codes = [c for c in px.columns if px[c].notna().sum() > 250 and c in op.columns]
    me = pd.Series(px.index, index=px.index).resample('ME').last().dropna()
    rebal = pd.DatetimeIndex(me.values)
    rebal = rebal[(rebal >= A.FACTOR_START) & (rebal <= A.FACTOR_END)]
    prices = A.make_prices(px[codes], op[codes]); cal = acna.Calendar(px.index)
    factor = A.build_factor(rebal, codes)

    r = forward_returns(prices, cal, list(A.HORIZONS),
                        model=ReturnModel(entry='same_open', exit_price='open',
                                          delist_policy='last_price'))
    cr = clean(factor, r, name='roe_pit')
    df = cr.data.join(acna.quantize(cr, n=5)[['q']]).rename(
        columns={'q': 'factor_quantile'})

    segs = {'前半 2019-2022': lambda d: d < SPLIT,
            '后半 2023-2026': lambda d: d >= SPLIT,
            '全样本': lambda d: pd.Series(True, index=d.index)}
    out, rows = {}, []
    for name, fn in segs.items():
        dates = df.index.get_level_values('date')
        sub = df[fn(pd.Series(dates, index=df.index)).values]
        tb = acna.tail_by_quantile(sub, threshold=ABS)
        sp = acna.crash_spread(sub, threshold=ABS)
        out[name] = {'tail': tb.reset_index().to_dict('records'),
                     'spread': sp.reset_index().to_dict('records')}
        print('=' * 78)
        print(f'{name}   清洗后 {len(sub):,} 行   '
              f'调仓 {sub.index.get_level_values("date").nunique()} 个')
        print('=' * 78)
        print(f'  {"h":>4} {"Q1":>7} {"Q2":>7} {"Q3":>7} {"Q4":>7} {"Q5":>7} '
              f'{"Q5-Q1":>8} {"t_NW":>7} {"均值差":>9}')
        for h in A.HORIZONS:
            t = tb.xs(h, level='h'); s = sp.loc[h]
            hs = {int(q): t.loc[q, 'hit_rate'] for q in t.index}
            print(f'  {h:>4} ' + ' '.join(f'{100 * hs[q]:>6.1f}%' for q in sorted(hs))
                  + f' {s["hit_spread"]:>+8.3f} {s["t_hit"]:>+7.2f} '
                    f'{s["mean_spread"]:>+9.4f}')
            rows.append({'seg': name, 'h': h,
                         **{f'Q{q}_hit': hs[q] for q in sorted(hs)},
                         'hit_spread': s['hit_spread'], 't_hit': s['t_hit'],
                         'mean_spread': s['mean_spread'],
                         'n_periods': s['n_periods']})
        print()

    tab = pd.DataFrame(rows)
    print('=' * 78)
    print('稳定性判定（腰斩率差 Q5−Q1，t_NW）')
    print('=' * 78)
    piv = tab.pivot(index='h', columns='seg', values='t_hit')
    print(piv.to_string())
    print()
    for h in A.HORIZONS:
        a = tab[(tab.h == h) & (tab.seg == '前半 2019-2022')].iloc[0]
        b = tab[(tab.h == h) & (tab.seg == '后半 2023-2026')].iloc[0]
        same = (a['hit_spread'] < 0) and (b['hit_spread'] < 0)
        print(f'  h={h:>3}  前半 {a["hit_spread"]:+.3f} (t={a["t_hit"]:+.2f}, n={a["n_periods"]})'
              f'   后半 {b["hit_spread"]:+.3f} (t={b["t_hit"]:+.2f}, n={b["n_periods"]})'
              f'   → {"同向 ✅" if same else "不同向 ❌"}')
    with open(os.path.join(HERE, 'outputs', 'tail_oos.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print('\n已存 outputs/tail_oos.json')


if __name__ == '__main__':
    main()
