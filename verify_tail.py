"""尾部统计在**真实 ROE 面板**上的表现 —— 看它能否揭穿 IC 的盲区。

对照两种退市收益约定，输出分位 × 尾部统计 + Q5−Q1 崩盘率差（带 NW t）。
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

LEVEL = 0.10

def main():
    px, op = A.load_panels()
    codes = [c for c in px.columns if px[c].notna().sum() > 250 and c in op.columns]
    me = pd.Series(px.index, index=px.index).resample('ME').last().dropna()
    rebal = pd.DatetimeIndex(me.values)
    rebal = rebal[(rebal >= A.FACTOR_START) & (rebal <= A.FACTOR_END)]
    prices = A.make_prices(px[codes], op[codes]); cal = acna.Calendar(px.index)
    factor = A.build_factor(rebal, codes)

    out = {}
    ABS = -0.50        # 绝对门槛：一年腰斩
    for pol in ('nan', 'last_price'):
        r = forward_returns(prices, cal, list(A.HORIZONS),
                            model=ReturnModel(entry='same_open', exit_price='open',
                                              delist_policy=pol))
        cr = clean(factor, r, name='roe_pit')
        df = cr.data.join(acna.quantize(cr, n=5)[['q']])
        tb = acna.tail_by_quantile(df, level=LEVEL)
        sp = acna.crash_spread(df, level=LEVEL)
        tb_abs = acna.tail_by_quantile(df, level=LEVEL, threshold=ABS)
        sp_abs = acna.crash_spread(df, level=LEVEL, threshold=ABS)
        out[pol] = {'tail': tb, 'spread': sp, 'tail_abs': tb_abs, 'spread_abs': sp_abs}
        print('=' * 76)
        print(f'退市约定 = {pol}   清洗后 {len(cr.data):,} 行   门槛 = 当日截面最差 {LEVEL:.0%}')
        print('=' * 76)
        for h in A.HORIZONS:
            sub = tb.xs(h, level='h')
            print(f'  h={h:>3}  均值: ' + ' '.join(f'Q{int(q)}={sub.loc[q, "mean"]:+.3f}' for q in sub.index))
            print(f'         崩盘率: ' + ' '.join(f'Q{int(q)}={sub.loc[q, "hit_rate"]:.1%}' for q in sub.index))
            print(f'         CVaR:  ' + ' '.join(f'Q{int(q)}={sub.loc[q, "cvar"]:+.3f}' for q in sub.index))
            s = sp.loc[h]
            print(f'   分位门槛 Q5−Q1 崩盘率差 = {s["hit_spread"]:+.3f} (t={s["t_hit"]:+.2f})'
                  f'   均值差 = {s["mean_spread"]:+.4f}')
            a = tb_abs.xs(h, level='h'); sa = sp_abs.loc[h]
            print(f'   绝对门槛(腰斩) 崩盘率: ' + ' '.join(
                f'Q{int(q)}={a.loc[q, "hit_rate"]:.1%}' for q in a.index)
                + f'   Q5−Q1 = {sa["hit_spread"]:+.3f} (t={sa["t_hit"]:+.2f})')
        print()
    print('=' * 76)
    print('对照汇总（h=252）')
    print('=' * 76)
    hdr = (f'  {"口径":<12}{"Q1腰斩率":>10}{"Q5腰斩率":>10}{"腰斩差":>9}{"t_NW":>8}'
           f'{"Q1 CVaR":>10}{"Q5 CVaR":>10}{"均值差":>10}')
    print(hdr)
    for pol in out:
        s = out[pol]['spread_abs'].loc[252]; t = out[pol]['tail'].xs(252, level='h')
        print(f'  {pol:<12}{s["hit_lo"]:>10.1%}{s["hit_hi"]:>10.1%}{s["hit_spread"]:>+9.3f}'
              f'{s["t_hit"]:>+8.2f}{t.loc[1, "cvar"]:>+10.3f}{t.loc[5, "cvar"]:>+10.3f}'
              f'{out[pol]["spread"].loc[252, "mean_spread"]:>+10.4f}')

    with open(os.path.join(HERE, 'outputs', 'tail_roe.json'), 'w') as f:
        json.dump({p: {k: v.reset_index().to_dict('records')
                       for k, v in out[p].items()}
                   for p in out}, f, ensure_ascii=False, indent=2, default=str)
    print('\n已存 outputs/tail_roe.json')

if __name__ == '__main__':
    main()
