"""换手 / 秩自相关口径核实 —— 用真实 ROE 月度面板。

回答三个问题：
1. alphalens 在**真实** A 股月频面板上到底返不返 NaN（D9 的真实证据）；
2. 本库默认口径（= alphalens 口径，单边）与旧的对称口径差多少；
3. 两种口径各自的经济含义 —— 年化成本估算该用哪个。

数据复用 ``accept_m0.py`` 的加载逻辑（缓存面板 + PIT parquet）。
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import accept_m0 as A                                            # noqa: E402
import alphalens_cna as acna                                     # noqa: E402
from alphalens_cna.engine.clean import clean                     # noqa: E402
from alphalens_cna.engine.returns import ReturnModel, forward_returns  # noqa: E402


def main():
    px, op = A.load_panels()
    cal_idx = px.index
    me = pd.Series(cal_idx, index=cal_idx).resample('ME').last().dropna()
    rebal = pd.DatetimeIndex(me.values)
    rebal = rebal[(rebal >= A.FACTOR_START) & (rebal <= A.FACTOR_END)]
    codes = [c for c in px.columns if px[c].notna().sum() > 250 and c in op.columns]

    prices = A.make_prices(px[codes], op[codes])
    cal = acna.Calendar(cal_idx)
    factor = A.build_factor(rebal, codes)

    # M0 验收口径：T 开盘 → T+h 开盘
    r = forward_returns(prices, cal, list(A.HORIZONS),
                        model=ReturnModel(entry='same_open', exit_price='open'))
    cr = clean(factor, r, name='roe_pit')
    q = acna.quantize(cr, n=A.QUANTILES)

    print(f'月度调仓 {len(rebal)} 个  股票 {len(codes)} 只  '
          f'清洗后 {len(cr.data):,} 行')
    print(f'调仓日索引 freq = {rebal.freq}  ← None 就是 D9 的触发条件\n')

    out = {'rebalances': int(len(rebal)), 'universe': len(codes),
           'freq_is_none': rebal.freq is None}

    # ── ① 本库两种口径 ──────────────────────────────────────────────
    print('=' * 74)
    print('本库 quantile_turnover —— 两种口径（月度，91 期）')
    print('=' * 74)
    to_al = acna.quantile_turnover(q['q'])                        # 默认
    to_sym = acna.quantile_turnover(q['q'], method='symmetric')
    hdr = '  分位    alphalens 口径    symmetric 口径    差'
    print(hdr)
    for c in to_al.columns:
        a, b = float(to_al[c].mean()), float(to_sym[c].mean())
        print(f'  Q{int(c):<5} {a:>13.4f} {b:>17.4f} {a - b:>+9.4f}')
    avg_al = float(to_al.mean().mean())
    avg_sym = float(to_sym.mean().mean())
    print(f'  {"平均":<6} {avg_al:>13.4f} {avg_sym:>17.4f} {avg_al - avg_sym:>+9.4f}')
    out['ours_alphalens_method'] = {int(c): float(to_al[c].mean()) for c in to_al.columns}
    out['ours_symmetric_method'] = {int(c): float(to_sym[c].mean()) for c in to_sym.columns}
    out['ours_avg_alphalens'] = avg_al
    out['ours_avg_symmetric'] = avg_sym

    # ── ② alphalens 本尊 ────────────────────────────────────────────
    print('\n' + '=' * 74)
    print('alphalens 本尊 —— 同一份面板')
    print('=' * 74)
    try:
        from alphalens import performance as aperf
        from alphalens import utils as autils

        wide = prices['adj_close'].unstack('asset')
        af = autils.get_clean_factor_and_forward_returns(
            factor['value'], wide, quantiles=A.QUANTILES,
            periods=tuple(A.HORIZONS), max_loss=0.7)
        al_to = {qq: float(aperf.quantile_turnover(
            af['factor_quantile'], qq, period=1).mean())
            for qq in range(1, A.QUANTILES + 1)}
        print('  quantile_turnover :', al_to)
        ra_al = aperf.factor_rank_autocorrelation(af, period=1)
        print(f'  factor_rank_autocorr.mean() = {ra_al.mean()}  '
              f'(有效值 {int(ra_al.notna().sum())}/{len(ra_al)})')
        out['alphalens_turnover'] = al_to
        out['alphalens_rank_autocorr_mean'] = (
            None if pd.isna(ra_al.mean()) else float(ra_al.mean()))
    except Exception as e:                                        # noqa: BLE001
        print(f'  alphalens 抛异常：{type(e).__name__}: {e}')
        out['alphalens_error'] = f'{type(e).__name__}: {e}'

    # ── ③ 本库的秩自相关 ────────────────────────────────────────────
    ra = acna.rank_autocorrelation(cr)
    print(f'\n  本库 rank_autocorrelation = {ra.mean():.4f}  '
          f'({len(ra)} 期，有效 {int(ra.notna().sum())})')
    out['ours_rank_autocorr_mean'] = float(ra.mean())

    # ── ④ 经济含义：年化交易成本 ────────────────────────────────────
    print('\n' + '=' * 74)
    print('落到钱上：年化单边交易成本（12 次调仓/年，单边 15bp）')
    print('=' * 74)
    for lab, v in (('alphalens 口径', avg_al), ('symmetric 口径', avg_sym)):
        # 每次调仓要动的名义 = 换手率；单边成本 = 换手 × 单边费率
        annual = v * 12 * 0.0015
        print(f'  {lab:<18} 月换手 {v:6.2%} → 年换手 {v * 12:6.1%} '
              f'→ 年成本 {annual:.3%}')
        out[f'annual_cost_{lab.split()[0]}'] = annual

    with open(os.path.join(HERE, 'outputs', 'turnover_check.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('\n结果已存 outputs/turnover_check.json')


if __name__ == '__main__':
    main()
