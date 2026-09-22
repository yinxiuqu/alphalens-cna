"""用**真实 A 股面板**跑一遍数据体检 —— 看它到底能抓出什么。

数据：``cache/px_daily.parquet``（全市场日频，含已退市股）+ 月度 ROE 因子。
输出：``outputs/health_real.md``
"""
from __future__ import annotations

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


def main():
    px, op = A.load_panels()
    codes = [c for c in px.columns if px[c].notna().sum() > 250 and c in op.columns]
    prices = A.make_prices(px[codes], op[codes])
    cal = acna.Calendar(px.index)

    me = pd.Series(px.index, index=px.index).resample('ME').last().dropna()
    rebal = pd.DatetimeIndex(me.values)
    rebal = rebal[(rebal >= A.FACTOR_START) & (rebal <= A.FACTOR_END)]
    factor = A.build_factor(rebal, codes)

    print(f'行情面板 {prices.shape}  股票 {len(codes)} 只  '
          f'{px.index[0].date()} ~ {px.index[-1].date()}')
    print(f'因子面板 {factor.shape}  调仓日 {len(rebal)} 个\n')

    rep = acna.health_check(prices=prices, factor=factor, calendar=cal,
                            name=f'全市场日频行情 + ROE 月度因子')
    print(rep)

    out = os.path.join(HERE, 'outputs', 'health_real.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(rep.to_markdown('数据体检报告 · 真实 A 股面板'))
    print(f'\n已存 {out}')


if __name__ == '__main__':
    main()
