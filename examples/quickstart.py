"""三分钟上手 —— **只用合成数据**，不依赖任何私有数据源。

    python examples/quickstart.py

会打印体检、结论、尾部风险，并在当前目录写出 report.md。
"""
import numpy as np
import pandas as pd

import alphalens_cna as acna


def make_data(n_days=300, n_assets=80, seed=7):
    """造一份**契约合法**的面板：三价并存 + 停牌 + 涨跌停 + 一个弱因子。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2022-01-03', periods=n_days)
    assets = [f'{i:06d}' for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    n = len(idx)

    ret = rng.normal(0.0003, 0.02, (n_days, n_assets))
    close = 10 * np.exp(np.cumsum(ret, axis=0))
    c = pd.DataFrame(close.ravel(), index=idx, columns=['c'])['c']
    o = c * (1 + rng.normal(0, 0.004, n))
    px = pd.DataFrame({
        'raw_open': o, 'raw_close': c,
        'raw_high': np.maximum(o, c) * 1.005, 'raw_low': np.minimum(o, c) * 0.995,
        'adj_factor': 1.0, 'volume': rng.integers(1e5, 1e7, n).astype(float),
    }, index=idx)
    for col in ('open', 'close', 'high', 'low'):
        px[f'adj_{col}'] = px[f'raw_{col}']
    px['prev_close'] = px['raw_close'].groupby(level='asset').shift(1).fillna(px['raw_open'])

    # 停牌：随机 0.5% 的日子没有价格
    susp = rng.random(n) < 0.005
    px.loc[susp, ['adj_open', 'adj_close', 'adj_high', 'adj_low',
                  'raw_open', 'raw_close', 'raw_high', 'raw_low', 'volume']] = np.nan

    # 因子：与下期收益**弱正相关**（让报告里能看到点东西）
    fwd = pd.Series(px['adj_close'].groupby(level='asset').pct_change().shift(-21),
                    index=idx)
    f = pd.DataFrame({
        'value': 0.10 * fwd.fillna(0).to_numpy() + rng.normal(size=n) * 0.05 +
                 rng.normal(size=n) * 0.02,
        'available_at': idx.get_level_values('date'),
    }, index=idx)
    return px, f, acna.Calendar(dates)


def main():
    px, f, cal = make_data()
    print('=' * 78)
    print('① 数据体检（防线 2）')
    print('=' * 78)
    h = acna.health_check(prices=px, factor=f, calendar=cal, name='合成数据')
    print(h)

    print()
    print('=' * 78)
    print('② 全流程报告（防线 3 + 4 + 5 在内部自动跑）')
    print('=' * 78)
    rep = acna.build_report(f, px, cal, horizons=(21, 63, 126),
                            quantiles=5, n_trials=3, crash_threshold=-0.30,
                            name='demo')
    print(rep.verdict)

    print()
    print('=' * 78)
    print('③ 尾部风险 —— IC 看不见的那一块')
    print('=' * 78)
    print(rep.crash[['hit_lo', 'hit_hi', 'hit_spread', 't_hit']].to_string())

    print()
    print('=' * 78)
    print('④ 诊断量：扰动鲁棒性 + 排名熵稳定性')
    print('=' * 78)
    ic = rep.ic.iloc[:, 0].dropna()
    print(f'  PFS(扰动鲁棒性) = {acna.pfs(ic, n_draws=200):.3f}')
    print(f'  RRE(排名稳定性) = {acna.rank_stability(f):.3f}')

    md = rep.to_markdown('合成数据 · 快速上手')
    with open('report.md', 'w', encoding='utf-8') as fh:
        fh.write(md)
    print(f'\n已写出 report.md（{len(md):,} 字符，共 '
          f'{md.count(chr(10) + "## ")} 节）')
    print('tidy 表：', ', '.join(sorted(rep.frames())))


if __name__ == '__main__':
    main()
