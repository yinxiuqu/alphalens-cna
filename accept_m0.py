"""M0 验收：用 alphalens-cna 复现《ROE月度调仓分析报告》的 RankIC。

口径对齐（关键）
----------------
原脚本用的是 ``op.shift(-h)/op - 1``，即 **T 开盘 → T+h 开盘**。
在本库里对应 ``ReturnModel(entry='same_open', exit_price='open')``。

原脚本的 ``close`` 口径是 ``px.shift(-h)/px - 1``，对应
``ReturnModel(entry='close', exit_price='close')``。

数据说明
--------
缓存面板 ``px_daily*.parquet`` 是**已复权**价（close×adj / open×adj）。
所以构造契约对象时令 ``raw_* = adj_*``、``adj_factor = 1`` —— 契约自洽成立。
IC 计算不使用原始价，这个简化不影响结论（但真实分析**必须**用原始价判涨跌停）。
"""
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alphalens_cna as acna
from alphalens_cna.engine.clean import clean
from alphalens_cna.engine.returns import ReturnModel, forward_returns

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
PIT = '/home/yinxiuqu/quantming/data/financial_pit/financial_pit.parquet'

# PIT 模式（三段对照，环境变量切换）：
#   original         原始 PIT（= 早前报告的基线，含存活偏差）
#   delisted         并入 250 只退市股（行情+财务都有了）
#   delisted+zerofix 再把 roe==0 视为缺失（0 是占位，不是真值）
MODE = os.environ.get('PIT_MODE', 'original')

# 退市收益约定（持有期内股票消失时怎么算）：
#   nan        保持旧行为：剔除（长持有期会系统性丢掉退市前那一段）
#   last_price 按最后成交价清算，之后视为现金
#   haircut    最后成交价再打 DELIST_RETURN 折扣
DELIST_POLICY = os.environ.get('DELIST_POLICY', 'nan')
DELIST_RETURN = float(os.environ.get('DELIST_RETURN', '-0.3'))


def load_pit():
    """按 MODE 返回 PIT 表（列与原表一致）。"""
    if MODE == 'original':
        return pd.read_parquet(PIT)
    import fetch_delisted_financials as FDF
    if MODE == 'delisted':
        return FDF.load_pit(with_delisted=True, zerofix=False)
    if MODE == 'delisted+zerofix':
        return FDF.load_pit(with_delisted=True, zerofix=True)
    raise SystemExit(f'未知 PIT_MODE={MODE!r}')

FACTOR_START, FACTOR_END = '2019-01-01', '2026-07-31'
HORIZONS = (21, 63, 126, 252)
QUANTILES = 5
STALENESS_DAYS = 400

# 原报告（open 口径）的 RankIC —— 验收目标
TARGET_OPEN = {21: -0.0108, 63: -0.0205, 126: -0.0391, 252: -0.0593}
# 原报告（open 口径）的 Q5−Q1
TARGET_SPREAD = {21: -0.0067, 63: -0.0168, 126: -0.0407, 252: -0.0728}


def load_panels():
    px = pd.read_parquet(os.path.join(CACHE, 'px_daily.parquet'))
    px.index = pd.to_datetime(px.index)
    op = pd.read_parquet(os.path.join(CACHE, 'px_daily_open.parquet'))
    op.index = pd.to_datetime(op.index)
    return px, op


def build_factor(rebal, codes):
    """与原脚本同法：avail_314 + merge_asof(backward)。"""
    pit = load_pit()
    pit = pit[['code', 'roe', 'avail_314']]
    pit = pit[pit['avail_314'].notna()].copy()
    pit['avail_314'] = pd.to_datetime(pit['avail_314'])
    pit = pit[pit['code'].astype(str).str.zfill(6).isin(codes)]
    pit['code'] = pit['code'].astype(str).str.zfill(6)
    pit = pit.sort_values('avail_314', kind='mergesort')

    panel = pd.DataFrame(
        [(d, c) for d in rebal for c in codes], columns=['date', 'asset'])
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values('date', kind='mergesort')
    m = pd.merge_asof(panel, pit, left_on='date', right_on='avail_314',
                      left_by='asset', right_by='code', direction='backward')
    m['staleness_days'] = (m['date'] - m['avail_314']).dt.days
    m = m[m['staleness_days'].notna() & (m['staleness_days'] <= STALENESS_DAYS)]
    f = pd.DataFrame({
        'value': pd.to_numeric(m['roe'], errors='coerce').values,
        'available_at': m['date'].values,
        'staleness_days': m['staleness_days'].values,
    }, index=pd.MultiIndex.from_arrays([m['date'].values, m['asset'].values],
                                       names=['date', 'asset']))
    return f.dropna(subset=['value']).sort_index()


def make_prices(px, op):
    """缓存已复权 → raw=adj, factor=1（契约自洽；IC 不用原始价）。

    缓存只有开/收盘两个面板，没有 high/low。这里用
    ``high = max(open, close)``、``low = min(open, close)`` 合成 ——
    **IC 计算不使用 high/low**，合成只为满足契约。
    （真实分析必须用真实 high/low：涨跌停判定要用它们。）
    """
    cols = {}
    for name, df in (('close', px), ('open', op)):
        s = df.stack()
        s.index = s.index.set_names(['date', 'asset'])
        cols[f'adj_{name}'] = s
    out = pd.DataFrame(cols).dropna()
    out['adj_high'] = out[['adj_open', 'adj_close']].max(axis=1)
    out['adj_low'] = out[['adj_open', 'adj_close']].min(axis=1)
    for c in ('open', 'high', 'low', 'close'):
        out[f'raw_{c}'] = out[f'adj_{c}']
    out['prev_close'] = out.groupby(level='asset')['raw_close'].shift(1)
    out['adj_factor'] = 1.0
    return out.sort_index()


def main():
    px, op = load_panels()
    cal_idx = px.index
    me = pd.Series(cal_idx, index=cal_idx).resample('ME').last().dropna()
    rebal = pd.DatetimeIndex(me.values)
    rebal = rebal[(rebal >= FACTOR_START) & (rebal <= FACTOR_END)]
    codes = [c for c in px.columns if px[c].notna().sum() > 250 and c in op.columns]
    print(f'收盘面板 {px.shape}  开盘面板 {op.shape}')
    print(f'月度调仓 {len(rebal)} 个: {rebal[0].date()} ~ {rebal[-1].date()}，股票 {len(codes)} 只')

    prices = make_prices(px[codes], op[codes])
    cal = acna.Calendar(cal_idx)
    factor = build_factor(rebal, codes)
    print(f'ROE 因子 {len(factor):,} 条 '
          f'({factor.index.get_level_values("date").nunique()} 个调仓日)')

    out = {'config': {'pit_mode': MODE, 'delist_policy': DELIST_POLICY,
                      'delist_return': DELIST_RETURN,
                      'horizons': list(HORIZONS),
                      'quantiles': QUANTILES,
                      'rebalances': int(len(rebal)), 'universe': len(codes)}}
    for tag, model in (('open', ReturnModel(entry='same_open', exit_price='open',
                                            delist_policy=DELIST_POLICY,
                                            delist_return=DELIST_RETURN)),
                       ('close', ReturnModel(entry='close', exit_price='close',
                                             delist_policy=DELIST_POLICY,
                                             delist_return=DELIST_RETURN))):
        r = forward_returns(prices, cal, list(HORIZONS), model=model)
        cr = clean(factor, r, name='roe_pit')
        ic = acna.information_coefficient(cr)
        summ = acna.ic_summary(ic)
        q = acna.quantize(cr, n=QUANTILES)
        qs = acna.quantile_stats(cr, quantiles=q)
        out[tag] = {
            'delist': {int(h): {k: v for k, v in led.items()
                                if k in ('delist_filled', 'delisted_dropped',
                                         'no_exit_price')}
                       for h, led in r.ledger.items()},
            'rankic': {int(h): float(summ.loc[h, 'mean']) for h in summ.index},
            'n': {int(h): int(summ.loc[h, 'n']) for h in summ.index},
            'spread': {int(h): float(qs.loc[h, 'spread']) for h in qs.index},
            'ledger': dict(cr.ledger.counts),
            'n_input': cr.ledger.n_input, 'n_output': cr.ledger.n_output,
        }
        print(f'\n--- {tag} 口径 ---')
        print(f'  清洗账: {cr.ledger.n_input:,} → {cr.ledger.n_output:,}')
        for h in HORIZONS:
            tgt = TARGET_OPEN.get(h) if tag == 'open' else None
            mark = ''
            if tgt is not None:
                d = abs(summ.loc[h, 'mean'] - tgt)
                mark = f'   原报告 {tgt:+.4f}  差 {d:.5f} {"✅" if d < 5e-4 else "⚠️"}'
            print(f'  h={h:>3}  RankIC {summ.loc[h, "mean"]:+.4f} '
                  f'(n={int(summ.loc[h, "n"])})  Q5-Q1 {qs.loc[h, "spread"]:+.4f}{mark}')

    os.makedirs(os.path.join(HERE, 'outputs'), exist_ok=True)
    tag = MODE.replace('+', '_')
    if DELIST_POLICY != 'nan':
        tag += '_' + DELIST_POLICY + (f'{DELIST_RETURN:+.2f}'.replace('.', '')
                                      if DELIST_POLICY == 'haircut' else '')
    fn = os.path.join(HERE, 'outputs',
                      'm0_acceptance.json' if MODE == 'original'
                      else f'm0_acceptance_{tag}.json')
    with open(fn, 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n结果已存 {os.path.relpath(fn, HERE)}')


if __name__ == '__main__':
    main()
