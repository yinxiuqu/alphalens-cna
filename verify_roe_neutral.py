"""中性化后重跑 ROE —— 崩盘率结论是否还站得住？

关键问题：低 ROE 组腰斩率更高，会不会只是"小市值 / 特定行业"的代理？
做法：行业 + 市值中性化之后再算一遍 RankIC / Q5−Q1 / 崩盘率差。
"""
from __future__ import annotations
import glob, json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('PIT_MODE', 'delisted+zerofix')
os.environ.setdefault('HOME', os.path.join(HERE, '.qh'))
import accept_m0 as A, alphalens_cna as acna
from alphalens_cna.engine.clean import clean
from alphalens_cna.engine.returns import ReturnModel, forward_returns

# 私有数据根目录（可用环境变量 ALPHALENS_DATA_ROOT 覆盖）
DATA_ROOT = os.environ.get(
    'ALPHALENS_DATA_ROOT',
    os.path.expanduser('~/alphalens-data'))

ABS = -0.50


def load_mv(rebal, codes):
    """从 stock_daily_basic 取总市值（date 有索引，单日 0.07s）。"""
    from pymongo import MongoClient
    db = MongoClient('127.0.0.1', 27017, serverSelectionTimeoutMS=15000)['quantaxis']
    rows = []
    t = time.time()
    for d in rebal:
        ds = d.strftime('%Y-%m-%d')
        cur = db.stock_daily_basic.find({'date': ds, 'code': {'$in': list(codes)}},
                                        {'code': 1, 'total_mv': 1, '_id': 0})
        for r in cur:
            rows.append((d, r['code'], r.get('total_mv')))
    print(f'  市值取数 {len(rows):,} 条，用时 {time.time() - t:.1f}s')
    m = pd.DataFrame(rows, columns=['date', 'asset', 'total_mv']).dropna()
    m = m[m['total_mv'] > 0]
    m['ln_mv'] = np.log(m['total_mv'].astype(float))
    return m.set_index(['date', 'asset'])['ln_mv'].sort_index()


def load_industry(rebal):
    """申万 L1 行业 —— **按调仓日 as-of 取**（members 自带 in_date/out_date）。

    比"最新快照"正确：行业分类会变，用今天的行业去解释 2019 年的股票是前视。
    """
    m = pd.read_parquet(os.path.join(DATA_ROOT, 'data/sw_industry/members.parquet'))
    c = pd.read_parquet(os.path.join(DATA_ROOT, 'data/sw_industry/classify.parquet'))
    m = m[(m['level'] == 'L1') & (m['con_code'].notna())].copy()
    m['con_code'] = m['con_code'].astype(str).str.zfill(6).str[:6]
    m['in_date'] = pd.to_datetime(m['in_date'], errors='coerce')
    m['out_date'] = pd.to_datetime(m['out_date'], errors='coerce')
    name = (c[c['level'] == 'L1'].drop_duplicates('index_code')
            .set_index('index_code')['industry_name'])
    m['industry'] = m['index_code'].map(name)
    m = m.dropna(subset=['in_date', 'industry'])
    print(f'  行业成分 {len(m):,} 条（L1，{m["industry"].nunique()} 个行业，'
          f'源 {m["src"].iloc[0]}）')

    rows = []
    for d in rebal:
        ok = (m['in_date'] <= d) & (m['out_date'].isna() | (m['out_date'] >= d))
        sub = m[ok].sort_values('in_date').drop_duplicates('con_code', keep='last')
        if len(sub):
            rows.append(pd.DataFrame({'date': d, 'asset': sub['con_code'].values,
                                      'industry': sub['industry'].values}))
    idx = pd.concat(rows).set_index(['date', 'asset'])['industry'].sort_index()
    print(f'  as-of 展开 {len(idx):,} 条（{idx.index.get_level_values("date").nunique()} 个调仓日）')
    return idx


def evaluate(df, tag, ind_map=None):
    """df: 含 factor / factor_quantile / forward_return_* 的面板。"""
    tb = acna.tail_by_quantile(df, threshold=ABS)
    sp = acna.crash_spread(df, threshold=ABS)
    ic = df.groupby(level='date').apply(
        lambda g: g['factor'].corr(g['forward_return_126'], method='spearman')
        if len(g) > 30 else np.nan)
    row = {'口径': tag, 'RankIC_126': float(ic.mean())}
    for h in A.HORIZONS:
        row[f'均值差_{h}'] = sp.loc[h, 'mean_spread']
        row[f'腰斩差_{h}'] = sp.loc[h, 'hit_spread']
        row[f't_{h}'] = sp.loc[h, 't_hit']
        row[f'Q1腰斩_{h}'] = tb.loc[(h, 1), 'hit_rate']
        row[f'Q5腰斩_{h}'] = tb.loc[(h, 5), 'hit_rate']
    return row


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
    base = cr.data.join(acna.quantize(cr, n=5)[['q']])
    base = base.rename(columns={'q': 'factor_quantile'})
    print(f'清洗后 {len(cr.data):,} 行，调仓 {len(rebal)} 个')

    print('取中性化数据…')
    lnmv = load_mv(rebal, set(codes))
    ind = load_industry(rebal)

    rows = [evaluate(base, '① 原始 ROE')]

    # ② 市值中性化
    f2 = base['factor']
    n2 = acna.neutralize(f2, exposures=lnmv.rename('ln_mv').to_frame())
    q2 = base.copy(); q2['factor'] = n2
    q2 = q2.dropna(subset=['factor'])
    q2['factor_quantile'] = acna.quantize(q2, n=5)['q']
    rows.append(evaluate(q2, '② +市值中性'))
    print(f'  ② 市值中性后剩 {len(q2):,} 行')

    # ③ 行业 + 市值
    if ind is not None:
        grp = ind.reindex(f2.index)
        n3 = acna.neutralize(f2, groups=grp)
        n3 = acna.neutralize(n3, exposures=lnmv.rename('ln_mv').to_frame())
        q3 = base.copy(); q3['factor'] = n3
        q3 = q3.dropna(subset=['factor'])
        q3['factor_quantile'] = acna.quantize(q3, n=5)['q']
        rows.append(evaluate(q3, '③ +行业+市值中性'))
        print(f'  ③ 行业+市值中性后剩 {len(q3):,} 行')

    tab = pd.DataFrame(rows).set_index('口径')
    pd.set_option('display.width', 200)
    print()
    print('=' * 100); print('中性化前后对比（腰斩门槛 −50%，open 口径，退市约定 last_price）'); print('=' * 100)
    show = tab[['RankIC_126'] + [f'Q1腰斩_{h}' for h in A.HORIZONS]
               + [f'Q5腰斩_{h}' for h in A.HORIZONS] + [f't_{h}' for h in A.HORIZONS]]
    print(show.to_string())
    tab.to_csv(os.path.join(HERE, 'outputs', 'roe_neutral.csv'), encoding='utf-8-sig')


if __name__ == '__main__':
    main()
