# -*- coding: utf-8 -*-
"""构建行情面板 (供因子分析使用) —— 日度 + 月度

为什么按 code 分块取数
----------------------
`quantaxis.stock_day` 上只有 `code_1_date_stamp_1` 复合索引, **没有独立 date 索引**:

    db.stock_day.find({'date': '2026-08-28'})   -> 93 秒 (1660 万条全表扫描)
    db.stock_day.find({'code': '000001', ...})  -> 0.17 秒 (走 code 前缀)

所以按 code 分块并行取, 是唯一可行的全市场取数方式。

复权
----
`stock_adj.adj` 是复权因子, 复权价 = close × adj (两者同日)。本模块**只保存复权价**,
因为因子分析要的是收益率, 未复权价在除权除息日会产生假跳空。

输出 (只写本工作区, 对 mongo 只读)
-----------------------------------
    cache/px_daily.parquet      日度复权价 (date × code)   ← 主产物
    cache/px_monthly.parquet    月末复权价 (date × code)
    cache/panel_meta.json       元信息
"""
import json
import os
import time
import warnings
from multiprocessing import Pool

import pandas as pd
from pymongo import MongoClient

warnings.filterwarnings('ignore')

MONGO, PORT, DB = '127.0.0.1', 27017, 'quantaxis'
START, END = '2016-01-01', '2026-12-31'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
WORKERS = 8
CHUNK = 200


def _client():
    return MongoClient(MONGO, PORT, serverSelectionTimeoutMS=20000)[DB]


def _fetch_chunk(args):
    i, codes = args
    db = _client()
    q = {'code': {'$in': list(codes)}, 'date': {'$gte': START, '$lte': END}}
    px = pd.DataFrame(list(db.stock_day.find(q, {'code': 1, 'date': 1, 'close': 1, '_id': 0})))
    aj = pd.DataFrame(list(db.stock_adj.find(q, {'code': 1, 'date': 1, 'adj': 1, '_id': 0})))
    return i, len(codes), px, aj




def _delisted_codes(db, start):
    """★ 退市股代码。

    ``stock_info`` 里**只有在市股** —— 实测 5,066 只中 0 只退市股。
    只按它取数，面板就没有一只中途消失的票，回测被系统性高估（存活偏差）。
    所以要从 ``stock_basic`` 的 ``list_status='D'`` 补回来。
    """
    out = set()
    for d in db.stock_basic.find({'list_status': 'D'},
                                 {'symbol': 1, 'delist_date': 1, '_id': 0}):
        code = str(d.get('symbol') or '').zfill(6)
        raw = str(d.get('delist_date') or '').replace('-', '')
        if len(code) == 6 and len(raw) >= 8 and raw[:8] >= start.replace('-', ''):
            out.add(code)
    return out


def _all_codes(db, start):
    """在市股 ∪ 退市股（退市日 ≥ start）。"""
    return sorted(set(d['code'] for d in db.stock_info.find({}, {'code': 1, '_id': 0}))
                  | _delisted_codes(db, start))

def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.time()
    codes = _all_codes(_client(), START)   # ★ 含退市股
    print('股票数 %d, 区间 %s ~ %s' % (len(codes), START, END), flush=True)

    chunks = [(i, codes[i:i + CHUNK]) for i in range(0, len(codes), CHUNK)]
    print('分块 %d × %d 只, 并发 %d' % (len(chunks), CHUNK, WORKERS), flush=True)

    px_parts, aj_parts, t0, done = [], [], time.time(), 0
    with Pool(WORKERS) as pool:
        for i, n, px, aj in pool.imap_unordered(_fetch_chunk, chunks):
            px_parts.append(px)
            aj_parts.append(aj)
            done += n
            print('  进度 %d/%d 只  行数 %d  用时 %.0fs'
                  % (done, len(codes), sum(len(x) for x in px_parts), time.time() - t0), flush=True)
    print('取数完成 %.1fs' % (time.time() - t0), flush=True)

    px = pd.concat(px_parts, ignore_index=True)
    aj = pd.concat(aj_parts, ignore_index=True)
    del px_parts, aj_parts

    for df, col in ((px, 'close'), (aj, 'adj')):
        df.drop(columns=[c for c in df.columns if c not in ('code', 'date', col)], inplace=True, errors='ignore')
        df.drop_duplicates(subset=['code', 'date'], inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print('行数 行情=%d 复权=%d' % (len(px), len(aj)), flush=True)

    m = px.merge(aj, on=['code', 'date'], how='left')
    print('合并后 %d 行, adj 缺失 %.2f%%' % (len(m), 100 * m['adj'].isna().mean()), flush=True)
    m['adj'] = m['adj'].fillna(1.0)
    m['px_adj'] = m['close'] * m['adj']
    m = m[m['px_adj'] > 0]

    daily = m.pivot(index='date', columns='code', values='px_adj').sort_index()
    daily = daily.astype('float32')
    print('日度面板 %s' % (daily.shape,), flush=True)

    # 月末最后交易日
    mo = daily.resample('ME').last()
    # 把索引改回"该月最后一个真实交易日"
    last_day = pd.Series(daily.index, index=daily.index).resample('ME').last()
    mo.index = pd.DatetimeIndex(last_day.values)
    mo = mo.astype('float32')
    print('月度面板 %s' % (mo.shape,), flush=True)

    daily.to_parquet(os.path.join(OUT, 'px_daily.parquet'))
    mo.to_parquet(os.path.join(OUT, 'px_monthly.parquet'))

    meta = {
        'built_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'mongo': '%s:%d/%s' % (MONGO, PORT, DB),
        'collections': {'price': 'stock_day(close)', 'adj': 'stock_adj(adj)'},
        'price_definition': 'px_adj = close * adj (同日) —— 复权价',
        'range': [START, END],
        'codes': len(codes),
        'daily_shape': list(daily.shape),
        'monthly_shape': list(mo.shape),
        'first_day': str(daily.index.min().date()),
        'last_day': str(daily.index.max().date()),
        'elapsed_sec': round(time.time() - t_all, 1),
        'note': 'stock_day 无 date 索引: 按日期查单日需 93s(全表扫描); 本面板按 code 分块并行取。',
    }
    with open(os.path.join(OUT, 'panel_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print('总耗时 %.1fs' % (time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
