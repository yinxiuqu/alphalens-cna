# -*- coding: utf-8 -*-
"""补拉**开盘价**面板（月度调仓的 T+1 开盘成交口径需要它）

现有 cache/px_daily.parquet 只有复权收盘价。本次要验证设计的成交模型
（信号 T 日收盘生成 → T+1 开盘成交），因此需要开盘价。

输出: cache/px_daily_open.parquet   (date × code, 复权开盘价 = open × adj)
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


def _fetch(args):
    i, codes = args
    db = _client()
    q = {'code': {'$in': list(codes)}, 'date': {'$gte': START, '$lte': END}}
    op = pd.DataFrame(list(db.stock_day.find(q, {'code': 1, 'date': 1, 'open': 1, '_id': 0})))
    aj = pd.DataFrame(list(db.stock_adj.find(q, {'code': 1, 'date': 1, 'adj': 1, '_id': 0})))
    return len(codes), op, aj




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
    t0 = time.time()
    codes = _all_codes(_client(), START)   # ★ 含退市股
    print('股票数 %d' % len(codes), flush=True)
    chunks = [(i, codes[i:i + CHUNK]) for i in range(0, len(codes), CHUNK)]

    ops, ajs, done = [], [], 0
    with Pool(WORKERS) as pool:
        for n, op, aj in pool.imap_unordered(_fetch, chunks):
            ops.append(op); ajs.append(aj); done += n
            print('  进度 %d/%d  %.0fs' % (done, len(codes), time.time() - t0), flush=True)

    op = pd.concat(ops, ignore_index=True)
    aj = pd.concat(ajs, ignore_index=True)
    for df, col in ((op, 'open'), (aj, 'adj')):
        df.drop_duplicates(subset=['code', 'date'], inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df[col] = pd.to_numeric(df[col], errors='coerce')

    m = op.merge(aj, on=['code', 'date'], how='left')
    m['adj'] = m['adj'].fillna(1.0)
    m['px'] = m['open'] * m['adj']
    m = m[m['px'] > 0]
    daily = m.pivot(index='date', columns='code', values='px').sort_index().astype('float32')
    daily.to_parquet(os.path.join(OUT, 'px_daily_open.parquet'))
    print('开盘面板 %s,  %.0fs' % (daily.shape, time.time() - t0), flush=True)
    with open(os.path.join(OUT, 'px_daily_open.meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'built_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'definition': 'px = open × adj (同日), 复权开盘价',
                   'shape': list(daily.shape),
                   'range': [str(daily.index.min().date()), str(daily.index.max().date())]},
                  f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
