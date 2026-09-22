#!/usr/bin/env python3
"""
退市股历史 K 线入库 —— 消除 A 股回测的生存者偏差。

背景
----
现有 quantaxis.stock_day 的 5220 个代码**最后交易日全部在 2026 年** ——
即库里只有当前仍在上市的股票，退市股一只都没有。
用这样的数据回测历史，等于**事先知道了哪些股票不会退市**，结论系统性偏乐观。

本脚本
------
1. tushare `stock_basic(list_status='D')` 取退市股清单（339 只，1999–2026）
2. 逐只拉 `daily`（不复权 OHLCV）+ `adj_factor`（复权因子）
3. 按 stock_day / stock_adj **现有字段与格式**写入 mongodb

字段口径（已逐项与库内实测比对确认）
------------------------------------
stock_day:
    open/close/high/low  tushare 1:1（不复权）
    vol                  tushare 1:1（单位: 手）
    amount               tushare × 1000（tushare 是千元，库内是元）
    date                 'YYYYMMDD' → 'YYYY-MM-DD'
    code                 ts_code 前 6 位
    date_stamp           (date − 1970-01-01).days × 86400 − 28800
                         = 该日 00:00 北京时间 的 Unix 秒（已在 1990–2026 验证）

stock_adj:
    adj                  前复权因子 = adj_factor(t) / adj_factor(该股最后交易日)
                         归一后该股最后一个交易日 adj = 1.0，与库内约定一致
                         且**只保留 stock_day 里存在的日期**（库内约定）

⚠️ 分片拉取（重要）
------------------
tushare 的 daily / adj_factor **单次最多返回 6000 行**，超出部分**静默丢弃**。
实测 000004 一次拉取只有 6000 条（2000-03-20 起），
分片后是 8214 条（1991-01-14 起）—— **一次静默丢了 9 年历史**。
本脚本因此先试一次，若正好 6000 条则判定触顶，改用 15 年窗口重取合并。

幂等
----
写入用 upsert（按 code+date 定位），重复执行不会产生重复文档。

用法
----
    python fetch_delisted.py --dry-run --limit 3     # 预览，不写库
    python fetch_delisted.py --limit 3               # 试跑 3 只
    python fetch_delisted.py                         # 全量
"""

import argparse
import datetime as dt
import json
import os
import sys
import time

import pandas as pd
import pymongo
import tushare as ts
from pymongo import UpdateOne

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
LIST_CACHE = os.path.join(CACHE, 'delisted_basic.parquet')

MONGO = 'mongodb://127.0.0.1:27017'
DB = 'quantaxis'
EPOCH = dt.date(1970, 1, 1)

TUSHARE_MAX_ROWS = 6000          # 单次返回上限
WINDOWS = [('19900101', '20041231'), ('20050101', '20141231'),
           ('20150101', '20241231'), ('20250101', '20261231')]


def read_token():
    """从 quantaxis 配置读 tushare token（不调 ts.set_token，它会写 ~/tk.csv）。"""
    cfg = os.path.expanduser('~/.quantaxis/setting/config.ini')
    if not os.path.exists(cfg):
        raise SystemExit(f'找不到 quantaxis 配置: {cfg}')
    in_tspro = False
    for line in open(cfg, encoding='utf-8'):
        s = line.strip()
        if s.startswith('['):
            in_tspro = s.lower().startswith('[tspro]')
        elif in_tspro and s.lower().startswith('token'):
            return s.split('=', 1)[1].strip()
    raise SystemExit('配置里没有 [TSPRO] token')


def load_delisted(pro, force=False):
    """退市股清单，带本地缓存。"""
    if os.path.exists(LIST_CACHE) and not force:
        df = pd.read_parquet(LIST_CACHE)
    else:
        df = pro.stock_basic(exchange='', list_status='D',
                             fields='ts_code,symbol,name,area,industry,market,'
                                    'list_date,delist_date')
        os.makedirs(CACHE, exist_ok=True)
        df.to_parquet(LIST_CACHE)
    df = df.copy()
    df['code'] = df['symbol'].astype(str).str.zfill(6)
    return df


def call_retry(fn, *args, retries=6, base=2.0, **kw):
    """tushare 有每分钟调用上限，指数退避重试。"""
    delay = base
    for i in range(retries):
        try:
            return fn(*args, **kw)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(delay)
            delay *= 1.8
    return None


def fetch_paged(pro, api_name, ts_code, sleep, verbose=False):
    """
    分片拉取全历史，绕开 tushare 6000 行静默截断。

    策略: 先试一次全区间。若条数 < 6000 说明没触顶，直接采用；
          正好 6000 则判定触顶，用 15 年窗口重取并合并去重。
    """
    api = getattr(pro, api_name)
    full = call_retry(api, ts_code=ts_code, start_date='19900101', end_date='20261231')
    time.sleep(sleep)
    if full is None:
        return None
    if len(full) < TUSHARE_MAX_ROWS:
        return full.sort_values('trade_date').reset_index(drop=True)

    if verbose:
        print(f'      [{api_name}] 触顶 {len(full)} 条 → 分片重取')
    parts = [full]
    for a, b in WINDOWS:
        d = call_retry(api, ts_code=ts_code, start_date=a, end_date=b)
        time.sleep(sleep)
        if d is not None and len(d):
            parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=['trade_date'])
    return out.sort_values('trade_date').reset_index(drop=True)


def to_date_stamp(datestr):
    """'YYYY-MM-DD' → 该日 00:00 北京时间 的 Unix 秒（float，与库内一致）。"""
    d = dt.date.fromisoformat(datestr)
    return float((d - EPOCH).days * 86400 - 28800)


def ymd2iso(td):
    td = str(td)
    return f'{td[:4]}-{td[4:6]}-{td[6:]}'


def build_stock_day(daily):
    """tushare daily → stock_day 文档列表（升序）。"""
    d = daily.sort_values('trade_date')
    out = []
    for r in d.itertuples(index=False):
        datestr = ymd2iso(r.trade_date)
        out.append({
            'open': float(r.open),
            'close': float(r.close),
            'high': float(r.high),
            'low': float(r.low),
            'vol': float(r.vol),
            'amount': round(float(r.amount) * 1000, 4),   # 千元 → 元
            'date': datestr,
            'code': str(r.ts_code)[:6],
            'date_stamp': to_date_stamp(datestr),
        })
    return out


def build_stock_adj(adjf, valid_dates):
    """
    tushare adj_factor → stock_adj 文档列表。

    归一: adj = adj_factor(t) / adj_factor(该股最后交易日) ⇒ 末位 = 1.0
    对齐: 只保留 valid_dates（= stock_day 的日期）中的记录 —— 库内约定
    """
    if adjf is None or len(adjf) == 0:
        return []
    a = adjf.copy()
    a['trade_date'] = a['trade_date'].astype(str)
    a = a.sort_values('trade_date')
    a = a[a['trade_date'].isin(valid_dates)]
    if len(a) == 0:
        return []
    last = float(a['adj_factor'].iloc[-1])
    if last == 0:
        return []
    return [{'date': ymd2iso(r.trade_date), 'code': str(r.ts_code)[:6],
             'adj': round(float(r.adj_factor) / last, 10)}
            for r in a.itertuples(index=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 只（调试）')
    ap.add_argument('--dry-run', action='store_true', help='不写库，只预览')
    ap.add_argument('--refresh-list', action='store_true', help='强制重取退市清单')
    ap.add_argument('--sleep', type=float, default=0.12, help='每次 API 调用后的间隔秒')
    ap.add_argument('--verbose', action='store_true', help='打印分片细节')
    args = ap.parse_args()

    pro = ts.pro_api(read_token())
    dl = load_delisted(pro, force=args.refresh_list)
    print(f'退市股清单: {len(dl)} 只', flush=True)

    db = pymongo.MongoClient(MONGO)[DB]
    existing = set(db.stock_day.distinct('code'))
    todo = dl[~dl['code'].isin(existing)].copy()
    print(f'库内已有代码 {len(existing)} 个；待处理退市股 {len(todo)} 只', flush=True)

    if args.limit:
        todo = todo.head(args.limit)
        print(f'--limit 生效 → 只处理 {len(todo)} 只', flush=True)

    stat = {'ok': 0, 'empty': 0, 'fail': 0,
            'day_docs': 0, 'adj_docs': 0, 'written': 0}
    failures = []

    for i, row in enumerate(todo.itertuples(index=False), 1):
        tc, code, name = row.ts_code, row.code, row.name
        try:
            daily = fetch_paged(pro, 'daily', tc, args.sleep, args.verbose)
            if daily is None or len(daily) == 0:
                stat['empty'] += 1
                print(f'  [{i}/{len(todo)}] {code} {name}: 无行情数据', flush=True)
                continue

            day_docs = build_stock_day(daily)
            valid = set(str(x.trade_date) for x in daily.itertuples(index=False))
            adjf = fetch_paged(pro, 'adj_factor', tc, args.sleep, args.verbose)
            adj_docs = build_stock_adj(adjf, valid)

            stat['day_docs'] += len(day_docs)
            stat['adj_docs'] += len(adj_docs)

            span = f'{day_docs[0]["date"]} ~ {day_docs[-1]["date"]}'
            tag = '(dry-run)' if args.dry_run else ''
            print(f'  [{i}/{len(todo)}] {code} {name}: '
                  f'{len(day_docs)} 日线 / {len(adj_docs)} 复权  {span} {tag}', flush=True)

            if args.dry_run:
                stat['ok'] += 1
                continue

            if day_docs:
                res = db.stock_day.bulk_write(
                    [UpdateOne({'code': d['code'], 'date': d['date']},
                               {'$set': d}, upsert=True) for d in day_docs],
                    ordered=False)
                stat['written'] += res.upserted_count + res.modified_count
            if adj_docs:
                res = db.stock_adj.bulk_write(
                    [UpdateOne({'code': d['code'], 'date': d['date']},
                               {'$set': d}, upsert=True) for d in adj_docs],
                    ordered=False)
                stat['written'] += res.upserted_count + res.modified_count
            stat['ok'] += 1
        except Exception as e:
            stat['fail'] += 1
            failures.append({'code': code, 'name': name, 'error': str(e)[:200]})
            print(f'  [{i}/{len(todo)}] {code} {name}: ❌ {str(e)[:140]}', flush=True)

    print()
    print('=' * 62)
    print(json.dumps(stat, ensure_ascii=False, indent=2))
    if failures:
        p = os.path.join(CACHE, 'delisted_failures.json')
        os.makedirs(CACHE, exist_ok=True)
        json.dump(failures, open(p, 'w'), ensure_ascii=False, indent=2)
        print(f'失败明细已存: {p}')

    if not args.dry_run:
        print()
        print('入库后计数: stock_day =', db.stock_day.estimated_document_count(),
              ' stock_adj =', db.stock_adj.estimated_document_count())


if __name__ == '__main__':
    main()
