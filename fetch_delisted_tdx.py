#!/usr/bin/env python3
"""
退市股入库（通达信复权口径）—— 消除 A 股回测的生存者偏差。

为什么不用 tushare 的 adj_factor
--------------------------------
核查结论（见 outputs/复权口径核查报告.md）：
  · 库内 stock_adj = 通达信 xdxr + quantaxis 前复权公式（QASU/save_tdx.py）
  · tushare adj_factor = tushare 自己的一套
  · 公式原理相同，**上游事件数据不同**
  · 实测差异：000001 在 1991 年差 58%，2003 年后收敛到 <0.1%
  · 退市股在原库 stock_xdxr 里是 0/339 —— 原库没能力给它们复权
→ 若用 tushare 口径，退市股与在市股会在时间序列上留下**两套口径的接缝**。

数据来源分工（实测确定）
------------------------
  stock_day   ← tushare。TDX 的 get_security_bars 对退市股返回空；
                且 tushare 的 OHLCV 已实测与库内在市股完全一致。
  stock_xdxr  ← 通达信。实测**仍提供退市股除权数据**（000003 有 7 次除权）。
  stock_adj   ← quantaxis `_QA_data_stock_to_fq`，吃上面两者，同款算法。

本脚本 = 把 QASU/save_tdx.py 的 __saving_work 跑在退市股清单上。

⚠️ 已知陷阱
-----------
`_QA_data_stock_to_fq` 内部用 `.loc[start:end]` 切片，**必须传 DatetimeIndex**。
传字符串索引会**静默**产生 10/11 的偏差（末位 adj = 0.909091 而非 1.0），
不报错，只是数错。本脚本显式转 DatetimeIndex。

幂等
----
按 (code,date) upsert；stock_adj 会先删该 code 的旧记录再写，
保证口径切换时不留残渣。

用法
----
    python fetch_delisted_tdx.py --dry-run --limit 3
    python fetch_delisted_tdx.py --limit 3
    python fetch_delisted_tdx.py                 # 全量
"""

import argparse
import datetime as dt
import json
import os
import shutil
import socket
import sys
import time

# quantaxis 导入时要写 ~/.quantaxis/log/，而这些目录在沙箱外只读。
# 把 HOME 强制指向工作区内的影子目录（必须覆盖，setdefault 不生效）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_REAL_QA = os.path.expanduser('~/.quantaxis')   # quantaxis 配置目录
_FAKE_HOME = os.path.join(_HERE, '.qh')
os.makedirs(os.path.join(_FAKE_HOME, '.quantaxis', 'log'), exist_ok=True)
os.makedirs(os.path.join(_FAKE_HOME, '.quantaxis', 'setting'), exist_ok=True)
_cfg_src = os.path.join(_REAL_QA, 'setting', 'config.ini')
_cfg_dst = os.path.join(_FAKE_HOME, '.quantaxis', 'setting', 'config.ini')
if os.path.exists(_cfg_src) and not os.path.exists(_cfg_dst):
    shutil.copy(_cfg_src, _cfg_dst)
os.environ['HOME'] = _FAKE_HOME
sys.path.insert(0, '/home/yinxiuqu/quantaxis')

import pandas as pd
import pymongo
from pymongo import UpdateOne
from pytdx.hq import TdxHq_API

import tushare as ts
import QUANTAXIS as QA
from QUANTAXIS.QAData.data_fq import _QA_data_stock_to_fq
from QUANTAXIS.QASU.save_tdx import _merge_xdxr_same_day

CACHE = os.path.join(_HERE, 'cache')
LIST_CACHE = os.path.join(CACHE, 'delisted_basic.parquet')
PROGRESS = os.path.join(CACHE, 'delisted_progress.json')

MONGO = 'mongodb://127.0.0.1:27017'
DB = 'quantaxis'
EPOCH = dt.date(1970, 1, 1)
TUSHARE_MAX_ROWS = 6000
WINDOWS = [('19900101', '20041231'), ('20050101', '20141231'),
           ('20150101', '20241231'), ('20250101', '20261231')]


# --------------------------------------------------------------------------- #
# 凭证与服务器
# --------------------------------------------------------------------------- #
def read_token():
    """读 tushare token。注意 HOME 已被重定向，这里用绝对路径。"""
    cfg = os.path.join(_REAL_QA, 'setting', 'config.ini')
    for line in open(cfg, encoding='utf-8'):
        s = line.strip()
        if s.lower().startswith('token'):
            return s.split('=', 1)[1].strip()
    raise SystemExit('配置里没有 token')


def tdx_servers():
    p = os.path.join(_REAL_QA, 'setting', 'stock_ip.json')
    ips = json.load(open(p))
    return [(x['ip'], x.get('port', 7709)) for x in ips if 'ip' in x]


def pick_server(servers, skip=()):
    for ip, port in servers:
        if (ip, port) in skip:
            continue
        try:
            s = socket.create_connection((ip, port), timeout=3)
            s.close()
            return ip, port
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# tushare：日线（分片，绕开 6000 行静默截断）
# --------------------------------------------------------------------------- #
def call_retry(fn, *a, retries=6, base=2.0, **kw):
    delay = base
    for i in range(retries):
        try:
            return fn(*a, **kw)
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(delay)
            delay *= 1.8
    return None


def fetch_daily_paged(pro, ts_code, sleep):
    full = call_retry(pro.daily, ts_code=ts_code,
                      start_date='19900101', end_date='20261231')
    time.sleep(sleep)
    if full is None:
        return None
    if len(full) < TUSHARE_MAX_ROWS:
        return full.sort_values('trade_date').reset_index(drop=True)
    parts = [full]
    for a, b in WINDOWS:
        d = call_retry(pro.daily, ts_code=ts_code, start_date=a, end_date=b)
        time.sleep(sleep)
        if d is not None and len(d):
            parts.append(d)
    out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=['trade_date'])
    return out.sort_values('trade_date').reset_index(drop=True)


def to_date_stamp(datestr):
    return float((dt.date.fromisoformat(datestr) - EPOCH).days * 86400 - 28800)


def build_stock_day(daily):
    out = []
    for r in daily.sort_values('trade_date').itertuples(index=False):
        td = str(r.trade_date)
        ds = f'{td[:4]}-{td[4:6]}-{td[6:]}'
        out.append({'open': float(r.open), 'close': float(r.close),
                    'high': float(r.high), 'low': float(r.low),
                    'vol': float(r.vol),
                    'amount': round(float(r.amount) * 1000, 4),   # 千元 → 元
                    'date': ds, 'code': str(r.ts_code)[:6],
                    'date_stamp': to_date_stamp(ds)})
    return out


# --------------------------------------------------------------------------- #
# 通达信：xdxr + quantaxis 复权算法
# --------------------------------------------------------------------------- #
def market_of(code):
    return 1 if str(code).startswith(('6', '9')) else 0


def fetch_xdxr_tdx(api, code):
    """通达信除权除息，列名与 quantaxis 落库格式一致。"""
    mk = market_of(code)
    data = api.to_df(api.get_xdxr_info(mk, code))
    if data is None or len(data) == 0:
        return None
    category = {'1': '除权除息', '2': '送配股上市', '3': '非流通股上市', '4': '未知股本变动',
                '5': '股本变化', '6': '增发新股', '7': '股份回购', '8': '增发新股上市',
                '9': '转配股上市', '10': '可转债上市', '11': '扩缩股', '12': '非流通股缩股',
                '13': '送认购权证', '14': '送认沽权证', '15': '小额股份上市'}
    d = (data.assign(date=pd.to_datetime(data[['year', 'month', 'day']], utc=False))
             .drop(['year', 'month', 'day'], axis=1)
             .assign(category_meaning=data['category'].apply(
                 lambda x: category.get(str(x), '未知类别%s' % x)))
             .assign(code=str(code))
             .rename(columns={'panhouliutong': 'liquidity_after',
                              'panqianliutong': 'liquidity_before',
                              'houzongguben': 'shares_after',
                              'qianzongguben': 'shares_before'}))
    d = d.assign(date=d['date'].apply(lambda x: str(x)[0:10]))
    return d.set_index('date', drop=False)


def compute_adj(stock_day_docs, xdxr):
    """
    用 quantaxis 原版算法算前复权因子。

    ⚠️ 必须 DatetimeIndex —— 传字符串索引会静默产生 10/11 偏差。
    """
    if xdxr is None or len(xdxr) == 0:
        return None
    d = pd.DataFrame(stock_day_docs).sort_values('date')
    d['date'] = pd.to_datetime(d['date'])
    d = d.set_index('date', drop=False)
    if 'volume' not in d.columns:
        d['volume'] = d['vol']
    x = xdxr.copy()
    x['date'] = pd.to_datetime(x['date'])
    x = x.set_index('date', drop=False)
    qfq = _QA_data_stock_to_fq(d.copy(), x, 'qfq')
    qfq = qfq.assign(date=qfq['date'].apply(lambda v: str(v)[0:10]))
    return qfq.loc[:, ['date', 'code', 'adj']]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--sleep', type=float, default=0.12)
    ap.add_argument('--redo', action='store_true',
                    help='已入库的也重做（口径切换时必须用）')
    args = ap.parse_args()

    pro = ts.pro_api(read_token())
    dl = pd.read_parquet(LIST_CACHE)
    dl['code'] = dl['symbol'].astype(str).str.zfill(6)
    print(f'退市股清单: {len(dl)} 只', flush=True)

    db = pymongo.MongoClient(MONGO)[DB]
    # 断点续传判据用 stock_adj 而非 stock_day：
    # stock_day 先写、stock_adj 后写，中途被杀会留下"日线写了一半"的残局。
    # 以 stock_adj 为准 ⇒ 残局会被重跑，upsert 补齐剩余日线，不会漏。
    done = set(db.stock_adj.distinct('code'))
    todo = dl.copy() if args.redo else dl[~dl['code'].isin(done)]
    print(f'已完整入库: {len(done & set(dl["code"]))} 只；待处理: {len(todo)} 只 (redo={args.redo})',
          flush=True)
    if args.limit:
        todo = todo.head(args.limit)
        print(f'--limit → {len(todo)} 只', flush=True)

    servers = tdx_servers()
    srv = pick_server(servers)
    print(f'TDX 服务器: {srv}', flush=True)
    if srv is None:
        raise SystemExit('没有可达的通达信服务器')
    api = TdxHq_API(raise_exception=False)

    stat = {'ok': 0, 'no_xdxr': 0, 'empty': 0, 'fail': 0,
            'day': 0, 'xdxr': 0, 'adj': 0}
    failures = []

    for i, row in enumerate(todo.itertuples(index=False), 1):
        code, tc, name = row.code, row.ts_code, row.name
        try:
            daily = fetch_daily_paged(pro, tc, args.sleep)
            if daily is None or len(daily) == 0:
                stat['empty'] += 1
                print(f'  [{i}/{len(todo)}] {code} {name}: 无行情', flush=True)
                continue
            day_docs = build_stock_day(daily)

            try:
                with api.connect(*srv):
                    xdxr = fetch_xdxr_tdx(api, code)
            except Exception:
                srv2 = pick_server(servers, skip={srv})
                if srv2 is None:
                    raise
                srv = srv2
                with api.connect(*srv):
                    xdxr = fetch_xdxr_tdx(api, code)

            if xdxr is None or len(xdxr) == 0:
                stat['no_xdxr'] += 1
                print(f'  [{i}/{len(todo)}] {code} {name}: 无除权数据（跳过复权）', flush=True)
                adj = None
            else:
                adj = compute_adj(day_docs, xdxr)

            stat['day'] += len(day_docs)
            stat['xdxr'] += 0 if xdxr is None else len(xdxr)
            stat['adj'] += 0 if adj is None else len(adj)
            span = f'{day_docs[0]["date"]} ~ {day_docs[-1]["date"]}'
            tag = '(dry-run)' if args.dry_run else ''
            print(f'  [{i}/{len(todo)}] {code} {name}: 日线{len(day_docs)} '
                  f"除权{0 if xdxr is None else len(xdxr)} "
                  f"复权{0 if adj is None else len(adj)}  {span} {tag}", flush=True)
            stat['ok'] += 1

            if args.dry_run:
                continue

            # 新代码用 insert_many（逐条 upsert 在 8000 条量级上极慢）；
            # 已有代码走 upsert 保证幂等。
            has_day = db.stock_day.find_one({'code': code}, {'_id': 1}) is not None
            if has_day:
                db.stock_day.bulk_write(
                    [UpdateOne({'code': d['code'], 'date': d['date']}, {'$set': d}, upsert=True)
                     for d in day_docs], ordered=False)
            else:
                db.stock_day.insert_many(day_docs, ordered=False)

            if xdxr is not None and len(xdxr):
                # ⚠️ 通达信同日可能有多条（如 1994-05-20 同时有 category 1 和 2）。
                # 直接按 (code,date) upsert 会让后写的覆盖先写的，
                # 而被覆盖的往往是 category==1（含分红数据）—— 实测 000003 丢了 4 条。
                # 用 save_tdx.py 同款的合并函数：以 category==1 为主记录，
                # 其余类别记进 extra_categories。
                merged = _merge_xdxr_same_day(xdxr)
                docs = QA.QA_util_to_json_from_pandas(merged)
                for d in docs:
                    d['code'] = code
                db.stock_xdxr.bulk_write(
                    [UpdateOne({'code': d['code'], 'date': d['date']}, {'$set': d}, upsert=True)
                     for d in docs], ordered=False)
            if adj is not None and len(adj):
                docs = QA.QA_util_to_json_from_pandas(adj)
                for d in docs:
                    d['code'] = code
                # 口径切换：先清旧记录，避免残留 tushare 口径的 adj
                db.stock_adj.delete_many({'code': code})
                db.stock_adj.insert_many(docs, ordered=False)

        except Exception as e:
            stat['fail'] += 1
            failures.append({'code': code, 'name': name, 'error': str(e)[:200]})
            print(f'  [{i}/{len(todo)}] {code} {name}: ❌ {str(e)[:130]}', flush=True)

    print()
    print('=' * 62)
    print(json.dumps(stat, ensure_ascii=False, indent=2), flush=True)
    json.dump({'stat': stat, 'failures': failures, 'finished_at': str(dt.datetime.now())},
              open(PROGRESS, 'w'), ensure_ascii=False, indent=2)
    if not args.dry_run:
        print('stock_day =', db.stock_day.estimated_document_count(),
              ' stock_adj =', db.stock_adj.estimated_document_count(),
              ' stock_xdxr =', db.stock_xdxr.estimated_document_count(), flush=True)


if __name__ == '__main__':
    main()
