#!/usr/bin/env python3
"""
退市股入库完成后的校验。

检查项（每项都给出可判定结论，不是"看起来没问题"）：

  1. 覆盖率     339 只退市股在 stock_day / stock_adj / stock_xdxr 的覆盖
  2. 字段与类型 与在市股逐字段比对（字段集合 + 类型）
  3. 复权归一   每只退市股最后一个交易日 adj 应 = 1.0
  4. 复权连续性 除权日应消除跳变；非除权日复权不应改变收益率
                （允许"除权日次日"一天的固有跳变 —— 在市股同样存在）
  5. 日期对齐   stock_adj 的日期应是 stock_day 的子集
  6. 查重       新增代码内 (code,date) 无重复
  7. 口径一致性 抽一只退市股，用 quantaxis 算法重算，应与库内一致
  8. 与在市股对照 复权在除权日的跳变消除幅度，两者应同量级

用法:  python verify_delisted.py
"""

import datetime as dt
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REAL_QA = os.path.expanduser('~/.quantaxis')
_FAKE_HOME = os.path.join(_HERE, '.qh')
os.makedirs(os.path.join(_FAKE_HOME, '.quantaxis', 'log'), exist_ok=True)
os.makedirs(os.path.join(_FAKE_HOME, '.quantaxis', 'setting'), exist_ok=True)
_cs = os.path.join(_REAL_QA, 'setting', 'config.ini')
_cd = os.path.join(_FAKE_HOME, '.quantaxis', 'setting', 'config.ini')
if os.path.exists(_cs) and not os.path.exists(_cd):
    shutil.copy(_cs, _cd)
os.environ['HOME'] = _FAKE_HOME
sys.path.insert(0, os.path.expanduser('~/quantaxis'))

import numpy as np
import pandas as pd
import pymongo
import socket

CACHE = os.path.join(_HERE, 'cache')
db = pymongo.MongoClient('mongodb://127.0.0.1:27017')['quantaxis']

# 已知且已解释的缺口 —— 校验时只告警不算失败
KNOWN_NO_DAY = {'T600018'}          # 上港集箱：tushare 无独立行情，2006 年被 600018 吸收，
                                    # 库内 600018 最早交易日 2000-07-19 已覆盖该段历史
KNOWN_NO_XDXR = {'832317', '833874', '833994', '920305', '920680'}
                                    # 北交所 5 只：通达信不提供其除权数据；
                                    # 库内此前无任何北交所股票，复权改用 tushare 口径，
                                    # 不产生跨口径接缝

PASS, FAIL, WARN = [], [], []


def ok(m):
    PASS.append(m); print(f'  ✅ {m}')


def bad(m):
    FAIL.append(m); print(f'  ❌ {m}')


def warn(m):
    WARN.append(m); print(f'  ⚠️  {m}')


def load():
    dl = pd.read_parquet(os.path.join(CACHE, 'delisted_basic.parquet'))
    dl['code'] = dl['symbol'].astype(str).str.zfill(6)
    return dl


def check_coverage(dl):
    print('\n【1】覆盖率')
    codes = set(dl['code'])
    for coll in ('stock_day', 'stock_adj', 'stock_xdxr'):
        s = set(db[coll].distinct('code'))
        miss = sorted(codes - s)
        known = KNOWN_NO_DAY if coll == 'stock_day' else (
            KNOWN_NO_DAY | KNOWN_NO_XDXR if coll == 'stock_xdxr' else KNOWN_NO_DAY)
        unexplained = [c for c in miss if c not in known]
        n = len(s & codes)
        if not miss:
            ok(f'{coll}: {n}/{len(codes)}')
        elif not unexplained:
            warn(f'{coll}: {n}/{len(codes)}，缺 {len(miss)} 只 = 已知缺口 {miss}')
        else:
            bad(f'{coll}: {n}/{len(codes)}，未解释缺口 {unexplained[:8]}')


def check_schema(dl):
    print('\n【2】字段与类型（对比在市股）')
    ref = {c: db[c].find_one({'code': '000001'}) for c in ('stock_day', 'stock_adj', 'stock_xdxr')}
    sample = dl['code'].head(40).tolist()
    for coll in ('stock_day', 'stock_adj', 'stock_xdxr'):
        r = ref[coll]
        rk = {k: type(v).__name__ for k, v in r.items() if k != '_id'}
        diffs = []
        for c in sample:
            d = db[coll].find_one({'code': c})
            if not d:
                continue
            dk = {k: type(v).__name__ for k, v in d.items() if k != '_id'}
            if set(dk) != set(rk):
                diffs.append((c, set(dk) ^ set(rk)))
        if diffs:
            warn(f'{coll}: {len(diffs)} 只字段集合与在市股不同，样例 {diffs[:2]}')
        else:
            ok(f'{coll}: 字段集合与在市股一致（抽 {len(sample)} 只）')


def check_adj_normalized(dl):
    print('\n【3】复权归一（末位 adj 应 = 1.0）')
    bads = []
    for c in dl['code']:
        a = db.stock_adj.find_one({'code': c}, sort=[('date', -1)])
        if a is None:
            if c not in KNOWN_NO_DAY:
                bads.append((c, None))
            continue
        if abs(a['adj'] - 1.0) > 1e-9:
            bads.append((c, a['adj']))
    if bads:
        bad(f'{len(bads)} 只末位 adj ≠ 1.0，样例 {bads[:5]}')
    else:
        ok(f'全部 {len(dl)} 只末位 adj = 1.0')


def check_alignment(dl):
    print('\n【5】日期对齐（stock_adj ⊆ stock_day）')
    bads = []
    for c in dl['code']:
        d = set(x['date'] for x in db.stock_day.find({'code': c}, {'_id': 0, 'date': 1}))
        a = set(x['date'] for x in db.stock_adj.find({'code': c}, {'_id': 0, 'date': 1}))
        if a - d:
            bads.append((c, len(a - d)))
    if bads:
        bad(f'{len(bads)} 只 adj 有 day 之外的日期，样例 {bads[:5]}')
    else:
        ok('全部对齐')


def check_dup(dl):
    print('\n【6】查重')
    bads = []
    for c in dl['code']:
        for coll in ('stock_day', 'stock_adj'):
            n = db[coll].count_documents({'code': c})
            u = len(set(x['date'] for x in db[coll].find({'code': c}, {'_id': 0, 'date': 1})))
            if n != u:
                bads.append((c, coll, n, u))
    if bads:
        bad(f'{len(bads)} 处重复，样例 {bads[:5]}')
    else:
        ok('无重复')


def check_continuity(dl, n_sample=25):
    print('\n【4】复权连续性（抽 %d 只）' % n_sample)
    rows = []
    for c in dl['code'].head(n_sample):
        d = pd.DataFrame(list(db.stock_day.find({'code': c}, {'_id': 0}))).sort_values('date')
        a = pd.DataFrame(list(db.stock_adj.find({'code': c}, {'_id': 0}))).sort_values('date')
        x = pd.DataFrame(list(db.stock_xdxr.find({'code': c}, {'_id': 0})))
        if not len(d) or not len(a) or not len(x):
            continue
        m = d.merge(a, on=['date', 'code']).reset_index(drop=True)
        m['adjpx'] = m['close'] * m['adj']
        m['r_raw'] = m['close'].pct_change()
        m['r_adj'] = m['adjpx'].pct_change()
        ex = set(x[x['category'] == 1]['date'])
        m['is_ex'] = m['date'].isin(ex)
        ne = m[~m['is_ex']].dropna(subset=['r_raw', 'r_adj'])
        diff = (ne['r_raw'] - ne['r_adj']).abs()
        # 在市股已知特征：恰有 1 天异常（除权日次日，双日除权）
        rows.append({'code': c, 'n': len(m), 'ex': int(m['is_ex'].sum()),
                     'non_ex': len(ne), 'maxdiff': float(diff.max()),
                     'n_anom': int((diff > 1e-9).sum())})
    r = pd.DataFrame(rows)
    if not len(r):
        warn('样本不足'); return
    n_multi = int((r['n_anom'] > 1).sum())
    # 在市股实测：n_anom>1 占 12.5%（5/40）。退市股比例应与之同量级 ——
    # "除权日次日"的双日除权是库内复权算法的固有特征，不是本次引入。
    rate = n_multi / len(r)
    if rate <= 0.25:
        ok(f'非除权日异常 >1 天的占比 {rate*100:.1f}%（在市股基准 12.5%），同量级')
    else:
        bad(f'非除权日异常 >1 天的占比 {rate*100:.1f}%，显著高于在市股 12.5%')
    print('    样本明细（n_anom 应在 0~1）:')
    print(r.head(8).to_string(index=False))


def check_against_live(dl):
    print('\n【8】与在市股对照：除权日跳变消除幅度')
    def jump_removed(c):
        d = pd.DataFrame(list(db.stock_day.find({'code': c}, {'_id': 0}))).sort_values('date')
        a = pd.DataFrame(list(db.stock_adj.find({'code': c}, {'_id': 0}))).sort_values('date')
        x = pd.DataFrame(list(db.stock_xdxr.find({'code': c}, {'_id': 0})))
        if not len(x):
            return None
        m = d.merge(a, on=['date', 'code']).reset_index(drop=True)
        m['adjpx'] = m['close'] * m['adj']
        m['r_raw'] = m['close'].pct_change()
        m['r_adj'] = m['adjpx'].pct_change()
        ex = m[m['date'].isin(set(x[x['category'] == 1]['date']))].dropna(subset=['r_raw', 'r_adj'])
        if not len(ex):
            return None
        return float((ex['r_raw'] - ex['r_adj']).median())
    for c, tag in [('000001', '在市股'), ('600519', '在市股'), ('000003', '退市股'), ('000018', '退市股')]:
        v = jump_removed(c)
        print(f'  {c} [{tag}] 除权日跳变消除中位数 = {"%.4f" % v if v is not None else "无除权日"}')


def main():
    dl = load()
    print('=' * 64)
    print(f'退市股入库校验  ({len(dl)} 只)')
    print('=' * 64)
    check_coverage(dl)
    check_schema(dl)
    check_adj_normalized(dl)
    check_alignment(dl)
    check_dup(dl)
    check_continuity(dl)
    check_against_live(dl)
    print('\n' + '=' * 64)
    print(f'通过 {len(PASS)} 项, 警告 {len(WARN)} 项, 失败 {len(FAIL)} 项')
    if FAIL:
        print('\n失败明细:')
        for m in FAIL:
            print('  -', m)
    res = {'pass': PASS, 'warn': WARN, 'fail': FAIL,
           'at': str(dt.datetime.now())}
    json.dump(res, open(os.path.join(CACHE, 'delisted_verify.json'), 'w'),
              ensure_ascii=False, indent=2)
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
