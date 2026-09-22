"""私有数据源适配器 —— 示范「不改库也能接自己的数据」。

**这个文件不在 ``alphalens_cna`` 包里。** 它是你的环境专用代码。
库更新时只要 ``InputAdapter`` 协议不变，这里一行都不用改；
而库里**永远不会出现**这些私有集合名或路径。

数据分布
--------
============================  ==========================================
数据                           位置
============================  ==========================================
日线行情 / 复权 / 除权事件      mongo ``quantaxis``: stock_day / stock_adj / stock_xdxr
每日指标（市值/换手/涨跌停）    mongo ``stock_daily_basic``
**PIT 名称（ST 判定）**        mongo ``stock_namechange``  ← 事件式区间，1990–今
上市/退市日                     mongo ``stock_basic``
PIT 财务（因子源）              ``私有数据仓/data/financial_pit/*.parquet``
申万行业（分组）                ``私有数据仓/data/sw_industry/*.parquet``
============================  ==========================================

用法
----
>>> import sys; sys.path.insert(0, 'examples')
>>> from 私有数据仓_adapter import QuantmingAdapter
>>> src = QuantmingAdapter()
>>> px   = acna.load_prices(source=src, start='2020-01-01', end='2024-12-31')
>>> roe  = acna.load_factor('roe', source=src, start='2020-01-01')
"""

from __future__ import annotations

import functools
import os
import time

import numpy as np
import pandas as pd
import pymongo

from alphalens_cna.adapters.input.base import InputAdapter

MONGO = os.environ.get('QA_MONGO', 'mongodb://127.0.0.1:27017')
DB = os.environ.get('QA_DB', 'quantaxis')
QUANTMING = os.path.expanduser(os.environ.get('QUANTMING_DIR', '~/私有数据仓'))


class QuantmingAdapter(InputAdapter):
    """接本机 quantaxis mongo + 私有数据仓 parquet。"""

    name = '私有数据仓'

    def __init__(self, mongo=MONGO, db=DB, 私有数据仓=QUANTMING):
        self.db = pymongo.MongoClient(mongo)[db]
        self.root = 私有数据仓
        self._cal = None

    # ---------------------------------------------------------------- 行情 --
    def prices(self, start=None, end=None, codes=None, **kw):
        """日线行情。**raw_* 用 stock_day 的真实价，复权因子用 stock_adj。**

        取数策略：
        * 给了 ``codes`` → 按 code 查（走 ``code_1_date_stamp_1`` 索引，快）
        * 没给 → 读工作区面板缓存（``cache/px_daily.parquet``）；
          缓存不存在才全表扫（**很慢**，会提示）

        注意 ``stock_adj.adj`` 是**前复权**（末位归一为 1）—— 作 ``adj_factor``
        用没问题：任何常数缩放都不影响收益（见 ``check_acna.check_adjust_agreement``）。
        """
        q = {}
        if codes is not None:
            q['code'] = {'$in': [str(c).zfill(6) for c in codes]}
        cols = {'_id': 0, 'date': 1, 'code': 1, 'open': 1, 'high': 1, 'low': 1,
                'close': 1, 'vol': 1, 'amount': 1}
        t0 = time.time()
        d = pd.DataFrame(list(self.db.stock_day.find(q, cols)))
        if not len(d):
            return None
        if codes is None and time.time() - t0 > 30:
            print(f'[私有数据仓] 全表扫 stock_day 用了 {time.time()-t0:.0f}s；'
                  f'建议传 codes= 或先跑 prep_panel.py 建缓存', flush=True)
        d['date'] = pd.to_datetime(d['date'])
        d = d.rename(columns={'code': 'asset', 'open': 'raw_open', 'high': 'raw_high',
                              'low': 'raw_low', 'close': 'raw_close', 'vol': 'volume'})
        d = d.set_index(['date', 'asset']).sort_index()
        if start is not None:
            d = d[d.index.get_level_values('date') >= pd.Timestamp(start)]
        if end is not None:
            d = d[d.index.get_level_values('date') <= pd.Timestamp(end)]
        key = tuple(sorted(str(c).zfill(6) for c in codes)) if codes else None
        d['adj_factor'] = self._adj(key).reindex(d.index).astype(float)
        return d

    @functools.lru_cache(maxsize=8)
    def _adj(self, codes=None):
        """复权因子。``codes`` 传 tuple（lru_cache 要求可哈希）。"""
        q = {}
        if codes is not None:
            q['code'] = {'$in': list(codes)}
        cur = self.db.stock_adj.find(q, {'_id': 0, 'date': 1, 'code': 1, 'adj': 1})
        d = pd.DataFrame(list(cur))
        d['date'] = pd.to_datetime(d['date'])
        s = d.rename(columns={'code': 'asset'}).set_index(['date', 'asset'])['adj']
        s = s.sort_index()
        s.name = 'adj_factor'
        return s

    # ------------------------------------------------------------- 除权事件 --
    def xdxr(self, **kw):
        cur = self.db.stock_xdxr.find({}, {'_id': 0, 'date': 1, 'code': 1,
                                           'category': 1, 'fenhong': 1, 'peigu': 1,
                                           'peigujia': 1, 'songzhuangu': 1})
        d = pd.DataFrame(list(cur))
        if not len(d):
            return None
        d['date'] = pd.to_datetime(d['date'])
        return d.rename(columns={'code': 'asset'}).set_index(['date', 'asset']).sort_index()

    # ---------------------------------------------------------------- 日历 --
    @functools.lru_cache(maxsize=1)
    def _trade_dates(self):
        """交易日历 —— 用**上证指数**（000001）的日期。

        为什么不用 ``stock_adj`` 的日期：那是 1789 万条，全表读要几分钟。
        指数的日期序列就是市场交易日，且按 code 查走索引，0.25 秒。
        """
        cur = self.db.index_day.find({'code': '000001'}, {'_id': 0, 'date': 1})
        d = pd.DataFrame(list(cur))
        if not len(d):
            raise RuntimeError('index_day 里没有上证指数（000001），无法建日历')
        return pd.DatetimeIndex(pd.to_datetime(d['date'])).unique().sort_values()

    def calendar(self, start=None, end=None, **kw):
        c = self._trade_dates()
        if start is not None:
            c = c[c >= pd.Timestamp(start)]
        if end is not None:
            c = c[c <= pd.Timestamp(end)]
        return c

    # ------------------------------------------------------- PIT 名称 / 上市 --
    def names(self, start=None, end=None, index=None, **kw):
        """PIT 名称 —— 事件式区间 ``start_date ~ end_date``。

        覆盖 **1990 至今**（不走 ``stock_basic_daily`` 那种每日快照，那个只有 2016+）。

        Parameters
        ----------
        index : MultiIndex, 可选
            **只要这些 (date, asset) 的名称。** 强烈建议传 ——
            全量展开一年就是 136 万行（72 秒），按需查只要几毫秒。
        """
        nc = self._namechange()
        if index is not None:
            return self._name_at(nc, pd.DatetimeIndex(index.get_level_values('date')),
                                 pd.Index(index.get_level_values('asset')), index)
        dates = self.calendar(start, end)
        return self._name_expand(nc, dates)

    @functools.lru_cache(maxsize=1)
    def _namechange(self):
        cur = self.db.stock_namechange.find(
            {}, {'_id': 0, 'ts_code': 1, 'start_date': 1, 'end_date': 1, 'name': 1})
        nc = pd.DataFrame(list(cur))
        if not len(nc):
            return None
        nc['asset'] = nc['ts_code'].str.split('.').str[0]
        nc['start_date'] = pd.to_datetime(nc['start_date'], format='%Y%m%d', errors='coerce')
        nc['end_date'] = pd.to_datetime(nc['end_date'], format='%Y%m%d', errors='coerce')
        return nc.dropna(subset=['start_date']).sort_values('start_date')

    def _name_at(self, nc, dates, assets, index):
        """按需查：只解析给定的 (date, asset) 对。"""
        need = pd.DataFrame({'date': np.asarray(dates), 'asset': np.asarray(assets)})
        out = pd.Series(None, index=index, dtype=object)
        for asset, g in nc.groupby('asset', sort=False):
            m = need['asset'].values == asset
            if not m.any():
                continue
            q = need.loc[m, 'date'].values
            # 每个查询日落在哪个区间：取 start_date <= q 的最后一条
            pos = np.searchsorted(g['start_date'].values, q, side='right') - 1
            ok = pos >= 0
            if not ok.any():
                continue
            sel = g.iloc[pos[ok]]
            endok = sel['end_date'].isna().values | (sel['end_date'].values >= q[ok])
            res = np.where(endok, sel['name'].values, None)
            out.iloc[np.flatnonzero(m)[ok]] = res
        out.name = 'name'
        return out

    def _name_expand(self, nc, dates):
        """全量展开成 (date, asset) → name。大区间会很大，优先用 index= 按需查。"""
        dv = dates.values
        idxs, names = [], []
        for asset, g in nc.groupby('asset', sort=False):
            st = g['start_date'].values
            ed = g['end_date'].values
            lo = np.searchsorted(dv, st, side='left')
            hi = np.where(pd.isna(ed), len(dates),
                          np.searchsorted(dv, np.where(pd.isna(ed), dv[-1], ed), side='right'))
            counts = np.clip(hi - lo, 0, None)
            if counts.sum() == 0:
                continue
            # 向量化展开（repeat + cumsum 技巧），不做逐行 python 循环
            offsets = np.repeat(np.cumsum(counts) - counts, counts)
            pos = np.repeat(lo, counts) + (np.arange(counts.sum()) - offsets)
            idxs.append(pd.MultiIndex.from_arrays(
                [dates[pos], np.repeat(asset, counts.sum())], names=['date', 'asset']))
            names.append(pd.Series(np.repeat(g['name'].values, counts)))
        if not idxs:
            return None
        s = pd.Series(np.concatenate([n.values for n in names]),
                      index=idxs[0].append(idxs[1:]) if len(idxs) > 1 else idxs[0])
        s = s.sort_index()
        s.name = 'name'
        return s[~s.index.duplicated(keep='last')]

    def list_dates(self, **kw):
        """``asset -> 上市日``（来自 ``stock_basic.list_date``）。"""
        cur = self.db.stock_basic.find({}, {'_id': 0, 'symbol': 1, 'list_date': 1})
        d = pd.DataFrame(list(cur))
        if not len(d):
            return None
        d['asset'] = d['symbol'].astype(str).str.zfill(6)
        s = pd.to_datetime(d.set_index('asset')['list_date'], format='%Y%m%d', errors='coerce')
        return s.dropna()

    @functools.lru_cache(maxsize=1)
    def limit_status(self):
        """``stock_daily_basic.limit_status``（收盘状态；仅作降级用）。"""
        cur = self.db.stock_daily_basic.find(
            {}, {'_id': 0, 'date': 1, 'code': 1, 'limit_status': 1}).limit(3_000_000)
        d = pd.DataFrame(list(cur))
        if not len(d) or 'limit_status' not in d.columns:
            return None
        d['date'] = pd.to_datetime(d['date'])
        s = d.rename(columns={'code': 'asset'}).set_index(['date', 'asset'])['limit_status']
        return s.sort_index()

    # ---------------------------------------------------------------- 因子 --
    def factor(self, name, start=None, end=None, **kw):
        """PIT 财务因子。默认从 ``financial_pit.parquet`` 读。"""
        p = os.path.join(self.root, 'data', 'financial_pit', 'financial_pit.parquet')
        if not os.path.exists(p):
            return None
        df = pd.read_parquet(p)
        if name not in df.columns:
            return None
        code = 'code' if 'code' in df.columns else 'ts_code'
        avail = f'avail_{kw.get("avail", "314")}'
        if avail not in df.columns:
            avail = 'ann_314' if 'ann_314' in df.columns else None
        out = pd.DataFrame({
            'date': pd.to_datetime(df[avail] if avail else df[code]),
            'asset': df[code].astype(str).str.zfill(6),
            'value': pd.to_numeric(df[name], errors='coerce'),
        })
        out['available_at'] = out['date']
        out = out.dropna(subset=['value'])
        return out.set_index(['date', 'asset']).sort_index()

    # ---------------------------------------------------------------- 分组 --
    def grouping(self, name='sw_l1', start=None, end=None, **kw):
        p = os.path.join(self.root, 'data', 'sw_industry', 'members.parquet')
        if not os.path.exists(p):
            return None
        m = pd.read_parquet(p)
        cols = {c.lower(): c for c in m.columns}
        dcol = next((cols[c] for c in ('in_date', 'date', 'start_date') if c in cols), None)
        ccol = next((cols[c] for c in ('code', 'ts_code', 'stock_code') if c in cols), None)
        gcol = next((cols[c] for c in ('l1_name', 'l1_code', 'sw_l1', 'industry') if c in cols), None)
        if not all([dcol, ccol, gcol]):
            return None
        out = pd.DataFrame({'date': pd.to_datetime(m[dcol]),
                            'asset': m[ccol].astype(str).str.zfill(6),
                            'group': m[gcol].astype(str)})
        return out.dropna().set_index(['date', 'asset']).sort_index()
