"""抓 **退市股** 财务数据，产出与现有 PIT 表**同口径**的 roe。

为什么不能直接用 tushare 的 ``roe``
-----------------------------------
实测对账（600519，55 个报告期）：现有 PIT 的 ``roe``（gpcw 字段 197）
与 tushare 的四个 ROE 变体**没有一期相同**：

    tushare roe         最大差 6.03   中位差 0.65   一致 0/55
    tushare roe_waa     最大差 6.03   中位差 0.77   一致 0/55
    tushare roe_dt      最大差 6.35   中位差 0.77   一致 0/55
    tushare roe_yearly  最大差 33.31  中位差 13.77  一致 0/55

而用**两个原始科目**自算：

    roe = 归母净利润(累计, 元) / 期末归母净资产(元) × 100
        → 最大差 **0.0005**，**55/55 期一致** ✅

所以本脚本只取两个原始科目，**不碰** tushare 的任何财务比率：

    income.n_income_attr_p              → net_income_attr_p
    balancesheet.total_hldr_eqy_exc_min_int → equity_attr_p

口径对齐（与 ``私有数据仓/tools/financial_pit.py`` 完全一致）
-----------------------------------------------------------
* ``report_date``  : ``int32``，形如 ``20240630``
* ``roe``          : ``float32``，百分数、**不年化、非加权平均**
* ``ann_314``      : 公告日（tushare ``ann_date``）
* ``avail_314``    : **严格晚于公告日的第 1 个交易日**
                     （用与 PIT 相同的 ``trade_date_sse`` 日历）
* 唯一性           : ``(code, report_date)`` 唯一（同 Tushare 多版本 → 取最新公告版）

不写 mongo
----------
现有 ``quantaxis.financial`` 是 **gpcw 数字键（001–584）** 格式，且是
``financial_pit.py`` 的**唯一只读真相源**。混入 tushare 英文字段会污染它，
所以本脚本只产出 parquet：

    cache/financial_delisted.parquet         完整版本史（可回溯，多版本并存）
    cache/financial_pit_with_delisted.parquet  PIT 同构合并表（原表不动）

用法::

    python fetch_delisted_financials.py verify        # 先用在市股验口径(快)
    python fetch_delisted_financials.py build         # 抓 250 只退市股
    python fetch_delisted_financials.py merge         # 合并成 PIT 同构表
    python fetch_delisted_financials.py info
"""
from __future__ import annotations

# 说明：roe 的两条修正规则在 merge() 里实现，见 ZERO_RUN 与报告 §

import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# 私有数据根目录（可用环境变量 ALPHALENS_DATA_ROOT 覆盖）
DATA_ROOT = os.environ.get(
    'ALPHALENS_DATA_ROOT',
    os.path.expanduser('~/alphalens-data'))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# QUANTAXIS 要在 import 前重定向 HOME（与库里其它脚本同一套处理）
os.environ['HOME'] = os.path.join(HERE, '.qh')
os.makedirs(os.path.join(HERE, '.qh', '.quantaxis', 'log'), exist_ok=True)
os.makedirs(os.path.join(HERE, '.qh', '.quantaxis', 'setting'), exist_ok=True)

MONGO_URI, MONGO_DB = 'mongodb://127.0.0.1:27017', 'quantaxis'
PIT_PATH = os.path.join(DATA_ROOT, 'data/financial_pit/financial_pit.parquet')
CACHE = os.path.join(HERE, 'cache')
OUT = os.path.join(CACHE, 'financial_delisted.parquet')
MERGED = os.path.join(CACHE, 'financial_pit_with_delisted.parquet')
META = os.path.join(CACHE, 'financial_delisted.meta.json')

START, END = '20160101', '20261231'
DELIST_FROM = '20160101'          # 只取退市日不早于面板起点的
WINDOW_YEARS = 6                  # 分窗取数，避开单次返回行数上限
SLEEP = 0.12                      # 调用间隔，别把接口打爆
ZERO_RUN = None                   # None = 所有 roe==0 一律视为缺失（业务判断）
VERIFY_FIELDS = ('report_date', 'roe')

INC_FIELDS = 'end_date,ann_date,f_ann_date,report_type,update_flag,n_income_attr_p'
BAL_FIELDS = ('end_date,ann_date,f_ann_date,report_type,update_flag,'
              'total_hldr_eqy_exc_min_int')


# --------------------------------------------------------------------------- #
def read_token():
    """从 quantaxis 配置的 [TSPRO] 段读 tushare token（不调 ts.set_token，它会写 ~/tk.csv）。"""
    cfg = os.path.expanduser('~/.quantaxis/setting/config.ini')
    if not os.path.exists(cfg):
        cfg = 'os.path.expanduser('~/.quantaxis')/setting/config.ini'
    in_tspro = False
    for line in open(cfg, encoding='utf-8'):
        s = line.strip()
        if s.startswith('['):
            in_tspro = s.lower().startswith('[tspro]')
        elif in_tspro and s.lower().startswith('token'):
            return s.split('=', 1)[1].strip()
    raise SystemExit('配置里没有 [TSPRO] token')


def pro_api():
    import tushare as ts
    return ts.pro_api(read_token())


def trading_calendar():
    """与 PIT 表同一个交易日历：``trade_date_sse``。"""
    try:
        from QUANTAXIS.QAUtil.QADate_trade import trade_date_sse
        cal = pd.Series(list(trade_date_sse)).astype('datetime64[ns]')
        return np.sort(cal.values.astype('int64')), 'trade_date_sse'
    except Exception as e:                                        # noqa: BLE001
        print(f'  ⚠️ 取不到 trade_date_sse（{type(e).__name__}），退回工作日近似')
        v = pd.bdate_range('1990-12-19', '2026-12-31').values.astype('int64')
        return v, 'bdate_range(近似)'


def next_trading_day(ann, cal_vals, name='avail_314'):
    """公告日 → 严格晚于它的第 1 个交易日（与 PIT 的 ``_next_trading_day`` 同法）。"""
    a = pd.to_datetime(ann).values.astype('datetime64[ns]').astype('int64')
    idx = np.searchsorted(cal_vals, a, side='right')
    nat = pd.isna(ann).values
    ok = (idx < len(cal_vals)) & (~nat)
    early = (~nat) & (a < cal_vals[0])
    out = np.full(len(a), np.datetime64('NaT'), dtype='datetime64[ns]')
    take = ok & (~early)
    out[take] = cal_vals[idx[take]].astype('datetime64[ns]')
    return pd.Series(out, index=ann.index), int(early.sum())


# --------------------------------------------------------------------------- #
def delisted_list(force=False):
    """退市股清单（带本地缓存）。"""
    cache = os.path.join(CACHE, 'delisted_basic.parquet')
    if os.path.exists(cache) and not force:
        df = pd.read_parquet(cache)
    else:
        from pymongo import MongoClient
        db = MongoClient(MONGO_URI, serverSelectionTimeoutMS=20000)[MONGO_DB]
        rows = list(db.stock_basic.find({'list_status': 'D'},
                                        {'ts_code': 1, 'symbol': 1, 'name': 1,
                                         'list_date': 1, 'delist_date': 1, '_id': 0}))
        df = pd.DataFrame(rows)
        df.to_parquet(cache)
    df['code'] = df['symbol'].astype(str).str.zfill(6)
    df['delist_date'] = pd.to_datetime(df['delist_date'], format='%Y%m%d', errors='coerce')
    df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')
    df = df[df['delist_date'] >= pd.Timestamp(DELIST_FROM)]
    return df.sort_values('code').reset_index(drop=True)


def _call(pro, fn, **kw):
    """带重试的接口调用（频次限制很常见）。"""
    for attempt in range(4):
        try:
            return fn(**kw)
        except Exception as e:                                    # noqa: BLE001
            msg = str(e)
            if attempt == 3:
                raise
            wait = 2.0 * (attempt + 1)
            if '分钟' in msg or 'limit' in msg.lower() or '频率' in msg:
                wait = 15.0
            time.sleep(wait)
    return pd.DataFrame()


def _fetch_windows(pro, fn, ts_code, fields, start, end):
    """按时间窗取数，避开单次返回行数上限。"""
    parts = []
    y0, y1 = int(start[:4]), int(end[:4])
    for a in range(y0, y1 + 1, WINDOW_YEARS):
        w = (f'{a}0101', f'{min(a + WINDOW_YEARS - 1, y1)}1231')
        df = _call(pro, fn, ts_code=ts_code, start_date=w[0], end_date=w[1],
                   fields=fields)
        if df is not None and len(df):
            parts.append(df)
        time.sleep(SLEEP)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def fetch_one(pro, ts_code, start=START, end=END):
    """一只票的 income + balancesheet。"""
    inc = _fetch_windows(pro, pro.income, ts_code, INC_FIELDS, start, end)
    bal = _fetch_windows(pro, pro.balancesheet, ts_code, BAL_FIELDS, start, end)
    return inc, bal


def compute_roe(inc, bal):
    """合并两表 → 版本史 → roe = 归母净利 / 期末归母净资产 × 100。

    同一 ``end_date`` 可能有多行（修正版本），**全部保留**：
    每行带自己的 ``ann_date``，PIT 合并时才按"当期可知"选版本（不会前视）。
    """
    if not len(inc) or not len(bal):
        return pd.DataFrame()
    a = inc[['end_date', 'ann_date', 'f_ann_date', 'report_type',
             'update_flag', 'n_income_attr_p']].copy()
    b = bal[['end_date', 'ann_date', 'f_ann_date', 'report_type',
             'update_flag', 'total_hldr_eqy_exc_min_int']].copy()
    a['ann'] = pd.to_datetime(a['ann_date'], format='%Y%m%d', errors='coerce')
    a['ann'] = a['ann'].fillna(
        pd.to_datetime(a['f_ann_date'], format='%Y%m%d', errors='coerce'))
    b['ann'] = pd.to_datetime(b['ann_date'], format='%Y%m%d', errors='coerce')
    b['ann'] = b['ann'].fillna(
        pd.to_datetime(b['f_ann_date'], format='%Y%m%d', errors='coerce'))
    # 净资产是时点值：同报告期多版本取**最新公告**的那个
    b = b.sort_values('ann').drop_duplicates('end_date', keep='last')
    m = a.merge(b[['end_date', 'total_hldr_eqy_exc_min_int']], on='end_date',
                how='inner', suffixes=('', '_b'))
    m = m.rename(columns={'n_income_attr_p': 'net_income_attr_p',
                          'total_hldr_eqy_exc_min_int': 'equity_attr_p'})
    m['net_income_attr_p'] = pd.to_numeric(m['net_income_attr_p'], errors='coerce')
    m['equity_attr_p'] = pd.to_numeric(m['equity_attr_p'], errors='coerce')
    eq = m['equity_attr_p']
    # ★ 口径与 gpcw 字段 197 **完全一致**：净资产 <= 0 时 gpcw 记 **0**
    #   （实测 000505：净资产 −3.96e7，gpcw 给的是 0.0000，不是 −176.5）
    #   也就是说 gpcw 用 0 编码"无定义"。要合并就必须照抄这个约定，
    #   否则补进去的退市股和既有股票是两套口径。
    #   真正的"0 → 缺失"由 load_pit() 的两条规则统一处理（见 README/报告）。
    m['roe'] = np.where(eq > 0, m['net_income_attr_p'] / eq * 100.0, 0.0)
    m['equity_le_zero'] = (eq <= 0)
    m['report_date'] = pd.to_numeric(m['end_date'], errors='coerce').astype('Int64')
    m = m[m['report_date'].notna() & m['ann'].notna()]
    return m


# --------------------------------------------------------------------------- #
def verify(pro=None, n=20, seed=0):
    """★ 用**在市股**验口径：同一份代码路径算出来的 roe，必须等于 PIT 里的 roe。

    这是全流程唯一的正确性保证 —— 口径不一致，补进去就是混口径。
    """
    pro = pro or pro_api()
    pit = pd.read_parquet(PIT_PATH, columns=['code', 'report_date', 'roe'])
    rng = np.random.default_rng(seed)
    codes = pit['code'].dropna().unique()
    # 刻意混入一部分 ST / 小票（净资产可能为负），把公式的压力测出来
    pick = list(rng.choice(codes, size=n, replace=False))
    rows = []
    for c in pick:
        ts_code = f'{c}.SH' if c[0] == '6' else f'{c}.SZ'
        try:
            inc, bal = fetch_one(pro, ts_code, '20100101', END)
            mine = compute_roe(inc, bal)
        except Exception as e:                                    # noqa: BLE001
            print(f'  {c} 取数失败: {str(e)[:60]}')
            continue
        if not len(mine):
            continue
        mine = mine.sort_values('ann').drop_duplicates('report_date', keep='last')
        mine['code'] = c
        j = mine.merge(pit[pit['code'] == c], on=['code', 'report_date'],
                       how='inner', suffixes=('_mine', '_pit'))
        if not len(j):
            continue
        d = (j['roe_mine'].astype(float) - j['roe_pit'].astype(float)).abs()
        rows.append({'code': c, 'periods': len(j), 'max_diff': float(d.max()),
                     'median_diff': float(d.median()),
                     'within_0.01': int((d < 0.01).sum())})
    rep = pd.DataFrame(rows)
    if not len(rep):
        print('  没有可比对的期次')
        return rep
    tot = int(rep['periods'].sum())
    good = int(rep['within_0.01'].sum())
    print(f'\n口径对账（{len(rep)} 只在市股，{tot} 个报告期）')
    print(f'  差 < 0.01 的期次：{good}/{tot} = {good / tot:.2%}')
    print(f'  全样本最大差 {rep["max_diff"].max():.4f}，'
          f'中位差 {rep["median_diff"].median():.4f}')
    bad = rep[rep['within_0.01'] < rep['periods']]
    if len(bad):
        print(f'  有差异的股票 {len(bad)} 只（前 5）:')
        print(bad.head(5).to_string(index=False))
    ok = good / tot >= 0.98
    print(f'  结论：{"✅ 口径一致，可以合并" if ok else "❌ 口径不一致，不许合并"}')
    return rep


def build(limit=None, force=False):
    """抓全部退市股。断点续跑：已抓的存 part 文件。"""
    os.makedirs(CACHE, exist_ok=True)
    parts_dir = os.path.join(CACHE, 'delisted_fin_parts')
    os.makedirs(parts_dir, exist_ok=True)
    d = delisted_list(force=force)
    if limit:
        d = d.head(limit)
    cal, cal_name = trading_calendar()
    print(f'退市股 {len(d)} 只（退市日 ≥ {DELIST_FROM}），交易日历 {cal_name}')

    pro = pro_api()
    t0, done, empty = time.time(), 0, []
    for i, r in enumerate(d.itertuples(), 1):
        part = os.path.join(parts_dir, f'{r.code}.parquet')
        if os.path.exists(part) and not force:
            done += 1
            continue
        try:
            inc, bal = fetch_one(pro, r.ts_code)
            m = compute_roe(inc, bal)
        except Exception as e:                                    # noqa: BLE001
            print(f'  [{i}/{len(d)}] {r.code} 失败: {str(e)[:70]}', flush=True)
            empty.append(r.code)
            continue
        if not len(m):
            empty.append(r.code)
            pd.DataFrame().to_parquet(part)
            continue
        m['code'] = r.code
        m['ts_code'] = r.ts_code
        m['delist_date'] = r.delist_date
        m['avail_314'], n_early = next_trading_day(m['ann'], cal)
        m['ann_314'] = m['ann']
        m = m.sort_values(['report_date', 'ann'])
        m.to_parquet(part)
        done += 1
        if i % 20 == 0 or i == len(d):
            print(f'  [{i}/{len(d)}] 已抓 {done}，空 {len(empty)}，'
                  f'{time.time() - t0:.0f}s', flush=True)

    # 汇总
    frames = []
    for f in sorted(os.listdir(parts_dir)):
        if f.endswith('.parquet'):
            x = pd.read_parquet(os.path.join(parts_dir, f))
            if len(x):
                frames.append(x)
    if not frames:
        raise SystemExit('一份数据都没抓到')
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(['code', 'report_date', 'ann'])
    out = out.sort_values(['code', 'report_date', 'ann']).reset_index(drop=True)
    out.to_parquet(OUT)
    meta = {
        'built_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'codes': int(out['code'].nunique()),
        'rows': int(len(out)),
        'periods': int(out.groupby(['code', 'report_date']).ngroups),
        'empty_codes': empty,
        'roe_formula': 'net_income_attr_p / equity_attr_p * 100 '
                       '(累计归母净利润 / 期末归母净资产)',
        'source': 'tushare income.n_income_attr_p + '
                  'balancesheet.total_hldr_eqy_exc_min_int',
        'calendar': cal_name,
        'window': [START, END],
    }
    with open(META, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f'\n写出 {OUT}  {out.shape}')
    print(f'  股票 {meta["codes"]} 只，记录 {meta["rows"]} 行，'
          f'报告期 {meta["periods"]} 个，空票 {len(empty)} 只')
    return out


def merge():
    """合并成 PIT 同构表（**原表不动**）。

    PIT 的 ``(code, report_date)`` 唯一 → 每期只保留**最新公告版本**。
    """
    add = pd.read_parquet(OUT)
    pit = pd.read_parquet(PIT_PATH)
    add = add.sort_values('ann').drop_duplicates(['code', 'report_date'],
                                                keep='last')
    rows = pd.DataFrame({
        'code': add['code'].astype(str),
        'report_date': add['report_date'].astype('int32'),
        'roe': add['roe'].astype('float32'),
        # 原始科目也带上 —— 便于事后复核口径，PIT 本身没有这两列
        'net_income_attr_p': add['net_income_attr_p'].astype('float64'),
        'equity_attr_p': add['equity_attr_p'].astype('float64'),
        'ann_314': pd.to_datetime(add['ann_314']),
        'avail_314': pd.to_datetime(add['avail_314']),
        'source': 'tushare-delisted',
    })
    rows = rows[rows['avail_314'].notna()]
    for c in pit.columns:
        if c not in rows.columns:
            rows[c] = np.nan
    # ⚠️ equity_attr_p 本来就在 PIT 里，重复列名会让 concat 直接崩
    extra = [c for c in ('net_income_attr_p', 'equity_attr_p', 'source')
             if c not in pit.columns]
    rows = rows[pit.columns.tolist() + extra]
    merged = pd.concat([pit, rows], ignore_index=True)
    merged = merged.sort_values(['code', 'report_date', 'avail_314'])
    dup = merged.duplicated(['code', 'report_date'], keep=False)
    if dup.any():
        n = int(merged.loc[dup, 'code'].nunique())
        print(f'  ⚠️ {int(dup.sum())} 行与既有 PIT 重复 (code, report_date)，'
              f'涉及 {n} 只 —— 只保留 PIT 原行')
        merged = merged[~(dup & (merged['source'] != 'tushare-delisted'))]
        merged = merged.drop_duplicates(['code', 'report_date'], keep='first')
    merged = merged.sort_values(['code', 'report_date']).reset_index(drop=True)

    # 掩码只**报告**不落盘 —— 保持与 PIT 口径一致，修正放到读取时做
    # （这样才跑得出"原始 / +退市股 / +零值修正"三段对照）
    _, rep = zero_roe_mask(merged)
    merged = merged.sort_values(['code', 'report_date']).reset_index(drop=True)
    merged.to_parquet(MERGED)
    with open(os.path.join(CACHE, 'roe_missing_report.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    n_new = int((merged['source'] == 'tushare-delisted').sum())
    print(f'写出 {MERGED}')
    print(f'  {pit.shape} + 退市股 {n_new} 行 → {merged.shape}')
    print(f'  股票数 {pit["code"].nunique()} → {merged["code"].nunique()}')
    print('  roe=0 修正（读取时生效）：%d 行 roe==0（%.2f%%）全部置 NaN，'
          '涉及 %d 只股票；其中净资产<0 的 %d 行、未披露占位 %d 行'
          % (rep['masked'], 100 * rep['zero_pct'], rep['codes_affected'],
             rep['why_negative_equity'], rep['why_positive_equity']))
    return merged


def zero_roe_mask(df, zero_run=None):
    """``roe == 0`` **一律视为缺失**（0 是占位，不是真值）。

    依据（业务判断 + 实测）

    * ROE = 归母净利润 / 归母净资产。要恰好等于 0，得净利润**精确为 0** ——
      现实中不会发生。所以 0 是"无定义 / 未披露"的**占位编码**。
    * 实测 PIT 全表 ``roe == 0`` 有 3,118 行（1.03%），拆开看：
      净资产 < 0 占 **78.8%**（ROE 数学上无定义）、净资产 = 0 占 0.5%、
      净资产 > 0 占 20.6%（未披露占位）。三类都不该当成"ROE 正好是 0"。

    不修正的后果（实测）：这批票的因子值会被读成 0（= 截面中位水平），
    而它们其实是**净资产为负的困境公司** —— 恰恰是 ROE 因子最该识别的样本；
    更糟的是因子值会连续几十个月冻结在 0（"因子冻结"告警的来源）。

    Parameters
    ----------
    df : DataFrame
        需含 ``roe``；``equity_attr_p`` 有则用于**归因统计**（不影响掩码）。
    zero_run : int, optional
        兼容旧行为的备用规则（净资产>0 且连续 N 期恒 0 才算）。
        默认 ``None`` = 不做区分，**所有 0 都置 NaN**。

    Returns
    -------
    (mask, report)
        ``mask`` 布尔 Series（True = 应置 NaN）；``report`` 计数 dict。
    """
    r = df
    is0 = (r['roe'] == 0).fillna(False)
    eq = (pd.to_numeric(r['equity_attr_p'], errors='coerce')
          if 'equity_attr_p' in r.columns else pd.Series(np.nan, index=r.index))
    neg = is0 & eq.notna() & (eq < 0)
    zero_eq = is0 & eq.notna() & (eq == 0)
    pos = is0 & eq.notna() & (eq > 0)
    unknown = is0 & eq.isna()

    mask = is0
    if zero_run is not None:
        # 备用规则：只挑"净资产>0 且连续 >= zero_run 期恒 0"
        s2 = pd.DataFrame({'code': r['code'].values, 'is0': pos.values},
                          index=r.index)
        grp = (~s2['is0']).cumsum()
        mask = pd.Series(False, index=r.index)
        if s2['is0'].any():
            sub = s2[s2['is0']]
            run = sub.groupby([grp[s2['is0']], sub['code']])['is0'].transform('size')
            mask.loc[run.index] = (run >= zero_run).values

    rep = {
        'rows': int(len(r)),
        'roe_zero_rows': int(is0.sum()),
        'zero_pct': float(is0.mean()) if len(r) else 0.0,
        'why_negative_equity': int(neg.sum()),
        'why_zero_equity': int(zero_eq.sum()),
        'why_positive_equity': int(pos.sum()),
        'why_equity_unknown': int(unknown.sum()),
        'masked': int(mask.sum()),
        'codes_affected': int(r.loc[mask, 'code'].nunique()),
        'mode': 'all_zeros' if zero_run is None else f'run>={zero_run}',
    }
    return mask, rep


def load_pit(with_delisted=True, zerofix=True):
    """读 PIT（可含退市股），按需施加 roe 零值修正。

    Parameters
    ----------
    with_delisted : bool
        ``True`` 读合并表（含 250 只退市股）；``False`` 读原始 PIT。
    zerofix : bool
        ``True`` 把"净资产<=0"与"连续多期恒 0"的 roe 置为 NaN。

    Returns
    -------
    DataFrame
        与原 PIT 同结构（修正模式下 ``roe`` 有 NaN）。
    """
    path = MERGED if (with_delisted and os.path.exists(MERGED)) else PIT_PATH
    df = pd.read_parquet(path)
    if zerofix and 'equity_attr_p' in df.columns:
        mask, rep = zero_roe_mask(df)
        df = df.copy()
        df.loc[mask, 'roe'] = np.nan
        df.attrs['zero_fix'] = rep
    return df


def info():
    for p in (OUT, MERGED, META):
        if os.path.exists(p):
            print(f'  {p}  ({os.path.getsize(p) / 1e6:.2f} MB)')
    if os.path.exists(META):
        print(json.dumps(json.load(open(META, encoding='utf-8')),
                         ensure_ascii=False, indent=2)[:900])


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'info'
    if cmd == 'verify':
        verify(n=int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif cmd == 'build':
        build(limit=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif cmd == 'merge':
        merge()
    else:
        info()
