# -*- coding: utf-8 -*-
"""事件研究 —— 财报公告（PEAD 检验）

事件研究能不能做？
    能。本质就是"把事件日前后 N 天的收益窗口对齐后取平均"，比因子分析简单。

事件源: quantming/data/financial_pit 的 `ann_314` = 财报公告日
问题:   财报公告前后，股价怎么走？按"业绩惊喜"分组后有没有漂移（PEAD）？

同时与 alphalens 的 average_cumulative_return_by_quantile **对拍**，
以确认它返回的到底是"累积收益路径"还是"平均日收益"（源码命名存疑）。

输出: outputs/event_*.json/.csv, outputs/charts/3x_*.png
"""
import json
import os
import warnings

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['FZLanTingHei-R-GBK', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from alphalens import performance as perf
from alphalens.utils import get_clean_factor

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, 'cache'), os.path.join(HERE, 'outputs')
CHARTS = os.path.join(OUT, 'charts')
PIT = '/home/yinxiuqu/quantming/data/financial_pit/financial_pit.parquet'

BEFORE, AFTER = 5, 20          # 事件窗口: -5 ~ +20 交易日
START, END = '2020-01-01', '2026-06-30'


def load_events():
    """事件表: (code, ann_date, surprise)"""
    pit = pd.read_parquet(PIT, columns=['code', 'report_date', 'ann_314', 'profit_yoy', 'revenue_yoy'])
    pit = pit[pit['ann_314'].notna()].copy()
    pit['ann_314'] = pd.to_datetime(pit['ann_314'])
    pit = pit[(pit['ann_314'] >= START) & (pit['ann_314'] <= END)]
    # 业绩惊喜: 净利润同比增速 (缺失用营收增速兜底)
    pit['surprise'] = pit['profit_yoy'].fillna(pit['revenue_yoy'])
    pit = pit[pit['surprise'].notna()]
    pit = pit.drop_duplicates(subset=['code', 'ann_314'])
    return pit[['code', 'ann_314', 'surprise']].rename(columns={'ann_314': 'date'})


def align_windows(daily_ret, events, before, after):
    """把每个事件日前后的收益窗口对齐到共同相对日。返回 DataFrame(index=相对日, columns=事件序号)。"""
    idx = daily_ret.index
    pos = {d: i for i, d in enumerate(idx)}
    cols = {}
    for k, (code, dt) in enumerate(zip(events['code'].values, events['date'].values)):
        i = pos.get(pd.Timestamp(dt))
        if i is None or code not in daily_ret.columns:
            continue
        s, e = i - before, i + after + 1
        if s < 0 or e > len(idx):
            continue
        w = daily_ret[code].iloc[s:e]
        if w.isna().any():
            continue
        cols[k] = pd.Series(w.values, index=np.arange(-before, after + 1))
    return pd.DataFrame(cols)


def main():
    os.makedirs(CHARTS, exist_ok=True)
    px = pd.read_parquet(os.path.join(CACHE, 'px_daily.parquet'))
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    dr = px.pct_change()
    print('日收益面板 %s' % (dr.shape,), flush=True)

    ev = load_events()
    print('财报公告事件 %d 条  (%s ~ %s)' % (len(ev), ev['date'].min().date(), ev['date'].max().date()), flush=True)

    W = align_windows(dr, ev, BEFORE, AFTER)
    print('可用事件窗口 %d 个 (剔除停牌/边界)' % W.shape[1], flush=True)
    if W.shape[1] < 50:
        raise SystemExit('事件样本太少')

    # 全样本
    mean_daily = W.mean(axis=1)                      # 每个相对日的平均日收益
    cum_path = (1 + mean_daily).cumprod() - 1        # 累积路径
    # 用事件首日归一 (相对 -before 的累积)
    cum_from_start = (1 + mean_daily).cumprod()
    cum_from_start = cum_from_start / cum_from_start.loc[-BEFORE] - 1

    res = {'config': {'event': '财报公告日 (financial_pit.ann_314)', 'before': BEFORE, 'after': AFTER,
                      'range': [START, END], 'events_used': int(W.shape[1]),
                      'total_events': int(len(ev))}}

    # 按业绩惊喜分组 (高于/低于中位)
    # 注意: W 的列名是 ev 的原始行号; iloc 取出后 reset_index, 行位置与 W 的列位置一一对应
    cols_used = list(W.columns)
    ev_used = ev.iloc[cols_used].reset_index(drop=True)
    med = ev_used['surprise'].median()
    hi_pos = list(np.where(ev_used['surprise'].values > med)[0])
    lo_pos = list(np.where(ev_used['surprise'].values <= med)[0])
    print('超预期组 %d 个 / 平淡组 %d 个' % (len(hi_pos), len(lo_pos)), flush=True)

    def path(pos):
        m = W.iloc[:, pos].mean(axis=1)
        p = (1 + m).cumprod()
        return m, p / p.loc[-BEFORE] - 1

    md_hi, cp_hi = path(hi_pos)
    md_lo, cp_lo = path(lo_pos)
    md_all = mean_daily

    res['all'] = {'mean_daily': {str(k): round(float(v), 6) for k, v in md_all.items()},
                  'cum_from_day0': {str(k): round(float(v), 6) for k, v in cum_from_start.items()}}
    res['by_surprise'] = {
        'high': {'cum': {str(k): round(float(v), 6) for k, v in cp_hi.items()}},
        'low': {'cum': {str(k): round(float(v), 6) for k, v in cp_lo.items()}},
        'spread_at_end': round(float(cp_hi.loc[AFTER] - cp_lo.loc[AFTER]), 6),
    }

    # ---------- 与 alphalens 对拍 ----------
    try:
        fac = pd.Series(np.where(ev_used['surprise'] > med, 1.0, 0.0),
                        index=pd.MultiIndex.from_arrays(
                            [pd.to_datetime(ev_used['date'].values), ev_used['code'].values],
                            names=['date', 'asset']))
        fac = fac[fac.index.get_level_values('date').isin(dr.index)]
        fake_fwd = pd.DataFrame({'1D': 0.0}, index=fac.index)
        fd = get_clean_factor(fac, fake_fwd, quantiles=2, max_loss=0.9)
        q = perf.average_cumulative_return_by_quantile(
            fd, dr, periods_before=BEFORE, periods_after=AFTER, demeaned=False)
        al_hi = q.loc[(2, 'mean')]
        res['cross_check'] = {
            'note': 'alphalens average_cumulative_return_by_quantile(quantile=2) 相对日 -5/0/+20 的值',
            'alphalens_vals': {str(k): round(float(v), 6) for k, v in
                               [(k, al_hi.loc[k]) for k in (-BEFORE, 0, BEFORE, AFTER) if k in al_hi.index]},
            'mine_cum': {str(k): round(float(v), 6) for k, v in
                         [(k, cp_hi.loc[k]) for k in (-BEFORE, 0, BEFORE, AFTER)]},
            'mine_mean_daily': {str(k): round(float(v), 6) for k, v in
                                [(k, md_hi.loc[k]) for k in (-BEFORE, 0, BEFORE, AFTER)]},
        }
    except Exception as e:
        res['cross_check'] = {'error': '%s: %s' % (type(e).__name__, e)}

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = md_all.index.values
    axes[0].bar(x, md_all.values * 100, color='#4C72B0', alpha=0.85)
    axes[0].axhline(0, color='k', lw=0.8); axes[0].axvline(0, color='r', ls='--', lw=1, label='公告日')
    axes[0].set_title('财报公告前后 · 平均日收益 (%%)')
    axes[0].set_xlabel('相对公告日的交易日'); axes[0].legend()

    axes[1].plot(cp_hi.index, cp_hi.values * 100, lw=1.8, label='业绩超预期 (高于中位)', color='#DD8452')
    axes[1].plot(cp_lo.index, cp_lo.values * 100, lw=1.8, label='业绩平淡 (低于中位)', color='#4C72B0')
    axes[1].axvline(0, color='r', ls='--', lw=1)
    axes[1].axhline(0, color='k', lw=0.8)
    axes[1].set_title('财报公告事件研究 · 累积收益路径（窗口首日归一）')
    axes[1].set_xlabel('相对公告日的交易日'); axes[1].set_ylabel('累积收益 (%)'); axes[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(CHARTS, '31_事件研究_财报公告.png'), dpi=130); plt.close()

    with open(os.path.join(OUT, 'event_earnings.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps(res.get('cross_check', {}), ensure_ascii=False, indent=2), flush=True)
    print('末端价差(高-低): %.2f%%' % (100 * res['by_surprise']['spread_at_end']), flush=True)
    print('完成 ->', OUT, flush=True)


if __name__ == '__main__':
    main()
