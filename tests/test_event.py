"""事件研究测试 —— 防线 5（已知答案）+ 事件聚集的 t 虚高。

**关键测试是 `test_clustered_events_inflate_naive_t`**：
把所有事件挤在少数几天里，横截面 t 会虚高好几倍，
而日历时间 t 才是可信的 —— 这正是本模块要解决的问题。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna  # noqa: E402

DATES = pd.bdate_range('2024-01-01', periods=12)


def mk_prices(series_by_asset):
    """``{asset: 价格数组}`` → 契约形状的价格面板。"""
    frames = []
    for a, v in series_by_asset.items():
        frames.append(pd.DataFrame({
            'adj_close': np.asarray(v, dtype=float),
            'adj_open': np.asarray(v, dtype=float),
            'adj_high': np.asarray(v, dtype=float),
            'adj_low': np.asarray(v, dtype=float),
            'raw_close': np.asarray(v, dtype=float),
            'raw_open': np.asarray(v, dtype=float),
            'raw_high': np.asarray(v, dtype=float),
            'raw_low': np.asarray(v, dtype=float),
            'prev_close': np.asarray(v, dtype=float),
            'adj_factor': 1.0,
        }, index=pd.MultiIndex.from_arrays([DATES, [a] * len(DATES)],
                                           names=['date', 'asset'])))
    return pd.concat(frames)


def mk_events(pairs, **cols):
    """``[(date, asset), ...]`` → Events 形状。"""
    idx = pd.MultiIndex.from_tuples(pairs, names=['date', 'asset'])
    d = {'event_type': ['ann'] * len(pairs)}
    d.update({k: v for k, v in cols.items()})
    return pd.DataFrame(d, index=idx)


# --------------------------------------------------------------------------- #
# 对齐（已知答案）
# --------------------------------------------------------------------------- #
def test_alignment_known_answer():
    """★ 价格 100×4 → 110×3 → 121×3，事件在第 5 天。

    相对日收益 = [0, 0, **+10%**, 0, 0, **+10%**, 0]
    ``base_day=0``（事件日收盘为基准）→ ``car`` = [−10%, −10%, 0, 0, 0, +11%, +11%]
    """
    px = mk_prices({'X': [100, 100, 100, 100, 110, 110, 110, 121, 121, 121, 121, 121]})
    ev = mk_events([(DATES[4], 'X')])
    w = acna.align_event_windows(px, ev, acna.Calendar(DATES),
                                 window=(-2, 4), base_day=0)
    d = w.df.droplevel('event_id')
    assert list(d.index) == [-2, -1, 0, 1, 2, 3, 4]
    assert np.allclose(d['ret'].values, [0, 0, 0.10, 0, 0, 0.10, 0], atol=1e-12)
    assert np.allclose(d['car'].values,
                       [-0.10, -0.10, 0.0, 0.0, 0.0, 0.11, 0.11], atol=1e-12)
    assert abs(d.loc[0, 'cum_ret'] - 0.10) < 1e-12
    assert len(w) == 1


def test_base_day_minus_one_includes_event_day():
    """``base_day=-1`` → 公告当天的涨跌也计入 car。"""
    px = mk_prices({'X': [100, 100, 100, 100, 110, 110, 110, 110, 110, 110, 110, 110]})
    ev = mk_events([(DATES[4], 'X')])
    w = acna.align_event_windows(px, ev, acna.Calendar(DATES),
                                 window=(-2, 3), base_day=-1)
    d = w.df.droplevel('event_id')
    assert abs(d.loc[-1, 'car']) < 1e-12          # 基准日自己为 0
    assert abs(d.loc[0, 'car'] - 0.10) < 1e-12    # 公告当天 +10% 计进来


def test_benchmark_subtraction():
    """基准与个股同涨同跌 → 异常收益为 0。"""
    px = mk_prices({'X': [100, 100, 100, 100, 110, 110, 110, 110, 110, 110, 110, 110]})
    ev = mk_events([(DATES[4], 'X')])
    w = acna.align_event_windows(px, ev, acna.Calendar(DATES), window=(-2, 3),
                                 base_day=-1)
    w2 = acna.align_event_windows(px, ev, acna.Calendar(DATES), window=(-2, 3),
                                  base_day=-1, benchmark='X')
    assert w.df['abn_ret'].isna().all(), '没给基准就不该凭空造异常收益'
    assert np.abs(w2.df['abn_ret']).max() < 1e-12
    assert np.abs(w2.df['car']).max() < 1e-12


def test_ledger_records_drops():
    """带原因的账：日历外的日期、窗口内停牌，都要记上。"""
    px = mk_prices({'X': [100] * 12, 'Y': [100, 100, 100, 100, 100, 100,
                                           np.nan, 100, 100, 100, 100, 100]})
    ev = mk_events([(DATES[4], 'X'),
                    (pd.Timestamp('2024-01-06'), 'X'),    # 周六，不在日历
                    (DATES[4], 'Y')])                     # Y 窗口内有缺失
    w = acna.align_event_windows(px, ev, acna.Calendar(DATES), window=(-2, 3))
    assert w.ledger['n_events'] == 3 and w.ledger['kept'] == 1
    assert w.ledger['not_on_calendar'] == 1
    assert w.ledger['missing_price'] == 1, '窗口内停牌必须被剔除，不能被前值填充糊过去'


def test_all_dropped_raises_with_ledger():
    """★ 一个事件都没对齐上 → **报错**（不是悄悄返回空表），且带剔除账。"""
    px = mk_prices({'X': [100] * 12})
    ev = mk_events([(DATES[0], 'X')])
    with pytest.raises(acna.ContractError, match='not_on_calendar'):
        acna.align_event_windows(px, ev, acna.Calendar(DATES), window=(-2, 3))


# --------------------------------------------------------------------------- #
# ★ 事件聚集 → 朴素 t 虚高
# --------------------------------------------------------------------------- #
def test_clustered_events_inflate_naive_t():
    """★★ 事件全挤在 3 天里：横截面 t 虚高数倍，日历时间 t 才可信。

    构造：120 只票的公告后一日收益 = 当日共同效应 + 个股噪声，
    当日共同效应 = [+10%, 0%, +5%]（**只有 3 个独立观测**）。
    横截面 t 把 120 个事件当独立样本 → 虚高；
    日历时间 t 只看 3 天 → 才是真实的不确定性。
    """
    rng = np.random.default_rng(3)
    event_days = [DATES[2], DATES[5], DATES[8]]
    effects = [0.10, 0.0, 0.05]
    series, pairs = {}, []
    for i in range(120):
        k = i % 3
        ed = event_days[k]
        pos = list(DATES).index(ed)
        r = effects[k] + rng.normal(0, 0.01)
        v = np.ones(12)
        v[pos + 1:] = 1.0 + r                  # 公告后一日涨 r，之后维持
        a = f'{i:06d}'
        series[a] = v
        pairs.append((ed, a))
    px = mk_prices(series)
    ev = mk_events(pairs)
    w = acna.align_event_windows(px, ev, acna.Calendar(DATES), window=(-1, 3),
                                base_day=0)
    s = w.summary()
    row = s.loc[1]
    assert row['n_events'] == 120
    assert row['n_days'] == 3, row['n_days']
    assert abs(row['mean'] - 0.05) < 0.01
    # 横截面 t 虚高：至少是日历时间 t 的 5 倍
    assert abs(row['t_naive']) > 5 * abs(row['t_ct']), (row['t_naive'], row['t_ct'])
    assert abs(row['t_naive']) > 8, row['t_naive']
    assert abs(row['t_ct']) < 5, row['t_ct']


def test_summary_columns_and_index():
    px = mk_prices({'X': [100, 100, 100, 100, 110, 110, 110, 110, 110, 110, 110, 110]})
    ev = mk_events([(DATES[4], 'X')])
    s = acna.align_event_windows(px, ev, acna.Calendar(DATES),
                                 window=(-2, 3)).summary()
    for c in ('n_events', 'mean', 'median', 'hit_rate', 't_naive', 't_ct'):
        assert c in s.columns
    assert s.index.name == 'rel_day'


# --------------------------------------------------------------------------- #
# 分组路径（PEAD 形态）
# --------------------------------------------------------------------------- #
def test_event_path_by_group():
    """按事件属性分位分组，给出各组累计路径与高低差。"""
    series, pairs, sur = {}, [], []
    for i in range(60):
        a = f'{i:06d}'
        r = (i / 60.0) * 0.10                  # 惊喜越大，公告后涨得越多
        v = np.ones(12)
        v[6:] = 1.0 + r
        series[a] = v
        pairs.append((DATES[4], a))
        sur.append(i / 60.0)
    px = mk_prices(series)
    ev = mk_events(pairs, surprise=sur)
    w = acna.align_event_windows(px, ev, acna.Calendar(DATES), window=(-2, 3),
                                 base_day=0)
    p = acna.event_path_by_group(w, 'surprise', quantiles=3)
    assert 'G1' in p.columns and 'G3' in p.columns
    assert 'spread_G3-G1' in p.columns
    # 惊喜越大、公告后收益越高 → 高低差为正
    assert p.loc[2, 'spread_G3-G1'] > 0.02
    assert p['n_G1'].iloc[0] == 20


def test_bad_inputs():
    px = mk_prices({'X': [100] * 12})
    ev = mk_events([(DATES[4], 'X')])
    cal = acna.Calendar(DATES)
    with pytest.raises(acna.ContractError):
        acna.align_event_windows(px, ev, cal, window=(3, -2))
    with pytest.raises(acna.ContractError):
        acna.align_event_windows(px, ev, cal, window=(-2, 3), base_day=9)
    with pytest.raises(acna.ContractError):
        acna.align_event_windows(px.drop(columns=['adj_close']), ev, cal)
    with pytest.raises(acna.ContractError):
        acna.align_event_windows(px, ev, cal, benchmark='NOT_THERE')
    w = acna.align_event_windows(px, ev, cal, window=(-2, 3))
    with pytest.raises(acna.ContractError):
        acna.event_path_by_group(w, 'no_such_col')
