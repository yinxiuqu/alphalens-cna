"""数值不变量与已知答案 —— 抓"算错但看起来正常"的那类 bug。

这些不是"跑通就行"的冒烟测试，而是**恒等式**：只要哪次改动把口径弄偏一点点，
它们就会失败（例如持有期错一格、成本重复扣、复利/单利混用）。
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna

D = pd.bdate_range('2023-01-02', periods=40)
A = [f'{i:06d}' for i in range(8)]
IDX = pd.MultiIndex.from_product([D, A], names=['date', 'asset'])


def mk(seed=0, n_asset=8, n_date=40):
    rng = np.random.default_rng(seed)
    d = D[:n_date]
    a = A[:n_asset]
    idx = pd.MultiIndex.from_product([d, a], names=['date', 'asset'])
    close = (np.tile(rng.uniform(5, 50, len(a)), len(d))
             * np.exp(np.cumsum(rng.normal(0, .01, len(idx)))))
    px = pd.DataFrame({'raw_close': close}, index=idx)
    for c, k in (('open', 1.0), ('high', 1.01), ('low', .99)):
        px[f'raw_{c}'] = close * k
    px['prev_close'] = px.groupby(level='asset')['raw_close'].shift(1).fillna(px['raw_close'])
    px['adj_factor'] = 1.0
    for c in ('open', 'high', 'low', 'close'):
        px[f'adj_{c}'] = px[f'raw_{c}']
    px['volume'] = 1e5
    f = pd.DataFrame({'value': rng.normal(size=len(idx)),
                      'available_at': idx.get_level_values('date')}, index=idx)
    u = pd.DataFrame({'in_universe': True}, index=idx)
    return f, px, u, pd.DatetimeIndex(d)


def test_ledger_identity_holds():
    """★ 输入 = 输出 + Σ各类剔除（台账是"每一步都可归因"的凭据）。

    同一行可能被多个判据命中，但 `clean` 只按**第一个**命中的原因记一次，
    所以这个恒等式必须严格成立 —— 不成立就说明有行被算了两次或漏了。
    """
    for mode in ('normal', 'universe', 'nan_factor'):
        f, px, u, cal = mk(1)
        if mode == 'universe':
            u.iloc[:200, 0] = False
        if mode == 'nan_factor':
            f.iloc[:150, 0] = np.nan
        ret = pd.DataFrame({'forward_return_1': np.nan}, index=f.index)
        led = acna.clean(f, ret, universe=u, horizons=[1]).ledger
        assert led.n_input == led.n_output + sum(led.counts.values()), \
            f'{mode}: 台账对不上 {led.n_input} vs {led.n_output + sum(led.counts.values())}'


def test_forward_return_known_answer():
    """★ 已知答案：``forward_return_h`` = adj_close(t+entry+h) / adj_open(t+entry) − 1。

    entry='next_open' → 入场是**次日开盘**；隔夜跳空单独放在 `overnight_gap_h`，
    `total_return_h` 才等于 (1+fwd)(1+gap)−1。三者别混。
    """
    f, px, u, cal = mk(2)
    fr = acna.forward_returns(px, acna.Calendar(cal), [1],
                              model=acna.ReturnModel(entry='next_open', policy='skip')).df
    got = fr['forward_return_1']
    entry = px['adj_open'].groupby(level='asset').shift(-1)      # t+1 开盘
    exit_ = px['adj_close'].groupby(level='asset').shift(-2)     # t+1+h 收盘
    manual = (exit_ / entry - 1)
    diff = (got.dropna() - manual.reindex(got.dropna().index)).abs().max()
    assert float(diff) < 1e-12, f'与手工公式差 {float(diff):.2e}'

    d2 = ((1 + fr['forward_return_1']) * (1 + fr['overnight_gap_1'])
          - 1 - fr['total_return_1']).abs().max()
    assert float(d2) < 1e-12, f'total != (1+fwd)(1+gap)-1，差 {float(d2):.2e}'


def test_constant_price_gives_zero_return():
    f, px, u, cal = mk(3)
    for c in ('open', 'high', 'low', 'close'):
        px[f'adj_{c}'] = 10.0
    fr = acna.forward_returns(px, acna.Calendar(cal), [1, 5]).df
    m = max(float(fr[c].abs().max()) for c in ('forward_return_1', 'forward_return_5'))
    assert m == 0.0, f'常数价格下前向收益应为 0，实际 {m}'


def test_cost_zero_means_net_equals_gross():
    """★ 成本语义：`net = Π(1+r−c)−1`、`gross = Π(1+r)−1` —— 所以
    **只有 c 全为 0 时** net 才恒等于 gross；一般情形 `net ≠ gross − cost`
    （报告里那行不是减法，差在复利）。"""
    f, px, u, cal = mk(4)
    rep = acna.build_report(f, px, acna.Calendar(cal), horizons=(1, 5), quantiles=3,
                            universe=u, name='x', cost_bps=0.0)
    v = rep.verdict
    assert abs(v.net_return - v.gross_return) < 1e-12, \
        f'cost_bps=0 时 net={v.net_return} 应等于 gross={v.gross_return}'


def test_report_is_deterministic():
    """★ 同输入两次 → markdown 逐字节相同（不许有集合序 / 随机导致的漂移）。"""
    f, px, u, cal = mk(5)

    def build():
        return acna.build_report(f, px, acna.Calendar(cal), horizons=(1, 5),
                                 quantiles=3, universe=u, name='x').to_markdown()
    h1 = hashlib.sha256(build().encode()).hexdigest()
    h2 = hashlib.sha256(build().encode()).hexdigest()
    assert h1 == h2, '两次生成的报告不一致'


def test_save_frames_roundtrip():
    """★ 存盘 round-trip：写出去的表读回来列与行数不变；**降级必须被记录**。

    ⚠️ CI 只装 `.[dev]`（**没有 pyarrow**）→ parquet 写不了，库会降级成 CSV。
    所以这里**按实际格式**读回，并且只断言库真正承诺的东西：
      · 每张表都记了格式（``save_report['formats']``）
      · 凡是降级成 CSV 的，都要在 ``save_report['downgraded']`` 里有原因
        —— 0.1.3 的约定：格式悄悄变了调用方必须知道
      · 读回来的行数一致、原列都在
    （此前这里硬读 parquet 且断言"零降级"，本地装了 pyarrow 才过 ——
      CI 无 pyarrow，三个 Python 版本一起挂，就是这条造成的。）
    """
    f, px, u, cal = mk(6)
    rep = acna.build_report(f, px, acna.Calendar(cal), horizons=(1, 5), quantiles=3,
                            universe=u, name='x')
    tmp = tempfile.mkdtemp()
    rep.save(tmp, kind='frames')
    fmts = rep.save_report.get('formats', {})
    downgraded = dict(rep.save_report.get('downgraded') or [])
    assert fmts, 'save_report 必须记录每张表的格式'
    n_checked = 0
    for name, frame in rep.frames().items():
        if frame is None:
            continue
        assert name in fmts, f'{name} 没有记录格式'
        fmt = fmts[name]
        if fmt == 'parquet':
            assert name not in downgraded, f'{name} 没降级却出现在 downgraded 里'
            back = pd.read_parquet(os.path.join(tmp, f'{name}.parquet'))
        elif fmt == 'csv':
            assert name in downgraded, f'{name} 降级成 CSV 却没记录原因'
            back = pd.read_csv(os.path.join(tmp, f'{name}.csv'))
        else:
            raise AssertionError(f'{name} 的格式异常：{fmt!r}')
        assert len(back) == len(frame), f'{name} 行数不一致'
        # ⚠️ CSV **不保 dtype**：整数列名（如持有期 1/5）读回来是字符串 "1"/"5"。
        #    所以按**字符串**比对列名 —— 这不是库的问题，是 CSV 格式的固有性质，
        #    也正是"降级必须被记录"那条约定存在的原因。
        have = {str(c) for c in back.columns}
        missing = [c for c in frame.columns if str(c) not in have]
        assert not missing, f'{name} 丢列：{missing}'
        n_checked += 1
    assert n_checked >= 10, f'只检查了 {n_checked} 张表，太少'


def test_hfq_qfq_returns_are_identical():
    """★ 后复权与前复权**收益完全相同**（常数倍数不影响收益）——
    这正是 `check_adjust_agreement` 能拿两种口径互验的前提。"""
    from alphalens_cna.engine.adjust import compute_adj_factor, DIVIDEND_FIELDS
    d = pd.bdate_range('2023-01-02', periods=30)
    a = ['600000']
    idx = pd.MultiIndex.from_product([d, a], names=['date', 'asset'])
    cl = np.linspace(10, 20, 30)
    px = pd.DataFrame({'raw_close': cl, 'raw_open': cl, 'raw_high': cl * 1.01,
                       'raw_low': cl * .99, 'prev_close': np.r_[cl[0], cl[:-1]],
                       'volume': 1e5}, index=idx)
    xr = pd.DataFrame({'category': [1, 1]},
                      index=pd.MultiIndex.from_arrays([[d[10], d[20]], a * 2],
                                                      names=['date', 'asset']))
    for c in DIVIDEND_FIELDS:
        xr[c] = 0.0
    xr['songgu'] = [10.0, 0.0]
    xr['peigu'] = 0.0
    xr['peigujia'] = 0.0
    hfq = compute_adj_factor(px, xr, method='hfq')
    qfq = compute_adj_factor(px, xr, method='qfq')
    rh = pd.Series(cl * hfq.values).pct_change().dropna()
    rq = pd.Series(cl * qfq.values).pct_change().dropna()
    assert np.allclose(rh, rq, atol=0, rtol=0), '两种复权口径的收益不一致'
