"""严格样本外：**先在训练段定规格，再在测试段验一次**。

与之前"分两段都算一遍"的区别
----------------------------
之前是**事后**看两段一致不一致；这里是**事前冻结**：
训练段（2019-2022）把 4 个持有期全试一遍并**记进研究台账**，
挑出最强的那个，冻结"持有期 + 门槛 + 方向"三件事；
测试段（2023-2026）**只算这一个假设**，并按台账记的 n_trials=4
给出 Bonferroni 门槛（|t| = 2.498），而不是 1.96。

这才是"样本外验证"该有的样子 —— 不是把两段都算一遍然后挑好看的。
"""
from __future__ import annotations
import json, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('PIT_MODE', 'delisted+zerofix')
import accept_m0 as A, alphalens_cna as acna
from alphalens_cna.engine.clean import clean
from alphalens_cna.engine.returns import ReturnModel, forward_returns

ABS = -0.50
SPLIT = pd.Timestamp('2023-01-01')
GOAL = 'roe_crash_oos'


def main():
    px, op = A.load_panels()
    codes = [c for c in px.columns if px[c].notna().sum() > 250 and c in op.columns]
    me = pd.Series(px.index, index=px.index).resample('ME').last().dropna()
    rebal = pd.DatetimeIndex(me.values)
    rebal = rebal[(rebal >= A.FACTOR_START) & (rebal <= A.FACTOR_END)]
    prices = A.make_prices(px[codes], op[codes]); cal = acna.Calendar(px.index)
    factor = A.build_factor(rebal, codes)
    r = forward_returns(prices, cal, list(A.HORIZONS),
                        model=ReturnModel(entry='same_open', exit_price='open',
                                          delist_policy='last_price'))
    cr = clean(factor, r, name='roe_pit')
    df = cr.data.join(acna.quantize(cr, n=5)[['q']]).rename(
        columns={'q': 'factor_quantile'})
    dates = pd.Series(df.index.get_level_values('date'), index=df.index)
    tr = df[(dates < SPLIT).values]
    te = df[(dates >= SPLIT).values]

    # ── 台账：训练段试过的每一个假设都要记 ──────────────────────────
    led = acna.ResearchLedger(os.path.join(HERE, 'outputs', 'ledger_oos.jsonl'))
    led.clear(GOAL)
    print('=' * 78)
    print('① 训练段 2019-2022 —— 把 4 个持有期都试一遍（**全部记进台账**）')
    print('=' * 78)
    tr_sp = acna.crash_spread(tr, threshold=ABS)
    print(f'  {"持有期":>6} {"Q1腰斩":>8} {"Q5腰斩":>8} {"差":>9} {"t_NW":>8}')
    for h in A.HORIZONS:
        s = tr_sp.loc[h]
        print(f'  {h:>6} {s["hit_lo"]:>8.2%} {s["hit_hi"]:>8.2%} '
              f'{s["hit_spread"]:>+9.4f} {s["t_hit"]:>+8.2f}')
        led.record(goal=GOAL, factor='roe', horizon=int(h), t=float(s['t_hit']),
                   note='训练段 2019-2022')
    n_trials = led.n_trials(GOAL)
    thr = led.threshold(GOAL, alpha=0.05)
    print(f'\n  台账：n_trials = {n_trials}  →  Bonferroni |t| 门槛 = {thr:.3f}')
    ok, msg = led.check_n_trials(n_trials, GOAL)
    print(f'  自检：{msg}')

    # ── 冻结规格 ───────────────────────────────────────────────────
    h_star = int(tr_sp['t_hit'].abs().idxmax())
    print(f'\n② **冻结规格**：持有期 = {h_star}，门槛 = {ABS:.0%}（腰斩），'
          f'方向 = Q1 腰斩率 > Q5。测试段不再改任何东西。')

    # ── 测试段：只算这一个假设 ──────────────────────────────────────
    print()
    print('=' * 78)
    print('③ 测试段 2023-2026 —— **只算这一个假设**')
    print('=' * 78)
    te_sp = acna.crash_spread(te, threshold=ABS)
    s = te_sp.loc[h_star]
    print(f'  持有期 {h_star}：Q1 腰斩 {s["hit_lo"]:.2%}  vs  Q5 腰斩 '
          f'{s["hit_hi"]:.2%}    差 {s["hit_spread"]:+.4f}')
    print(f'  t_NW = {s["t_hit"]:+.2f}   （n_periods = {int(s["n_periods"])}，'
          f'lags = {int(s["lags"])}）')
    same_dir = s['hit_spread'] < 0
    strong = abs(s['t_hit']) > thr
    print(f'\n  方向一致（Q1 更容易腰斩）: {"✅" if same_dir else "❌"}')
    print(f'  超过台账门槛 |t| > {thr:.3f}: {"✅" if strong else "❌"}')
    verdict = same_dir and strong
    print(f'\n  结论：{"✅ 样本外通过" if verdict else "❌ 样本外不通过"}')
    led.record(goal=GOAL, factor='roe', horizon=h_star, t=float(s['t_hit']),
               note='测试段 2023-2026（冻结规格后只此一次）')
    print(f'\n  台账：{led.n_trials(GOAL)} 个假设（训练 4 + 测试 1，'
          f'其中 h={h_star} 重复 → 去重后 {led.n_trials(GOAL)}）')
    out = {'frozen': {'horizon': h_star, 'threshold': ABS, 'direction': 'Q1>Q5'},
           'n_trials': int(n_trials), 't_threshold': float(thr),
           'train': tr_sp.reset_index().to_dict('records'),
           'test': te_sp.reset_index().to_dict('records'),
           'test_hit_lo': float(s['hit_lo']), 'test_hit_hi': float(s['hit_hi']),
           'test_t': float(s['t_hit']), 'passed': bool(verdict)}
    with open(os.path.join(HERE, 'outputs', 'oos_prereg.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print('已存 outputs/oos_prereg.json')


if __name__ == '__main__':
    main()
