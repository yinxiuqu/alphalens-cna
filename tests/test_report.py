"""报告层测试 —— 零绘图依赖 + 内容完整性。"""

from __future__ import annotations

import os
import sys
import subprocess

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna

rng = np.random.default_rng(0)
DATES = pd.bdate_range('2024-01-01', periods=120)
ASSETS = [f'{i:06d}' for i in range(40)]
IDX = pd.MultiIndex.from_product([DATES, ASSETS], names=['date', 'asset'])


def make():
    o = np.tile(rng.uniform(5, 50, 40), len(DATES)) * (1 + rng.normal(0, .02, len(IDX)))
    px = pd.DataFrame({'raw_open': o, 'raw_close': o * 1.001, 'raw_high': o * 1.02,
                       'raw_low': o * .98, 'adj_factor': 1.0,
                       'prev_close': np.r_[np.full(40, np.nan), o[:-40]]}, index=IDX)
    for c in ('open', 'close', 'high', 'low'):
        px[f'adj_{c}'] = px[f'raw_{c}'] * px['adj_factor']
    f = pd.DataFrame({'value': rng.normal(size=len(IDX)),
                      'available_at': IDX.get_level_values('date')}, index=IDX)
    return f, px


def build(**kw):
    f, px = make()
    kw.setdefault('universe', pd.DataFrame({'in_universe': True}, index=IDX))
    return acna.build_report(f, px, acna.Calendar(DATES), horizons=(1, 5),
                             quantiles=5, name='demo', **kw)


def test_build_report_end_to_end():
    rep = build(n_trials=12)
    assert isinstance(rep, acna.Report)
    assert rep.ic is not None and len(rep.ic) > 0
    assert rep.verdict is not None and rep.verdict.n_trials == 12
    assert rep.quantile_stats is not None


def test_frames_are_tidy():
    rep = build(n_trials=12)
    fr = rep.frames()
    for k, v in fr.items():
        assert isinstance(v, pd.DataFrame), k
    assert {'ic', 'ic_summary', 'newey_west', 'quantile_stats',
            'ledger', 'verdict', 'long_short', 'health',
            'tail', 'crash'} <= set(fr)


def test_markdown_contains_key_sections():
    md = build(n_trials=12).to_markdown()
    for s in ('# demo 因子分析报告', '## 一、结论', '## 二、数据体检',
              '## 三、样本账', '## 四、IC', '## 五、Newey-West',
              '## 六、分层', '## 七、因子衰减与稳定性', '## 八、尾部风险',
              '## 九、换手与成本', '## 十、剔除明细',
              '怎么读这份报告'):
        assert s in md, s
    assert 'n_trials' in md and 'p_adj' in md
    assert '1.0000 |' not in md, '持有期不应渲染成 1.0000'


def test_markdown_says_uncorrected_when_no_n_trials():
    md = build().to_markdown()
    assert '未做多重假设检验校正' in md


def test_equity_curves_chart_ready():
    c = build(n_trials=12).equity_curves()
    assert isinstance(c, pd.DataFrame) and len(c) > 0
    assert (c.iloc[0] > 0).all()


def test_save_markdown(tmp_path):
    p = tmp_path / 'r.md'
    build(n_trials=12).save(str(p))
    assert p.exists() and '因子分析报告' in p.read_text(encoding='utf-8')


def test_save_frames(tmp_path):
    d = tmp_path / 'frames'
    build(n_trials=12).save(str(d), kind='frames')
    files = os.listdir(d)
    assert any('ic' in f for f in files) and any('verdict' in f for f in files)


def test_zero_plotting_dependency():
    """★ D3 回归：import 本库**不得**拉起任何绘图库。

    alphalens 的 `import alphalens` 会连带 seaborn/matplotlib。
    """
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import alphalens_cna as acna;"
        "acna.build_report;"          # 触碰一下报告层
        "bad=[m for m in ('matplotlib','seaborn','plotly','bokeh') if m in sys.modules];"
        "print('LEAKED:'+','.join(bad) if bad else 'CLEAN')"
        % os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert 'CLEAN' in out.stdout, out.stdout + out.stderr


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))


def test_save_frames_reports_actual_formats(tmp_path):
    """★ 第 10 项回归：存盘不许静默改格式。

    优先 parquet，没引擎时降级 CSV —— 但**实际格式与原因必须可查**
    （此前是静默 `except: to_csv`）。
    """
    rep = build(n_trials=3)
    out = str(tmp_path / 'frames')
    rep.save(out, kind='frames')
    sr = rep.save_report
    assert sr['formats'], '必须记录实际格式'
    assert set(sr['formats'].values()) <= {'parquet', 'csv'}
    # 降级了就必须有原因，没降级就该是空的
    assert bool(sr['downgraded']) == any(v == 'csv' for v in sr['formats'].values())
    # 文件真的存在，且扩展名与实际格式一致
    import os
    for nm, fmt in sr['formats'].items():
        assert os.path.exists(os.path.join(out, f'{nm}.{fmt}'))


def test_save_markdown_creates_parent_dir(tmp_path):
    """存 markdown 时父目录不存在也要建（此前会抛 FileNotFoundError）。"""
    rep = build(n_trials=3)
    p = str(tmp_path / 'deep' / 'nest' / 'report.md')
    rep.save(p)
    assert rep.save_report['formats']
    import os
    assert os.path.exists(p)


def test_save_bad_kind_rejected():
    import pytest
    rep = build(n_trials=3)
    with pytest.raises(acna.ContractError):
        rep.save('/tmp/x', kind='nope')


def test_hold_period_column_name_is_unified():
    """★ 第 11 项：全文「持有期」列名必须统一成 `h`。

    此前同一份报告里有三种叫法：`horizon`（IC/分层）、`index`（NW，索引未命名）、
    `h`（尾部/稳定性）—— 读者每换一节都要重新对表头。
    """
    import re
    md = build(n_trials=3).to_markdown()
    sec = None
    seen = {}
    for line in md.split('\n'):
        if line.startswith('## '):
            sec = line[3:].strip()
            continue
        if line.startswith('| ') and sec and '---' not in line:
            first = [c.strip() for c in line.strip('|').split('|')][0]
            if first in ('index', 'horizon', 'h'):
                seen.setdefault(first, set()).add(sec[:14])
    assert 'index' not in seen, f'仍有 pandas 默认列名 index：{seen.get("index")}'
    assert 'horizon' not in seen, f'仍有 horizon：{seen.get("horizon")}'
    assert seen.get('h'), '应当用 h 作为持有期列名'
    # 图例里要解释 h
    assert '| `h` |' in md and '持有期' in md


# ── 零剔除：报告必须照样出得来 ────────────────────────────────────
def test_zero_drop_ledger_keeps_columns():
    """★ 零剔除时 to_frame() 也必须带列名（真 bug 的根因）。

    修前 ``pd.DataFrame(rows)`` 在 counts 为空时给出 ``(0, 0)`` —— 连列名都没有，
    报告层接着 ``.set_index('reason')`` 就
    ``KeyError: "None of ['reason'] are in the columns"``。
    """
    from alphalens_cna.engine.clean import DropLedger
    led = DropLedger(n_input=48, n_output=48, counts={}, examples={})
    df = led.to_frame()
    assert list(df.columns) == ['reason', 'count', 'pct', 'meaning', 'sample'], \
        list(df.columns)
    assert len(df) == 1, '零剔除也要有「—— 保留 ——」那一行（0 剔除本身就是结论）'
    assert df.iloc[0]['reason'] == '—— 保留 ——'
    assert df.iloc[0]['count'] == 48
    assert df.set_index('reason').shape[0] == 1        # 崩点：修前这里 KeyError

    # 真正空台账（输入 0 行）：允许 0 行，但列名必须在，下游不许崩
    empty = DropLedger().to_frame()
    assert list(empty.columns) == ['reason', 'count', 'pct', 'meaning', 'sample']
    assert len(empty) == 0
    empty.set_index('reason')


def test_zero_drop_report_still_renders(tmp_path):
    """★ 一条都没剔时报告要出得来（修前：整份报告渲染直接崩）。

    之前一直没暴露，是因为真实研究几乎总会剔掉点什么（区间末尾的前向收益缺失）；
    稠密面板 + 短持有期才撞得出来。
    """
    import dataclasses
    rep = build()
    dates = pd.DatetimeIndex(IDX.get_level_values('date').unique())[:6]
    sub = IDX[IDX.get_level_values('date').isin(dates)]
    fac = pd.DataFrame({'factor': rng.normal(size=len(sub))}, index=sub)
    ret = pd.DataFrame({'forward_return_1': rng.normal(0, .02, len(sub))}, index=sub)
    zero = acna.clean(fac, ret, horizons=[1])
    assert zero.ledger.counts == {}, '用例前提：必须真的一条都没剔'
    rep2 = dataclasses.replace(rep, clean=zero)

    md = rep2.to_markdown()
    assert '## 十、剔除明细' in md
    assert '零剔除' in md, '零剔除要写明，不能留一片空白'

    # 两条存盘路也都得走通
    assert rep2.save(str(tmp_path / 'r.md'), kind='markdown')
    rep2.save(str(tmp_path / 'frames'), kind='frames')
