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
              '## 三、样本账', '## 四、IC',
              '## 五、Newey-West', '## 六、分层', '## 七、尾部风险',
              '## 八、换手与成本', '## 九、剔除明细',
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
