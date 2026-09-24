"""公开 API 承诺的完整性 —— 按**设计文档承诺的清单**断言。

为什么需要这个（两道不同的问题）
--------------------------------
1. **写进 `__all__` 却取不到** —— 会让 `from alphalens_cna import *` 抛异常。
   由 `tests/test_fixes_api.py::test_every_all_name_is_reachable` 守。
2. **文档承诺了、却根本没导出** —— 用"查 `__all__`"的方法**查不出来**。
   本文件就是为这一条存在的：清单来自设计文档的 L2/L3 示例，
   它承诺 `acna.forward_returns(...)` 这么写就能用。

   实测教训：用户审计了 `__all__` 全部 88 项、只发现 1 个问题；
   而 `forward_returns` / `clean` / `ReturnModel` / `compute_tradability` /
   `compute_adj_factor` 这 5 个核心函数**一个都没在 `__all__` 里**，
   那份审计完全看不到。
"""

from __future__ import annotations
import os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphalens_cna as acna  # noqa: E402


#: 设计文档「三档公开 API」里承诺的写法 —— `acna.<名字>` 必须可直接使用
PROMISED = [
    # L0 契约对象
    'Calendar', 'PricePanel', 'FactorPanel', 'Tradability',
    'Universe', 'Grouping', 'Exposures', 'Events', 'ContractError',
    # L2 显式流水线
    'forward_returns', 'clean', 'ReturnModel',
    'compute_tradability', 'compute_adj_factor',
    # L3/L4 预处理与分析
    'winsorize', 'standardize', 'neutralize', 'orthogonalize', 'combine',
    'quantize', 'quantile_returns', 'quantile_stats',
    'information_coefficient', 'ic_summary', 'ic_decay', 'rolling_ic',
    'factor_returns', 'cumulative_returns', 'quantile_turnover',
    'cross_sectional_regression', 'fama_macbeth',
    'align_event_windows', 'event_summary',
    'tail_by_quantile', 'crash_spread', 'cvar', 'value_at_risk',
    # L5 推断
    'assess', 'Verdict', 'nw_tstat', 'effective_n',
    'bonferroni', 'holm', 'benjamini_hochberg', 'benjamini_yekutieli',
    'dsr', 'psr', 'subsample_stability', 'decay_test',
    'rank_stability', 'pfs', 'robustness_report',
    # 体检 / 台账 / 报告 / 对拍
    'health_check', 'HealthReport', 'check_parity',
    'ResearchLedger', 'build_report', 'Report',
]


@pytest.mark.parametrize('name', PROMISED)
def test_promised_api_is_reachable(name):
    """★ 文档承诺的写法必须真的能用。"""
    assert hasattr(acna, name), (
        f'设计文档承诺 `acna.{name}` 可用，实际取不到 —— '
        f'要么补导出，要么改文档，二者不能不一致')


def test_promised_names_are_all_in_dunder_all():
    """承诺的 API 也应当出现在 `__all__` 里（否则 `import *` 拿不到）。"""
    missing = [n for n in PROMISED if n not in acna.__all__]
    assert not missing, f'这些承诺的 API 不在 __all__ 里: {missing}'


def test_engine_package_reexports():
    """引擎层必须能从包路径取到（曾经只有一行 docstring）。"""
    from alphalens_cna.engine import (                  # noqa: F401
        ReturnModel, clean, compute_adj_factor, compute_tradability,
        forward_returns,
    )


def test_pipeline_pieces_compose():
    """★ 用文档承诺的**显式流水线**写法端到端跑通（不用 build_report）。"""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0)
    cal = pd.bdate_range('2023-01-02', periods=120)
    assets = [f'{i:06d}' for i in range(30)]
    idx = pd.MultiIndex.from_product([cal, assets], names=['date', 'asset'])
    n = len(idx)
    px = pd.DataFrame({c: rng.uniform(5, 30, n) for c in
                       ('raw_open', 'raw_close', 'raw_high', 'raw_low')}, index=idx)
    px['raw_high'] = np.maximum(px['raw_open'], px['raw_close']) * 1.01
    px['raw_low'] = np.minimum(px['raw_open'], px['raw_close']) * 0.99
    for k in ('open', 'close', 'high', 'low'):
        px[f'adj_{k}'] = px[f'raw_{k}']
    px['adj_factor'] = 1.0
    px['volume'] = 1e6
    px['prev_close'] = px.groupby(level='asset')['raw_close'].shift(1).fillna(px['raw_open'])
    f = pd.DataFrame({'value': rng.normal(size=n),
                      'available_at': idx.get_level_values('date')}, index=idx)

    calendar = acna.Calendar(cal)
    prices = acna.PricePanel(px)
    factor = acna.FactorPanel(f)
    trad = acna.compute_tradability(prices, calendar=calendar)
    ret = acna.forward_returns(prices, calendar, [5, 21], tradability=trad,
                               model=acna.ReturnModel(entry='next_open'))
    cr = acna.clean(factor, ret, name='demo')
    ic = acna.information_coefficient(cr)
    assert len(ic) > 0 and ic.shape[1] == 2
    assert acna.ResearchLedger(path=False).n_trials() == 0
