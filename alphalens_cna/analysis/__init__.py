"""L4 分析层 —— 分层 / 双重排序 / IC / 截面回归 / 多空组合。

**本层只算，不下结论。** 显著性判定（Newey-West、多重检验、有效样本量）
在 ``inference/`` 层 —— 见设计原则 1「推断与计算分离」。
"""

from .ic import (
    ic_decay,
    ic_summary,
    information_coefficient,
    quantile_turnover,
    rank_autocorrelation,
    return_cols,
)
from .portfolio import (
    cumulative_returns,
    factor_returns,
    portfolio_summary,
    turnover_summary,
    weighted_returns,
)
from .event import (
    EventWindows,
    align_event_windows,
    event_path_by_group,
    event_summary,
)
from .tail import (
    crash_spread,
    crash_stats,
    cvar,
    expected_shortfall,
    downside_deviation,
    tail_by_quantile,
    tail_ratio,
    value_at_risk,
)
from .regression import (
    FMResult,
    cross_sectional_regression,
    fama_macbeth,
    shanken_inflation,
)
from .quantile import (
    double_sort,
    monotonicity_test,
    quantile_returns,
    quantile_stats,
    quantize,
)

__all__ = [
    'information_coefficient', 'ic_summary', 'ic_decay',
    'rank_autocorrelation', 'quantile_turnover', 'return_cols',
    'quantize', 'quantile_returns', 'quantile_stats',
    'double_sort', 'monotonicity_test',
    'factor_returns', 'cumulative_returns', 'portfolio_summary',
    'turnover_summary', 'weighted_returns',
    'cross_sectional_regression', 'fama_macbeth', 'FMResult',
    'shanken_inflation',
    'value_at_risk', 'cvar', 'expected_shortfall', 'downside_deviation',
    'tail_ratio', 'crash_stats', 'tail_by_quantile', 'crash_spread',
    'align_event_windows', 'event_summary', 'event_path_by_group',
    'EventWindows',
]
