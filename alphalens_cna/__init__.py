"""alphalens-cna —— 中国 A 股的因子研究框架。

设计原点（三个问题）
--------------------
1. **算得对不对？**   A 股制度正确：涨跌停 / 停牌 / T+1 / 退市 / 复权三价并存
2. **算得可信吗？**   推断层：多重检验 / Newey-West / 有效样本量 / 稳健性
3. **别人凭什么信？** 六道防线：契约校验 / 体检 / 不变量对账 / 等价性回归 /
                      已知答案测试 / 稳健性报告

核心零绘图依赖 —— 只产出 tidy DataFrame，画图交给外部工具。
"""

from .analysis import (
    EventWindows,
    FMResult,
    align_event_windows,
    crash_spread,
    crash_stats,
    cvar,
    downside_deviation,
    event_path_by_group,
    event_summary,
    expected_shortfall,
    cross_sectional_regression,
    cumulative_returns,
    double_sort,
    fama_macbeth,
    factor_returns,
    ic_decay,
    ic_summary,
    information_coefficient,
    monotonicity_test,
    portfolio_summary,
    quantile_returns,
    quantile_stats,
    quantile_turnover,
    quantize,
    rank_autocorrelation,
    shanken_inflation,
    tail_by_quantile,
    tail_ratio,
    turnover_summary,
    value_at_risk,
    weighted_returns,
)
from .adapters.input import (
    AdapterBase,
    DataFrameAdapter,
    InputAdapter,
    ParquetAdapter,
    load_calendar,
    load_exposures,
    load_factor,
    load_grouping,
    load_prices,
    load_tradability,
    load_universe,
)
from .compat import ParityReport, check_parity
from .preprocess import (
    combine,
    neutralize,
    orthogonalize,
    preprocess_log,
    standardize,
    winsorize,
)
from .ledger import Entry, ResearchLedger
from .health import Finding, HealthReport
from .health import check as health_check
from .report import Report, build_report
from .inference import (
    DSRResult,
    Estimate,
    Verdict,
    decay_test,
    dsr,
    min_track_record_length,
    pfs,
    perturb_series,
    psr,
    subsample_stability,
    rank_entropy,
    rank_entropy_summary,
    rank_stability,
    robustness_report,
    adjust,
    assess,
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    dependency_factor,
    auto_lags,
    effective_n,
    estimate_from_series,
    estimates_to_frame,
    hlz_threshold,
    holm,
    newey_west_summary,
    nw_tstat,
    t_threshold,
    to_frame,
)
from .contract import (
    Calendar,
    ContractError,
    Events,
    Exposures,
    FactorPanel,
    Grouping,
    PricePanel,
    Tradability,
    Universe,
    check_adjust_agreement,
    validate_inputs,
)

__version__ = '0.1.0.dev0'

__all__ = [
    '__version__',
    # 载入
    'load_prices', 'load_factor', 'load_calendar', 'load_universe',
    'load_tradability', 'load_exposures', 'load_grouping',
    'DataFrameAdapter', 'ParquetAdapter', 'InputAdapter', 'AdapterBase',
    # 分析
    'information_coefficient', 'ic_summary', 'ic_decay',
    'rank_autocorrelation', 'quantile_turnover',
    'cross_sectional_regression', 'fama_macbeth', 'FMResult',
    'shanken_inflation',
    'quantize', 'quantile_returns', 'quantile_stats',
    'double_sort', 'monotonicity_test',
    'factor_returns', 'cumulative_returns', 'portfolio_summary',
    'turnover_summary', 'weighted_returns',
    # 对拍（防线 4）
    'check_parity', 'ParityReport',
    'health_check', 'HealthReport', 'Finding',
    'ResearchLedger', 'Entry',
    'winsorize', 'standardize', 'neutralize', 'orthogonalize', 'combine',
    'preprocess_log',
    'rank_entropy', 'rank_stability', 'rank_entropy_summary',
    'pfs', 'perturb_series', 'robustness_report',
    'dsr', 'psr', 'DSRResult', 'min_track_record_length',
    'subsample_stability', 'decay_test', 'rolling_ic',
    # 报告
    'build_report', 'Report',
    # 推断 ★
    'assess', 'Verdict', 'Estimate',
    'adjust', 'bonferroni', 'holm', 'benjamini_hochberg', 'benjamini_yekutieli',
    't_threshold', 'hlz_threshold', 'dependency_factor',
    'nw_tstat', 'effective_n', 'auto_lags', 'newey_west_summary',
    'estimate_from_series', 'estimates_to_frame', 'to_frame',
    # 契约
    'Calendar',
    'ContractError',
    'FactorPanel',
    'PricePanel',
    'Tradability',
    'Universe',
    'Grouping',
    'Exposures',
    'Events',
    'validate_inputs',
    'check_adjust_agreement',
]
