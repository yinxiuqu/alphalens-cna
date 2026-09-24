"""L5 推断层 ★ —— **本书的灵魂，本项目区别于所有同类的地方。**

分层理由（设计原则 1）：**推断与计算分离。**
``analysis/`` 只算 IC / 分层 / 收益；**判断可不可信是这一层的事**。
绝不在计算里偷偷下结论。

包含
----
* :mod:`.multiplicity` —— Bonferroni / Holm / BH / **BHY** + HLZ 阈值
* :mod:`.newey_west` —— 重叠观测修正、**有效样本量**
* :mod:`.estimate` —— 带不确定性标注的估计量（不给单一数字）
* :mod:`.verdict` —— 结论收口
* :mod:`.rank_entropy` —— **RRE** 排名熵稳定性（因子新鲜度诊断）
* :mod:`.robustness` —— **PFS** 扰动鲁棒性（结论怕不怕挪椅子）
"""

from .estimate import Estimate, estimate_from_series, estimates_to_frame
from .verdict import Verdict, assess
from .multiplicity import (
    METHODS,
    adjust,
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    dependency_factor,
    hlz_threshold,
    holm,
    p_from_t,
    t_threshold,
    to_frame,
)
from .deflated import (
    DSRResult,
    dsr,
    expected_max_sharpe,
    min_track_record_length,
    psr,
    sharpe_variance_estimate,
)
from .stability import decay_test, subsample_stability
from .rank_entropy import (
    rank_entropy,
    rank_entropy_series,
    rank_entropy_summary,
    rank_stability,
)
from .robustness import (
    MODES,
    perturb_series,
    pfs,
    robustness_report,
)
from .newey_west import (
    auto_lags,
    effective_n,
    newey_west_summary,
    nw_tstat,
    nw_variance,
    variance_inflation,
)
from .verdict import Verdict, assess

__all__ = [
    'Estimate', 'estimate_from_series', 'estimates_to_frame',
    'adjust', 'bonferroni', 'holm', 'benjamini_hochberg', 'benjamini_yekutieli',
    't_threshold', 'hlz_threshold', 'p_from_t', 'dependency_factor',
    'to_frame', 'METHODS',
    'nw_variance', 'nw_tstat', 'auto_lags', 'effective_n',
    'variance_inflation', 'newey_west_summary',
    'Verdict', 'assess',
    'rank_entropy', 'rank_entropy_series', 'rank_stability',
    'rank_entropy_summary',
    'perturb_series', 'pfs', 'robustness_report', 'MODES',
    'dsr', 'psr', 'DSRResult', 'expected_max_sharpe',
    'min_track_record_length', 'sharpe_variance_estimate',
    'subsample_stability', 'decay_test',
]
