"""L2 引擎层 —— 复权 / 可成交性 / 收益 / 带原因的清洗。

本模块只做**再导出**，实现分在四个子模块里：

* :mod:`.adjust`      —— 复权因子（与 QUANTAXIS 逐位一致）
* :mod:`.tradability` —— A 股可成交性（涨跌停 / ST / 停牌 / 新股 / T+1）
* :mod:`.returns`     —— 收益模型（含**退市收益约定**）
* :mod:`.clean`       —— 带原因的清洗（不变量：输入 = 输出 + 各类剔除之和）

⚠️ 这里曾经只有一行 docstring、没有任何再导出，导致设计文档承诺的
``acna.forward_returns(...)`` 这种写法取不到函数 —— 文档说能用、实际不能用。
现在补齐，并由 ``tests/test_api_surface.py`` 按文档承诺的清单守着。
"""

from .adjust import DIVIDEND_FIELDS, compute_adj_factor
from .clean import CleanResult, DropLedger, clean
from .returns import (
    DELIST_POLICIES,
    ENTRY_MODES,
    POLICIES,
    ReturnModel,
    Returns,
    forward_returns,
)
from .tradability import (
    BOARD_LIMIT,
    EPS,
    ST_LIMIT,
    board_pct,
    compute_tradability,
    is_st_name,
    limit_prices,
    limit_ratio_of,
)

__all__ = [
    'compute_adj_factor', 'DIVIDEND_FIELDS',
    'compute_tradability', 'board_pct', 'is_st_name', 'limit_prices',
    'limit_ratio_of', 'BOARD_LIMIT', 'ST_LIMIT', 'EPS',
    'ReturnModel', 'Returns', 'forward_returns',
    'ENTRY_MODES', 'POLICIES', 'DELIST_POLICIES',
    'clean', 'CleanResult', 'DropLedger',
]
