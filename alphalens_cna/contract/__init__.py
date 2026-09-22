"""L0 契约层。

八个契约对象 + 交易日历 + 跨对象校验。

任何数据源转成这些对象才能进本库；不符合契约的输入**直接拒绝，不产出数字**。
规格见 ``docs/输入数据规格.md``。
"""

from .calendar import Calendar
from .errors import ContractError
from .panels import (
    Events,
    Exposures,
    FactorPanel,
    Grouping,
    PricePanel,
    Tradability,
    Universe,
)
from .validate import check_adjust_agreement, validate_inputs

__all__ = [
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
