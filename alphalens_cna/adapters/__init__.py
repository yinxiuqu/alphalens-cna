"""适配层。输入适配见 ``adapters.input``。"""

from .input import *          # noqa: F401,F403
from .input import __all__ as _input_all

__all__ = list(_input_all)
