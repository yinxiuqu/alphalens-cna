"""契约层错误类型。

设计原则：**错误信息必须能定位到具体是哪一行、哪个字段**。
只说"校验失败"等于没说 —— 使用者要知道去哪儿改。
"""


class ContractError(ValueError):
    """契约校验失败。违反契约时抛出，**不产出任何数字**。

    Attributes
    ----------
    contract : str
        哪个契约对象（如 ``'PricePanel'``）。
    rule : str
        违反了哪条规则（如 ``'raw_positive'``）。
    detail : str
        人话说明，含具体行/字段。
    """

    def __init__(self, contract, rule, detail):
        self.contract = contract
        self.rule = rule
        self.detail = detail
        super().__init__(f'[{contract}] 违反契约 `{rule}`：{detail}')

    def __str__(self):
        return (f'[{self.contract}] 违反契约 `{self.rule}`\n  {self.detail}')


def fail(contract, rule, detail):
    """抛契约错误。"""
    raise ContractError(contract, rule, detail)


def describe_rows(idx, n=3):
    """把若干行索引渲染成人话，用于错误信息。"""
    try:
        total = len(idx)
    except TypeError:
        return repr(idx)
    shown = list(idx[:n])
    more = f' …（共 {total} 行）' if total > n else ''
    return '; '.join(str(x) for x in shown) + more
