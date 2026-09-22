"""对拍层测试 —— 防线 4（等价性回归）。

**这是"取代 alphalens"这句话唯一的硬证据**：在退化配置下
（日频 + 收盘价成交 + 关掉 A 股规则），三个核心量的数值必须与
alphalens **逐位相同**。不是"接近"，是 ``max_diff == 0.0``。

alphalens 是**可选**依赖，没装就整体 skip —— 测试套件不该因为
对拍目标缺席而变红。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alphalens_cna as acna  # noqa: E402
from alphalens_cna.compat import (  # noqa: E402
    ParityCheck, _freq_label, _guard, alphalens_available,
)

pytestmark = pytest.mark.skipif(
    not alphalens_available(),
    reason='没装 alphalens-reloaded，对拍测试跳过（它是可选依赖）')


def synth(dates, n_assets=20, seed=7):
    """造一份最小可用的面板：几何随机游走 + 随机因子。"""
    rng = np.random.default_rng(seed)
    T, N = len(dates), n_assets
    assets = [f'{i:06d}' for i in range(N)]
    idx = pd.MultiIndex.from_product([dates, assets], names=['date', 'asset'])
    close = 30.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (T, N)), axis=0))
    c = pd.DataFrame(close.ravel(), index=idx, columns=['c'])['c']
    op = c * (1 + rng.normal(0, 0.004, T * N))
    px = pd.DataFrame({
        'raw_open': op, 'raw_close': c,
        'raw_high': np.maximum(op, c) * 1.01, 'raw_low': np.minimum(op, c) * 0.99,
        'adj_factor': 1.0,
    }, index=idx)
    for col in ('open', 'close', 'high', 'low'):
        px[f'adj_{col}'] = px[f'raw_{col}']
    px['prev_close'] = px['raw_close'].groupby(level='asset').shift(1)
    f = pd.Series(rng.normal(size=T * N), index=idx, name='factor')
    return f, px


DAILY = pd.bdate_range('2023-01-02', periods=60)
MONTH_END = pd.bdate_range('2020-01-31', periods=30, freq='ME')


# --------------------------------------------------------------------------- #
# 主线：三个量逐位相同
# --------------------------------------------------------------------------- #
def test_parity_daily_passes():
    """★ 日频退化场景下对拍必须整体通过。"""
    f, px = synth(DAILY)
    rep = acna.check_parity(f, px, acna.Calendar(DAILY), horizons=(1, 5), quantiles=5)
    assert rep.ok, f'对拍未通过：\n{rep}'
    assert {c.name for c in rep.checks} == {'IC', '分层均值', '换手'}


def test_parity_bit_identical():
    """★ 不是"接近" —— 三个量的最大差必须**恰好** 0.0。"""
    f, px = synth(DAILY, n_assets=30, seed=11)
    rep = acna.check_parity(f, px, acna.Calendar(DAILY), horizons=(1, 5, 21), quantiles=5)
    for c in rep.checks:
        assert c.max_diff == 0.0, f'{c.name} 最大差 {c.max_diff}，应当逐位相同'
        assert c.n_compared > 0, f'{c.name} 一个点都没比上'


def test_parity_sample_count_matches():
    """清洗后样本数必须与 alphalens 一模一样（丢的行都一样）。"""
    f, px = synth(DAILY, n_assets=25, seed=3)
    rep = acna.check_parity(f, px, acna.Calendar(DAILY), horizons=(1, 5), quantiles=5)
    assert rep.n_ours == rep.n_alphalens, (rep.n_ours, rep.n_alphalens)
    # 60 天 × 25 只，h=5 丢掉前 5 天 → 55 × 25
    assert rep.n_ours == 55 * 25


def test_parity_monthly_with_freq_still_passes():
    """月频但索引**带 freq** 时 alphalens 也正常 —— D9 的触发条件是 freq=None。

    真实 A 股月频面板的日期是"每月首个交易日"，freq 为 None，那才是 D9；
    这里用一个规则的月末索引把"月频本身没问题"钉住。
    """
    f, px = synth(MONTH_END, n_assets=20, seed=5)
    rep = acna.check_parity(f, px, acna.Calendar(MONTH_END), horizons=(1, 3), quantiles=5)
    assert rep.ok, f'对拍未通过：\n{rep}'
    assert rep.freq == '月频'


def test_parity_quantile_column_names():
    """报告里必须显示真实频率，不能把月频写成日频（D2 反面）。"""
    f, px = synth(MONTH_END, n_assets=20, seed=5)
    rep = acna.check_parity(f, px, acna.Calendar(MONTH_END), horizons=(1,), quantiles=5)
    assert '月频' in str(rep)
    assert '日频' not in str(rep)


# --------------------------------------------------------------------------- #
# 参数与降级行为
# --------------------------------------------------------------------------- #
def test_parity_checks_subset():
    """只跑指定检查。"""
    f, px = synth(DAILY)
    rep = acna.check_parity(f, px, acna.Calendar(DAILY), horizons=(1, 5),
                            quantiles=5, checks=('ic',))
    assert [c.name for c in rep.checks] == ['IC']


def test_parity_tol_can_be_tightened():
    """容差收紧到 0 也不该翻车 —— 因为我们本来就是 0 差。"""
    f, px = synth(DAILY)
    rep = acna.check_parity(f, px, acna.Calendar(DAILY), horizons=(1, 5),
                            quantiles=5, tol=0.0)
    assert rep.ok, str(rep)


def test_parity_alphalens_missing(monkeypatch):
    """没装 alphalens → 明确报错，且提示它是可选依赖。"""
    monkeypatch.setattr('alphalens_cna.compat.alphalens_available', lambda: False)
    f, px = synth(DAILY)
    with pytest.raises(acna.ContractError, match='alphalens'):
        acna.check_parity(f, px, acna.Calendar(DAILY))


def test_guard_reraises_our_own_errors():
    """本库自己的 ContractError 必须照常炸，不能被"降级"吞掉。"""
    def boom():
        raise acna.ContractError('TestContract', 'test_rule', 'boom')
    with pytest.raises(acna.ContractError):
        _guard('IC', 1e-10, boom)


def test_guard_degrades_on_third_party_error():
    """alphalens 一侧炸了 → 变成"无法对拍"的检查结果，不漏堆栈。"""
    def boom():
        raise ValueError('alphalens went boom')
    c = _guard('IC', 1e-10, boom)
    assert isinstance(c, ParityCheck) and not c.ok
    assert c.n_compared == 0 and 'boom' in c.detail


def test_freq_label():
    """频率标签。"""
    assert _freq_label(acna.Calendar(DAILY)) == '日频'
    assert _freq_label(acna.Calendar(MONTH_END)) == '月频'
    assert _freq_label(acna.Calendar(pd.bdate_range('2024-01-05', periods=20, freq='W-FRI'))) == '周频'


def test_empty_report_is_not_ok():
    """一个检查都没跑 → 不能算通过（不能把"没查"当"没问题"）。"""
    from alphalens_cna.compat import ParityReport
    assert ParityReport().ok is False
