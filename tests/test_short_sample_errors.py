"""极短样本的报错必须**指向真正的原因**（第 11 项）。

反馈原文：2~3 期时抛的是 `multiplicity/nan_p: p 值里有 NaN/Inf`，
提示让人"剔除算不出 p 的检验" —— 而真正该做的是缩短持有期。
另一处：`ic_summary(pd.DataFrame())` 抛原始 `KeyError: 'horizon'`。
"""

from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphalens_cna as acna  # noqa: E402


def test_ic_summary_empty_frame_is_friendly():
    """★ 空表不该抛 KeyError。"""
    with pytest.raises(acna.ContractError) as e:
        acna.ic_summary(pd.DataFrame())
    assert 'empty_ic' in str(e.value) and 'horizons' in str(e.value)


def test_assess_rejects_all_nan_ic():
    """★ IC 全 NaN → 在进入多重检验**之前**拦住，并指向根因。"""
    ic = pd.DataFrame({'21': [np.nan, np.nan], '63': [np.nan, np.nan]},
                      index=pd.bdate_range('2024-01-01', periods=2))
    with pytest.raises(acna.ContractError) as e:
        acna.assess(ic=ic, n_trials=3)
    msg = str(e.value)
    assert 'all_nan_ic' in msg
    assert 'horizons' in msg and '台账' in msg


def test_assess_rejects_empty_ic():
    with pytest.raises(acna.ContractError) as e:
        acna.assess(ic=pd.DataFrame(), n_trials=3)
    assert 'empty_ic' in str(e.value)


def test_error_points_at_horizons_not_multiplicity():
    """★ 反馈的核心：报错里必须出现"持有期/horizons"，而不是只谈 p 值。"""
    ic = pd.DataFrame({'21': [np.nan] * 3}, index=pd.bdate_range('2024-01-01', periods=3))
    with pytest.raises(acna.ContractError) as e:
        acna.assess(ic=ic, n_trials=5)
    msg = str(e.value)
    assert '持有期' in msg or 'horizons' in msg
    assert 'nan_p' not in msg, '不该再漏到多重检验那一层'


def _ultra_short(daily_prices=True, periods=3, horizons=(21, 63)):
    rng = np.random.default_rng(0)
    cal = pd.bdate_range('2023-01-02', periods=200)      # 日频日历，跨度足够
    me = pd.Series(cal, index=cal).resample('ME').last().dropna()
    rebal = pd.DatetimeIndex(me.values)[:periods]
    assets = [f'{i:06d}' for i in range(20)]
    if daily_prices:
        idx = pd.MultiIndex.from_product([cal, assets], names=['date', 'asset'])
    else:
        idx = pd.MultiIndex.from_product([rebal, assets], names=['date', 'asset'])
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
    fidx = pd.MultiIndex.from_product([rebal, assets], names=['date', 'asset'])
    f = pd.DataFrame({'value': rng.normal(size=len(fidx)),
                      'available_at': fidx.get_level_values('date')}, index=fidx)
    return px, f, acna.Calendar(cal)


@pytest.mark.parametrize('periods', [1, 2, 3])
@pytest.mark.parametrize('horizons', [(21, 63), (1,)])
def test_ultra_short_either_works_or_explains(periods, horizons):
    """★ 极短样本的**不变量**：要么跑通，要么报错必须指向根因。

    绝不允许出现：
      · `KeyError: None of ['horizon'] are in the columns`（pandas 原始异常）
      · `multiplicity/nan_p: p 值里有 NaN/Inf`（把用户引向错误的方向）

    实测：日频价格 + 3 期是能算通的（IC 有 3 个有效值）——
    所以这里不预设"必须报错"，只钉住"报错必须有用"。
    """
    px, f, cal = _ultra_short(periods=periods, horizons=horizons)
    try:
        rep = acna.build_report(f, px, cal, horizons=horizons, quantiles=3, name='x')
        assert len(rep.ic) > 0
    except acna.ContractError as e:
        msg = str(e.value)
        assert 'KeyError' not in msg, f'漏出 pandas 原始异常: {msg[:140]}'
        assert 'NaN/Inf' not in msg, f'把用户引向多重检验那一层: {msg[:140]}'
        assert any(k in msg for k in ('horizons', '持有期', '台账', '样本')), msg[:160]
    except Exception as e:                                       # noqa: BLE001
        pytest.fail(f'抛了非 ContractError: {type(e).__name__}: {e}')


def test_nan_p_message_points_upstream():
    """万一真到达多重检验那一层，提示也要指向"上游样本不足"。"""
    with pytest.raises(acna.ContractError) as e:
        acna.inference.multiplicity.adjust(
            p=[0.01, float('nan')], method='bhy')
    msg = str(e.value)
    assert '样本' in msg or '持有期' in msg, msg[:160]


def test_nan_p_error_codes_distinguish_cases():
    """★ 错误码必须与文案一致（测试者反馈：文案换了、码没换）。

    「全部 NaN」与「部分 NaN」该做的事不同，程序化调用者靠 `err.rule` 分支，
    不能共用一个 `nan_p`。
    """
    from alphalens_cna.inference import multiplicity as mult
    # 全部 NaN → no_valid_p
    with pytest.raises(acna.ContractError) as e:
        mult.adjust(p=[np.nan, np.nan], method='bhy')
    assert e.value.rule == 'no_valid_p', e.value.rule
    assert '没有任何可校正的检验' in str(e.value)
    # 部分 NaN → partial_nan_p
    with pytest.raises(acna.ContractError) as e:
        mult.adjust(p=[0.01, np.nan], method='bhy')
    assert e.value.rule == 'partial_nan_p', e.value.rule
    assert '1/2' in str(e.value) and 'n_trials' in str(e.value)
    # 都不该再出现旧的 nan_p
    for bad in ([np.nan], [0.01, np.inf]):
        with pytest.raises(acna.ContractError) as e:
            mult.adjust(p=bad, method='bhy')
        assert e.value.rule != 'nan_p', '旧的 nan_p 码不该再出现'


def test_error_codes_are_programmatically_distinguishable():
    """两种情况的 contract/rule 组合必须可区分（供 except 分支使用）。"""
    from alphalens_cna.inference import multiplicity as mult
    codes = []
    for p in ([np.nan, np.nan], [0.01, np.nan]):
        try:
            mult.adjust(p=p, method='bhy')
        except acna.ContractError as e:
            codes.append((e.contract, e.rule))
    assert len(set(codes)) == 2, codes
