"""研究台账测试 —— 它守的是"n_trials 不许凭记忆填"这条底线。"""

from __future__ import annotations
import os, sys
import numpy as np, pandas as pd, pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import alphalens_cna as acna  # noqa: E402


def test_record_and_count(tmp_path):
    """★ 校正基数按 (goal, factor, horizon) 去重，不是数跑了多少次。"""
    led = acna.ResearchLedger(str(tmp_path / 'l.jsonl'))
    for h in (21, 63, 126, 252):
        led.record(goal='roe', factor='roe', horizon=h, t=-1.4)
    assert led.n_trials('roe') == 4
    # 同一假设重复记录 → 不算新假设
    led.record(goal='roe', factor='roe', horizon=21, t=-1.9, note='重跑')
    assert led.n_trials('roe') == 4
    assert len(led) == 5
    # 换因子 = 新假设
    led.record(goal='roe', factor='mom20', horizon=21, t=2.0)
    assert led.n_trials('roe') == 5
    # 换 goal = 另一个家族
    led.record(goal='other', factor='roe', horizon=21, t=1.0)
    assert led.n_trials('roe') == 5 and led.n_trials('other') == 1
    assert led.n_trials(family='*') == 6


def test_persistence_roundtrip(tmp_path):
    p = str(tmp_path / 'l.jsonl')
    a = acna.ResearchLedger(p)
    a.record(goal='g', factor='f', horizon=21, t=-2.0, note='第一次')
    b = acna.ResearchLedger(p)
    assert b.n_trials('g') == 1
    assert b.entries('g')[0].note == '第一次'
    assert '第一次' in b.frame('g')['note'].iloc[0]


def test_memory_only_ledger():
    led = acna.ResearchLedger(path=False)
    led.record(goal='g', factor='f', horizon=1)
    assert led.path is False and led.n_trials('g') == 1
    assert not os.path.exists('.alphalens_ledger.jsonl')


def test_threshold_matches_bonferroni(tmp_path):
    led = acna.ResearchLedger(str(tmp_path / 'l.jsonl'))
    for i, h in enumerate((21, 63, 126, 252)):
        led.record(goal='g', factor=f'f{i}', horizon=h)
    assert abs(led.threshold('g') - acna.inference.t_threshold(4)) < 1e-12
    assert abs(led.threshold('g') - 2.498) < 0.001


def test_empty_ledger_threshold_is_nan(tmp_path):
    """★ 台账为空 → 门槛是 NaN，**不许**假装 1.96。"""
    led = acna.ResearchLedger(str(tmp_path / 'l.jsonl'))
    assert np.isnan(led.threshold('nothing'))
    assert '为空' in led.summary()
    assert len(led.frame()) == 0


def test_check_n_trials_catches_underreport(tmp_path):
    """★★ 报了比台账少的数 = 低报校正基数 → 必须拦。"""
    led = acna.ResearchLedger(str(tmp_path / 'l.jsonl'))
    for h in (21, 63, 126, 252, 504):
        led.record(goal='g', factor='roe', horizon=h)
    ok, msg = led.check_n_trials(3, 'g')
    assert not ok and '少于' in msg
    ok, msg = led.check_n_trials(5, 'g')
    assert ok and '一致' in msg
    ok, msg = led.check_n_trials(20, 'g')
    assert ok and '保守' in msg
    ok, msg = led.check_n_trials(None, 'g')
    assert not ok and '没报' in msg


def test_clear(tmp_path):
    p = str(tmp_path / 'l.jsonl')
    led = acna.ResearchLedger(p)
    led.record(goal='a', factor='f'); led.record(goal='b', factor='f')
    led.clear('a')
    assert led.n_trials('a') == 0 and led.n_trials('b') == 1
    led.clear()
    assert len(led) == 0 and not os.path.exists(p)


def test_bad_row_does_not_break_ledger(tmp_path):
    """坏行跳过，整本台账不许因为一行坏掉。"""
    p = tmp_path / 'l.jsonl'
    led = acna.ResearchLedger(str(p))
    led.record(goal='g', factor='f', horizon=1)
    with open(p, 'a') as f:
        f.write('{ this is not json\n')
    led.record(goal='g', factor='f2', horizon=2)
    assert led.n_trials('g') == 2


def test_requires_names(tmp_path):
    led = acna.ResearchLedger(str(tmp_path / 'l.jsonl'))
    with pytest.raises(ValueError):
        led.record(goal='', factor='f')
    with pytest.raises(ValueError):
        led.record(goal='g', factor='')


def test_fingerprint_stable(tmp_path):
    p = str(tmp_path / 'l.jsonl')
    a = acna.ResearchLedger(p)
    a.record(goal='g', factor='f', horizon=1); a.record(goal='g', factor='f', horizon=2)
    b = acna.ResearchLedger(p)
    assert a.fingerprint() == b.fingerprint()
    before = a.fingerprint()          # 文件型台账共享同一份存储，先取快照
    b.record(goal='g', factor='f', horizon=3)
    assert b.fingerprint() != before
