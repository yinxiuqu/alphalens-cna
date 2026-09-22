"""从「原始价 + 除权事件」现算复权因子。

这是 D5 决策里的 ``computed`` 路径 —— 默认、可复现。
与之相对的是 ``stored``：直接读现成的复权因子（快，但快照会漂）。

算法来源
--------
复刻 QUANTAXIS ``QAData/data_fq.py:_QA_data_stock_to_fq``（MIT, yutiansut/QUANTAXIS）。
选择复刻而非重写，是为了与既有数据库**同源** —— 本会话已实测在在市股上
**bit 级复现**库内 ``stock_adj``（4/4，误差 1e-8 ~ 1e-10）。

除权参考价公式
--------------
.. code-block:: text

    preclose[t] = (close[t-1] × 10 − 分红 + 配股数 × 配股价) / (10 + 配股数 + 送转股数)

前复权：``adj = (preclose[t+1] / close[t]).fillna(1)[::-1].cumprod()``
后复权：``adj = (close[t] / preclose[t+1]).cumprod().shift(1).fillna(1)``

⚠️ 必须传 DatetimeIndex
----------------------
``.loc[start:end]`` 切片在**字符串索引**上会静默产生 10/11 的偏差
（末位 adj 变 0.909091 而非 1.0），**不报错，只是数错**。
本模块显式强制 DatetimeIndex —— 本会话踩过这个坑。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contract.errors import fail

__all__ = ['compute_adj_factor', 'DIVIDEND_FIELDS']

DIVIDEND_FIELDS = ('fenhong', 'peigu', 'peigujia', 'songzhuangu')
#: 除权除息类别（通达信 xdxr 的 category==1）
XDXR_CATEGORY = 1


def compute_adj_factor(prices, xdxr, method='hfq', date_col='date'):
    """现算复权因子。

    Parameters
    ----------
    prices : DataFrame
        原始不复权行情，需含 ``raw_close``（或 ``close``），索引为
        ``MultiIndex(date, asset)``，**date 层必须是 DatetimeIndex**。
    xdxr : DataFrame
        除权事件，索引 ``MultiIndex(date, asset)``，列含 ``category`` 与
        :data:`DIVIDEND_FIELDS`。只用到 ``category == 1`` 的行。
    method : {'hfq', 'qfq'}
        ``'hfq'``（默认）后复权：首日 = 1，随除权递增，**不随新数据回溯改变**。
        ``'qfq'`` 前复权：末位 = 1（归一化到数据末尾）。

    Returns
    -------
    pd.Series
        复权因子，索引与 ``prices`` 相同的 ``MultiIndex(date, asset)``，
        名称 ``adj_factor``。

    Notes
    -----
    同一只股票内 ``adj`` 的**比值**才决定收益；整体缩放不影响收益。
    所以 ``hfq`` 与 ``qfq``（乃至任何常数倍数）在收益上完全等价 ——
    这正是 ``check_adjust_agreement`` 能拿两种口径互相验证的依据。
    """
    if method not in ('hfq', 'qfq'):
        fail('adjust', 'bad_method', f"method 只能是 'hfq' 或 'qfq'，收到 {method!r}")

    close_col = 'raw_close' if 'raw_close' in prices.columns else 'close'
    if close_col not in prices.columns:
        fail('adjust', 'no_close',
             f'prices 里找不到收盘价（试过 raw_close / close）；'
             f'实际列：{list(prices.columns)[:12]}')

    _require_datetime_index(prices, 'prices')
    if xdxr is not None and len(xdxr):
        _require_datetime_index(xdxr, 'xdxr')

    out = []
    for asset, g in prices.groupby(level='asset', sort=False):
        px = g.droplevel('asset').sort_index()
        x = None
        if xdxr is not None and len(xdxr):
            try:
                xs = xdxr.xs(asset, level='asset')
                x = xs[xs['category'] == XDXR_CATEGORY]
            except KeyError:
                x = None
        out.append(_one_asset(px, x, close_col, method).assign(asset=asset))

    if not out:
        fail('adjust', 'empty', 'prices 为空')
    # 注意：set_index(..., append=True) 之后索引已是 (date, asset)，
    # **不要再 swaplevel** —— 那会把值换成 (asset, date) 而名字仍写 (date, asset)，
    # 造成"名字与值不符"的隐性错误（下游按 level='asset' 分组会分到日期上去）。
    res = pd.concat(out).set_index('asset', append=True)
    res.index = res.index.set_names(['date', 'asset'])
    if list(res.index.names) != ['date', 'asset']:
        fail('adjust', 'index_order', f'索引层级异常：{res.index.names}')
    res = res.sort_index()['adj_factor']
    if not pd.api.types.is_datetime64_any_dtype(res.index.get_level_values('date')):
        fail('adjust', 'index_corrupt',
             'date 层不是 datetime —— 索引层级顺序可能被弄反了。')
    res.name = 'adj_factor'
    return res


def _require_datetime_index(df, what):
    """强制 DatetimeIndex —— 字符串索引会静默算错（10/11 偏差）。"""
    d = df.index.get_level_values('date')
    if not pd.api.types.is_datetime64_any_dtype(d):
        fail('adjust', 'index_not_datetime',
             f'{what} 的 date 层必须是 DatetimeIndex，收到 {d.dtype}。\n'
             f'  ⚠️ 传字符串索引**不会报错**，但会静默产生 10/11 的偏差'
             f'（末位复权因子变成 0.909091 而非 1.0）。\n'
             f'  修法：df.index = df.index.set_levels('
             f'pd.to_datetime(df.index.levels[0]), level="date")')


def _one_asset(px: pd.DataFrame, x: pd.DataFrame, close_col: str,
               method: str) -> pd.DataFrame:
    """单只股票的复权因子。**逐行复刻** QUANTAXIS 的算法。

    ⚠️ 关键点：除权日**不一定在行情索引里**（停牌日、上市前、未来的公告）。
    QUANTAXIS 用 ``pd.concat`` 取**并集**，把这些日子加进索引再 ``ffill``；
    若改用 ``reindex`` 就会**丢掉这些除权**，导致更早的历史整体偏大。
    实测 600519 丢了 2006-05-19/05-24 两次除权，前 1078 天的因子偏大 2.262 倍。
    """
    data = px.copy()
    data['close'] = data[close_col].astype(float)
    data['if_trade'] = 1

    empty = pd.DataFrame(columns=['category'] + list(DIVIDEND_FIELDS))
    if x is not None and len(x):
        info = x.sort_index()
        info = info[~info.index.duplicated(keep='last')]
        lo, hi = data.index[0], data.index[-1]
        info = info.loc[lo:hi]          # 区间切片（要求 DatetimeIndex）
    else:
        info = empty

    if len(info):
        # ① 先把 category 并进来（会引入"只在除权日存在"的行）
        data = pd.concat([data, info[['category']]], axis=1)
        data['if_trade'] = data['if_trade'].fillna(0)
        data = data.ffill()             # 含 close —— 补出来的行沿用前一日收盘
        # ② 再把分红字段并进来。【不做 ffill】—— 非除权日必须为 0
        data = pd.concat([data, info[list(DIVIDEND_FIELDS)]], axis=1)
    else:
        data = pd.concat([data, info[['category'] + list(DIVIDEND_FIELDS)]], axis=1)

    # 只对需要的列做数值化 + 补零。
    # 不做整表 fillna —— 那会把 object 列（code / date）也卷进来，
    # 触发 pandas 的 silent-downcasting 弃用警告。
    for c in DIVIDEND_FIELDS:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors='coerce').fillna(0.0)
        else:
            data[c] = 0.0
    data['close'] = pd.to_numeric(data['close'], errors='coerce')

    data['preclose'] = (
        data['close'].shift(1) * 10
        - data['fenhong']
        + data['peigu'] * data['peigujia']
    ) / (10 + data['peigu'] + data['songzhuangu'])

    if method == 'qfq':
        data['adj_factor'] = (data['preclose'].shift(-1) / data['close']) \
            .fillna(1)[::-1].cumprod()
    else:
        data['adj_factor'] = (data['close'] / data['preclose'].shift(-1)) \
            .cumprod().shift(1).fillna(1)

    # 只保留真实交易日（并集引入的"仅除权日"行要丢掉，那两天没有行情）
    return data.loc[px.index, ['adj_factor']]
