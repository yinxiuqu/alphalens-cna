"""八个契约对象。

这是 L0 层，**整个库的入口**。任何数据源转成这些对象才能进本库。

设计要点
--------
1. **只吃 pandas DataFrame**，不定义新容器 —— 能 ``print()`` 出来就能喂进来。
2. **构造即校验**，违反契约直接抛 ``ContractError``，不产出任何数字。
3. **薄包装**：``.df`` 拿到原表，不复制、不改写（除必要的排序）。

统一约定（见 docs/输入数据规格.md）
----------------------------------
* 索引：``MultiIndex(date, asset)``，date 为 ``datetime64[ns]`` 且 tz-naive
* 比率用小数（0.05 = 5%）、金额用元、成交量用手
* 缺失用 ``NaN``
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .errors import ContractError, describe_rows, fail

DATE, ASSET = 'date', 'asset'


# --------------------------------------------------------------------------- #
# 基类
# --------------------------------------------------------------------------- #
class _Panel:
    """DataFrame 型契约的基类。"""

    contract = '_Panel'
    required = ()          # 必需列
    optional = ()          # 可选列
    numeric = ()           # 需要是数值、且不得为 NaN 的列
    positive = ()          # 需要 > 0 的列

    def __init__(self, df: pd.DataFrame, validate: bool = True, copy: bool = False):
        if not isinstance(df, pd.DataFrame):
            fail(self.contract, 'type',
                 f'需要 pandas.DataFrame，收到 {type(df).__name__}')
        self._df = df.copy() if copy else df
        self._normalize_index()
        # ★ 记下"到底校验过没有"：`XxxPanel(df, validate=False)` 造出来的对象
        #   如果不留痕，下游（入口层统一校验）就只能靠猜，于是"上游关掉校验、
        #   下游以为校验过了"这种静默漏洞就有了藏身处。
        self._validated = False
        if validate:
            self.validate()
            self._validated = True

    # -- 索引 -----------------------------------------------------------------
    def _normalize_index(self):
        df = self._df
        names = list(df.index.names)
        if not isinstance(df.index, pd.MultiIndex):
            fail(self.contract, 'index',
                 f'索引必须是 MultiIndex(date, asset)，收到 {type(df.index).__name__}。'
                 f'提示：df.set_index(["date", "asset"])')
        if names != [DATE, ASSET]:
            fail(self.contract, 'index_names',
                 f'索引层级名必须是 ["date", "asset"]，收到 {names}。'
                 f'提示：df.index = df.index.set_names(["date", "asset"])')

    # -- 校验 -----------------------------------------------------------------
    def validate(self):
        self._check_unique()
        self._check_columns()
        self._check_dtypes()
        self._check_numeric()
        self._check_finite()
        self._check_positive()
        self.validate_extra()

    def validate_extra(self):
        """子类补充校验。"""

    def _check_unique(self):
        if self._df.index.has_duplicates:
            dup = self._df.index[self._df.index.duplicated()].unique()
            fail(self.contract, 'index_unique',
                 f'索引 (date, asset) 必须唯一，发现 {len(dup)} 个重复：'
                 f'{describe_rows(dup)}')

    def _check_columns(self):
        missing = [c for c in self.required if c not in self._df.columns]
        if missing:
            fail(self.contract, 'missing_columns',
                 f'缺少必需列 {missing}；实际列：{list(self._df.columns)}')

    def _check_dtypes(self):
        d = self._df.index.get_level_values(DATE)
        if not pd.api.types.is_datetime64_any_dtype(d):
            fail(self.contract, 'date_dtype',
                 f'date 层级必须是 datetime64，收到 {d.dtype}。'
                 f'提示：pd.to_datetime(df["date"])')
        if getattr(d.dtype, 'tz', None) is not None:
            fail(self.contract, 'timezone',
                 f'date 必须 tz-naive（全库统一），收到 tz={d.dtype.tz}。'
                 f'提示：df.index = df.index.set_levels(d.tz_localize(None), level="date")')
        a = self._df.index.get_level_values(ASSET)
        if not (a.dtype == object or pd.api.types.is_string_dtype(a)):
            fail(self.contract, 'asset_dtype',
                 f'asset 层级必须是字符串（6 位代码），收到 {a.dtype}')

    def _check_numeric(self):
        for col in self.numeric:
            if col not in self._df.columns:
                continue
            s = self._df[col]
            if not pd.api.types.is_numeric_dtype(s):
                fail(self.contract, 'dtype',
                     f'列 `{col}` 必须是数值，收到 {s.dtype}')

    def _check_finite(self):
        """±inf 不是合法数值。**NaN 是合法缺失**（停牌 / 未上市），inf 不是。

        `_check_positive` 判的是 `<= 0`，正的 inf 满足 `> 0` 会直接穿过去，
        然后在收益 / 相关系数 / 方差里算出 NaN 或 inf —— 属于垃圾进垃圾出，
        宁可在门口拦下。

        ⚠️ 检查范围是 **`numeric` 声明的列 ∪ 所有数值 dtype 的列** —— 不能只看
        `numeric`：`Exposures` / `Universe` / `Grouping` / `Events` 的 `numeric`
        是空的（它们的列名不固定），只看 `numeric` 等于**一个列都不查**。
        实测：把 inf 塞进 `exposures`，`build_report` 会一路跑通，且结果与把
        inf 换成 NaN **逐位相同** —— 非法值被静默当成缺失，用户拿不到任何信号。
        （标签 / 字符串列是 object dtype，天然跳过；bool 列不会有 inf。）
        """
        cols = list(self.numeric) + [
            c for c in self._df.columns
            if c not in self.numeric and pd.api.types.is_numeric_dtype(self._df[c])]
        for col in cols:
            if col not in self._df.columns:
                continue
            v = pd.to_numeric(self._df[col], errors='coerce').to_numpy(
                dtype='float64', na_value=np.nan)
            bad = np.isinf(v)
            if bad.any():
                fail(self.contract, 'non_finite',
                     f'列 `{col}` 有 {int(bad.sum())} 行是 ±inf：'
                     f'{describe_rows(self._df.index[bad])}\n'
                     f'  含义：inf 不是缺失（缺失应当用 NaN），'
                     f'它会让收益、相关系数、方差全部变成 NaN / inf。')

    def _check_positive(self):
        for col in self.positive:
            if col not in self._df.columns:
                continue
            s = self._df[col]
            bad = s.notna() & (s <= 0)
            if bad.any():
                n = int(bad.sum())
                fail(self.contract, 'positive',
                     f'列 `{col}` 必须 > 0，发现 {n} 行非正：'
                     f'{describe_rows(s.index[bad])}')

    # -- 便捷 -----------------------------------------------------------------
    @property
    def df(self) -> pd.DataFrame:
        """底层 DataFrame（**只读用，不要就地改**）。"""
        return self._df

    def __len__(self):
        return len(self._df)

    def __repr__(self):
        d = self._df.index.get_level_values(DATE)
        assets = self._df.index.get_level_values(ASSET).nunique()
        span = f'{d.min():%Y-%m-%d} ~ {d.max():%Y-%m-%d}' if len(d) else '空'
        return (f'<{self.contract} {len(self._df):,} 行 · {assets} 只 · {span}>')


# --------------------------------------------------------------------------- #
# 1. FactorPanel
# --------------------------------------------------------------------------- #
class FactorPanel(_Panel):
    """待检验的因子。

    ============  =============  =====================================
    列            类型            说明
    ============  =============  =====================================
    ``value``     float64        因子值
    ``available_at`` datetime64  该值**何时可知**（硬闸门）
    ============  =============  =====================================
    """

    contract = 'FactorPanel'
    required = ('value', 'available_at')
    numeric = ('value',)

    def validate_extra(self):
        a = self._df['available_at']
        if not pd.api.types.is_datetime64_any_dtype(a):
            fail(self.contract, 'available_at_dtype',
                 f'`available_at` 必须是 datetime64，收到 {a.dtype}')
        d = self._df.index.get_level_values(DATE)
        if getattr(a.dtype, 'tz', None) is not None:
            fail(self.contract, 'available_at_tz', '`available_at` 必须 tz-naive')
        # ★ 前视硬闸门
        late = a.notna() & (a > d)
        if late.any():
            n = int(late.sum())
            ex = self._df.index[late][:2]
            detail = '; '.join(
                f'{dt:%Y-%m-%d} {c}: available_at={a.loc[(dt, c)]:%Y-%m-%d}'
                for dt, c in ex)
            fail(self.contract, 'lookahead',
                 f'发现前视：{n} 行的 `available_at` 晚于 `date`（{detail} …）。\n'
                 f'  含义：这个因子值在标注日期时**还不知道**，用它做分析会高估收益。\n'
                 f'  修法：把 `date` 改成 `available_at` 当天或之后。')
        if self._df['value'].isna().all():
            fail(self.contract, 'all_nan', '`value` 全为 NaN，无法分析')


# --------------------------------------------------------------------------- #
# 2. PricePanel
# --------------------------------------------------------------------------- #
class PricePanel(_Panel):
    """行情。**三价并存** —— 原始价、复权因子、复权价。

    ====================  ========  ===================================
    组                    列         用途
    ====================  ========  ===================================
    原始不复权            ``raw_*``  **只用于制度判定**（涨跌停/报价单位）
    前收                  ``prev_close``
    复权因子（唯一真相）   ``adj_factor``
    复权价（派生）         ``adj_*``  **只用于收益计算**
    成交                  ``volume`` / ``amount``
    ====================  ========  ===================================

    ⚠️ 缺 ``raw_*`` 或 ``adj_factor`` 会被拒绝 —— 前者判不了涨跌停，
    后者换不了口径。
    """

    contract = 'PricePanel'
    required = ('raw_open', 'raw_high', 'raw_low', 'raw_close', 'prev_close',
                'adj_factor', 'adj_open', 'adj_high', 'adj_low', 'adj_close')
    optional = ('volume', 'amount')
    numeric = required + optional
    positive = ('raw_open', 'raw_high', 'raw_low', 'raw_close',
                'adj_factor', 'adj_open', 'adj_high', 'adj_low', 'adj_close')

    ADJ_TOL = 1e-8

    def validate_extra(self):
        # ⚠️ 这里原本有两个"raw 与 adj 必须成对"的分支（只给 adj_* / 只给 raw_*），
        #    但它们**永远执行不到** —— `validate()` 先跑 `_check_columns()`，而
        #    `required` 里 raw_* 与 adj_* 全都列着，缺哪个都会先在那边报
        #    `missing_columns`。留着不只是冗余，还会撒谎：其中一句写着
        #    "或只给 raw_* 让库自算"，而实测只给 raw_* 会被 missing_columns 拒绝
        #    （规格里 adj_* 本来就是必需列，示例就是让用户自己乘出复权价）。
        #    成对性的**正确表述**在类 docstring 里，这里删掉以免又教错人。
        # ★ 自洽：adj_* == raw_* * adj_factor（含缺失检查）
        self._check_self_consistent()

    def _check_self_consistent(self):
        df = self._df
        for raw, adj in (('raw_open', 'adj_open'), ('raw_high', 'adj_high'),
                         ('raw_low', 'adj_low'), ('raw_close', 'adj_close')):
            if raw not in df.columns or adj not in df.columns:
                continue
            # ★ 缺失也要查 —— 这是本检查此前的**洞**：
            #   判据是 `diff > tol`，而 `NaN > tol` 恒为 False，
            #   于是 adj 或 adj_factor 缺行会被"自洽"放行。后果不只是漏检：
            #   体检层算复权收益时**前值填充**，缺失行算出 0% 收益 →
            #   被读成"复权后正常（复权价连续）"，直接给出假 all-clear。
            #   口径按规格：缺失一律 NaN；停牌日 raw 与 adj **两边都缺**，
            #   所以只查"raw 在、adj 或 factor 不在"的行 —— 停牌不受影响。
            raw_ok = df[raw].notna()
            miss_adj = raw_ok & df[adj].isna()
            miss_fac = raw_ok & df['adj_factor'].isna()
            if miss_adj.any() or miss_fac.any():
                n_adj, n_fac = int(miss_adj.sum()), int(miss_fac.sum())
                which = miss_adj if miss_adj.any() else miss_fac
                i = df.index[which][0]
                fail(self.contract, 'adjust_incomplete',
                     f'有原始价却缺复权价：`{adj}` 缺 {n_adj} 行、'
                     f'`adj_factor` 缺 {n_fac} 行（这些行的 `{raw}` 有值）。\n'
                     f'  例：{i[0]:%Y-%m-%d} {i[1]}  '
                     f'{raw}={df.loc[i, raw]:.6f}，'
                     f'adj_factor={df.loc[i, "adj_factor"]}，'
                     f'{adj}={df.loc[i, adj]}\n'
                     f'  含义：复权价本该由 `{raw} × adj_factor` 算得出来，'
                     f'缺了就无法对账；体检层还会把它读成"复权后正常"。\n'
                     f'  修法：补齐该行复权价/因子。**停牌**要把 raw 也置 NaN'
                     f'（两边同时缺才算停牌）。')
            expect = df[raw] * df['adj_factor']
            diff = (df[adj] - expect).abs()
            scale = expect.abs().clip(lower=1e-12)
            bad = diff > self.ADJ_TOL * scale.max()
            if bad.any():
                n = int(bad.sum())
                i = df.index[bad][0]
                fail(self.contract, 'adjust_self_consistent',
                     f'列 `{adj}` ≠ `{raw}` × `adj_factor`，{n} 行不符。\n'
                     f'  例：{i[0]:%Y-%m-%d} {i[1]}  '
                     f'{adj}={df.loc[i, adj]:.6f}，'
                     f'{raw}×factor={expect.loc[i]:.6f}\n'
                     f'  含义：复权价与因子不自洽，收益会算错。')


# --------------------------------------------------------------------------- #
# 3. Tradability
# --------------------------------------------------------------------------- #
class Tradability(_Panel):
    """可成交性。不给就由库从 PricePanel + Calendar 推断。

    ==============  =========  ==========================================
    列              类型       说明
    ==============  =========  ==========================================
    ``suspended``   bool       停牌
    ``is_st``       bool       名称含 ST/*ST → 涨跌停 5%
    ``listed_days`` int32      上市第几个交易日（1 = 首日）
    ==============  =========  ==========================================

    ⚠️ **不要**用「收盘在板」字段判开盘成交 —— 那是前视。
    """

    contract = 'Tradability'
    required = ()
    optional = ('suspended', 'is_st', 'listed_days',
                'limit_up_price', 'limit_down_price')
    numeric = ('listed_days', 'limit_up_price', 'limit_down_price')
    # 至少要有**一个**能用的成交信号，否则这张表毫无意义 ——
    # "什么都能通过"等于没有契约。用户反馈里"错层报错"的最后一处就是它：
    # 一张只有无关列的表会一路穿到 `forward_returns` 才报 bad_tradability。
    signals = ('can_buy_open', 'can_sell_open', 'suspended', 'is_st', 'listed_days')

    def validate_extra(self):
        if not any(c in self._df.columns for c in self.signals):
            fail(self.contract, 'missing_signal',
                 f'可成交性表至少要有一列成交信号 {list(self.signals)}；'
                 f'实际列：{list(self._df.columns)}\n'
                 f'  修法：用 `compute_tradability(prices, calendar=…)` 生成，'
                 f'或自行提供 `can_buy_open` / `can_sell_open` / `suspended`。')


# --------------------------------------------------------------------------- #
# 4. Universe
# --------------------------------------------------------------------------- #
class Universe(_Panel):
    """as-of 股票池。**防生存者偏差的硬闸门。**"""

    contract = 'Universe'
    required = ('in_universe',)


# --------------------------------------------------------------------------- #
# 5. Grouping
# --------------------------------------------------------------------------- #
class Grouping(_Panel):
    """行业 / 分组标签。必须是 as-of 的（按当日实际所属，不是最新）。

    ⚠️ 列名**不强制叫 `group`** —— 要求是「**恰好一列**」：
    行业表叫 `industry` / `sw_l1` 都是常见写法，硬性要求改名会打断合法用法。
    若同时有 `group` 列则优先用它（`group_col` 会告诉你用的是哪一列）。
    """

    contract = 'Grouping'
    required = ()

    @property
    def group_col(self):
        """分组标签所在的列名（有 `group` 用它，否则就是唯一那一列）。"""
        if 'group' in self._df.columns:
            return 'group'
        if not len(self._df.columns):
            # 只在 `Grouping(df, validate=False)` + 空表时走得到 —— 别漏裸 IndexError。
            fail(self.contract, 'empty',
                 'Grouping 没有列，取不到分组标签列名。（空表请用 '
                 '`Exposures` 传控制变量，或补上分组列。）')
        return self._df.columns[0]

    def _check_columns(self):
        super()._check_columns()
        if not len(self._df.columns):
            fail(self.contract, 'empty',
                 'Grouping 至少要有一列分组标签（如 industry）；'
                 '或改用 `Exposures` 传控制变量。')
        if len(self._df.columns) > 1 and 'group' not in self._df.columns:
            fail(self.contract, 'ambiguous_group',
                 f'分组标签只能有一列，收到 {len(self._df.columns)} 列：'
                 f'{list(self._df.columns)}。\n'
                 f'  含义：`neutralize` 的组内去均值需要**一个**分组维度。\n'
                 f'  修法：只保留那一列；或把要用的那列命名为 `group`。')


# --------------------------------------------------------------------------- #
# 6. Exposures
# --------------------------------------------------------------------------- #
class Exposures(_Panel):
    """控制变量（市值、换手、PB…）。做中性化或 Fama-MacBeth 时必填。"""

    contract = 'Exposures'
    required = ()

    def _check_columns(self):
        super()._check_columns()
        if not len(self._df.columns):
            fail(self.contract, 'empty',
                 'Exposures 至少要有一列控制变量（如 ln_mv）')


# --------------------------------------------------------------------------- #
# 7. Events
# --------------------------------------------------------------------------- #
class Events(_Panel):
    """事件日。只保留事件发生那一天的行（不是每日一行）。"""

    contract = 'Events'
    required = ('event_type',)
