# alphalens-reloaded 改造设计分析

> 配套文档：`ROE因子分析报告.md`（本次实测的完整结果）
> 结论基于**真实跑通一次标准因子分析**过程中撞到的问题，不是纸面推演。

---

## 0. 结论先行

**推荐：另起炉灶做 `alphalens-cna`，但只重写"引擎"，不重写"定义"。**

一句话理由：**alphalens 的问题不在算法，而在它把"美股 / 日频 / T+0 / 无涨跌停 / 无停牌 /
无退市 / 无生存者偏差"这一整套市场假设，焊死在了数据契约里。**
改它等于拆地基；而 IC、分位、换手这些**定义**是学术共识，直接沿用即可。

具体分工：

| | 处理方式 |
|---|---|
| IC / RankIC / 分位收益 / 换手 / 衰减 **的定义** | **沿用**（数学定义，不受版权约束） |
| 前向收益引擎、清洗、可成交性、日历 **的实现** | **重写**（这些正是缺陷所在） |
| 绘图 | 拆成可选依赖 |
| 迁移 | 提供兼容适配器 |

---

## 1. alphalens-reloaded 现状测绘

| 项 | 事实 |
|---|---|
| 血统 | Quantopian `alphalens`（2015-2018 停更）→ 社区维护的 `alphalens-reloaded`（0.4.6，持续发版到 2025） |
| 规模 | `utils` 1066 + `performance` 1209 + `plotting` 1014 + `tears` 706 ≈ **4000 行** |
| 许可 | **Apache 2.0**，Copyright 2018 Quantopian, Inc. |
| 硬依赖 | numpy / pandas / scipy / statsmodels / **seaborn / matplotlib** / IPython / empyrical-reloaded |
| 导入名 | 仍是 `alphalens`（与已停更的原包同名，靠发行版名区分） |

**公开 API 面**（迁移时必须覆盖的）：

- `alphalens.utils`：`get_clean_factor_and_forward_returns`、`get_clean_factor`、
  `compute_forward_returns`、`quantize_factor`、`infer_trading_calendar`、
  `demean_forward_returns`、`rate_of_return`、`get_forward_returns_columns`…
- `alphalens.performance`：`factor_information_coefficient`、`mean_information_coefficient`、
  `mean_return_by_quantile`、`factor_returns`、`factor_alpha_beta`、`quantile_turnover`、
  `factor_rank_autocorrelation`、`average_cumulative_return_by_quantile`、`create_pyfolio_input`…
- `alphalens.tears`：7 个 tear sheet（summary / returns / information / turnover / full / event×2）
- `alphalens.plotting`：20+ 绘图函数

---

## 2. 实测缺陷（每条都有本次运行的证据）

### D1 · 非日频因子索引直接崩 —— **阻断级**

原计划做"月末调仓"，alphalens 直接抛异常：

```
File "alphalens/utils.py", line 358, in compute_forward_returns
    df.index.levels[0].freq = freq
ValueError: Inferred frequency None from passed values
            does not conform to passed frequency C
```

**成因**：`infer_trading_calendar` 把索引里出现过的每个日历日当作交易日，未被覆盖的工作日
全被当成"假期"，推出来的 `CustomBusinessDay` 与月度索引不可能通过 pandas 2.x 的
`freq` 一致性校验。

**影响**：**"月度调仓"这个 A 股基本面因子最常用的设定，在 alphalens 上根本跑不了。**
本次分析被迫退化成"日度观测"。

### D2 · 列名把"1 个月"标成 "1D"

同一份数据，两种价格面板的频率，alphalens 输出：

| 价格面板 | 输出的收益列名 | 实际含义 |
|---|---|---|
| 月度 | `['1D', '3D', '6D', '12D']` | 1 / 3 / 6 / 12 **个月**（≈30/91/182/365 天） |
| 日度 | `['21D', '63D', '126D', '252D']` | 21/63/126/252 **天** ✓ |

月度下 `diff_custom_calendar_timedeltas` 在退化的日历上算错跨度，列名随之错标。
下游 `rate_of_return` / `std_conversion` 会按"天"做年化 —— **数字直接错**。

### D3 · 计算与绘图强耦合

`alphalens/__init__.py` 里 `from . import plotting`，而 `plotting.py` 顶部就
`import seaborn` / `import matplotlib`。结果是：

> **只想算 IC 的人，也必须装上 matplotlib + seaborn。**

### D4 · 重叠观测无任何提示 —— **方法论级**

日度观测 + 20 日持有期 → 相邻样本高度重叠。实测：

| 因子 | 因子秩自相关 | 朴素 t | Newey-West t(lag=20) | 虚高 |
|---|---|---|---|---|
| ROE | **0.9948** | 0.16 | 0.04 | **3.7×** |
| 20日动量 | 0.9279 | −21.93 | **−6.76** | **3.2×** |

alphalens 全程用朴素 t 值，**没有任何有效样本量或自相关的提示**。
（注：修正后动量仍强显著、ROE 仍不显著，结论稳健——但这是我自己补算的，库不管。）

### D5 · `dropna()` 一刀切，丢弃原因不可观测

```python
merged_data["factor"] = factor_copy
merged_data = merged_data.dropna()          # ← 任一持有期缺失即整行丢弃
...
merged_data["factor_quantile"] = quantile_data
merged_data = merged_data.dropna()
```

- **AND 语义**：四个持有期里只要有一个 NaN，这一行对**所有**持有期都消失
- 只打印一个总百分比，**不告诉你是停牌、退市、还是新股**
- 无法区分"真的没有数据"和"数据在但不该用"

本次运行丢弃 3.4%，看不出构成。

### D6 · 完全没有交易规则概念

实测本次数据（2019-2026，5066 只）：

| 现象 | 占比 | alphalens 的处理 |
|---|---|---|
| \|涨跌\| ≥ 9.8%（触及涨跌停） | **2.31%** 的(股,日) | 当作正常价格算收益 |
| \|涨跌\| ≥ 19.6%（创业/科创） | 0.14% | 同上 |
| 面板空格（停牌/未上市/已退市） | **13.28%** | 静默变 NaN 后 dropna |

**涨跌停意味着次日大概率买不进/卖不出**，把这段收益算进去就是策略高估。
T+1、ST（±5%）、上市首日无涨跌幅限制等规则同样为零。

### D7 · 生存者偏差

退市股从行情面板里直接消失，**它的最后一段亏损不会被计入**。
`stock_info` 只有当前在市的股票 —— 用当前股票池回测历史，天然幸存者偏差。
alphalens 对此既无检查也无提示。

### D8 · 有 `groupby`，但没有中性化

`get_clean_factor(..., groupby=...)` + `by_group=True` 只做**分组统计**，
不做回归中性化（剥离行业/市值暴露）。A 股因子研究里行业中性化基本是默认动作。

---

## 3. quantaxis 侧的印证：合规逻辑只能写在调用方

`QUANTAXIS/QAFactor/featureAnalysis.py`：

```python
# 2026-09-02 A股 T+1 合规: 用"次日开盘价"作为成交价,
# prices[t] = open[t+1], 使 forward_ret[t] = open[t+1+period]/open[t+1] - 1
# 原实现用收盘价面板=假设 T 日收盘即可成交, 对 T+1 市场会高估
panelprice = deepcopy(self.stock_data.openpanel).shift(-1)
```

**A 股的 T+1 合规，QuantAxis 只能靠在调用方把价格面板整体平移一天来"骗"过 alphalens。**
这是个很典型的信号：**该逻辑本该属于收益引擎，却被迫外置。**
一旦调用方忘记平移，结果就悄悄错掉——而且没有任何报错。

---

## 4. 三条路线对比

| 维度 | A. fork 改源码 | B. 外层包一层 | C. 另起炉灶 alphalens-cna |
|---|---|---|---|
| 工作量 | 中 | **小** | 大 |
| 能治 D1（非日频崩） | 能 | **不能** | 能 |
| 能治 D2（列名错标） | 能 | **不能** | 能 |
| 能治 D4（重叠观测） | 能（加个修正输出） | **不能** | 能 |
| 能治 D6/D7（交易规则/生存偏差） | 要改数据契约 | 只能外围打补丁 | 契约层原生支持 |
| 能治 D3（绘耦合） | 要重构 import | 不能 | 分层即可 |
| 上游同步成本 | **持续**（reloaded 仍在发版） | 无 | 无 |
| 命名空间冲突 | 与 `alphalens` 同名 | 无 | 干净 |
| 许可负担 | 需保留 Apache 声明 + 标注修改 | 低 | 无（仅致谢） |
| 开源价值 | "又一个 fork" | **低**（是脚本不是库） | **高** |

**三条关键判断（第 1 条为 2026-09-16 更正）：**

1. ~~**路线 B 不可行**~~ → **更正：路线 B 可行，我原来的论断是错的。**
   实测发现 D1 的确切触发条件是「**因子非日频 + 价格日频**」，而不是"月度调仓不能做"：

   | 配置 | 结果 |
   |---|---|
   | 月度因子 + 日频价格 → `get_clean_factor_and_forward_returns` | ❌ 崩 |
   | 月度因子 + 月频价格 → 同上 | ✅ 不崩，但列名错标 `['1D','3D','6D']` |
   | **月度因子 + 自建 forward_returns → `get_clean_factor`** | ✅ **完全正常**，列名自定 `['1M','3M','6M']` |

   原因：`get_clean_factor_and_forward_returns` = `compute_forward_returns` +
   `get_clean_factor`，而崩溃只在 `compute_forward_returns` 的
   `df.index.levels[0].freq = freq`。**`get_clean_factor` 接受外部传入的
   forward_returns，于是 D1 与 D2 都能在外层彻底绕开。**

   外层包装**能**解决的：D1 / D2 / D5 / D6 / D7 / D8（自建 forward_returns 时
   就把涨跌停、停牌、生存偏差、持有期口径全部处理掉；行业中性化可先对因子做）。
   外层包装**解决不了**的只有：**D3（`import alphalens` 必拉 seaborn/matplotlib）**，
   以及 D4 需要自己补算 Newey-West。

   **结论修正：B 不是 C 的替代品，而是 C 的第一阶段。**
   但若目标只是"把 A 股因子分析做对"（内部工具），B 就已经够了。

2. **A 与 C 的分水岭在"数据契约"** —— alphalens 的核心抽象是
   `(factor, prices) → forward_returns`。要支持 A 股，必须把这个抽象换成
   **`(factor, prices, tradability, calendar) → 带可成交性标注的 forward_returns`**。
   契约一换，实现里能复用的比例就掉到很低，**却还得背着 fork 的同步包袱和同名冲突**。

3. **"取代"与"包装"是两件事** —— 用户目标是**取代** alphalens 并开源。
   包装方案虽然能用，但它**仍然依赖 alphalens-reloaded**，达不到"取代"。
   所以路径应当是**绞杀者模式**：先用包一层拿到正确性与等价性验证，
   再逐步把 alphalens 的内部实现替换掉，最终去掉依赖。

---

## 5. 推荐设计：alphalens-cna

### 5.1 分层（依赖单向，核心零绘图）

```
alphalens_cna/
├── core/                  纯 pandas/numpy/scipy，不 import matplotlib
│   ├── contract.py        输入契约与校验（PIT 可用日 / 索引规范 / 股票池 as-of）
│   ├── calendar.py        交易日历（自持，不依赖 pandas freq 推断）
│   ├── tradability.py     A股可成交性：涨跌停 / 停牌 / 新股 / ST / T+1
│   ├── returns.py         前向收益引擎（next_open | vwap | close，可插拔）
│   ├── clean.py           清洗：带原因统计，不静默 dropna
│   ├── quantize.py        分层（支持行业内中性分层）
│   └── metrics.py         IC / RankIC / ICIR / NW修正t / 分层 / 换手 / IC衰减
├── plot/                  optional extra —— 只有这里才拉 matplotlib/seaborn
└── compat/                与 alphalens 的输入输出适配器（迁移用）
```

### 5.2 七条设计决策（每条对应一个实测缺陷）

| # | 决策 | 治哪个缺陷 |
|---|---|---|
| 1 | **契约以"可用日(as-of)"为核心**：因子值必须带一个"何时可知"的时点，库负责校验不越界 | 前视偏差（本次用 PIT 层已做到，应固化为库的要求） |
| 2 | **可成交性是一等公民**：`tradability` 掩码与收益同时返回；不可成交记为 `NaN` **并附 `reason` 列**，单独统计张数 | D5 + D6 |
| 3 | **收益引擎可插拔**，默认 `next_open`（T+1 语义），不再要求调用方平移面板 | quantaxis 的外置补丁（§3） |
| 4 | **非日频原生**：月度/周度调仓直接用交易日历切片，**不依赖 pandas `freq` 推断** | D1 + D2 |
| 5 | **默认输出 Newey-West 修正 t 与有效样本量**，并在自相关过高时显式警告 | D4 |
| 6 | **清洗输出丢弃原因分布**（停牌 X 条 / 涨跌停 Y 条 / 新股 Z 条 / 退市 W 条） | D5 |
| 7 | **行业中性化内置**，直接吃申万事件态成份股（`sw_industry/members.parquet`，as-of 取成分） | D8 |

D3（绘耦合）由分层天然解决；D7（生存者偏差）需要**股票池做成 as-of 的历史成分表**，
这属于数据层责任 —— alphalens-cna 负责**拒绝**用当前股票池回测历史（契约校验时报警）。

### 5.3 明确**不做**（防止范围爆炸）

- ❌ 不做回测引擎 / 撮合 / 资金管理 —— 那是 quantaxis / QARSBridge 的职责
- ❌ 不做数据获取与存储 —— 用现有的 mongo + `quantming/data/` parquet 层
- ❌ 不做 Barra 风格风险模型
- ❌ 不重新定义 IC / 分位 / 换手 —— 沿用学术共识，保证结论可比

---

## 6. 许可与合规

- alphalens-reloaded 是 **Apache 2.0**。若**移植代码片段**，必须：
  保留版权声明、附上许可证副本、并**在修改过的文件里显著标注改动**（Apache 2.0 §4b/c）
- 若只是**沿用定义**（IC = 因子与收益的相关系数这类数学定义），不受版权约束
- 建议：`alphalens-cna` 采用 **MIT**（与 quantaxis 一致），
  在 `NOTICE` / README 里致谢 Quantopian 与 alphalens-reloaded，
  并注明"算法定义参考，实现独立重写"

---

## 7. 分阶段计划（建议）

| 阶段 | 内容 | 估计 |
|---|---|---|
| **P0** | `contract` + `returns` + `metrics`；**用本次报告的数据复现 alphalens 的结论**作为等价性验证 | 1-2 天 |
| P1 | `tradability` + `clean`（带原因）+ NW 修正 | 2-3 天 |
| P2 | 行业中性分层 + 非日频原生化 + `compat` 适配器 | 2-3 天 |
| P3 | `plot` 独立 + 文档 + 开源发布 | 2 天 |

### 等价性验证方法（**这一步不能省**）

1. **退化配置对齐**：同一份 `factor`/`prices`，在「日度 + 关闭全部 A 股规则」下，
   alphalens-cna 与 alphalens-reloaded 的 IC / 分层收益必须数值一致（容差 1e-10）
2. **打开规则后逐项归因**：差异必须能被解释清楚 —— 剔除涨跌停 X 条、停牌 Y 条、
   新股 Z 条，逐项对账。**解释不了的差异就是 bug。**

---

## 8. 一句话总结

> **它算得对的部分（IC / 分位 / 换手）是学术共识，抄定义即可；
> 它算得不对的部分（收益引擎、清洗、日历、可成交性）全是 A 股特有的，必须自己写。**

**路径建议不是"一次性重写"，而是绞杀者模式（B → C）：**

| 阶段 | 做什么 | 对 alphalens 的依赖 |
|---|---|---|
| 阶段一 | 自建**收益引擎 + 契约 + 指标**，但**同时**用 `get_clean_factor` 跑一遍做交叉验证 | 依赖（用于对账） |
| 阶段二 | 自有实现覆盖常用路径后，切默认路径 | 依赖（仅回归测试用） |
| 阶段三 | 移除依赖，开源发布纯自研库 | 无 |

这样做有三个好处：
1. **等价性验证是免费的** —— 阶段一里新旧两套同时在跑，逐项对账
2. **风险可控** —— 任何一步都能退回上一个可用状态
3. **不受 Apache 2.0 传染** —— 阶段三产出的是独立实现，只需在 NOTICE 里致谢

> 第一个里程碑**不是"功能更多"，而是"用新库复现旧库的结论"**——先证明等价，再谈改进。
