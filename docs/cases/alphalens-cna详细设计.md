# alphalens-cna 详细设计（不动手 · 设计稿）

> 目标：**具备 alphalens-reloaded 的全部功能并更优**，贴合 A 股隔夜交易实际，
> 正确排除涨跌停与停牌，兼容 quantaxis 数据层，且**让使用者能判断分析对不对**。
> 不内置绘图，只产出可被任意绘图工具消费的数据结构。

---

## 0. 先说最要紧的：怎么才能"知道分析对不对"

这是你问的最关键的一句。alphalens 给不了这个 —— 它只给数字，不给"这个数字可不可信"。
本设计把它当成**一等需求**，构建**六道防线**，每一道都独立可验证：

| # | 防线 | 时机 | 具体做什么 | 失败表现 |
|---|---|---|---|---|
| **1** | **契约校验** | 输入时 | 因子值必须带"何时可知"时点；索引唯一；价格为正；股票池必须 as-of；时区一致 | 直接拒绝运行，不产出数字 |
| **2** | **数据体检** | 输入时 | 复权连续性、涨跌停占比、停牌占比、覆盖率、极端收益 —— 生成体检报告 | 超阈值告警并列出明细 |
| **3** | **不变量检查** | 运行中 | ①账要平：输入条数 = 输出条数 + 各类剔除之和 ②分层样本数均衡 ③前向收益自洽（重算对账）| 不变量破 ⇒ 抛异常 |
| **4** | **等价性回归** | 测试 | 退化配置（关闭全部 A 股规则、日频、收盘成交）下与 alphalens **数值一致（容差 1e-10）** | CI 直接红 |
| **5** | **已知答案测试** | 测试 | 手工构造 3 只股票 × 2 个调仓日的小数据集，含涨停/停牌，答案可**手算** | 期望值不符即失败 |
| **6** | **稳健性报告** | 输出时 | 同一结论在 ①Newey-West 修正 ②去极值 ③子样本分期 下是否稳定 | 报告里并列展示，不给单一数字 |

**第 4 条是最有说服力的**：先把新库调成"和 alphalens 一模一样"，证明引擎没写错；
再逐步打开 A 股规则，**每一处差异都能被逐条归因**（剔除涨停 X 条、停牌 Y 条……）。
解释不了的差异就是 bug。

**第 6 条直接回应当下的困惑**：本次 ROE 分析里，朴素 t=0.16、Newey-West 修正后 t=0.04 ——
你看不出哪个可信，因为库只给了一个数。新库默认**同时给两个**，并标注有效样本量。

---

## 1. 与 alphalens-reloaded 的功能对照

### 1.1 必须齐平（不能少）

| 能力 | alphalens-reloaded | alphalens-cna |
|---|---|---|
| 因子清洗 | `get_clean_factor` | ✅ |
| 前向收益 | `compute_forward_returns` | ✅（重写，见 §3） |
| 分层 | `quantize_factor`（quantiles/bins/zero_aware/by_group） | ✅ 全支持 |
| IC | `factor_information_coefficient` | ✅ |
| 分层收益 | `mean_return_by_quantile` | ✅ |
| 多空组合 | `factor_returns` / `factor_alpha_beta` | ✅ |
| 换手 | `quantile_turnover` | ✅ |
| 因子自相关 | `factor_rank_autocorrelation` | ✅ |
| IC 衰减 | `mean_information_coefficient` + decay | ✅ |
| 事件研究 | `create_event_returns_tear_sheet` | ✅ |
| 分组分析 | `groupby` / `by_group` | ✅（并升级为行业中性，见 §1.2） |
| pyfolio 对接 | `create_pyfolio_input` | ✅（可选） |
| 绘图 | `plotting`（20+ 函数） | ❌ **不做**，改为输出 tidy DataFrame |

### 1.2 必须更优（逐条对应实测缺陷）

| # | alphalens 的问题 | alphalens-cna 的做法 |
|---|---|---|
| D1 | 非日频因子索引直接崩（`freq` 校验） | 自持交易日历，**不依赖 pandas freq 推断** |
| D2 | 列名把"1个月"标成 "1D" | 持有期用**显式语义**（`1M`/`3M` 或 `21D`），由调用方声明 |
| D3 | `import alphalens` 必拉 seaborn/matplotlib | **核心零绘图依赖**，绘图完全外置 |
| D4 | 重叠观测无提示，只给朴素 t | 默认**同时给朴素 t 与 Newey-West t**，并给有效样本估计 |
| D5 | `dropna()` 一刀切，丢弃原因不可观测 | 输出**丢弃原因明细表**（停牌/涨跌停/新股/退市/无财报…） |
| D6 | 无涨跌停/停牌/T+1 概念 | **成交模型**内建（见 §3） |
| D7 | 生存者偏差（退市股消失） | 契约层面**强制股票池 as-of**，用当前池回测历史直接拒绝 |
| D8 | 有 groupby 无中性化 | **行业中性化内置**（吃申万事件态成份股） |
| **新增** | 无 | **隔夜跳空显式分离**（见 §3.2） |
| **新增** | 无 | **六道防线**（见 §0） |

---

## 2. 数据契约（A 股优先）

### 2.1 核心抽象的改变

alphalens 的抽象是：

```
(factor, prices) → forward_returns
```

**这个抽象装不下 A 股**，因为它没有"什么时候能成交"这个概念。改成：

```
(factor, prices, calendar, tradability) → 带成交标注的 forward_returns
```

### 2.2 三个输入对象

**① Factor（因子）**——必须携带"可知时点"

```
index: MultiIndex(date, asset)
values: float
attrs:
  available_at : 该值何时可知（公告日+1 交易日 / 收盘后 / 盘中）
  freq         : 'D' | 'W' | 'M'
```
> 若 `available_at` 晚于 `date`，契约校验直接拒绝 —— 这是**前视偏差的硬闸门**。
> 本次 ROE 分析用 `avail_314` 就是这个思想，应固化为库的要求而非使用者的自觉。

**② Prices（行情）**——**长表**，不是宽表

```
columns: [date, asset, open, high, low, close, prev_close, volume, amount, adj_factor]
```
> 为什么用长表而不是 alphalens 的宽表？
> ① 涨跌停判定需要 `high/low/prev_close`，宽表装不下多字段
> ② 避免 pivot/unpivot 的隐性错误（本次就踩过 `prices` 必须宽表、`factor` 必须长表的不一致）
> ③ 内存更省（可直接用 float32 分块）

**③ Tradability（可成交性）**——显式输入或由库推断

```
columns: [date, asset, suspended, limit_up_price, limit_down_price,
          open_at_limit, close_at_limit, is_st, listed_days]
```

### 2.3 关于 `limit_status` 的解码（实测，可直接用）

quantaxis 的 `stock_daily_basic.limit_status` 实测语义：

| 值 | 语义 | 证据（近 3 个月 40.9 万条） |
|---|---|---|
| 0 | **停牌** | p50 涨幅 0.00%，收 = 前收 |
| 1 | 上涨（未涨停） | p50 +1.73% |
| **2 / 3** | **涨停（收盘封板）** | p50 +10.00%（主板）/ +20.00%（创业/科创） |
| 4 | 下跌（未跌停） | p50 −1.92% |
| **5 / 6** | **跌停（收盘封板）** | p50 −10.00% / −20.00% |

> ⚠️ **两个重要提醒**：
> 1. **2 与 3、5 与 6 的区别尚未确定**（价格行为完全一致，可能是"一字板 vs 打开过"之类）。
>    本设计**不依赖该区分** —— 可成交性只需要"是否在板"，所以按 {2,3}→涨停、{5,6}→跌停 归并即可。
>    原始值仍原样透出，供将来拿到 data dictionary 后细化。
> 2. **这是收盘状态，不能判断次日开盘能否买入** —— 见 §3.3。

---

## 3. 成交模型（本项目最核心的设计）

### 3.1 A 股的真实时间线

```
T 日 15:00 收盘
   │
   ├─ 因子在 T 日收盘后生成（只能用 T 日及之前已公开的信息）
   │
   ├─ 隔夜跳空（T 收盘 → T+1 开盘）：★ 这段时间的收益你拿不到 ★
   │     因为信号刚生成，你还没买进去
   │
T+1 日 09:30 开盘 ← 最早可成交时点
   │
   └─ 持有 h 个交易日后卖出
```

### 3.2 隔夜跳空必须显式分离

alphalens 默认 `prices.pct_change(h).shift(-h)`，即 **T 日收盘买入** —— 对 T+1 市场**高估**。
quantaxis 现在的补丁是把面板整体 `.shift(-1)` 用次日开盘，**方向对了但不完整**。

本设计的口径：

```
成交价        = P_open(T+1)
前向收益      = P(卖出日开盘) / P_open(T+1) − 1
隔夜跳空      = P_open(T+1) / P_close(T) − 1        ← 单独输出一列, 不计入因子收益
```

**输出三列并列**：`forward_return`（可获取）、`overnight_gap`（不可获取）、
`total_return`（两者之和）。这样"因子到底赚的是哪一段"一目了然。

> 这一步能让很多"看起来有效"的因子现出原形 —— 如果收益主要来自隔夜跳空，
> 那它根本不可交易。

### 3.3 可成交判定（T+1 开盘能不能买）

`limit_status` 是**收盘**状态，不能直接用于开盘判定。必须自算：

```
涨停价 = round(prev_close × (1 + limit_ratio), 2)
跌停价 = round(prev_close × (1 − limit_ratio), 2)

买入不可成交 ⟺ 停牌
              ∨ open == 涨停价            (开盘即封板, 买不到)
              ∨ open ≥ 涨停价 − ε
              ∨ 上市未满 N 日             (可选, 默认 N=0)
```

`limit_ratio` 按板块判定：

| 板块 | 代码前缀 | 限制 |
|---|---|---|
| 沪深主板 | 60 / 00 | 10% |
| 创业板 | 300 / 301 | 20% |
| 科创板 | 688 | 20% |
| 北交所 | 8x / 4x | 30% |
| ST / *ST | 需 ST 标记 | 5% |

> **ST 识别的数据来源：已解决（原设计里标为"缺口"，现已确认有现成数据）**
>
> **集合**：`quantaxis.stock_basic_daily`（1072.9 万条，2016-08-09 ~ 2026-09-11）
> **来源**：tushare `bak_basic`（股票历史列表），由 `QASU/save_basic.py` 落库
> **字段**：`code, date, name, industry, area, pe, eps, total_assets, total_share, ts_code` …
> **索引**：`date_1_code_1`（唯一）+ `code_1_date_1` —— **双向都有索引**，
> PIT 单点查询实测 **0.004 秒**（对比 `stock_day` 没有 date 索引，按日查要 93 秒）
> **读取接口**：`QA_fetch_stock_basic(date, code, start, end, st=True)`
> —— 其中 `st=False` **直接剔除当日名称含 'ST'/'退' 的股票**，能力已经内建
>
> **关键：它记录了名称的完整变更历史**，因此可做 **PIT 正确的 ST 判定**。实测：
>
> ```
> 600539  2016-08-09  ST狮头      ← 被 ST
>         2017-04-17  狮头股份     ← 摘帽
>         2018-04-25  *ST狮头     ← 退市风险警示
>         2019-04-08  ST狮头
>         2021-05-19  狮头股份     ← 摘帽
> ```
> 用 `(code, date)` 查当日名称 → `*ST`/`ST`/`S*ST` → 得到当日正确的 ±5% 限制。
>
> **规模**：ST/*ST 占全市场 **2.3%（2018，81 只）~ 3.7%（2026，204 只）** —— 量足够大，
> 不处理会系统性高估这批股票的涨跌停判定。
>
> **实测取值分布**（近 3 个月全市场）：`*ST` 195,915 条 / `ST` 142,895 条 /
> `S*ST·SST` 851 条。
>
> ⚠️ 注意区分两类字段，别混用：
> - `stock_basic_daily.name` → **判断 ST**（决定涨跌幅限制是 5% 还是 10%/20%）
> - `stock_daily_basic.limit_status` → **判断当日是否在板**（收盘状态）

### 3.4 无法成交时怎么办（可配置策略）

```
policy = 'skip'      → 该样本标记为不可成交, 从分析中剔除 (默认, 最保守)
       = 'delay:N'   → 顺延到之后 N 日内第一个可成交日, 并输出延迟天数
       = 'mark_only' → 不剔除, 只加标记列 (供敏感性对比)
```

**默认 `skip`**，并在报告里明确写出剔除了多少条、占比多少 —— 让使用者知道
"这个结论是在多少可交易样本上得出的"。

---

## 4. 与 quantaxis 的集成设计

### 4.1 三档适配器（按数据来源）

```python
from alphalens_cna.adapters import (
    from_qa_datastruct,   # QA_DataStruct_Stock_day (MultiIndex date,code)
    from_mongo,           # quantaxis mongo: stock_day + stock_adj + stock_daily_basic
    from_quantming,       # quantming/data 下的 parquet (financial_pit / sw_industry)
)
```

**关键：适配器负责把 quantaxis 的约定翻译成契约**

| quantaxis 现状 | 契约要求 | 适配器做什么 |
|---|---|---|
| `stock_day` 未复权 | 复权价 | 连 `stock_adj` 计算 `close×adj` |
| 数值混有字符串 | 纯数值 | `pd.to_numeric(errors='coerce')` + 记录转换失败数 |
| 无 `date` 索引（按日期查 93 秒） | 高效取数 | 内部按 code 分块并行 + 本地 parquet 缓存 |
| `limit_status` 是收盘状态 | 开盘可成交性 | **自算**开盘涨跌停价（§3.3） |
| `stock_basic_daily.name` 有 ST 历史 | `is_st`（决定 5% 还是 10%/20%） | 按 `(code, date)` PIT 取名判 ST（§3.3） |
| 因子是 (date,code) 长表 | 同上 | 直接复用 |
| `financial_pit.avail_314` | `available_at` | 映射（这正是 PIT 语义） |

### 4.2 复用 quantming 已建好的数据层

| quantming 资产 | 在新库里的角色 |
|---|---|
| `financial_pit`（PIT 财务，25 因子） | 因子来源；其 `avail_314` 直接作为 `available_at` |
| `sw_industry/members`（事件态成份股） | **行业中性化**的分组输入（as-of 取成分） |
| `sw_industry/classify`（L1/L2/L3） | 分层统计的层级 |
| `momentum_cache`（月度面板约定） | 印证"月度调仓"是主用法，需原生支持 |

---

## 5. 工程结构（核心零绘图依赖）

```
alphalens_cna/
├── core/
│   ├── contract.py      输入契约与校验（available_at / 索引 / 股票池 as-of）
│   ├── calendar.py      交易日历（自持，不依赖 pandas freq）
│   ├── tradability.py   涨跌停/停牌/ST/新股 判定
│   ├── returns.py       成交模型与前向收益（含隔夜跳空分离）
│   ├── clean.py         清洗（带原因的账）
│   ├── quantize.py      分层（含行业中性分层）
│   └── metrics.py       IC / RankIC / ICIR / NW-t / 分层 / 换手 / 衰减
├── adapters/            quantaxis / mongo / quantming 适配
├── validate/            六道防线（体检 + 不变量 + 稳健性报告）
└── compat/              alphalens 输入输出适配（对拍用）
```

**绘图策略（按你的要求）**：核心**只产出 tidy DataFrame / dict**，
例如 `metrics.to_frame()`、`quantile_returns.to_frame()`，
任何绘图工具（matplotlib / plotly / 你自己 quantming 的 `plot_tools.py`）都能直接消费。
**不引入任何绘图依赖，也不内置画图函数。**

---

## 6. 开源协议选择

### 6.1 现状事实（已核实）

| 项目 | 协议 |
|---|---|
| **quantaxis** | **MIT**（`LICENSE`: The MIT License, Copyright 2016-2021 yutiansut/QUANTAXIS）|
| **alphalens-reloaded** | **Apache 2.0**（Copyright 2018 Quantopian, Inc.）|

### 6.2 三条候选

| 协议 | 优点 | 缺点 | 适合吗 |
|---|---|---|---|
| **MIT** | 极简、与 quantaxis 一致、无摩擦 | **无专利授权条款** | 可选 |
| **Apache 2.0** | 显式**专利授权**（§3）+ NOTICE 致谢机制 + 允许商业闭源使用 | 略长、要求标注修改 | **推荐** |
| **AGPL/GPL** | 强 copyleft，保护开源 | 机构用户基本会直接劝退（量化库多为闭源系统内使用） | ❌ 不推荐 |

### 6.3 推荐：**Apache 2.0**

理由：

1. **专利授权是刚需** —— 量化/算法领域专利风险真实存在，MIT 完全没有这一层保护。
2. **与 alphalens 生态一致** —— 万一将来需要移植 alphalens 的任何片段，
   不必换协议、不必处理双协议共存。
3. **商业友好** —— 允许闭源商用，机构才敢用（对比 AGPL 会被法务直接否掉）。
4. **NOTICE 机制** —— 可以在 NOTICE 里正式致谢 Quantopian / alphalens-reloaded / QUANTAXIS，
   既合规又体面。
5. **与 quantaxis 的 MIT 不冲突** —— MIT 项目依赖 Apache-2.0 库是完全正常的
   （现在 quantaxis 依赖 alphalens-reloaded 就是这个组合）。

### 6.4 但有一条硬约束

> **只要移植了 alphalens 的任何代码，就必须保留其版权声明、附上许可证副本、
> 并在被修改的文件里显著标注改动（Apache 2.0 §4b/§4c）。**

所以推荐路径是绞杀者模式的终态：**独立实现**（仅沿用 IC/分层这类数学定义，
不受版权保护），此时协议选择是自由的 —— 仍建议 Apache 2.0，理由同上。

---

## 7. 分阶段计划

| 阶段 | 内容 | 出口标准 |
|---|---|---|
| **P0 · 对拍基线** | `contract` + `returns`（先只做收盘成交，复刻 alphalens 口径）+ `metrics` + `compat` | **与 alphalens 数值一致（1e-10）**，复现本次 ROE 报告 |
| **P1 · A 股规则** | `tradability` + 成交模型（次日开盘 + 隔夜跳空分离）+ `clean` 带原因 | 逐条归因"与基线差了多少、为什么" |
| **P2 · 科学框架** | 六道防线（体检 / 不变量 / 稳健性报告）+ `adapters` | 任何一次分析都自动带体检与稳健性报告 |
| **P3 · 行业中性** | 接 `sw_industry` 事件态成份股，行业中性分层 | 中性化前后对比可一键产出 |
| **P4 · 开源** | 文档 + NOTICE + 协议 + PyPI | `pip install alphalens-cna` |

**P0 是最关键的一步**：它把"新库对不对"这个问题变成可判定的 ——
**先用新库复现旧库的结论，再谈改进。**

---

## 8. 明确不做（防止范围爆炸）

- ❌ **不做绘图**（按你的要求）—— 只输出数据，画图交给外部工具
- ❌ 不做回测引擎 / 撮合 / 资金管理（quantaxis / QARSBridge 的职责）
- ❌ 不做数据获取与存储（用现有 mongo + quantming parquet 层）
- ❌ 不做 Barra 风险模型（可后续以适配器形式接）
- ❌ **不重新定义 IC / 分层 / 换手** —— 沿用学术共识，保证结论与业界可比

---

## 9. 一句话

> **alphalens 的问题不是算错了，而是它假设了一个没有涨跌停、没有停牌、没有 T+1、
> 也不存在退市的市场。** 本设计的全部重心就是把这几件事显式建模，
> 并且**让每一步剔除都可见、可归因、可对账** —— 这样你才能知道分析到底对不对。
>
> 第一件事不是"功能更多"，而是**用新库复现旧库的结论**。
