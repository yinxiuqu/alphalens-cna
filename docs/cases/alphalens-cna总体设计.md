# alphalens-cna 总体设计

> 整合《因子投资：方法与实践》方法论 + A 股制度现实 + 可对拍性。
> 状态：**设计稿，未动手**。
> 本文取代此前的 `alphalens-cna详细设计.md`（那份只覆盖了 A 股制度层）。
>
> **修订记录**
> - v2：§3.3 三价并存契约（复权口径核查后新增）
> - v3：§6.3 FM 引擎改为自持内核 + `linearmodels` 作可选裁判
> - **v4：§7.4 新增三条推断维度 PFS / RRE / DH（借自 AlphaEval）；
>   Logic Score 与"合成单一评分"明确不采纳（§15）**

---

## 0. 设计原点：三个问题

整个架构不是"功能堆叠"，而是围绕三个问题长出来的：

| # | 问题 | 谁没回答 | 对应设计 |
|---|---|---|---|
| **1** | **算得对不对？** | alphalens 假设了一个没有涨跌停/停牌/T+1/退市的市场 | §4 成交模型、§3 三价契约 |
| **2** | **算得可信吗？** | alphalens 只给数字，不给"这个数字可不可信" | §7 **推断层**、§8 研究台账 |
| **3** | **别人凭什么信？** | 所有 A 股 alphalens 改造都没有可对拍性 | §10 六道防线 |

**第 2 条是本书带来的、也是本项目区别于市面上所有同类项目的核心。**

---

## 1. 五条设计原则

1. **推断与计算分离** —— 算 IC 是一个函数，判断 IC 可不可信是另一个阶段。**绝不在计算里偷偷下结论。**
2. **每一步剔除都可归因** —— 任何被丢掉的样本，都必须能回答"为什么丢的、丢了多少"。损失不可见 = bug 不可见。
3. **契约强制执行，不靠使用者自觉** —— 前视偏差、复权口径这类错误必须是**硬拒绝**，不是文档提醒。
4. **核心零绘图依赖** —— 只产出 tidy DataFrame；画图交给外部工具。
5. **纯函数 + 小型不可变对象** —— 每个阶段可单独对拍、单独测试。拒绝 god object。

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────┐
│  L6  报告层    tidy frames / Markdown        零绘图依赖       │
├─────────────────────────────────────────────────────────────┤
│  L5  推断层 ★  多重检验 / NW / Shanken / 有效样本 / 稳定性    │  ← 书的灵魂
│                + PFS扰动 / RRE排名熵 / DH多样性 (§7.4)       │
│                结论收口 Verdict                              │
├─────────────────────────────────────────────────────────────┤
│  L4  分析层    分层 / 双重排序 / IC / FM回归 / 事件 / 换手     │
├─────────────────────────────────────────────────────────────┤
│  L3  预处理层  去极值 / 标准化 / 中性化 / 正交化 / 合成        │
├─────────────────────────────────────────────────────────────┤
│  L2  引擎层    可成交性判定 / 成交模型 / 带原因的清洗          │
├─────────────────────────────────────────────────────────────┤
│  L1  体检层    复权连续性 / 停牌率 / 涨跌停率 / 覆盖率         │
├─────────────────────────────────────────────────────────────┤
│  L0  契约层 ★  FactorPanel / PricePanel / Tradability /       │
│                Calendar / Universe / Grouping                │
├─────────────────────────────────────────────────────────────┤
│      适配层    quantaxis / quantming / alphalens(对拍)        │
└─────────────────────────────────────────────────────────────┘
        ▲                                        ▲
        │           研究台账 Ledger ★             │
        └────── 记录每一次假设检验 → n_trials ─────┘
```

**数据自下而上流，推断自上而下收口。**

---

## 3. L0 契约层

### 3.1 为什么契约是 L0 而不是工具函数

本次实测踩的坑全部源于"契约太弱"：

| 踩过的坑 | 根因 |
|---|---|
| 用了 `avail_314` 但靠自觉 | 契约没有 `available_at` |
| 前复权价算涨停价差 23% | 契约没区分"制度价"与"收益价" |
| `prices` 必须宽表、`factor` 必须长表 | 契约不对称 |
| 非日频索引直接崩 | 契约隐含"日频"假设 |

→ **契约层是 L0，不是 utils。**

### 3.2 六个契约对象

```python
FactorPanel      # 因子值 + available_at（何时可知）+ freq
PricePanel       # 长表 OHLCV + prev_close + adj_factor
Tradability      # 可成交性：停牌/涨跌停/ST/上市天数
Calendar         # 自持交易日历（不依赖 pandas freq 推断）
Universe         # as-of 股票池（防生存者偏差）
Grouping         # 行业/分组标签（as-of 事件态）
```

### 3.3 ★ 三价并存（v3 修订的核心）

```python
PricePanel.columns = [
    'date', 'asset',
    'raw_open', 'raw_high', 'raw_low', 'raw_close',   # 原始不复权
    'prev_close',                                      # 原始前收
    'adj_factor',                                      # 累计后复权因子（唯一真相）
    'adj_open', 'adj_high', 'adj_low', 'adj_close',   # 复权价（由前两者导出）
    'volume', 'amount',
]
```

**三条硬规则，写进契约校验：**

1. **制度判定**（涨跌停价、报价单位、ST）**只允许读 `raw_*`**
2. **收益计算只允许读 `adj_*`**
3. **只存复权价、不存原始价 + 因子 ⇒ 契约校验直接拒绝**

> **为什么**：实测 600519 在 2016-08-26 的涨停价，
> 用原始价算是 **333.31**（正确），用当时的前复权价算是 **270.01** —— **差 23%**。
> v1 设计只写了 `round(prev_close × 1.1, 2)`，**没规定 `prev_close` 用哪种价**。
> 实现者顺手用复权面板，涨跌停判定就全线静默失效。

**口径**：统一用**后复权**（`adj_factor` 存累计后复权因子）。
后复权价格水平与真实价同量纲，且**不随新数据回溯改变** —— 两个独立来源（外部文章、`zer0factor` 的 `adjust="hfq"`）都指向它。

### 3.4 契约校验（防线 1）

启动即拒绝，不产出任何数字：

| 校验 | 拒绝条件 |
|---|---|
| **前视** | `available_at > date` |
| 索引 | 重复 `(date, asset)` |
| 价格 | `raw_* <= 0` 或 `adj_* <= 0` |
| **复权完整** | 有 `adj_*` 却无 `raw_*` 或 `adj_factor` |
| 股票池 | 用当前成分回测历史（须 as-of） |
| 交易日历 | `date` 不在 `Calendar` 上 |
| 时区 | 索引 tz-aware 混用 |

### 3.5 输入从哪来：适配器分层

**核心原则：核心包不认识任何具体数据源，只认契约对象。**

这条对开源项目是硬要求 —— 库不能绑死在任何人的私有目录结构上。

```
adapters/input/
├── base.py                 InputAdapter 协议 —— 核心只认这个
├── dataframe.py         ★  直接吃 DataFrame（零依赖，默认路径）
├── parquet.py              读【公开目录约定】
└── optional/               可选依赖，装了才用
    ├── quantaxis.py        quantaxis 公开 mongo schema（MIT 项目）
    ├── tushare.py          pip install alphalens-cna[tushare]
    ├── baostock.py
    └── akshare.py
```

| 层 | 认识什么 | 谁能用 |
|---|---|---|
| **核心** | 只认契约对象 | 所有人（数据从哪来都行） |
| **通用适配器** | 公开目录约定 | 照约定摆数据的人 |
| **可选适配器** | 公开数据源 | 装了对应 extras 的人 |
| **❌ 绝不认识** | **任何人的私有目录** | — |

**公开目录约定**（`parquet.py` 认这个）：

```
mydata/
├── prices.parquet      date, asset, raw_open/high/low/close,
│                       prev_close, adj_factor, volume, amount
├── xdxr.parquet        date, asset, category, fenhong, peigu, ...
└── factors/
    └── <name>.parquet  date, asset, value, available_at
```

**私有数据源怎么接**：写一个适配器放 `examples/`，**不进核心包**。
库更新时只要协议不变，私有适配器一行不用改；库里也**永远不会出现私有目录的名字**。

**默认路径是 `dataframe.py`** —— 零依赖、零假设，最通用。

### 3.6 数据放在哪（实现时天天要查）

| 数据 | 位置 | 性质 |
|---|---|---|
| 日线行情 | mongo `stock_day` | quantaxis 公开 schema |
| 复权 | mongo `stock_adj` | quantaxis 公开 schema |
| 除权事件 | mongo `stock_xdxr` | quantaxis 公开 schema |
| 每日指标（市值/换手/涨跌停） | mongo `stock_daily_basic` | 私有扩展 |
| **PIT 名称（ST 判定）** | mongo **`stock_namechange`** | 私有扩展 · **事件式区间** |
| 上市 / 退市日 | mongo `stock_basic` | 私有扩展 |
| 业绩预告 | mongo `stock_forecast` | 私有扩展 |
| 交易日历 | mongo `index_day`（上证指数） | 公开 schema |
| PIT 财务（因子源） | parquet `quantming/data/financial_pit` | **私有布局** |
| 申万行业（分组） | parquet `quantming/data/sw_industry` | **私有布局** |
| 面板缓存 | workspace `cache/*.parquet` | 本库自产 |

> **公开 schema 的可以内置适配器；私有的走 `examples/`。**
> 本表随调查更新 —— 它回答"这个数从哪读"，是接入时第一个要查的东西。

#### ⚠️ ST 判定数据源已更换（2026-09）

旧方案用 `stock_basic_daily`（每日名称快照）—— **只有 2016-08 起**，
2016 年前无法做 PIT 的 ST 判定，会让早期回测的涨跌停判定整体失真。

现改用 **`stock_namechange`（事件式名称区间）**：

```
ts_code    start_date  end_date    name       change_reason
600539.SH  20110630    20140707    ST狮头      ST
600539.SH  20150421    20160510    *ST狮头     *ST
600539.SH  20190408    20210518    ST狮头      撤消*ST并实行ST
```

查某日名称：找 `start_date ≤ d ≤ (end_date or 至今)` 的那条。

**实测覆盖 99.8%–100%，跨 1990–2026 全年代**：

| 日期 | 覆盖率 | ST 占比 |
|---|---|---|
| 1995-04-03 | 100.0% | 0%（当时尚未实行 ST 制度 ✓） |
| 2005-01-04 | 99.8% | 5.5% |
| 2010-01-04 | 100.0% | 6.5% |
| 2020-01-02 | 100.0% | 4.5% |
| 2026-01-05 | 100.0% | 4.5% |

→ **2016 的时间断层已消除，ST 判定可用于全历史回测。**

---

## 4. L2 引擎层：A 股成交模型

### 4.1 真实时间线

```
T 日 15:00 收盘
   ├─ 因子生成（只能用 T 日及之前已公开信息）
   ├─ ★ 隔夜跳空（T 收 → T+1 开）：这段收益你拿不到 ★
T+1 09:30 开盘 ← 最早可成交时点
   └─ 持有 h 个交易日后卖出
```

### 4.2 前向收益输出**三列并列**

```
forward_return   = P_adj(卖出日开盘) / P_adj(T+1 开盘) − 1     ← 可获取
overnight_gap    = P_adj(T+1 开盘) / P_adj(T 收盘) − 1          ← 不可获取
total_return     = forward_return + overnight_gap 的复合
```

> 这一步能让很多"看起来有效"的因子现出原形。
> 本次实测：**隔夜跳空 RankIC 0.0038 (t=0.59)** —— 它不构成主要收益来源。

### 4.3 可成交判定

```python
# limit_ratio 按板块 + ST 判定
主板 60/00 → 10%    创业板 300/301 → 20%
科创板 688 → 20%    北交所 8x/4x → 30%    ST/*ST → 5%

涨停价 = round(raw_prev_close × (1 + ratio), 2)   # ★ 必须 raw
买入不可成交 ⟺ 停牌 ∨ raw_open ≥ 涨停价 − ε ∨ 上市未满 N 日
```

**ST 数据源已解决**：`quantaxis.stock_basic_daily`（1072.9 万条，PIT 名称历史，
单点查询 0.004 秒）。ST/*ST 占全市场 **2.3%（2018）~ 3.7%（2026）**。
**新股默认剔除上市前 60 个交易日**。

### 4.4 无法成交的策略（可配置）

```
'skip'      剔除并计入原因账（默认，最保守）
'delay:N'   顺延到之后 N 日内首个可成交日，输出延迟天数
'mark_only' 不剔除，只加标记列（供敏感性对比）
```

### 4.5 ★ 带原因的账

```python
CleanResult(
    data=...,                  # 清洗后的面板
    ledger=DropLedger(         # 每一类剔除：条数 + 占比 + 明细索引
        suspended=...,
        limit_up_buy=...,
        limit_down_sell=...,
        new_stock=...,
        st=...,
        no_factor=...,
        no_price=...,
        delisted=...,
    ),
)
assert len(input) == len(output) + ledger.total    # 防线 3 的不变量
```

---

## 5. L3 预处理层

### 5.1 标准化流水线（可组合、可配置、可跳过）

```
去极值 → 标准化 → 中性化 → 正交化 → 合成
```

| 阶段 | 方法 |
|---|---|
| 去极值 | MAD(n=3) / 分位截断 / 不处理 |
| 标准化 | z-score / rank / 不处理 |
| **中性化** | 对**行业哑变量 + ln市值** 做截面回归取残差 |
| **正交化** | **对称正交**（顺序无关）—— 施密特正交结果依赖变量顺序，书里特别强调 |
| 合成 | 等权 / IC 加权 / **最大 ICIR** |

**中性化顺序（采纳外部建议）**：在分析**之前**完成，用截面回归取残差，
而非在已中性化的因子上再做回归。

---

## 6. L4 分析层

### 6.1 分层：一个可组合的原语

```python
quantize(factor, n=5, by=None, method='independent')
```

| 用法 | `by` | `method` | 得到 |
|---|---|---|---|
| 单变量分层 | `None` | — | 全样本 5 层 |
| **行业中性分层** | industry | `'independent'` | 行业内分层，再聚合 |
| **独立双重排序** | size | `'independent'` | 5×5 宫格，**每行内部独立分位** |
| **条件双重排序** | size | `'conditional'` | 先按 size 分组，**组内**再分位 |

> **为什么双重排序必要**：本次 FM 证明 **ROE 与规模相关 0.360**，
> 且 ROE 的负向是规模混淆出来的 —— 但这**只靠参数方法（回归）支撑**。
> 双重排序是**非参数交叉验证**。**可证伪预测**：
> 若"ROE 负向=规模混淆"成立，则**同一规模组内** ROE 单调性应消失。

### 6.2 IC 系列

单期 IC / RankIC、ICIR、**IC 衰减曲线（多 horizon）**、**因子自相关**、**换手**。

> alphalens 的**清洗阶段就会崩**（D9 真相：`utils.py:358` 硬写
> `index.levels[0].freq`，真实 A 股月频索引 `freq is None` → ValueError），
> 不只是 `factor_rank_autocorrelation` / `quantile_turnover` 返 NaN。
> 自算值：秩自相关 **0.9227**，月换手 **20.03%**。详见 `outputs/换手口径与D9核实.md`。

### 6.3 Fama-MacBeth 截面回归

**两阶段**：逐期截面 OLS 得 γₜ → 时序均值 + 协方差。

**引擎决策：自持内核，不依赖第三方包。**

| 做法 | 评价 |
|---|---|
| ❌ `fama_macbeth` 包 | **PyPI 上 404**（两个拼写都没有）；2 stars；README 让你"拷单个 .py"；作者自称 *"rather naive"* |
| 🟡 `linearmodels` 当引擎 | 专业且在维护，但 **Shanken 仍要自写**，且对拍时是黑盒 |
| ✅ **自持内核 + statsmodels** | 已有跑通实现（约 40 行）；无数学难点；**可读可测可对拍** |
| ✅ **`linearmodels` 作可选裁判** | 独立实现交叉验证，`extras[validate]` |

> 这个依赖是**负债不是资产**：项目卖点是可对拍性（1e-10），
> 底层换成第三方包后，**对拍失败时进不去它内部定位**。

**输出**：逐期 γ 序列、逐期 R²、逐期样本数、γ 的均值与 t 值。

### 6.4 事件研究

`ann_314` 等公告日 → 事件窗口 → 市场调整后异常收益。

> 本次实测：财报信息在 **T+1 日内被完全定价**（第 +1 日见顶 +1.848%，之后反转），
> **无 PEAD**。这个结论直接决定：**财务因子要靠事件驱动，不是周期调仓**。

---

## 7. L5 推断层 ★ 本书的灵魂

### 7.1 为什么必须独立成层

alphalens 给出 `IC = 0.004`，但**不告诉你这个数可不可信**。
本次实测：同一个 ROE，朴素 t = 0.16，Newey-West 后 t = 0.04 —— **差 4 倍**。
库里只给一个数，使用者无从判断。

**设计原则：每个统计量都自带不确定性标注。**

```python
Estimate(
    name          = 'RankIC',
    value         = -0.0108,
    t_naive       = -0.68,
    t_nw          = -0.64,      # 重叠观测修正
    n_obs         = 91,         # 名义样本
    n_effective   = 7,          # ★ 有效样本（重叠调整后）
    p_raw         = 0.497,
    p_adjusted    = None,       # 多重检验后
)
```

### 7.2 四类修正

| 修正 | 解决什么 | 本项目证据 |
|---|---|---|
| **Newey-West** | 重叠观测导致 t 虚高 | ROE 0.16→0.04（**3.7×**）；动量 −21.93→−6.76（3.2×） |
| **有效样本量** | 名义 91 个月，12 月口径下**有效独立样本仅约 7** | 决定性：91 和 7 是两个世界 |
| **Shanken** | 因子相关时 FM 标准误低估 | ROE↔规模 **0.360**，不是可选项 |
| **多重假设检验** | 测多了必然出假阳性 | 见 §7.3 |

### 7.3 多重假设检验 —— 本项目最有说服力的演示

用本次 FM 四因子真实数据：

| 因子 | t_NW | 原始 p | BH | **BHY** | 结论 |
|---|---|---|---|---|---|
| roe_z | +0.177 | 0.8595 | 0.9188 | 1.0000 | ✗ |
| pb_z | −0.102 | 0.9188 | 0.9188 | 1.0000 | ✗ |
| **ln_mv_z** | **−3.687** | 0.0002 | 0.0009 | **0.0019** | ✓ |
| **turnover_rate_z** | **−3.257** | 0.0011 | 0.0023 | **0.0047** | ✓ |

**但只要"总共测过几个"变了，结论就翻转：**

| 若共测试 | Bonferroni 阈值 | 换手因子 |
|---|---|---|
| 4 个（本次） | 2.498 | 存活 |
| 20 个 | 3.023 | 存活 |
| **100 个** | **3.481** | **掉出去** |
| 316 个（HLZ 统计的已发表因子数） | 3.778 | 掉出去 |

> **同一因子、同一数据，只因"你一共测了多少个"，显著性就没了。**

**要实现**：Bonferroni / Holm / BH / **BHY**；输出校正后 p 值与有效 t 阈值；
**强制记录 `n_trials`**；Harvey-Liu-Zhu 式 t > 3.0 基准线。

### 7.4 三条补充维度（借自 AlphaEval）

来源：AlphaEval，arXiv 2508.13174（因子挖掘模型评测框架）。
它从五个维度评估因子，其中三条**本设计原本没有**，且**计算便宜、不依赖回测**，故纳入推断层。

> **只借维度，不借它的"合成单一评分"。**
> 合成分数适合给模型排序；判断因子真假时必须能看出**是哪个维度挂了**（见 §1 原则 1）。
> 它的第五维 Logic Score（GPT-4o 打分）**明确不采纳**，理由见本节末。

#### ① PFS · 扰动鲁棒性（Perturbation Fidelity Score）

**测什么**：因子对数据微小扰动敏不敏感。排名一变就说明因子不稳。

```python
noise  = N(0, sigma)                      # 高斯
noisy  = factor * (1.0 + noise)           # ★ 乘性 —— 与因子量纲无关
PFS    = corr(factor, noisy) 按日算后取均值
```

**照搬它的两个好设计**：

1. **乘性噪声**（`× (1+noise)` 而非 `+noise`）⇒ 尺度不变，
   同一个 σ 对不同量纲的因子含义一致。
2. **σ 标定到真实市场波动**，而非拍脑袋：
   ```python
   close_norm = (hs300_close - min) / (max - min)   # 归一化到 [0,1]
   sigma      = sqrt(var(close_norm, ddof=1))       # 实测约 0.15–0.22
   ```
   再补一组 **t 分布(dof=3)** 捕捉厚尾。

**落点**：`inference/robustness.py` · **进 `Verdict`**（这是"这个因子稳不稳"的直接回答）

#### ② RRE · 时序稳定性（Relative Rank Entropy）

**测什么**：横截面排名分布随时间重排的程度。

```python
probs = ranks.div(ranks.sum(axis=1), axis=0)          # 每日排名 → 概率分布
kl    = (probs * log(probs / probs.shift(1))).sum(1)  # 相邻日 KL 散度
RRE   = (1 / (1 + kl)).mean()                          # 映射到 (0,1]，越高越稳
```

**与已有维度的分工**：

| 指标 | 看什么 |
|---|---|
| 因子秩自相关（本次实测 0.9261） | **值的延续性** |
| **RRE** | **排名分布的重排程度** |

两者互补：秩自相关高但 RRE 低，说明整体序稳定、但结构在换手。

> AlphaEval 论文中验证：**RRE 与换手率负相关** —— 稳定性高的因子交易频率更低。
> 这条与 §6.2 的换手率互为交叉验证。

**落点**：`analysis/ic.py`（与秩自相关并列）

#### ③ DH · 多样性熵（Diversity Entropy）

**测什么**：一组因子之间是否冗余。

```python
eigs = eigenvalues(cov(factor_matrix))     # 协方差特征值
p    = eigs / eigs.sum()
DH   = -(p * log(p)).sum() / log(n_factors)   # ★ 除以 log(n) 归一化
```

**除 `log(n_factors)` 这个归一化是关键** —— 让不同规模的因子集可比。
本设计原本只有"相关性矩阵 + 有效独立因子数"，这条更干净。

**落点**：`analysis/` 因子集合层 · **接 §8 研究台账**做冗余检测
（台账里已记录每个测过的因子，正好可直接算 DH）

#### 明确不采纳：Logic Score

AlphaEval 用 GPT-4o 给因子表达式打"金融逻辑性"分。**不进核心**，理由两条：

1. **它的 prompt 本身混入了长度偏好** —— 原文要求
   *"We also prefer longer factors, as this aligns with the goal of automated search."*
   于是分数同时反映"表达式长度"，**不再是纯粹的金融逻辑度量**。
2. **不可复现** —— LLM 打分（temperature=0.2 仍有随机性）与 §10 防线 4
   "与 alphalens 1e-10 一致"的可对拍原则正面冲突。

> 若将来确实需要，只能作为 `extras[llm]` 可选插件，并**强制标注"非可复现"**。

---

## 8. ★ 研究台账 Ledger：让 `n_trials` 变成测量值

**这是本设计里最有新意的一环。**

多重检验的前提是"知道测了多少次"。但这个数通常靠回忆 —— 而人的回忆系统性偏低。

**设计**：一个持久化的研究台账，每次检验自动登记。

```python
ledger = Ledger('~/.alphalens_cna/ledger.parquet')

ledger.record(
    factor='roe_pit', spec_hash='a3f9...', universe='all_a',
    horizon=21, result=Estimate(...),
)

ledger.n_trials(family='financial_factors')   # → 真实测过的次数
verdict = assess(stats, ledger=ledger, family='financial_factors')
```

**台账带来的三件事**：

1. **`n_trials` 从猜测变成测量** —— 校正才有意义
2. **防止"换个参数再试一次"** —— 每次重试都被计数，这不是纪律问题而是机制问题
3. **可审计的研究历史** —— 半年后能回答"这个因子我当时测过几版"

> 这直接落地了书里那句话：**因子投资不是找因子，而是验证因子。**

---

## 9. L6 报告层

### 9.1 零绘图依赖

核心只产出 tidy DataFrame / dict：

```python
report.ic_series.to_frame()
report.quantile_returns.to_frame()
report.estimates.to_frame()
report.ledger.to_frame()
```

任何绘图工具（matplotlib / plotly / 你自己的 `plot_tools.py`）可直接消费。
**不引入任何绘图依赖，也不内置画图函数。**

### 9.2 `Verdict`：结论收口

```python
Verdict(
    significant     = True,
    p_adjusted      = 0.0047,
    n_trials        = 4,
    threshold_used  = 2.498,      # |t| 阈值
    effective_n     = 7,
    stability       = 0.71,       # 子样本成立比例
    net_return      = 0.0612,     # 扣成本后
    tradable_ratio  = 0.8687,     # 可交易样本占比（13.28% 空格的另一面）
    # §7.4 三条补充维度
    pfs             = 0.9821,     # 扰动鲁棒性（乘性噪声 + 真实波动标定 σ）
    rre             = 0.8412,     # 排名熵稳定性（相邻日 KL 散度）
    dh              = None,       # 多样性熵 —— 仅在评估因子集合时有值
    warnings        = ['有效样本仅 7，统计功效低'],
)
```

> **alphalens 告诉你"因子 IC 是 0.004"；
> alphalens-cna 要告诉你"这个 0.004 可不可信"。**

---

## 10. 六道防线（落到具体接口）

| # | 防线 | 落点 | 失败表现 |
|---|---|---|---|
| **1** | 契约校验 | `contract.validate()` | **拒绝运行**，不产出数字 |
| **2** | 数据体检 | `health.check()` | 超阈值告警 + 明细 |
| **3** | 不变量对账 | `CleanResult.__post_init__` | **抛异常**（条数对不上） |
| **4** | **等价性回归** | `compat/` + CI | 与 alphalens **1e-10** 不一致 ⇒ CI 红 |
| **5** | 已知答案测试 | `tests/` 手算数据集 | 3 股 × 2 日含涨停/停牌 |
| **6** | 稳健性报告 | `inference.stability()` | 并列展示，不给单一数字 |

**第 4 条最有说服力**：先把新库调成"和 alphalens 一模一样"，证明引擎没写错；
再逐步打开 A 股规则，**每一处差异都能被逐条归因**。解释不了的差异就是 bug。

**关键：本会话已有的 6 份报告 + 21 张图 + 5 份 JSON 就是现成的回归基线**，
不必另造测试数据。

---

## 11. 模块结构

```
alphalens_cna/
├── contract/        ★ L0 契约（panels / calendar / validate）
├── adapters/input   ★ 适配器分层（见 §3.5）
│                      base / dataframe / parquet / optional/*
├── health/            数据体检
├── engine/            tradability / returns / clean
├── preprocess/        winsorize / standardize / neutralize
│                      / orthogonal / combine
├── analysis/          quantile / ic / portfolio / regression / event
│                      / diversity（§7.4 DH 多样性熵）
├── inference/       ★ newey_west / shanken / multiplicity
│                      / effective_n / deflated / verdict
│                      / robustness（§7.4 PFS 扰动鲁棒性）
│                      / rank_entropy（§7.4 RRE 排名熵）
├── ledger/          ★ 研究台账
├── report/            tidy frames / markdown（零绘图）
└── compat/            alphalens 等价性回归
```

---

## 12. 三档公开 API（服务三类使用者）

**L1 一行式 —— 日常**
```python
res = acna.quick_report(factor=roe, prices=px, spec=Spec(...))
res.verdict.significant     # 校正后是否显著
```

**L2 显式流水线 —— 研究**
```python
clean = acna.clean_factor(factor, prices, tradability, universe)
fwd   = acna.forward_returns(prices, calendar, tradability, spec.return_model)
prep  = acna.preprocess(clean, spec.preprocess)

stats = [acna.ic(prep, fwd),
         acna.quantile_stats(prep, fwd, n=5),
         acna.double_sort(prep, fwd, by=ln_mv),
         acna.fama_macbeth(prep, fwd, controls=[ln_mv, turnover])]

verdict = acna.assess(stats, ledger=ledger, family='financial')
```

**L3 单函数 —— 对拍与测试**
```python
acna.inference.bhy(pvalues)
acna.engine.tradability.limit_price(raw_prev_close, board, is_st)
```

---

## 13. 配置与可复现

```python
spec = Spec(
    adjust          = 'hfq',
    return_model    = ReturnModel(entry='next_open', policy='skip', new_stock_days=60),
    preprocess      = Preprocess(winsorize=('mad', 3.0), standardize='zscore',
                                 neutralize=('industry', 'ln_mv')),
    quantiles       = 5,
    horizons        = [21, 63, 126, 252],
    inference       = Inference(nw_lags=3, shanken=True, multiplicity='bhy'),
)
```

**所有选择集中在一个 `Spec`，并哈希进结果**（`result.spec_hash`）——
半年后能回答"这个数字是在什么口径下算出来的"。

---

## 14. 依赖策略

| 类型 | 依赖 | 理由 |
|---|---|---|
| **核心** | pandas / numpy | 不可避 |
| 核心 | **statsmodels** | OLS + HAC（NW），FM 底层 |
| 可选 `[validate]` | **linearmodels** | 独立裁判，交叉验证 FM |
| 可选 `[io]` | pyarrow | parquet |
| **绝不** | matplotlib / seaborn | 核心零绘图依赖（修 D3） |

---

## 15. 明确不做

- ❌ **不做绘图**（核心零绘图依赖）
- ❌ 不做回测引擎 / 撮合 / 资金管理（quantaxis / QARSBridge 的职责）
- ❌ 不做数据获取与存储（用现有 mongo + quantming parquet 层）
- ❌ 不做 Barra 风险模型（可后续以适配器形式接）
- ❌ **不重新定义 IC / 分层 / 换手** —— 沿用学术共识，保证与业界可比
- ❌ 不做因子挖掘（遗传算法等）—— 那是 `FactorHub` 的赛道，不是"验证内核"
- ❌ **不做 LLM 逻辑打分**（AlphaEval 的 Logic Score）—— 不可复现，与防线 4 冲突；
  且其 prompt 混入长度偏好，测的不纯是逻辑。详见 §7.4
- ❌ **不合成单一总分** —— 可给模型排序，不可用于判断因子真假（§1 原则 1）

---

## 16. 分期落地（对齐 `项目实施路线图.md`）

| 阶段 | 交付层 | 出口标准 |
|---|---|---|
| **P0** | L0 契约 + L2 收盘口径 + L4 基础分析 + `compat/` | **与 alphalens 1e-10 一致**，复现 6 份报告 |
| **P1** | L2 完整成交模型（次日开盘 + 隔夜跳空 + ST/新股） | **逐条归因**：打开每条规则后差异全部可解释 |
| **P2** | **L5 推断层** + L1 体检 + 防线 | 任何分析自动带体检 + 校正后 p 值 + `n_trials` |
| **P2.5** | **§7.4 三条补充维度**（PFS 扰动 / RRE 排名熵 / DH 多样性） | 三条指标可算且进 `Verdict`；DH 接台账做冗余检测 |
| **P3** | L3 完整预处理 + L4 双重排序 / FM / Shanken | 结果与本次手工实现一致；可证伪预测可执行 |
| **P4** | **L6 台账** + 稳健性 + 成本 | 一键产出结论稳定性评分与净收益 |
| **P5** | 适配器 + quantaxis 集成 + 开源 | `pip install alphalens-cna`，CI 全绿 |

**P0 是最关键的一步**：它把"新库对不对"从**信仰问题**变成**可判定问题**。

---

## 17. 一句话

> **算得对**（§4 A 股制度正确）→ **算得可信**（§7 推断层）→ **别人能验证**（§10 可对拍）。
>
> 前两条是内容，第三条是信任。**没有第三条，前两条没人敢用。**
>
> 而 §8 的研究台账是这套设计里最小、也最关键的一块 ——
> 它把"我测了多少次"从**人的记忆**变成**可测量的数字**。
> 没有它，多重检验无从谈起；有了它，**"换个参数再试一次"就不再是纪律问题，而是机制问题**。
