# 更新日志

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 待办
- 分组 IC（`grouped_ic` / `group_consistency`）—— 触发条件见 `outputs/功能增补清单`
- 退市收益约定的行业维度复核

## [0.4.2] - 2026-09-29

### 修复
- **`is_oos` 的 purge 改成按日历精确剔除**（修 0.4.0 引入的**单位错配**）。
  `split_is_oos` 的 `embargo` 数的是"分析日期**期数**"，而 `build_report` 用
  **交易日数**（`horizons`）算它 —— 日频同量纲，**月末调仓差约 21 倍**：
  实测月末 60 期面板 `horizons=(21,63)` 时默认 `embargo=64` → **样本内 0 期、直接跑不起来**；
  `(21,)` 时挖掉 22 个月末（本意 22 个交易日）。
  现在默认改为**按日历精确 purge**：只剔掉"前向收益窗口碰到切分日之后"的样本内样本
  （实测同一面板 purge 1 / 2 / 5 个，三种 horizons 全部跑通）；`entry_lag` 跟
  `ReturnModel.entry` 走；显式 `embargo=N` 仍走老口径（尊重显式意图），两者同时给则报错。
  并在报错里给出**下界**与**可行 ratio 区间**（"我到底需要多长样本"直接有答案）。
- **两条报错改成指向下一步**：
  * `double_sort` / `grouped_ic` 缺 `forward_return_*` 时不再抛到下游 `ic` 层
    （`ic/no_returns`，容易被误读成"没跑 forward_returns"），改为入口报
    `group/missing_returns` 并给两步处方（`forward_returns` → `clean` → 调用）；
  * `clean` 的 `ambiguous_factor` 补上"**本库一次分析一个因子**"与多因子入口指引
    （`factor_correlation` / `redundancy_check`）。
- **澄清一个易误判的依赖问题**（实测为**误报**，故只改文档、不加代码）：
  `Report.to_markdown()` / `HealthReport.to_markdown()` **自带渲染、零可选依赖** ——
  在屏蔽 `tabulate` 的环境里实测正常（6099 / 2071 字符）。需要 `tabulate` 的是
  **pandas 的** `DataFrame.to_markdown()`。README 与输入规格已写明，避免使用者去装
  一个本库不需要的包。

### 新增
- **`Report.sections()` + `SECTION_KEYS`** —— 报告各节的**稳定键**
  （`verdict` / `health` / `ledger` / `ic` / `newey_west` / `quantile` / `portfolio` /
  `stability` / `tail` / `turnover` / `is_oos` / `dropped`）。0.4.0 起节号是自动编号、
  会随版本变化（加「组合绩效」后稳定性从 `## 七` 挪到 `## 八`），下游若按节号解析
  Markdown 会**静默取空**。现在按 key 定位即可；`to_markdown` 文档也加了"勿按节号解析"。
- 7 个契约类的 docstring 各补一句：**入口层（`build_report` / `check_parity`）会补校验，
  传裸 `DataFrame` 也会被自动包装** —— 不必自己记着构造契约对象。

## [0.4.1] - 2026-09-28

### 修复
发布后自检（专门找测试覆盖不到的地方）发现三处，其中两处是 0.4.0 引入的：

- **`is_oos` 的默认 `embargo` 少一天**（`max(horizons)` → `max(horizons) + 1`）。
  `entry='next_open'` 下样本 t 的前向收益覆盖 `[t+1, t+1+h]`，所以 `t+1+h` 要
  **严格早于**切分日。实测 960 行面板上：`embargo = max(horizons)` 时样本内**最后一个**
  样本的 h=21 出场日**正好落在切分日**上 —— 读到样本外价格，而这个功能的全部意义
  就是不让它读到。回归测试按日历验算（不是断言魔数），切片算法一变就会失败。
- **多列 `groupby` 被误拒（回归）**：0.4.0 把 `groupby` 一律按 `Grouping` 校验，
  而 `Grouping` 要求"恰好一列"→ 0.3.0 能跑的"多列控制帧（不中性化）"被拒了。
  现改为：单列（或名为 `group`）按 `Grouping` 校验；**多列按控制帧**校验
  （形状 + 有限性）并**留痕**；真要中性化时 `neutralize` 会自己报
  `preprocess/ambiguous_group`（可执行报错）。
  （顺带更正一条我写错的注释：多列控制帧的作用是"分组值缺失就剔行"，
  那些列**不会**进 `CleanResult.data` —— 0.3.0 也一样。）
- **`Grouping(df, validate=False)` + 空表时 `group_col` 漏裸 `IndexError`** ——
  现在报 `Grouping/empty`（`group_col` 是 0.4.0 新加的属性，这是它的边界）。

## [0.4.0] - 2026-09-28

### 破坏性变更
- **契约层的三处补口**（用户反馈，已逐条实测）：
  * **`±inf` 现在在每个契约上都拦得住**：`_check_finite` 原先只遍历 `numeric` 声明的列，
    而 `Exposures` / `Universe` / `Grouping` / `Events` 的 `numeric` 是空的 ——
    即**一个列都不查**。实测：把 `inf` 塞进 `exposures`，0.3.0 会一路跑通，
    且结果与把 `inf` 换成 `NaN` **逐位相同**（非法值被静默当成缺失，用户拿不到任何信号）。
    现在检查 `numeric` ∪ **所有数值 dtype 的列**。
  * **`build_report` 的可选契约入参一并进边界**：`exposures` / `groupby` 以前不进
    `ensure_contract`，形状不对时漏的是**裸异常**（实测 `groupby='industry'` →
    `AttributeError: 'str' object has no attribute 'iloc'`；空表 → `IndexError`），
    且错误指向下游 `neutralize` 那一层。现在在入口报 `validate/type`、`Grouping/empty`、
    `Grouping/ambiguous_group`。
  * **`Grouping` 放宽为「恰好一列」**：不再强制列名 `group`（行业表叫 `industry` /
    `sw_l1` 都是常见写法）。同时 `preprocess.neutralize(groups=…)` 改为同口径
    （有 `group` 用它，否则取唯一那一列）—— 契约与消费方**必须一起改**，
    否则放宽契约会造出 `KeyError` 的新 bug。
  * **`Tradability` 收紧**：至少要有一列成交信号（`can_buy_open` / `can_sell_open` /
    `suspended` / `is_st` / `listed_days`）。此前"什么表都能通过"，一张无关列的表会
    一路穿到 `forward_returns` 才报错。
- **`double_sort` 换实现（旧的三行包装已从 `analysis/quantile.py` 删除）**：
  原先那个包装返回 `quantile_returns` 的立方，现由 `analysis/group.py` 的新实现取代 —— 后者给
  **组内单调性**、**缺格逐格记账**、`n_by`（控制变量层数）。名字不变，但**返回类型变了**
  （`DataFrame` → `DoubleSortResult`），默认 `method` 也从 `'conditional'` 改为
  `'independent'`。（仓库内无使用者；设计稿里的调用点已同步。）

### 新增
- **`analysis/correlation.py`：因子相关性 / 冗余度**（补上评估流程的第 6 步）
  * `factor_correlation(factors, *, method='spearman', min_overlap=20)` ——
    **逐期截面相关再对时间汇总**（mean / median / positive_rate / 期数），
    **不做整体池化相关**（池化会混入共同的时间漂移、系统性高估，把互补因子误判成冗余）；
    重叠期数不足的对给 NaN 并**写明原因**。
  * `redundancy_check(candidate, library, *, threshold=0.7, …)` —— 判定"是否与库内因子冗余"，
    给结论 + 最相关的几个因子 + 阈值依据（与 `Verdict` 同风格）。
- **`analysis/group.py`：分组 IC / 组内一致性 / 双重排序**（补上待办里的分组口径）
  * `grouped_ic(data, *, by, …)` —— 逐期**组内**截面 IC → 时间汇总；组内样本不足的期
    **记账**而不是静默跳过。
  * `group_consistency(grouped, …)` —— 各组 IC 的同号比例与离散度，回答
    "因子是不是只在某一组里有效"。
  * `double_sort(data, *, by, n_by=5, n=5, method='independent')` —— `by` × `q` 宫格平均收益
    + **组内**单调性（`monotonicity`）与池化单调性（`pooled_monotonicity`）分开给；
    缺格是 NaN 并逐格记原因，**不填 0**。
  * 三个函数共用同一条骨架（按列分组 → 逐期截面统计 → 时间汇总），不另写一套口径。
- **`analysis/split.py`：`split_is_oos`** —— 样本内 / 样本外拆分，补上评估流程第 5 步：
  * ⚠️ **只按时间顺序切**（时序数据随机切等于没切）；
  * ⚠️ `embargo` / purge：切分点前挖掉若干期 —— 那些样本的前向收益**跨过切分点**，
    留着就是把样本外的价格泄漏进样本内；
  * 两段各不足 `min_periods`（默认 8）期会**报错**，而不是给一段空洞的统计。
- **`build_report` 两个新入参**：
  * `tradability=` —— 自带可成交性表（给了就不再自动算，且先在边界校验）；
  * `is_oos=` / `embargo=` —— 要样本内外对照时给（比例或**切分日**；不给就不做，
    切分是研究决策，库不替你定）。
- **报告两节**（节次编号改成**自动生成**，以后加节不必再手改编号）：
  * 「组合绩效（统计口径，非可交易净值）」—— 年化 / 波动 / 最大回撤 / 夏普 / 胜率，
    按观测频率年化（相邻日期间隔中位数，月频调仓不会被当成年频的错）；
    节内**明写不是可交易净值、未扣成本** —— 不能让读者把它当回测曲线。
  * 「样本内 / 样本外对照」—— 两段的 IC 均值 / ICIR / IC 胜率 / 期数 / 单调性并排，
    并写明切分日与挖掉了几期。
- `Report.portfolio` / `Report.is_oos`；`frames()` 新增 `portfolio` / `is_oos_is` / `is_oos_oos`
  三张表（`save(kind='frames')` 从 14 张变 17 张；**新增而非改名**）。

### 修复
- **`portfolio_summary` 在累计净值为非正时开分数次方** → `NaN` + `RuntimeWarning`
  （`(-0.3) ** 0.4` 落到复数域）。这条路径 0.4.0 起会被 `build_report` **默认**走到，
  所以不能留一个往 stdout 喷警告的静默 NaN；现在判成"无法年化"并在报告该节写明原因。
- `align_event_windows` 的事件表**进契约**：以前只做 `_as_df`，缺 `event_type` 要拖到下游；
  现在在入口按 `Events` 报 `missing_columns`。
- `build_report` 不做事件研究这件事**写进文档**（它没有 `events` 入参；事件走
  `align_event_windows`）—— 加了入参反而会让人以为报告里有对应的一节。

## [0.3.0] - 2026-09-28

### 破坏性变更
- **一条龙入口补上防线 1** —— `build_report` / `check_parity` 现在默认先做契约校验
  （`validate=True`），以前能跑通的**坏数据**现在会被**拒绝运行**：
  前视（`available_at > date`）、因子日期不在交易日历、`raw_*` 有值而 `adj_*` 缺行、
  价格含 `±inf`、factor 的 `(date, asset)` 在 prices 里找不到、
  代码带交易所后缀导致资产无交集。

  **为什么这是对的**：这些数据算出来的数字本身就是错的，**静默放行才是 bug**。
  此前防线 1 靠"构造即校验"，只在走 loader 或用户显式构造面板时生效；
  一条龙入口把裸 DataFrame 解包后直接往下传，契约层从未被调用 ——
  上述坏数据能生成一份看着完全正常的报告。

  逃生门 `validate=False`（报告首屏会写明防线 1 已关闭）。
  另：`validate_inputs(..., grouping=…, exposures=…, tradability=…, events=…)`
  这几个参数以前是**收了但不用**（传什么都行），现在会真的包装并校验 ——
  例如 `grouping` 现在必须带 `group` 列，否则报 `missing_columns`。
  ⚠️ 因子缺 `available_at` 时按 `= date` 合成（与 loader 既有约定一致），
  但**不许静默**：这个事实会进 `Report.contract['notices']` 并在报告首屏印出
  "前视检查因此未生效"。

### 新增
- `contract.validate.ensure_contract()` —— 入口层统一契约校验：把裸 DataFrame
  包成契约对象 + 跨表校验，并**返回包装好的对象**
  （`factor` / `prices` / `checked` / `cross` / `notices` …），供一条龙入口复用，
  免得二次包装。`validate_inputs()` 成为它的薄封装（仍返回 `True`，向后兼容）。
- `Report.contract` —— 防线 1 的收据：`{'enabled','strict','checked','cross','notices'}`，
  报告 Markdown **首屏**打印一行（正常 / 缺 `available_at` / 校验已关闭 三种形态）。

### 修复
- **防线 1 的入口缺口**：契约层从未在一条龙入口被调用（见上），
  前视、非交易日、复权缺行、`+inf`、因子多出行五类坏数据在 `build_report` 下静默跑通。
- `validate_inputs()` 现在也接受裸 `DatetimeIndex` 当 `calendar`
  （原先抛 `AttributeError: 'DatetimeIndex' object has no attribute 'index'`）。
- **跨表检查性能**：原先 `set(index.get_level_values(...).unique())` 在 960,000 行
  面板上代价很高，现改为整数 code 去重（`bincount`）——
  两项集合检查从 1.1 s 降到毫秒级，且唯一值只算一次、两处检查共用。

## [0.2.0] - 2026-09-25

### 破坏性变更
- **契约补上缺失校验**：`raw_*` 有值的行上，`adj_*` 或 `adj_factor` 缺行现在会被
  `PricePanel` 拒绝（`adjust_incomplete`）。
  此前自洽校验的判据是 `diff > tol`，而 `NaN > tol` 恒为 False —— 缺行被"自洽"放行，
  一路穿到体检层就成了上面那条「复权价连续」的假 all-clear。
  停牌（raw 与 adj **两边都缺**）不受影响，仍按输入规格允许。
  ⚠️ 会拒绝以前能跑的面板，属破坏性变更：按约定进 0.2.0，不走 0.1.x patch。

### 新增
- **`ReturnModel.exit_policy` —— 出场侧可成交性**（默认 `'assume'`，**向后兼容**）。
  此前 `can_sell_open` 只出现在体检报告里，收益计算**完全不用它**：出场日一字跌停
  照样按当日价成交 —— 而设计文档自己写着"涨跌停意味着次日大概率买不进/**卖不出**，
  把这段收益算进去就是策略高估"。

  * `'assume'`（默认）—— 保持旧口径。已与已发布的 0.1.7 做过 A/B：收益、台账、
    parity（`0.0`）、verdict **全部逐位相同**，报告只差下面那条文案。
  * `'drop'` —— 该行剔除，计入原因账 `exit_not_sellable`。
  * `'delay'` —— 顺延到 `max_delay` 个交易日内第一个**可卖且有价**的日子，
    实际持有期因此长于 `h`；顺延不到则剔除，计入 `exit_blocked`。

  ⚠️ 非 `'assume'` 时**必须**给 `tradability`，否则明确报错
  （`exit_policy_needs_tradability`），不静默降级。
  ⚠️ **退市行不归它管**：退市按 `delist_policy` 清算 —— "股票没了"与"今天卖不掉"
  是两件事。实现中实测到两者互撞会把 `delist_policy='last_price'` 整个推翻
  （tradable 12 → 7），已用 `exempt` 隔离并加回归测试。
  台账键的文档同步说清分两类：**剔除原因**（⚠️ **不构成严格划分**，同一行可能被
  入场侧与出场侧各记一次，故 `Σ >= dropped`；恒成立的是 `total − tradable`）
  与**处理说明**（`delist_filled` / `exit_delayed` —— 那些行仍在样本里）。

### 修复
- 体检报告的「可卖 X%」补上限定语「（体检指标，是否门控收益见 exit_policy）」——
  不加这句会被读成"收益里已经扣掉了卖不掉的那些天"。实测 A/B：整份报告**只有
  这一行**变化，其余逐字节相同。

- **`calendar` 参数在不同入口的接受度不一致**（连带 4 个入口崩溃）。
  `forward_returns` / `compute_tradability` / `build_report` / `check_parity` 只认
  `Calendar` 对象，传裸 `DatetimeIndex` 会抛
  `AttributeError: 'DatetimeIndex' object has no attribute 'index'`；
  而 `health_check` / `align_event_windows` 一直两种都收（库里另有三处写着
  `getattr(calendar, 'index', calendar)`）。口径统一到
  `contract.calendar.as_calendar`，两处崩溃点（`_shift_dates`、
  `compute_tradability` 的 `validate_dates`）修好 —— 实测两种输入的前向收益、
  可成交性、parity 结果与**整份报告**完全一致。
- **契约补上 ±inf 检查**：`_check_positive` 判的是 `<= 0`，负 inf 会被拦、
  **正的 inf 会直接穿过去**，然后在收益 / 相关系数 / 方差里算出 NaN 或 inf。
  新增 `_check_finite`（**NaN 仍是合法缺失** —— 停牌照旧放行，只有 inf 被拒）。
- 文档：`Verdict.cost` 的注释说清「毛 → 净（成本）」**不是减法** ——
  毛是 `Π(1+r)−1`、净是 `Π(1+r−c)−1`（逐期扣成本后复利），
  `净 ≈ 毛 − 成本` 只在一阶近似下成立。
- 新增 `tests/test_invariants.py`：把 7 条恒等式固化下来 —— 台账对账
  （输入 = 输出 + Σ剔除）、前向收益的**已知答案**（`adj_close(t+1+h)/adj_open(t+1)−1`）、
  `total = (1+fwd)(1+gap) − 1`、`cost_bps=0 ⇒ net == gross`、报告确定性（同输入逐字节相同）、
  存盘 frames round-trip、hfq/qfq 收益等价。

- **loader 的构造参数没转发**（文档教的用法必崩）。`resolve()` 调
  `get_source(name)` 时漏传了 `**kw`，而注册表 `get_source(name, **kw)`
  本来就收 —— 于是 `acna.load_prices(source='parquet', root='~/mydata')`
  （模块 docstring 与 `no_source` 报错文案里都这么写）必定
  `TypeError: missing 'root'`，且 `root` 还被 `**kw` 塞给了 `.prices()`。
  现在明确路由：**字符串/类**源的 `**kw` 是**构造参数**；**实例**源的 `**kw`**
  交给它的 `prices()` / `factor()`。五个 loader（prices/factor/calendar/universe/
  tradability）一并修好。
- **体检层缺列/空面板的裸异常**：`health_check(factor=…)` 在列名不是 `value` 时抛
  `KeyError: 'value'`，`rank_stability` 同样；现在给 `factor_column` 契约错误并
  说明改成什么名字。`check_factor_panel` 的"因子冻结"分支在**空面板**上
  `int(run.max())` 抛 `ValueError: cannot convert float NaN to integer`，现在报
  `skip: 无因子观测`。
- **顶层少导出 `EventWindows`**：它和兄弟（`FMResult`/`Verdict`/`DSRResult`…）
  都在 `from .analysis import (...)` 里，却只有它没进 `__all__` ——
  `import *` 拿不到。
- **`docs/输入数据规格.md` 的"跑一次分析"示例根本跑不通**：它写的是设计稿里的
  目标 API（`acna.analyze` / `acna.Spec`），库从未实现；且那块 5 天 × 3 只的
  示意数据价格不随时间变、没有收益，真跑会报"样本不足以判断"。已换成
  `build_report` 的自足可运行示例（已在文档上逐块执行验证）。
- 清掉三处死代码：`PricePanel.validate_extra` 两个**永不执行**的成对性分支
  （其中一句还谎称"只给 raw_* 让库自算"，实测会被 `missing_columns` 拒绝）、
  `analysis/event.py` 的 `idx_ret`、`compat` 的 `common`，以及
  `inference/__init__.py` 的重复导入。

## [0.1.7] - 2026-09-25

### 修复
- **极端涨跌的归因：把判反的两支调回来，并把"缺失"单列一档**（非破坏性）。
  `adj = raw × adj_factor` —— 除权日因子抬升、复权价才连续，所以「复权后正常」是
  **因子已覆盖**的签名，不是缺陷。原实现却把它报成硬错误（"adj_factor 没盖住除权日"），
  反而把真正可疑的"复权后也大"只记为告警；在 2016-2022（高送转密集期）上 400 只样本
  一次报出 38 条，其中 31 条经 `stock_xdxr` 核对全部是 `category==1` 的正常除权除息日。

  随这次修正一并解决评测里暴露的三个连带问题：

  - **`pass` 不再隐去未覆盖的条数**：占比 > 80% 时汇总行会补一句
    「另有 N 条复权后仍大/复权价缺失 —— 见下方明细」。
  - **复权价缺失不再算作"已覆盖"**：`ar` 是**前值填充**后算的，缺失行会得到 0% 收益，
    此前被并进 `explained`，一旦占比过 80% 就输出 `pass`＋"复权价连续" —— **假 all-clear**。
    现在缺失单列一档，且**缺失不为 0 就不给 `pass`**（"没检验"与"检验通过"必须分开）。
  - **`metric` 与文案首数对齐**：warn 支此前报的是"未覆盖数"，汇总行却以总数开头，
    报告里渲染成「13 条 … [11]」。现在两支统一为**总触发数**。

  明细表新增 `问题` 列（`已覆盖（因子已跟上）` / `复权后仍大` / `复权价缺失`），
  读者不必自己回头数。函数 docstring 与 `docs/cases/体检层报告.md`（含 `outputs/`
  同名拷贝）里同样判反的说法一并改正。

## [0.1.6] - 2026-09-24

### 修复
- **零剔除时报告渲染直接崩**（0.1.0 ~ 0.1.5 全都有；确定性可复现，非偶发）。
  `DropLedger.to_frame()` 在 `counts` 为空时返回 `(0, 0)` —— 连列名都没有，
  于是 `Report.to_markdown()` 里的 `.set_index('reason')` 抛
  `KeyError: "None of ['reason'] are in the columns"`，**整份报告出不来**。

  触发条件是"一次清洗一条都没剔"：因子截面落在价格日历**内部**、
  没有涨跌停、没有新股即可（走公开入口 `build_report` 就能撞到，不必手工搭对象）。
  真实研究几乎总在面板边缘剔掉点东西（`no_return`），所以藏了六个版本。

  修两层：
  - **根因**：`to_frame()` 列名钉死；零剔除时也补「—— 保留 ——」那一行
    （「0 剔除、全部保留」本身就是结论，值得单独占一行）。
  - **渲染层**：加守卫 —— 真拿到空表就渲染"台账为空"，不再甩异常；
    零剔除时在报告里写明「零剔除」。

  影响面（同一次零剔除）：`to_markdown()` / `save(kind='markdown')` 直接失败；
  `frames()['ledger']` 与 `save(kind='frames')` 的 ledger 是 `(0, 0)` 空表。
  两个回归测试（根因层 + 渲染层，含两条存盘路）对 0.1.5 实测会失败。

## [0.1.5] - 2026-09-24

### 修复
- **pandas 3.0 兼容**（全量回归里冒出的 2 类弃用告警；**数值零变化**，已逐位核对）：
  - `pct_change()` 的默认前值填充，涉及 4 个调用点（`health/prices.py`、
    `analysis/event.py` 两处、`contract/validate.py`）。改为自己 `ffill` 后手写
    `s / s.shift(1) - 1`：既不告警，也不依赖 `fill_method` 关键字 ——
    该关键字 pandas 3.0 已移除，`event.py` 里原有的 `pct_change(fill_method=None)`
    届时会直接 `TypeError`（本次一并拆掉）。
    ⚠️ 别照抄"写成 `fill_method=None`"：那会**改变语义**（不复权归因要的正是跨停牌区间比）。
  - 对象列 `fillna` 的向下转型告警（`engine/clean.py` 的 universe / exposures 两处）。
    ⚠️ 也**不能**照抄提示里的 `result.infer_objects(copy=False)` —— 告警由 `fillna`
    自己发出，追加它一条都不会少（实测）。改用 `.where(notna(), 填充值)`。
  - 清掉 `build/lib/`、`.pypi_test/`、`.pylibs/alphalens_cna` 三份整包副本：
    它们含旧代码，会让 `grep` 扫出假命中，也会在"不在仓库根跑"时被静默用上。

- **`复权连续性` 的 fail 不再吞掉"跳变"**（全量回归暴露的连带问题）。
  该检查里 `n_down` 优先 `return`，于是 3 处存储舍入造成的假"倒退"把 3 处真实的
  2 倍跳变盖成了沉默项 —— 报告读起来像"只有复权问题"。
  现在 fail 的文案会补一句"另有 N 处单日跳变 > 2 倍"（`metric` 仍是 `n_down`，
  硬错误数不被跳变稀释）；`detail` 里本来就带着两批行，汇总文案现在也说全了。

## [0.1.4] - 2026-09-24

### 修复
- **多重检验的缺失 p 现在分两个错误码**。此前不管「全部 NaN」还是「部分 NaN」，抛的都是
  `multiplicity/nan_p`；文案已经改成指向「上游样本不足」，错误码却没跟着换，程序化调用者
  拿 `err.rule` 分不出该走哪条路（观感/可编程性问题，不影响结论正确性）。现在拆成：
  `no_valid_p`（**全部** p 是 NaN/Inf —— 没有任何可校正的检验，该回去查数据或缩短 horizons）、
  `partial_nan_p`（**部分** p 是 NaN —— 提示里给出两种处理，并强调剔除后 `n_trials`
  仍按全部假设数计）。
  ⚠️ 破坏性变更：匹配旧 `nan_p` 的 `except` 分支需同步改。

## [0.1.3] - 2026-09-24

### Fixed
- **极短样本仍漏出下游报错**（用户反馈：第 7 项"只修了一半"）——
  2~3 期时抛的是 `multiplicity/nan_p: p 值里有 NaN/Inf`，
  提示让人"剔除算不出 p 的检验"，而真正该做的是**缩短持有期**；用户拿到这条信息
  仍然不知道问题在哪。另外 `ic_summary(pd.DataFrame())` 直接漏
  `KeyError: "None of ['horizon'] are in the columns"`。
  修法：在 **`assess` 入口加前置判断**（IC 无列 / 全 NaN → 明确报"样本不足以判断"
  并列出三种常见原因 + 台账排查路径）；`ic_summary` 加同样的守卫；
  `multiplicity` 的 `nan_p` 提示补上"最可能的原因不是 p 值本身，而是上游样本不足"。
  新增 `tests/test_short_sample_errors.py`：钉住不变量 ——
  **极短样本要么跑通，要么报错必须指向根因，绝不允许 KeyError / NaN-Inf 漏出**。

- **同一份报告里「持有期」列有三种叫法** —— `horizon`（四、IC / 六、分层）、
  `index`（五、Newey-West，因为 `newey_west_summary` 的索引没命名，
  pandas 一 `reset_index()` 就暴露成默认名）、`h`（七、稳定性 / 八、尾部）。
  同一种东西三个名字，读者每换一节都要重新对表头。
  修法：渲染层统一成 **`h`**（一处改动覆盖全部节），
  源头给 `newey_west_summary` 的索引命名，
  并在「怎么读这份报告」里补一行图例解释 `h`。
  顺带补上第 6 项漏掉的第八节第二张表（`tail_by_quantile`）的双 `reset_index`。

- **`ledger` 表无法写成 parquet** —— 它的 `sample` 列装 Python 元组，
  整张表触发 `ArrowTypeError`，于是 `save(kind='frames')` 存出
  "14 张 parquet + 1 张 csv" 的**格式混杂**目录。
  `sample` 本来就是给人看的字符串，改为在 `DropLedger.to_frame()` 里 `str()` 化。
  修后 15 张表**全部 parquet、零降级**，报告渲染不变。
  （这一条是上一项 `save_report` 报出来的 —— 修之前它是静默降级，根本看不见。）

- **剔除明细的 `sample` 列直接渲染 Python repr** —— 公众号报告里出现
  `[(Timestamp('2023-01-31 00:00:00'), '000000'), ...]`。虽能看，但对读者不友好。
  新增 `_fmt_sample()` 排成人话：`2023-01-31 000000; 2023-02-28 000001 …（共 6 条）`
  （最多列 3 条，超出标注总数；非列表原样返回）。
  （纯排版问题，改动之前就存在。）

## [0.1.2] - 2026-09-24

### Fixed（10 个缺陷：3 个用户反馈 + 3 个从产物反查 + 2 个端到端回测 + 2 个同类扫描）
- **`rolling_ic` 顶层导不出** —— 名字写进了 `__all__` 却漏了 import 块。
  后果分两层：`acna.rolling_ic` 报 AttributeError；
  **`from alphalens_cna import *` 直接抛异常** —— 星号导入的 notebook 一升级就炸。
  审计 88 项 `__all__`，只有这一个取不到。
  ★ 顺带补上一道**自省式防线**测试：`__all__` 里每个名字都必须真的取得到。
  325 个测试当初一个都没抓到，说明缺的就是这类测试。
- **`__version__` 硬编码 `'0.1.0.dev0'`** —— 发行元数据说 0.1.1、运行时自报 0.1.0.dev0。
  改为 `importlib.metadata.version('alphalens-cna')` + 取不到时回退。
- **复权连续性假阳性** —— `health/prices.py` 用绝对容差 `d < -1e-12`，
  而 `adj_factor` 常以 10 位小数存储，舍入本身就有 ~1e-10，
  于是纯舍入被**判成硬错误**，让整个体检结论不可信。
  改为相对容差（`ADJ_REL_TOL = 1e-8`）。
  （同一类问题早前在 `engine/adjust.py` 的 `check_adjust_agreement` 修过，这里漏了一处。）
- **第七节在短样本上整节失效** —— `subsample_stability` 要求 `n_splits × min_obs = 20`、
  `decay_test` 要求 `n >= 20`，而 14 期月频面板是常见规模，于是整节渲染成全破折号。
  门槛降到 5 / 8（2 段各 7 期、回归 2 参数 14 点，本来就够算）。
  ★ 更关键：**算不出来时 `decaying` 改为 `None`**（渲染成"—"），不再返回 `False`。
  `False` 的含义是"检验过、没衰减"，而那是"根本没检验" ——
  把未检验报成已检验且通过，正是本库一路在抓的那类错误。
- **`auto_lags` 的 `horizon-1` 下限在短序列上撑爆** —— 14 期 + h=63 → 下限 62 阶
  → 被截到 `T-2=12`，等于用 14 个点估 12 阶自协方差，由此算出的 `vif`/`n_eff` 全是垃圾。
  加比例上限 `lags ≤ max(1, n//4)`（14 期 → 3 阶）。样本足够时行为不变
  （`auto_lags(91,21)` 仍是 20）。
- **第七、八节表格多出 `index` 列** —— 调用处已 `reset_index` 过，
  `_md_table` 内部又 reset 一次，于是多出一列 0/1 与 `h` 重复。改为 `index=False`。

- **空样本抛 pandas 原始异常** —— 持有期超过样本跨度时（如月频价格却要 63 个
  *交易日*的前向收益），清洗后一条不剩，`information_coefficient` 抛的是
  `ValueError: If using all scalar values, you must pass an index` —— 用户完全看不懂。
  改为 `ContractError` 并说明原因与排查方法（看 `clean()` 台账 / 缩短 horizons）。
- **设计文档承诺的 L2 流水线 API 没导出** —— 文档写 `acna.forward_returns(...)`，
  实际必须写 `from alphalens_cna.engine.returns import forward_returns`。
  `forward_returns` / `clean` / `ReturnModel` / `compute_tradability` /
  `compute_adj_factor` 五个核心函数**都不在 `__all__` 里**。
  根因：`engine/__init__.py` 当时只有一行 docstring、没有任何再导出。
  另补齐 11 个"可取到但漏在 `__all__` 外"的名字（事件研究 / 尾部统计）。
  ★ 并新增 `tests/test_api_surface.py`：按**设计文档承诺的清单**逐项断言。
  这类问题用"查 `__all__` 是否可取到"的方法**查不出来** —— 用户审计了全部
  88 项只发现 1 个，而这 5 个核心函数一个都没被看到。

- **DSR 算不出来时静默省略整节** —— `build_report` 里 `except Exception: dsr_res = None`，
  报告那一节直接消失，用户分不清"不用算"和"算失败"。
  而实测退化序列抛的是明确可解释的 `ContractError: 收益序列标准差为 0`。
  改为：失败原因写进 `Report.dsr_note` + `Verdict.warnings`，
  报告显示「**未计算** —— <原因>」。
  （这一条是"再扫一遍"扫出来的第 9 个缺陷，与前面 8 个同族：声明与实际不一致。）

- **`Report.save()` 静默降级格式** —— 优先写 parquet，没 pyarrow 时
  `except Exception: to_csv`，**格式悄悄变了调用方不知道**。
  这个兜底本身是必要的（parquet 引擎不在本库依赖里），错的是它不留痕。
  改为：实际格式与降级原因记在 `Report.save_report`
  （`{'formats': {...}, 'downgraded': [(名字, 原因)]}`）。
  顺带修一个真 bug：`save(kind='markdown')` 在父目录不存在时抛
  `FileNotFoundError`，现在会先建目录；`kind` 非法值也改为报 ContractError。


## [0.1.1] - 2026-09-24

### Added（补齐两个洞）
- **`inference/deflated.py`：DSR 紧缩夏普比率** —— 补齐"多重检验只治了一半"：
  `multiplicity` 治 t 值，**DSR 治"挑组合挑出来的夏普"**。含
  `psr` / `dsr` / `expected_max_sharpe` / `min_track_record_length`。
  N 可直接取自 `ResearchLedger.n_trials()` —— 这正是多数人算不出 DSR 的原因。
  实测效果：年化夏普 0.99、PSR=0.827 的策略，在"挑过 2850 个组合"下
  **DSR=0.005**（纯噪声下期望最大夏普 0.2308，比它还高）。
- **`inference/stability.py`：因子失效监控** —— `subsample_stability`
  （连续子段与全样本同号的比例）+ `decay_test`（衰减斜率，Newey-West 标准误）。
  **接上了此前悬空的 `Verdict.stability`** —— 该字段一直存在、报告也会渲染成
  "稳定性 X% 子样本成立"，但全库没有任何地方计算它，永远显示 NaN。
- **`analysis.rolling_ic()`** —— 跨**时间**的滚动 IC。
  （与跨**持有期**的 `ic_decay` 是两件事，docstring 已写明区别。）
- 报告新增「七、因子衰减与稳定性」一节（共十节）；
  `Report.stability_detail` / `Report.dsr`；`frames()` 新增 `stability` / `dsr`。

### Fixed
- `decay_test` 的 HAC 标准误最初写成 `S / sxx`，正确是 `T·S / sxx²`
  （`X'X` 对角化后 `Var(β)=Ŝ₂₂/sxx²`，而 `nw_variance` 返回的是已除 T−1 的**平均**量）。
  自检方式：同方差下必须退化为经典 `σ²/Σtc²` —— 修前 t=−0.50、修后 −37.28，
  与经典公式差 3.9%（那 3.9% 正是自相关修正）。
- 稳定性表改为百分比显示：`1.0` 原先被渲染成 `1.0000`。


## [0.1.0] - 2026-09-22

首个版本。M0（能用）/ M1（可信）完成，M2 进行中。

**已在 PyPI 发布**：`pip install alphalens-cna`（2026-09-23）
· wheel 139 KB + sdist 158 KB · Apache-2.0 · 依赖仅 numpy/pandas/scipy

### Added — 核心
- **契约层**：`FactorPanel` / `PricePanel` / `Tradability` / `Universe` / `Grouping` /
  `Exposures` / `Events` / `Calendar` 八个契约对象；自维护交易日历，
  **不依赖 pandas 的 `freq` 推断**（这正是 alphalens 在月频上崩掉的原因）
- **适配器**：`dataframe` / `parquet` 两个通用输入适配器；核心包不认识任何具体数据源
- **引擎**：复权因子计算（与 QUANTAXIS 逐位一致）、A 股可成交性判定
  （涨跌停/ST/停牌/新股/T+1）、收益模型（含**退市收益约定**）、带原因的清洗
- **分析**：IC / 分层 / 双重排序 / 换手 / 多空组合 / 截面回归 / Fama-MacBeth /
  事件研究 / **尾部统计（VaR / CVaR / 崩盘命中率）**
- **推断**：Newey-West（含 `vif ≥ 1` 保守下限）/ 多重检验（Bonferroni/Holm/BH/BHY）/
  有效样本量 / **排序熵稳定性 RRE** / **扰动鲁棒性 PFS**
- **预处理**：去极值 / 标准化 / 中性化 / 正交化 / 合成
- **数据体检**：10 项（复权连续性 / 存活偏差 / 覆盖率 / OHLC / 因子冻结 …）
- **研究台账**：`n_trials` 自动记账，可抓"低报校正基数"
- **报告**：tidy 表 + Markdown 九节，**零绘图依赖**
- **对拍**：`compat.check_parity()` —— 退化配置下与 alphalens 三个核心量
  **逐位相同（0.000e+00）**，作为 CI 常驻 job

### Changed — 三次结论自我修正（都保留了证据）
- **换手口径**：默认从"对称口径"改为 **alphalens 单边口径**，以守住"可替换"的承诺；
  对称口径保留为 `method='symmetric'`
- **存活偏差**：修正了"补上退市股会加强信号"的推测 —— 实测**反而弱 20%**，
  因为长持有期的崩盘落在窗口外（退市股最后 3 个调仓日的 h=252 实现率仅 14%）
- **ROE 结论**：市值中性化后 RankIC **符号翻转**（−0.027 → +0.029），
  原先的"ROE 是负向因子"实为**规模效应**；而崩盘率结论在行业+市值双重中性后
  反而更强（t = −8.28）

### Fixed — 实测抓到的缺陷
- `compute_adj_factor` 的 `reindex` 会丢掉非交易日除权日（改用 `concat` 取并集）
- `compute_adj_factor` 索引层级错位导致下游 `groupby` 分组错乱
- `returns` 的 `entry='close'` 实际用的是开盘价（名字与行为不符）
- `quantize` 条件双重排序在 `groupby.apply` 下静默产出全 NaN
- Newey-West 的 `γ₀` 与 `sd²` 分母不一致，导致"无自相关时 vif=1"失守；
  且短样本**系统性低估方差**（T=30 时 vif 均值 0.898）→ 加 `vif ≥ 1` 下限
- `t_naive` 与 `t_nw` 走两条方差计算路径，在不同 BLAS 上差 1 ULP
  （本地"靠运气通过"，CI 暴露）→ 统一源头
- `pct_change()` 默认前值填充把停牌伪装成 0 收益 —— 两处语义不同，分别处理：
  事件研究用 `fill_method=None`，体检的极端涨跌改用"上一个有效价"作分母
- RRE 第一版按**排名**分桶，导致直方图恒为均匀、RRE 恒等于 1（什么都测不出）
  → 改用排名转移矩阵对角线
- `roe == 0` 是占位而非真值（净资产为负时 gocw 记 0）→ 一律视为缺失

### Security
- `aws.env` / `.aws-article/` / `drafts/` 加入 `.gitignore`（仓库为 public）
- 移除全部私有目录引用：核心包 0 处，复现脚本改用 `ALPHALENS_DATA_ROOT` 环境变量
