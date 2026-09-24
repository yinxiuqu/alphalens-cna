# 更新日志

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Fixed（修掉 6 个缺陷：3 个来自用户反馈，3 个从产物反查）
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

### 待办
- 分组 IC（`grouped_ic` / `group_consistency`）—— 触发条件见 `outputs/功能增补清单`
- 退市收益约定的行业维度复核

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
