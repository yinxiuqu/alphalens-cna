# 更新日志

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 待办
- 分组 IC（`grouped_ic` / `group_consistency`）—— 触发条件见 `outputs/功能增补清单`
- 退市收益约定的行业维度复核

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
