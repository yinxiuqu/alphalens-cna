# M1 进展 & Newey-West 小样本偏差修正

> 日期：2026-09-22
> 全库测试：**216 passed**（M0 的 154 → 现在 216）
> 相关：`outputs/换手口径与D9核实.md`（对拍层）、`outputs/体检层报告.md`（体检层）

---

## 一、M1 模块进度

| 模块 | 状态 | 产出 |
|---|---|---|
| `contract/` | ✅ M0 | 8 个契约对象 + 跨对象校验 |
| `adapters/input/` | ✅ M0 | dataframe / parquet + `examples/` 私有适配器 |
| `engine/` | ✅ M0 | 复权 / 可成交性 / 收益 / 带原因清洗 |
| `analysis/ic,quantile,portfolio` | ✅ M0 | IC / 分层 / 换手 / 多空 |
| `inference/` | ✅ M0 | NW / 多重检验 / Estimate / Verdict |
| **`compat/`** | ✅ **M1** | 与 alphalens 三量对拍，**0.000e+00 逐位相同** |
| **`health/`** | ✅ **M1** | 10 项数据体检，接入 `build_report` 第二节 |
| **`analysis/regression`** | ✅ **M1** | 截面回归 / Fama-MacBeth / Shanken |
| `analysis/event` | ⏳ | 事件研究 |
| `inference/robustness,rank_entropy` | ⏳ | PFS 扰动鲁棒性 / RRE 排名熵 |
| `ledger/` | ⏳ | 研究台账（n_trials 自动记账） |
| `preprocess/` | ⏳ M2 | 去极值 / 标准化 / 中性化 / 正交化 |

---

## 二、截面回归 / Fama-MacBeth（用户点名要的）

`alphalens_cna/analysis/regression.py`

```python
coef = acna.cross_sectional_regression(y, X)        # 逐期 OLS
res  = acna.fama_macbeth(y, X, horizon=21)          # 两步法
print(res)          # 均值 + 朴素t + NW t + 虚高倍数
res.summary         # 每个回归元的完整统计
res.coef            # 每期系数（可直接画时序）
```

**三条设计立场**：

1. **t 值不自己算** —— 直接调 `inference.newey_west.nw_tstat`，
   与库里其它地方同一把尺子。`summary` 同时给 `t_naive` 与 `t_nw`，
   并在输出里明写"**只用 `t_nw` 下结论**"。
2. **样本不够就不回归** —— 单期样本 < `min_obs` 时该期系数是
   **NaN，不是 0**。造一个 0 出来就是伪造数据。
3. **Shanken 修正默认关闭** —— `shanken_inflation()` 只给膨胀因子
   `1 + λ'Σ_f⁻¹λ`，用不用由使用者显式决定；文档写明它是一阶近似。

**验证（防线 5）**：

| 检查 | 结果 |
|---|---|
| 三点共线 `y = 2x` | 截距 0.000000、斜率 2.000000（误差 < 1e-12） |
| 逐期系数 vs `numpy.linalg.lstsq` | **最大差 0.0**（逐位相同） |
| 真 λ = (0.02, −0.01) 的合成数据 | 复原 0.0202 / −0.0104（误差 < 2e-3） |
| 样本不足的日期 | 整行 NaN ✅ |
| 索引不合法 / 长度不匹配 | 抛 `ContractError` ✅ |

---

## 三、发现并修掉：Newey-West 的小样本偏差 ★

对拍层收尾时顺手压测 NW 估计量，发现一个**方向错误**的偏差。

### 现象

iid（无自相关）数据上，`vif = S/γ₀` 本应 ≈ 1。实测 300 次模拟：

| 样本长度 | 自动滞后阶数 | vif 均值 | 偏离 |
|---|---|---|---|
| T = 30 | 3 | **0.898** | −10.2% |
| T = 91 | 3 | 0.968 | −3.2% |
| T = 252 | 4 | 0.995 | −0.5% |

T=30 时 **62% 的模拟 vif < 1** —— 即系统性**低估方差**。

### 原因（不是实现 bug，是估计量的性质）

因为减掉了样本均值，自协方差估计量的期望是

```
E[γ̂_j] = −σ²/T   (j ≥ 1)
```

于是

```
E[S] ≈ γ₀ · (1 − 2Σw_j/T)
```

T=30、L=3 时 `Σw = 1.5` → `E[S] ≈ 0.9γ₀` —— 与实测 0.898 完全吻合。

试过换成无偏分母 `÷(T−j)`：**没用**（实测 0.898，反而更差）。
因为偏差来自"估计均值"这一步，不是分母。

### 后果与修法

低估方差 = **高估显著性** —— 恰好是本库最不该犯的错。
而且本库的主战场（重叠观测）只用得上 vif > 1，短样本的负向偏差纯粹是害处。

```python
S = max(S, γ₀)      # 等价于 vif ≥ 1：NW 永不比朴素方差更小
```

**代价（明写在 docstring 里）**：真实的负自相关也会被当成 vif = 1 ——
此时 `t_nw = t_naive`，不会比朴素 t 更宽松。**这是刻意的保守。**

### 影响面

* 既有真实测量**全部不受影响**（那些 vif 都 > 1）：
  mom20 的 1.02 / 1.75 / 2.94 倍修正照旧。
* 复现验证（合成自相关序列）：

| 持有期 | T | vif | t_naive | t_nw | 虚高 |
|---|---|---|---|---|---|
| h=1 | 330 | 1.085 | −0.273 | −0.262 | 1.04× |
| h=5 | 120 | 1.823 | −0.789 | −0.585 | 1.35× |
| h=21 | 40 | 2.883 | 3.155 | 1.858 | 1.70× |

* 顺带修掉一处 1 ulp 问题：`nw_tstat` 里 `vif` 原本另用 `numpy.std`
  算分母，与 `nw_variance` 的 `np.dot(d,d)/(T−1)` 求和顺序不同，
  导致"无自相关时 vif 精确为 1"在末位失守。现在两者**同源**。

---

## 四、下一步

1. **补 250 只退市股的财务数据** → 重建 PIT → 重跑 M0
   （这是让 ROE 结论真正可信的**唯一**前置条件，见体检层报告 §二.发现 2）
2. `analysis/event.py` 事件研究
3. `inference/robustness.py`（PFS）+ `rank_entropy.py`（RRE）
4. `ledger/` 研究台账（n_trials 自动记账）
