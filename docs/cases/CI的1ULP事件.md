# CI 上的 1 ULP：一个"应该精确相等"的断言逼出的结构缺陷

> 日期：2026-09-22｜修复提交：`f1f813c`
> 症状：GitHub Actions 三个 Python 版本的测试 job 同时红，本地 302 全绿

---

## 一、症状

首次把仓库推上 GitHub，CI 五个 job：

| job | 结果 |
|---|---|
| 等价性回归（与 alphalens 逐位对拍） | ✅ 一次就过 |
| 零绘图依赖 | ✅ 一次就过 |
| 测试 (3.9 / 3.11 / 3.12) | ❌ **三个版本全挂，且挂的是同一条** |

日志（Python 3.11 job）：

```
tests/test_regression.py::test_lags_zero_is_exactly_naive
    def test_lags_zero_is_exactly_naive():
        """lags=0 → t_nw 与 t_naive 精确相等（回归测试，防分母再走样）。"""
        x = np.random.default_rng(1).normal(size=50)
        st = nw_tstat(x, lags=0)
        assert st['vif'] == 1.0
>       assert st['t_nw'] == st['t_naive']
E       assert -0.2866165090394768 == -0.28661650903947683

1 failed, 289 passed, 12 skipped
```

差 **1 ULP**（最后一位二进制位）。CI 用的 Python 是 3.11.16，**与本地完全相同**。

## 二、这不是精度问题，是"两个真相源"

```
t_naive = mean / (x.std(ddof=1) / sqrt(n))            # numpy 的两遍算法
S       = np.dot(d, d) / (n - 1)                      # 直接点积
t_nw    = mean / sqrt(S / n)                          # lags=0 时 S 就等于 γ₀
```

两者**数学上完全相等**（都是 Σ(x−x̄)²/(n−1)），但浮点**求和顺序不同**，
结果可能差最后一位。差多少取决于 BLAS 实现 —— 所以：

* 本机：`np.dot` 法 = 0.7922356343171691
* 本机：`np.std` 法 = 0.7922356343171694
* 本机是否相同：**False**（CI 的 BLAS 上不同）

**同一个数学量被算了两遍，就是缺陷本身。** 平台差异只是把它暴露出来。

## 三、修法：统一源头，而不是放宽断言

```python
g0 = nw_variance(x, 0, horizon)      # 唯一的方差来源
t_naive = mean / np.sqrt(g0 / n)
S       = nw_variance(x, lags, horizon)
t_nw    = mean / np.sqrt(S / n)
```

于是 `lags = 0` 时 `t_nw == t_naive` 由**构造**保证，不再靠运气。

### 为什么不直接把断言改成近似相等

那是最省事的做法，也能让 CI 立刻变绿 —— 但它会把一个真实的
"两处算同一个量"留进库里，而且**从此再也没人会发现**。
这条断言的用途恰恰是钉住"NW 在 lags=0 时应当退化为朴素 t"这个性质；
改成 `approx` 等于把这个性质从"保证"降级成"大致如此"。

> **测试太严不是问题，测试太严而你又把它改松才是问题。**

## 四、可复用的教训

1. **"应该精确相等"的断言是探针。** 它逼问的是"这两条路径是否真的同源"，
   而不只是"数值是否接近"。近似断言永远不会问出这个问题。
2. **平台差异先当线索，不当借口。** "CI 和本地环境不同"解释不了
   *为什么同一个量会有两条计算路径*。
3. 本库里同一类问题此前出现过一次：`nw_variance` 的 γ₀ 用 `T−1` 而
   `sd²` 用 `T`，导致"无自相关时 vif 应为 1"在末位失守 ——
   那次也是靠"精确相等"的断言抓到的（见 `M1进展与NW修正.md`）。
   **同一种病，同一个探针，抓到两次。**

## 五、复现

```bash
pytest tests/test_regression.py::test_lags_zero_is_exactly_naive -q
```
修复前在不同 BLAS 上会间歇性失败；修复后在**任何平台**都精确相等。
