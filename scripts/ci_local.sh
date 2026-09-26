#!/usr/bin/env bash
# 本地自检 —— 把 CI 的三件事在本地原样跑一遍，**推之前先跑这个**。
#
# 为什么需要它：CI 的依赖集和日常开发环境**不一样**，这个项目吃过两次亏——
#
#   · `pyproject.toml` 的 dev extras 里**没有 pyarrow**，而 CI 只装 `.[dev]`。
#     于是 `Report.save(kind='frames')` 按约定降级写 CSV（并把原因记进
#     `save_report['downgraded']`），而本地因为装了 pyarrow 会写 parquet ——
#     任何"硬依赖 parquet"的测试都会**本地绿、CI 红**。
#   · CI 装的是**最新的** pandas/numpy（约束只写了 `pandas>=1.5`），
#     实测 CI 上是 pandas 3.0.6 + numpy 2.4.6，而本地可能还是 2.x ——
#     弃用与行为变更只在 CI 暴露。
#
# 本脚本用独立的 venv（默认 `.ci-venv/`，已 gitignore）复刻 CI 的依赖集，
# 依次跑 CI 的三个 job：测试（3 个 Python 版本里挑当前解释器）、
# 等价性回归、零绘图依赖。
#
# 用法：
#   scripts/ci_local.sh              # 全套（≈ CI 三个 job）
#   scripts/ci_local.sh --quick      # 跳过对拍 job（不装 alphalens-reloaded）
#   PYTHON=python3.12 scripts/ci_local.sh
#   CI_VENV_DIR=/tmp/x scripts/ci_local.sh     # 换个放 venv 的地方
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV_ROOT="${CI_VENV_DIR:-$ROOT/.ci-venv}"
PARITY=1
for arg in "$@"; do
    case "$arg" in
        --quick) PARITY=0 ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "未知参数：$arg（支持 --quick）" >&2; exit 2 ;;
    esac
done

say() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
warn() { printf '\033[33m%s\033[0m\n' "$*" >&2; }

# 解释器版本：项目要求 >= 3.9。系统里的 `python3` 可能是 3.8（实测踩过），
# 那时 pip 会抛一句很难懂的 "requires a different Python" —— 这里提前拦下。
if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
    echo "★ 解释器不满足要求（需要 >= 3.9）：" >&2
    "$PYTHON" -c 'import sys; print("   ", sys.executable, sys.version.split()[0])' >&2 2>/dev/null || echo "    $PYTHON" >&2
    echo "   换一个：PYTHON=python3.11 scripts/ci_local.sh" >&2
    exit 3
fi

# ── 过期构建残留提醒 ──────────────────────────────────────────────────────
# `python -m build` / `pip wheel` 会在仓库根留下 `alphalens_cna.egg-info/` 与
# `build/`。它们**不影响安装**，但躺在 cwd 上会被 `importlib.metadata` 当成本包的
# 元数据 —— 版本号一改，从仓库根 import 就会报**旧版本**
# （实测：pyproject 已 0.2.0，`python -c "import alphalens_cna; print(__version__)"`
#  仍报 0.1.6，因为读的是那份残留）。
# 所以这里只**提示**、不自动删（删目录这种事该由人拍板）。
# >>> stale-artifact-hint
STALE_HINT=""
for d in alphalens_cna.egg-info build; do
    [ -e "$d" ] && STALE_HINT="${STALE_HINT:+$STALE_HINT }$d"
done
if [ -n "$STALE_HINT" ]; then
    warn "⚠️  发现过期构建残留：$STALE_HINT"
    _py=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)
    _egg=$(sed -n 's/^Version: //p' alphalens_cna.egg-info/PKG-INFO 2>/dev/null | head -1)
    if [ -n "$_egg" ] && [ -n "$_py" ] && [ "$_egg" != "$_py" ]; then
        warn "    它自报 $_egg，而 pyproject 是 $_py —— 从仓库根 import 时 __version__ 会报旧值"
    fi
    warn "    清理（不影响任何安装）：rm -rf $STALE_HINT"
    warn "    （提示而已，不会自动删 —— 但不清掉的话，下面测试里读到的版本号是旧的）"
fi
# <<< stale-artifact-hint

# ── venv 准备 ────────────────────────────────────────────────────────────
# extra 参数 → venv 目录名。测试 job 用 [dev]，对拍 job 用 [dev,compat]。
#
# ⚠️ 判据是**能不能真的 import**，不是"目录在不在"：上一次失败（比如解释器版本
#    不对、装到一半断网）会留下半成品 venv，只看目录会把它当好的用，然后报出
#    一句莫名其妙的 ModuleNotFoundError。
make_venv() {                       # $1=目录名  $2=extras  $3=说明  $4=导入检查
    local dir="$VENV_ROOT/$1" extras="$2" what="$3" check="$4"
    if [ -x "$dir/bin/python" ] && \
       ! "$dir/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        echo "  （$dir 里是 3.9 以下的解释器，重建）"
        rm -rf "$dir"
    fi
    if [ ! -x "$dir/bin/python" ]; then
        "$PYTHON" -m venv "$dir"
    fi
    if ! "$dir/bin/python" -c "$check" >/dev/null 2>&1; then
        say "准备 venv（$what）：pip install -e \".[$extras]\""
        "$dir/bin/python" -m pip install -q --upgrade pip
        # shellcheck disable=SC2086
        "$dir/bin/python" -m pip install -q -e ".[$extras]"
    fi
    if ! "$dir/bin/python" -c "$check" >/dev/null 2>&1; then
        echo "★ venv 准备失败（$what）：$dir —— 手工跑一次看报错：" >&2
        echo "    $dir/bin/python -m pip install -e \".[$extras]\"" >&2
        exit 4
    fi
    echo "  $what → $dir"
}

show_env() {                        # 把"会不会踩坑"的环境事实打出来
    "$1/bin/python" - <<'PY'
import importlib.util as u
import sys
import numpy, pandas, scipy
print(f"  python {sys.version.split()[0]} | pandas {pandas.__version__} "
      f"| numpy {numpy.__version__} | scipy {scipy.__version__}")
print("  pyarrow: " + ("已装" if u.find_spec("pyarrow") else "未装（与 CI 一致）"))
PY
}

# ── job 1：测试（CI 的 test matrix）──────────────────────────────────────
make_venv dev dev "测试 job" "import sys; assert sys.version_info >= (3, 9); import alphalens_cna, pytest, pandas"
say "job 1/3  测试（pytest -q）"
show_env "$VENV_ROOT/dev"
"$VENV_ROOT/dev/bin/python" -m pytest -q

# ── job 2：零绘图依赖 ───────────────────────────────────────────────────
say "job 2/3  零绘图依赖（import 后不许出现绘图库）"
"$VENV_ROOT/dev/bin/python" - <<'PY'
import sys
import alphalens_cna  # noqa: F401
bad = [m for m in ('matplotlib', 'seaborn', 'plotly') if m in sys.modules]
assert not bad, f'核心包不该拉起绘图库，实际拉起了：{bad}'
print('  ✓ 零绘图依赖')
PY

# ── job 3：等价性回归 ───────────────────────────────────────────────────
if [ "$PARITY" = "1" ]; then
    make_venv compat "dev,compat" "对拍 job" "import sys; assert sys.version_info >= (3, 9); import alphalens_cna, pytest, alphalens"
    say "job 3/3  等价性回归（与 alphalens 逐位对拍）"
    show_env "$VENV_ROOT/compat"
    "$VENV_ROOT/compat/bin/python" -m pytest tests/test_compat.py -q -v
else
    say "job 3/3  已跳过（--quick）"
fi

say "全部通过 ✅  （CI 等价：$(basename "$VENV_ROOT")）"
if [ -n "$STALE_HINT" ]; then
    warn "提醒：仓库根仍有构建残留 $STALE_HINT —— 建议 rm -rf $STALE_HINT（否则 __version__ 会报旧值）"
fi
