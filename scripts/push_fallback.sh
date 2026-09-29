#!/usr/bin/env bash
# 推送不通时用这个 —— 按"最可能成功"的顺序逐个通道试，成功即停。
#
# 为什么需要它：2026-09-29 推 0.4.2 的收尾文档时，github.com 的 **HTTPS(HTTP/2)**
# 通道被反复掐断，表现有四种（同一条命令不同次还不一样）：
#
#   · 卡住不返回（`git ls-remote` 直接挂死，要 `timeout` 兜底）
#   · `GnuTLS recv error (-110): TLS 链接非正常地终止`
#   · `Failed to connect to github.com port 443 after 134833 ms`
#   · 偶尔又能过一次
#
# 实测定位结果是**传输层**问题，不是凭证/仓库/带宽：
#
#   git 默认（HTTP/2）        → ls-remote rc=124（挂住）
#   http.version=HTTP/1.1     → ✓ 正常
#   http.sslBackend=openssl   → rc=128（该后端没编译进去）
#
# 而 GitHub 的另外几条通道一直是通的：github.com:22、ssh.github.com:443、
# api.github.com:443。所以这里把可用通道排成梯子。
#
# ★ 各通道的 URL 由 `origin` **现场推导**，不依赖本地 remote 配置
#   （`.git/config` 不进版本控制，换台机器克隆就没有 ssh/ssh443 了 ——
#    而这个脚本恰恰是在"本机环境不顺"时才用，不能自己先挂）。
#
# 用法：
#   scripts/push_fallback.sh              # 推 main
#   scripts/push_fallback.sh my-branch
#   PUSH_TIMEOUT=60 scripts/push_fallback.sh     # 每个通道的超时（默认 120s）
#
# 退出码：0 = 某个通道推成功；1 = 全部失败（此时看输出末尾的提示）。
#
# 注：SSH 两条通道需要先有密钥并加到 GitHub 上，见 docs/发版核对清单.md 附录 A。
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="${1:-main}"
PUSH_TIMEOUT="${PUSH_TIMEOUT:-120}"

# ── 从 origin 推导各通道 URL（不依赖本地 remote 配置）────────────────────────
ORIGIN_URL="$(git config --get remote.origin.url || true)"
if [[ "$ORIGIN_URL" =~ github\.com[:/]+([^/]+)/(.+)$ ]]; then
  SLUG="${BASH_REMATCH[1]}/${BASH_REMATCH[2]%.git}"
else
  printf '✗ 认不出 origin 的 owner/repo：%s\n' "${ORIGIN_URL:-（空）}" >&2
  exit 1
fi
URL_HTTPS="https://github.com/${SLUG}.git"
URL_SSH="git@github.com:${SLUG}.git"
URL_SSH443="ssh://git@ssh.github.com:443/${SLUG}.git"

SSH_KEY_PRESENT=0
compgen -G "$HOME/.ssh/id_*" >/dev/null 2>&1 && SSH_KEY_PRESENT=1

head_() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# 端口通不通 —— 先探一下，别让 git 白等一两分钟
reachable() {
  timeout 6 bash -c "echo > /dev/tcp/$1/$2" 2>/dev/null
}

# 用 API 查远端分支 SHA。★ 这是本脚本的关键一环：git 传输被掐时
# api.github.com 往往还通（0.4.2 那次全程如此），所以"到底同没同步"
# 不能只靠 git —— 否则这个脚本恰好在最该发挥作用的场景里帮不上忙。
remote_sha_via_api() {
  local tok="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
  if [ -z "$tok" ] && [ -f "$HOME/.git-credentials" ]; then
    tok="$(sed -nE 's#^https?://[^:]+:([^@]+)@.*$#\1#p' "$HOME/.git-credentials" | head -1)"
  fi
  [ -n "$tok" ] || return 1
  curl -s --retry 2 --retry-delay 2 -m 25 -H "Authorization: token $tok" \
       "https://api.github.com/repos/$SLUG/git/ref/heads/$BRANCH" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["object"]["sha"])' 2>/dev/null
}

# 试一个通道：try <说明> <host> <port> <url> [git push 之前的额外参数…]
try() {
  local desc="$1" host="$2" port="$3" url="$4"; shift 4

  if ! reachable "$host" "$port"; then
    printf '  ✗ %s —— %s:%s 不可达，跳过\n' "$desc" "$host" "$port"
    return 1
  fi

  printf '  → %s（%s:%s 可达，推送中，上限 %ss）\n' "$desc" "$host" "$port" "$PUSH_TIMEOUT"
  local out rc
  out="$(GIT_TERMINAL_PROMPT=0 timeout "$PUSH_TIMEOUT" \
         git "$@" push "$url" "$BRANCH" 2>&1)"
  rc=$?

  # 判定不看单一信号：实测遇到过一次 `rc=1` 却输出 `Everything up-to-date`
  # （瞬时抖动，事后无法复现）。既然"已是最新"本身就是目标状态，就别判成失败。
  if [ $rc -eq 0 ] || printf '%s' "$out" | grep -q 'Everything up-to-date'; then
    printf '%s\n' "$out" | sed 's/^/      /'
    printf '  ✓ %s 成功\n' "$desc"
    return 0
  fi

  # 报错时先滤掉那条无害的凭证锁提示，否则它会把真正的错误挤到最后一行之外
  local why
  why="$(printf '%s\n' "$out" | grep -v '获得凭证存储锁' | grep -v '^[[:space:]]*$' | head -1)"
  [ -n "$why" ] || why="$(printf '%s' "$out" | tail -1)"
  printf '  ✗ %s 失败（rc=%s）：%s\n' "$desc" "$rc" "$(printf '%s' "$why" | cut -c1-130)"
  return 1
}

ladder() {
  try "① HTTPS（HTTP/1.1，走仓库配置）" github.com 443 "$URL_HTTPS" && return 0
  try "② HTTPS（强制 HTTP/1.1，防配置被改回）" github.com 443 "$URL_HTTPS" \
      -c http.version=HTTP/1.1 && return 0
  if [ "$SSH_KEY_PRESENT" -eq 1 ]; then
    try "③ SSH（22 端口）" github.com 22 "$URL_SSH" && return 0
    try "④ SSH over 443（22 被挡时用）" ssh.github.com 443 "$URL_SSH443" && return 0
  else
    printf '  ⏭ ③④ SSH 跳过：~/.ssh 下没有密钥（附录 A 第 3 节有生成步骤）\n'
  fi
  return 1
}

head_ "推送 $BRANCH（按通道依次尝试）"
printf '  仓库: %s\n' "$SLUG"

# 先看有没有要推的：已经同步就别报"失败"（否则发版流程会被无谓地中止）。
# 信息源优先 git fetch；git 传输不通时回落到 API —— 两条都拿不到才放弃判断。
FETCH_OK=0
GIT_TERMINAL_PROMPT=0 timeout 45 git fetch -q origin "$BRANCH" 2>/dev/null && FETCH_OK=1
if [ "$FETCH_OK" -eq 1 ]; then
  REMOTE_HEAD="$(git rev-parse --verify -q "origin/$BRANCH" || true)"
  SRC="git fetch"
else
  REMOTE_HEAD="$(remote_sha_via_api || true)"
  SRC="GitHub API"
fi
if [ -n "$REMOTE_HEAD" ] && [ "$REMOTE_HEAD" = "$(git rev-parse HEAD)" ]; then
  printf '  ✓ 已是最新（经 %s 确认）：远端 %s == HEAD（%s），无需推送\n' \
         "$SRC" "$BRANCH" "$(git rev-parse --short HEAD)"
  exit 0
fi

if ladder; then SUCCESS=1; else SUCCESS=0; fi

head_ "结果"
printf '  本地 HEAD      = %s\n' "$(git rev-parse HEAD 2>/dev/null || echo '?')"
REMOTE_SHA="$(GIT_TERMINAL_PROMPT=0 timeout 45 git ls-remote origin -h "refs/heads/$BRANCH" 2>/dev/null | cut -f1)"
HOW=""
if [ -z "$REMOTE_SHA" ]; then
  REMOTE_SHA="$(remote_sha_via_api || true)"       # git 不通就问 API
  HOW=" （经 API 查得）"
fi
printf '  远端 %s = %s%s\n' "$BRANCH" "${REMOTE_SHA:-（取不到）}" "$HOW"
# ⚠️ 别用 `origin/$BRANCH..HEAD` 数：本脚本推的是**URL**，git 不会更新本地的
#    origin/main 跟踪引用 —— 刚推成功也会显示"未推送 1 个"。改用刚查到的远端 SHA。
if [ -n "$REMOTE_SHA" ]; then
  printf '  未推送提交数   = %s\n' "$(git rev-list --count "$REMOTE_SHA..HEAD" 2>/dev/null || echo '?')"
else
  printf '  未推送提交数   = （远端取不到，无法判断）\n'
fi

if [ "$SUCCESS" -eq 1 ]; then
  GIT_TERMINAL_PROMPT=0 timeout 45 git fetch -q origin "$BRANCH" 2>/dev/null || true
  printf '\n✅ 已推送成功\n'
  exit 0
fi

cat <<'EOF'

❌ 全部通道都没成功。接下来按这个顺序（详见 docs/发版核对清单.md 附录 A）：

  1. 换个网络（最干净：SHA 与线性历史都不变）—— 0.4.2 那次就是这么解决的
  2. scripts/push_via_api.py --apply   （api.github.com 通常最稳；SHA 会变，推完要 reset）
  3. git bundle 导出，走别的机器/介质中转，再在那边 push
EOF
exit 1
