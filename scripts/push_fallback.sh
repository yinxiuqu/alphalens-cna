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
SSH_KEY_PRESENT=0
compgen -G "$HOME/.ssh/id_*" >/dev/null 2>&1 && SSH_KEY_PRESENT=1

REMOTE=origin

head_() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# 端口通不通 —— 先探一下，别让 git 白等一两分钟
reachable() {
  timeout 6 bash -c "echo > /dev/tcp/$1/$2" 2>/dev/null
}

# 试一个通道：try <说明> <host> <port> [git push 之前的额外参数…]
try() {
  local desc="$1" host="$2" port="$3"; shift 3

  if ! reachable "$host" "$port"; then
    printf '  ✗ %s —— %s:%s 不可达，跳过\n' "$desc" "$host" "$port"
    return 1
  fi

  printf '  → %s（%s:%s 可达，推送中，上限 %ss）\n' "$desc" "$host" "$port" "$PUSH_TIMEOUT"
  local out rc
  out="$(GIT_TERMINAL_PROMPT=0 timeout "$PUSH_TIMEOUT" \
         git "$@" push "$REMOTE" "$BRANCH" 2>&1)"
  rc=$?
  if [ $rc -eq 0 ]; then
    printf '%s\n' "$out" | sed 's/^/      /'
    printf '  ✓ %s 成功\n' "$desc"
    return 0
  fi
  printf '  ✗ %s 失败（rc=%s）：%s\n' "$desc" "$rc" \
         "$(printf '%s' "$out" | tail -1 | cut -c1-110)"
  return 1
}

ladder() {
  REMOTE=origin
  try "① HTTPS（HTTP/1.1，走仓库配置）" github.com 443 && return 0
  try "② HTTPS（强制 HTTP/1.1，防配置被改回）" github.com 443 \
      -c http.version=HTTP/1.1 && return 0
  if [ "$SSH_KEY_PRESENT" -eq 1 ]; then
    REMOTE=ssh
    try "③ SSH（22 端口）" github.com 22 && return 0
    REMOTE=ssh443
    try "④ SSH over 443（22 被挡时用）" ssh.github.com 443 && return 0
  else
    printf '  ⏭ ③④ SSH 跳过：~/.ssh 下没有密钥（附录 A 第 3 节有生成步骤）\n'
  fi
  return 1
}

head_ "推送 $BRANCH（按通道依次尝试）"
if ladder; then SUCCESS=1; else SUCCESS=0; fi

head_ "结果"
LOCAL="$(git rev-parse HEAD 2>/dev/null || echo '?')"
printf '  本地 HEAD      = %s\n' "$LOCAL"
REMOTE_SHA="$(git ls-remote "$REMOTE" -h "refs/heads/$BRANCH" 2>/dev/null | cut -f1)"
printf '  远端 %s = %s\n' "$BRANCH" "${REMOTE_SHA:-（取不到）}"
printf '  未推送提交数   = %s\n' "$(git rev-list --count "origin/$BRANCH..HEAD" 2>/dev/null || echo '?')"

if [ "$SUCCESS" -eq 1 ]; then
  printf '\n✅ 已推送成功\n'
  exit 0
fi

cat <<'EOF'

❌ 全部通道都没成功。接下来按这个顺序（详见 docs/发版核对清单.md 附录 A）：

  1. 换个网络（最干净：SHA 与线性历史都不变）—— 0.4.2 那次就是这么解决的
  2. 用 API 兜底推（api.github.com 通常最稳；会产生新 SHA，推完要 reset 对齐）
  3. git bundle 导出，走别的机器/介质中转，再在那边 push
EOF
exit 1
