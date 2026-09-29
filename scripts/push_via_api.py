#!/usr/bin/env python3
"""HTTPS/SSH 全不通时的兜底推送 —— 走 GitHub API（api.github.com）。

★ 什么时候用它
   2026-09-29 实测：github.com:443 的 git 传输被反复掐断（HTTP/2 直接挂住、
   TLS 中断、连接超时），而 **api.github.com:443 全程 200**。所以当
   `scripts/push_fallback.sh` 四条通道都失败、但 `curl https://api.github.com/` 正常时，
   用这个把提交推上去。

⚠️ 代价：SHA 会变
   API 生成的提交由 GitHub 用**当前时间**做 committer，所以 SHA 与你本地的
   commit 不同（内容/提交信息/parent 完全一致，只是 SHA 不同）。推完必须对齐：

       git fetch origin && git reset --hard origin/main

   因此它是**最后手段**，能换网络就换网络 —— 换网络不会动 SHA。

用法：
    scripts/push_via_api.py                 # 默认 dry-run，只说要做什么
    scripts/push_via_api.py --apply         # 真推
    scripts/push_via_api.py --branch dev --apply

退出码：0 成功（或 dry-run 正常）；非 0 失败。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

API = 'https://api.github.com'


def sh(*args: str) -> str:
    out = subprocess.run(args, capture_output=True).stdout
    return out.decode('utf-8', 'replace').strip()


def changed_files(base: str, tip: str):
    """``[(状态, 路径, 旧路径或 None)]`` —— 用 ``-z`` 取，别让 git 转义中文路径。

    ⚠️ 默认的 ``git diff --name-status`` 会把非 ASCII 路径输出成
    ``"docs/\\345\\217\\221..."``（``core.quotePath`` 默认为真）。那样不光显示难看，
    **拿去 open() 会直接 FileNotFoundError** —— 中文文件名全推不上去。
    """
    raw = subprocess.run(['git', 'diff', '--name-status', '-z', base, tip],
                         capture_output=True).stdout.decode('utf-8', 'replace')
    toks = [t for t in raw.split('\0') if t]
    res, i = [], 0
    while i < len(toks):
        st = toks[i]
        if st[0] in ('R', 'C'):            # 重命名/复制：状态 \0 旧 \0 新 \0
            res.append((st, toks[i + 2], toks[i + 1]))
            i += 3
        else:
            res.append((st, toks[i + 1], None))
            i += 2
    return res


def get_token() -> str:
    tok = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if tok:
        return tok.strip()
    cred = os.path.expanduser('~/.git-credentials')
    if os.path.exists(cred):
        line = open(cred, encoding='utf-8').readline()
        m = re.match(r'^https?://[^:]+:([^@]+)@', line)
        if m:
            return m.group(1)
    sys.exit('找不到 token：设 GITHUB_TOKEN，或确认 ~/.git-credentials 里有凭证')


def call(tok: str, method: str, path: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        API + path, data=data, method=method,
        headers={'Authorization': f'token {tok}',
                 'Accept': 'application/vnd.github+json',
                 'Content-Type': 'application/json',
                 'User-Agent': 'alphalens-cna-push-via-api'})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
        return json.loads(body) if body else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--branch', default='main')
    ap.add_argument('--apply', action='store_true',
                    help='真的推送；不加则只 dry-run')
    args = ap.parse_args()

    # ★ 必须先切到仓库根：`git diff` 给的是「仓库根相对路径」，而 open() 是 cwd 相对的。
    #   从子目录运行时两者对不上 —— 直接 FileNotFoundError，什么都推不上去。
    top = sh('git', 'rev-parse', '--show-toplevel')
    if not top:
        sys.exit('✗ 当前目录不在 git 仓库里')
    os.chdir(top)

    slug = sh('git', 'config', '--get', 'remote.origin.url')
    m = re.search(r'github\.com[:/]+([^/]+)/([^/]+?)(?:\.git)?$', slug)
    if not m:
        sys.exit(f'从 remote.origin.url 认不出 owner/repo：{slug!r}')
    owner, repo = m.groups()
    branch = args.branch

    tok = get_token()
    head = call(tok, 'GET', f'/repos/{owner}/{repo}/git/ref/heads/{branch}')['object']['sha']
    local = sh('git', 'rev-parse', 'HEAD')

    if head == local:
        print(f'✓ 已同步：远端 {branch} == 本地 HEAD == {local[:7]}')
        return 0

    # 只有在"远端是本地祖先"时才安全：否则是分叉，别用 API 硬推
    anc = subprocess.run(['git', 'merge-base', '--is-ancestor', head, local]).returncode == 0
    if not anc:
        sys.exit(f'✗ 远端 {head[:7]} 不是本地 {local[:7]} 的祖先 —— 分叉了，'
                 f'先 git fetch 看清楚，不要用这个脚本硬推')

    changed = changed_files(head, local)
    commits = sh('git', 'rev-list', '--count', f'{head}..{local}')
    print(f'仓库      : {owner}/{repo}  分支 {branch}')
    print(f'远端 HEAD : {head[:7]}')
    print(f'本地 HEAD : {local[:7]}  （{commits} 个提交待推）')
    print(f'涉及文件  : {len(changed)} 个')
    for st, path, old in changed:
        print(f'   {st:>4}  {path}' + (f'   ← {old}' if old else ''))

    if not args.apply:
        print('\n（dry-run）加 --apply 才会真推。')
        print('⚠️ 推完 SHA 会变，记得对齐：git fetch origin && git reset --hard origin/main')
        return 0

    base_tree = call(tok, 'GET', f'/repos/{owner}/{repo}/git/commits/{head}')['tree']['sha']
    entries = []

    def add_blob(path: str):
        blob = call(tok, 'POST', f'/repos/{owner}/{repo}/git/blobs',
                    {'content': base64.b64encode(open(path, 'rb').read()).decode(),
                     'encoding': 'base64'})
        mode = '100755' if os.access(path, os.X_OK) else '100644'
        entries.append({'path': path, 'mode': mode,
                        'type': 'blob', 'sha': blob['sha']})

    for st, path, old in changed:
        if st == 'D':
            entries.append({'path': path, 'mode': '100644',
                            'type': 'blob', 'sha': None})
        elif st[0] in ('R', 'C'):
            # 重命名：旧路径置空 + 新路径写内容（API 的 tree 不认"移动"，只能一删一加）
            if old:
                entries.append({'path': old, 'mode': '100644',
                                'type': 'blob', 'sha': None})
            add_blob(path)
        else:
            add_blob(path)

    tree = call(tok, 'POST', f'/repos/{owner}/{repo}/git/trees',
                {'base_tree': base_tree, 'tree': entries})
    msg = sh('git', 'log', '-1', '--format=%B', local)
    commit = call(tok, 'POST', f'/repos/{owner}/{repo}/git/commits',
                  {'message': msg, 'tree': tree['sha'], 'parents': [head]})
    call(tok, 'PATCH', f'/repos/{owner}/{repo}/git/refs/heads/{branch}',
         {'sha': commit['sha'], 'force': False})

    print(f'\n✅ 已通过 API 推送到 {branch}：{commit["sha"][:7]}')
    print('⚠️ SHA 与本地不同（GitHub 重新生成）。对齐本地：')
    print(f'   git fetch origin && git reset --hard origin/{branch}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except urllib.error.HTTPError as e:
        sys.exit(f'GitHub API {e.code}: {e.read().decode()[:400]}')
