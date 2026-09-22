#!/usr/bin/env python3
"""
主动推送到 IM —— 不依赖任何用户消息触发。

原理: dsh-im 插件在 Host 的 WebServer 上暴露了主动投递接口
      POST http://127.0.0.1:3080/api/dsh-im/delivery/messages
      只需 { botId, targetId, text }，不需要 sessionId / 聊天引用 / Webhook。

限制（官方文档口径）:
  * 只发**非空文字**，不支持图片/文件/卡片/富文本
  * 请求体上限 1 MiB  → 长文档可以整篇当文字推
  * 无鉴权、无 CORS，仅限本机/可信网络
  * 不保存投递历史、不自动重试；调用方需自己防重复
  * {sent:true} 只表示平台接口接受了，不承诺送达或已读

用法:
    python push.py "消息内容"                 # 文字也走微信优先/QQ兜底
    python push.py -f outputs/项目实施路线图.md   # 发文档 ★推荐: 自动选最优方式
    python push.py -f outputs/xxx.md -t          # 强制推文字全文
    python push.py "消息" --channel weixin        # 指定渠道
    python push.py --list                         # 列出已配目标

发文档的自动降级链（-f 模式，无需你操心）:
    ① 微信真文件     ilink CDN 上传(AES-128-ECB) → 最好看、保真、可存档
    ② 微信文字全文   ①失败时，同渠道退文字（分块，带【文件名】i/n）
    ③ QQ 文字全文    微信整条路不通时的兜底

两条通道的脾气:
    微信  支持真文件；但受 iLink 规则约束 —— 需该机器人**最近收到过用户消息**。
          若报 ret=-2 prepare failed，让目标用户发一条消息即可恢复；
          **不要反复重试续期**（官方文档明确不建议）。
    QQ    只支持文字（主动投递接口限制），但没有窗口约束，最稳。
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

HOST = os.environ.get('DSH_WEB_URL', 'http://127.0.0.1:3080')
ENDPOINT = HOST + '/api/dsh-im/delivery/messages'
INTEGRATIONS = os.path.expanduser('~/.dsh/integrations')

# 已保存的投递目标由设置页管理，存在各渠道的 workspaces.json:
#   {"deliveryTargets": {"<botId>": {"<targetId>": {"name":.., "kind":.., "route":..}}}}
CHANNEL_DIRS = {
    'weixin': 'dsh-weixin',
    'qq': 'dsh-qq',
    'feishu': 'dsh-feishu',
    'dingtalk': 'dsh-dingtalk',
    'wecom': 'dsh-wecom',
    'telegram': 'dsh-telegram',
}

MAX_TEXT = 1_000_000        # 保守: 官方上限 1 MiB


def list_targets():
    """扫描各渠道 workspaces.json，返回 [(channel, botId, targetId, name, kind)]。"""
    out = []
    for ch, d in CHANNEL_DIRS.items():
        p = os.path.join(INTEGRATIONS, d, 'workspaces.json')
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                ws = json.load(f)
        except Exception:
            continue
        for bot_id, targets in (ws.get('deliveryTargets') or {}).items():
            for tid, t in (targets or {}).items():
                out.append((ch, bot_id, tid, t.get('name', ''), t.get('kind', '')))
    return out


def pick(channel=None):
    """挑一个目标: 指定渠道优先, 否则 QQ 优先(不受 iLink 窗口约束)。"""
    ts = list_targets()
    if channel:
        ts = [t for t in ts if t[0] == channel]
    if not ts:
        raise SystemExit('没有已配置的投递目标。请到 设置 → IM机器人 → 齿轮 → 新建目标。')
    ts.sort(key=lambda t: 0 if t[0] == 'qq' else 1)
    return ts[0]


def send(text, channel=None, verbose=True):
    """主动发送。返回 True/False。"""
    if not text or not text.strip():
        raise SystemExit('消息不能为空（接口只接受非空文字）')
    if len(text.encode('utf-8')) > MAX_TEXT:
        raise SystemExit('文本超过 1 MiB 上限，请先分块')
    ch, bot_id, tid, name, kind = pick(channel)
    body = json.dumps({'botId': bot_id, 'targetId': tid, 'text': text}).encode('utf-8')
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            ok = json.loads(r.read().decode('utf-8')).get('sent') is True
            if verbose:
                print(f'[{ch}] {name or tid} → {"已投递" if ok else "未确认"}')
            return ok
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', 'replace')[:300]
        if verbose:
            print(f'[{ch}] 失败 HTTP {e.code}: {detail}', file=sys.stderr)
        return False
    except Exception as e:
        if verbose:
            print(f'[{ch}] 失败: {type(e).__name__} {e}', file=sys.stderr)
        return False


def send_text(text, channel=None):
    """主动推文字。"""
    return send(text, channel)


def send_text_smart(text, channel=None, verbose=True):
    """
    文字也走「微信优先 → QQ 兜底」，与文件路径保持一致 ——
    否则文件在微信、文字在 QQ，使用者要在两个 app 之间来回看。

    显式指定 channel 时不降级。
    """
    if channel:
        return send(text, channel, verbose=verbose)
    if any(t[0] == 'weixin' for t in list_targets()):
        if send(text, 'weixin', verbose=verbose):
            return True
        if verbose:
            print('[weixin] 文字失败 → 兜底到 QQ（窗口可能已过期，'
                  '让用户发一条消息可恢复）', file=sys.stderr)
    return send(text, 'qq', verbose=verbose)


def send_document(path, title=None, channel=None, max_chars=4000, delay=3.0, verbose=True):
    """
    把文档全文当**文字**主动推过去。

    主动投递接口不支持文件附件, 但支持 1 MiB 文字 —— 常规报告(几十 KB)完全放得下。
    超过 max_chars 时按段落分块, 依次发送。

    delay: 块间隔秒数。微信对连续发送敏感, 默认 3 秒; QQ 可设 0。
    """
    with open(path, encoding='utf-8') as f:
        doc = f.read()
    head = title or os.path.basename(path)
    chunks, buf = [], ''
    for para in doc.split('\n'):
        if len(buf) + len(para) + 1 > max_chars and buf:
            chunks.append(buf)
            buf = ''
        buf = (buf + '\n' + para) if buf else para
    if buf:
        chunks.append(buf)

    ok_all = True
    for i, c in enumerate(chunks, 1):
        tag = f'【{head}】{i}/{len(chunks)}\n\n' if len(chunks) > 1 else f'【{head}】\n\n'
        ok_all &= send(tag + c, channel=channel, verbose=verbose)
        if i < len(chunks) and delay:
            time.sleep(delay)
    return ok_all


def send_real_file(path, channel=None, verbose=True):
    """
    发送文档，按优先级**自动降级**：

        ① 微信真文件      —— 最好看、保真、可存档
        ② 微信文字全文    —— 文件发不了时，同一渠道退文字
        ③ QQ 文字全文     —— 微信整条路都不通时的兜底

    显式指定 channel='qq' 时跳过 ①②，直接走文字。

    微信真文件走 ilink CDN：AES-128-ECB 加密 → CDN → file_item，
    支持任意类型、无小文件限制；但需该机器人**最近收到过用户消息**。

    返回 True/False。
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise SystemExit(f'文件不存在: {path}')
    name = os.path.basename(path)

    targets = list_targets()
    has_wx = any(t[0] == 'weixin' for t in targets)
    has_qq = any(t[0] == 'qq' for t in targets)

    # 显式指定非微信渠道 → 直接文字
    if channel and channel != 'weixin':
        return send_document(path, channel=channel)

    if has_wx:
        # ① 微信真文件
        try:
            import wechat_send as wx
            wx.RETRY_MAX = 2                       # 短退避, 别把降级拖太久
            mid = wx.send_file(path)
            if mid is not None:
                if verbose:
                    print(f'[weixin] 文件 {name} → 已投递 ({mid})')
                return True
            if verbose:
                print('[weixin] 真文件未确认, 降级为文字…', file=sys.stderr)
        except Exception as e:
            if verbose:
                print(f'[weixin] 真文件失败: {type(e).__name__} {e}'
                      f'\n         → 降级为文字全文（窗口可能已过期，让用户发一条消息可恢复）',
                      file=sys.stderr)
        # ② 微信文字（同渠道优先）
        if send_document(path, channel='weixin', verbose=verbose):
            return True
        if verbose:
            print('[weixin] 文字也失败 → 兜底到 QQ', file=sys.stderr)

    # ③ QQ 文字
    if has_qq:
        return send_document(path, channel='qq', verbose=verbose)
    if verbose:
        print('没有可用渠道', file=sys.stderr)
    return False


def main():
    a = sys.argv[1:]
    if not a or a[0] in ('-h', '--help'):
        print(__doc__)
        return
    if a[0] == '--list':
        ts = list_targets()
        if not ts:
            print('（无已配置目标）')
        for ch, bot, tid, name, kind in ts:
            print(f'  {ch:<10} {tid:<26} {name:<12} {kind}')
        return

    # 渠道选择: --channel weixin / -c qq
    channel = None
    for i, x in enumerate(a):
        if x in ('--channel', '-c') and i + 1 < len(a):
            channel = a[i + 1]
            del a[i:i + 2]
            break

    # 文件模式: -f 真文件 (微信) / -t 强制文字全文
    text_mode = '-t' in a or '--text' in a
    a = [x for x in a if x not in ('-t', '--text')]

    if a and a[0] == '-f':
        if len(a) < 2:
            raise SystemExit('用法: push.py -f <文件路径> [--channel weixin|qq] [-t]')
        fn = send_document if text_mode else send_real_file
        print('已投递' if fn(a[1], channel=channel) else '投递失败')
        return
    if not a:
        raise SystemExit('用法: push.py "消息" [--channel weixin|qq]')
    print('已投递' if send_text_smart(a[0], channel=channel) else '投递失败')


if __name__ == '__main__':
    main()
