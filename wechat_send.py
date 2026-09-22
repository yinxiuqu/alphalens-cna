# -*- coding: utf-8 -*-
"""微信直发模块（ilink bot 协议）—— 文本 / 图片 / 文件

为什么需要它
------------
DSH 的 `dsh_im_return_file` 把产物绑定在 **(sessionId, turn)** 上，只有
**由 IM 发起的那一轮**才会有渠道适配器来取走并发出去。本会话的回合是从
Web GUI 发起的，所以文件只被"注册"、没有渠道来取 —— 微信收不到。

本模块绕开这套回传机制，直接调用微信机器人接口发送。
协议实现与 `@xmanrui/dsh-im/src/channels/weixin/weixin-api.mjs` 对齐：

    文本:  POST ilink/bot/sendmessage              item type=1 text_item
    图片:  getuploadurl(media_type=1) -> AES-128-ECB 加密 -> CDN 上传
           -> sendmessage                         item type=2 image_item
    文件:  getuploadurl(media_type=3) -> 同上
           -> sendmessage                         item type=4 file_item

凭据: ~/.dsh/integrations/dsh-weixin/config.json + ~/.dsh/.credentials.yaml

用法:
    python wechat_send.py text "消息"
    python wechat_send.py image /path/to.png
    python wechat_send.py file  /path/to.md
"""
import base64
import hashlib
import json
import os
import random
import re
import sys
import time
import uuid

import requests
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

CONFIG_PATH = '/home/yinxiuqu/.dsh/integrations/dsh-weixin/config.json'
CRED_PATH = '/home/yinxiuqu/.dsh/.credentials.yaml'
STATE_PATH_TMPL = '/home/yinxiuqu/.dsh/integrations/dsh-weixin/accounts/%s/state.json'
CDN_BASE = 'https://novac2c.cdn.weixin.qq.com/c2c/upload'
CHANNEL_VERSION = '2.4.6'
BOT_AGENT = 'DeepSeekHarness/1.1.0'
MAX_TEXT_CHARS = 1800          # 与插件 DEFAULT_WEIXIN_MAX_MESSAGE_CHARS 一致


# --------------------------------------------------------------------------- #
# 凭据
# --------------------------------------------------------------------------- #
def _cred_token(ref):
    raw = open(CRED_PATH, encoding='utf-8').read()
    for line in raw.split('\n'):
        m = re.match(r'^\s*([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$', line)
        if m and m.group(1) == ref:
            return m.group(2).strip().strip('"\'')
    return None


def _account():
    cfg = json.load(open(CONFIG_PATH, encoding='utf-8'))
    wx = cfg['accounts'][0]
    token = _cred_token(wx['tokenRef'])
    if not token:
        raise RuntimeError('credentials 中找不到 %s' % wx['tokenRef'])
    return wx['baseUrl'], token, wx['ownerUserId']


def _context_token(owner):
    """取用户最近一条入站消息的 context_token。

    微信机器人用的是"客服消息"模型: **只能在用户发消息后的一段时间内主动推送**,
    且发送时必须回带最近一条入站消息的 context_token。
    不带它会得到 {'ret': -2, 'errmsg': 'prepare failed'} —— 我实际踩过。

    token 由 dsh-im 插件维护在账号 state.json 的 contextTokens.users[userId].token。
    """
    try:
        path = STATE_PATH_TMPL % _bot_id()
        st = json.load(open(path, encoding='utf-8'))
        return (st.get('contextTokens') or {}).get('users', {}).get(owner, {}).get('token')
    except Exception:
        return None


def _bot_id():
    cfg = json.load(open(CONFIG_PATH, encoding='utf-8'))
    return cfg['accounts'][0]['botId']


def _headers(token):
    return {
        'iLink-App-Id': 'bot',
        'iLink-App-ClientVersion': str((2 << 16) | (4 << 8) | 6),
        'content-type': 'application/json',
        'AuthorizationType': 'ilink_bot_token',
        'Authorization': 'Bearer %s' % token,
        'X-WECHAT-UIN': base64.b64encode(
            str(random.randint(0, 2 ** 32 - 1)).encode()).decode(),
    }


def _base_info():
    return {'channel_version': CHANNEL_VERSION, 'bot_agent': BOT_AGENT}


# 微信对连续主动推送会限流: HTTP 200 但 body 是 {'ret': -2, 'errmsg': 'prepare failed'}。
# 实测连续发 10 条左右就会触发, 与请求格式无关(用其它脚本发同样被拒)。
RETRY_MAX = 5
RETRY_BASE_DELAY = 8          # 秒, 指数退避


def _sendmessage(base_url, token, body):
    """带限流重试的 sendmessage。"""
    delay = RETRY_BASE_DELAY
    last = None
    for attempt in range(1, RETRY_MAX + 1):
        r = requests.post(base_url + 'ilink/bot/sendmessage',
                          headers=_headers(token), json=body, timeout=30)
        r.raise_for_status()
        j = r.json()
        if j.get('ret') in (None, 0):
            return j.get('message_id')
        last = j
        if j.get('ret') == -2 and attempt < RETRY_MAX:
            print('  [限流] ret=-2 prepare failed, %.0fs 后重试 (%d/%d)'
                  % (delay, attempt, RETRY_MAX), flush=True)
            time.sleep(delay)
            delay *= 2
            continue
        raise RuntimeError('微信拒绝: %s' % j)
    raise RuntimeError('微信拒绝(重试耗尽): %s' % last)


# --------------------------------------------------------------------------- #
# 文本
# --------------------------------------------------------------------------- #
def split_text(text, max_chars=MAX_TEXT_CHARS):
    """按段落切分, 尽量不切断句子。"""
    if len(text) <= max_chars:
        return [text]
    chunks, buf = [], ''
    for para in text.split('\n'):
        if len(buf) + len(para) + 1 > max_chars and buf:
            chunks.append(buf)
            buf = ''
        while len(para) > max_chars:          # 单段超长, 硬切
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        buf = (buf + '\n' + para) if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def send_text(text):
    base_url, token, owner = _account()
    ct = _context_token(owner)
    body = {
        'msg': {
            'from_user_id': '', 'to_user_id': owner,
            'client_id': 'dsh-weixin-%s' % uuid.uuid4(),
            'message_type': 2, 'message_state': 2,
            'item_list': [{'type': 1, 'text_item': {'text': text}}],
            **({'context_token': ct} if ct else {}),
        },
        'base_info': _base_info(),
    }
    return _sendmessage(base_url, token, body)


def send_text_chunked(text, prefix='', delay=2.0):
    """分段发送长文本。段间留间隔, 降低触发限流的概率。"""
    ids = []
    parts = split_text(text)
    for i, p in enumerate(parts, 1):
        head = '%s(%d/%d)\n' % (prefix, i, len(parts)) if len(parts) > 1 else prefix
        ids.append(send_text(head + p))
        if i < len(parts):
            time.sleep(delay)
    return ids


# --------------------------------------------------------------------------- #
# 图片 / 文件
# --------------------------------------------------------------------------- #
def _aes_ecb_encrypt(data, key):
    padder = padding.PKCS7(128).padder()
    padded = padder.update(data) + padder.finalize()
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    return enc.update(padded) + enc.finalize()


def _send_artifact(path, media_type, build_item):
    base_url, token, owner = _account()
    data = open(path, 'rb').read()
    if not data:
        raise RuntimeError('文件为空: %s' % path)

    file_key = os.urandom(16).hex()
    aes_key = os.urandom(16)
    raw_md5 = hashlib.md5(data).hexdigest()
    padded_size = ((len(data) + 1 + 15) // 16) * 16    # = ceil((n+1)/16)*16

    # 1) 申请上传地址
    r = requests.post(base_url + 'ilink/bot/getuploadurl', headers=_headers(token), timeout=30,
                      json={
                          'filekey': file_key, 'media_type': media_type, 'to_user_id': owner,
                          'rawsize': len(data), 'rawfilemd5': raw_md5, 'filesize': padded_size,
                          'no_need_thumb': True, 'aeskey': aes_key.hex(),
                          'base_info': _base_info(),
                      })
    r.raise_for_status()
    up = r.json()
    if up.get('ret') not in (None, 0):
        raise RuntimeError('getuploadurl 被拒: %s' % up)
    url = up.get('upload_full_url') or (
        '%s?encrypted_query_param=%s&filekey=%s' % (CDN_BASE, up['upload_param'], file_key))

    # 2) 加密并上传（下载参数从响应头 x-encrypted-param 取）
    ciphertext = _aes_ecb_encrypt(data, aes_key)
    cu = requests.post(url, data=ciphertext, timeout=120,
                       headers={'content-type': 'application/octet-stream'})
    if cu.status_code != 200:
        raise RuntimeError('CDN 上传失败 HTTP %s: %s' % (cu.status_code, cu.text[:200]))
    download_param = cu.headers.get('x-encrypted-param')
    if not download_param:
        raise RuntimeError('CDN 响应缺少 x-encrypted-param')

    # 3) 发送
    media = {
        'encrypt_query_param': download_param,
        'aes_key': base64.b64encode(aes_key.hex().encode()).decode(),
        'encrypt_type': 1,
    }
    seed = os.urandom(8).hex()
    ct = _context_token(owner)
    body = {
        'msg': {
            'from_user_id': '', 'to_user_id': owner,
            'client_id': 'dsh-weixin-%s' % hashlib.sha256(seed.encode()).hexdigest()[:32],
            'message_type': 2, 'message_state': 2,
            'item_list': [build_item(os.path.basename(path), data, media, len(ciphertext))],
            **({'context_token': ct} if ct else {}),
        },
        'base_info': _base_info(),
    }
    return _sendmessage(base_url, token, body)


def send_image(path):
    return _send_artifact(path, 1, lambda name, data, media, size: {
        'type': 2, 'image_item': {'media': media, 'mid_size': size}})


def send_file(path):
    return _send_artifact(path, 3, lambda name, data, media, size: {
        'type': 4,
        'file_item': {'media': media, 'file_name': name, 'len': str(len(data))}})


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'text'
    arg = sys.argv[2] if len(sys.argv) > 2 else '测试'
    try:
        if kind == 'text':
            mid = send_text(arg)
        elif kind == 'image':
            mid = send_image(arg)
        elif kind == 'file':
            mid = send_file(arg)
        else:
            raise SystemExit('用法: wechat_send.py text|image|file <内容或路径>')
        print('发送成功 message_id=%s' % mid, flush=True)
    except Exception as e:
        print('发送失败: %s: %s' % (type(e).__name__, e), flush=True)
        sys.exit(1)
