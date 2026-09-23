# -*- coding: utf-8 -*-
"""生成公众号配图（图片模型未配置 → matplotlib 图表式降级）。

配色对齐排版预设「技术 / 蓝」：主色 #2563EB，辅 #E5E7EB，强调 #DC2626。
所有数字来自底稿，脚本里逐个标注出处，避免"图上的数对不上文"。
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.lines import Line2D

OUT = '/home/yinxiuqu/alphalens-cna/.wechat-draft'
BLUE, GRAY, RED, DARK = '#2563EB', '#9CA3AF', '#DC2626', '#0B1F3A'
NAVY2 = '#12305A'
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def cover():
    """封面 2.35:1 —— K 线在右侧断裂成空白 + 放大镜停在断裂处。"""
    fig, ax = plt.subplots(figsize=(17.92, 7.63), dpi=100)
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    # 渐变底
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[0, 100, 0, 100], aspect='auto', cmap='Blues_r',
              alpha=0.30, zorder=0)
    # 标题
    ax.text(4.5, 74, 'IC看不见崩盘', fontsize=58, color='white',
            fontweight='bold', zorder=5)
    ax.text(4.5, 56, 'ROE 因子全身检查', fontsize=27, color='#93C5FD', zorder=5)
    ax.text(4.5, 40, '一个 A 股因子的三次翻案：从 −0.059 到 +0.029',
            fontsize=17, color='#CBD5E1', zorder=5)
    # 走势线：右侧断裂
    x = np.linspace(46, 82, 160)
    y = 16 + 12 * np.exp(-(x - 46) / 26) + 3.2 * np.sin((x - 46) / 3.1) \
        + 0.9 * np.sin((x - 46) / 1.15)
    ax.plot(x, y, color='#60A5FA', lw=3.2, zorder=3)
    ax.plot([82, 90], [y[-1], y[-1] - 5.5], color='#60A5FA', lw=3.2,
            ls=(0, (4, 3)), zorder=3)                    # 虚线：之后的空白
    ax.plot([90, 96], [y[-1] - 5.5, y[-1] - 5.5], color='#334155', lw=2, zorder=3)
    ax.axvline(82, color=RED, lw=1.6, ls=':', alpha=.9, zorder=2)
    # 放大镜
    ax.add_patch(Circle((82, y[-1]), 7.4, fill=False, ec='white', lw=2.6, zorder=6))
    ax.add_patch(Circle((82, y[-1]), 7.4, fc='white', alpha=.06, zorder=5))
    ax.plot([87.2, 92.5], [y[-1] - 5.6, y[-1] - 9.6], color='white', lw=3.0, zorder=6)
    ax.text(82, y[-1] - 0.7, '?', fontsize=30, color='white', ha='center',
            va='center', fontweight='bold', zorder=7)
    ax.text(4.5, 6, '数据：A 股全市场 · ROE 月度调仓 · 2019–2026 · 10 项数据体检',
            fontsize=12.5, color='#64748B', zorder=5)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(f'{OUT}/cover.jpg', facecolor=DARK, bbox_inches='tight', pad_inches=0)
    plt.close()


def fig1():
    """信息图①：有行情 5,905 → 标记退市 339 → 进分析 0（底稿：体检层报告）。"""
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=110)
    ax.axis('off'); ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.text(2, 94, '退市股为什么没进分析', fontsize=22, fontweight='bold', color=DARK)
    ax.text(2, 86, '三段数字，断在最后一环', fontsize=13, color=GRAY)
    rows = [('数据库里有行情的股票', '5,905 只', BLUE, 66, 74, False),
            ('其中已标记退市', '339 只', BLUE, 42, 50, False),
            ('进入分析缓存的退市股', '0 只', RED, 18, 26, True)]
    for label, val, color, y, w, empty in rows:
        if empty:
            ax.add_patch(FancyBboxPatch((3, y), w, 15,
                         boxstyle='round,pad=0.4,rounding_size=1.4',
                         fc='none', ec=color, lw=2.0, ls='--'))
        else:
            ax.add_patch(FancyBboxPatch((3, y), w, 15,
                         boxstyle='round,pad=0.4,rounding_size=1.4',
                         fc=color, ec='none', alpha=.13))
            ax.add_patch(FancyBboxPatch((3, y), 1.4, 15,
                         boxstyle='round,pad=0,rounding_size=0.6',
                         fc=color, ec='none'))
        ax.text(5.5, y + 9.6, label, fontsize=15.5, color='#111827', va='center')
        ax.text(5.5, y + 4.0, val, fontsize=21, color=color, va='center',
                fontweight='bold')
    ax.annotate('', xy=(88, 14), xytext=(88, 78),
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=2.2))
    ax.text(90.5, 46, '断在这里', fontsize=15, color=RED, rotation=90,
            va='center', fontweight='bold')
    ax.text(2, 5, '建缓存的脚本只从「在市公司」表取代码 —— 数据在库里，却从没进过分析。',
            fontsize=13, color='#374151')
    fig.tight_layout()
    fig.savefig(f'{OUT}/imgs/01-退市股断层.png', facecolor='white')
    plt.close()


def fig2():
    """信息图②：腰斩率 U 型（底稿：尾部统计与 IC 盲区，12 个月口径）。"""
    labels = ['Q1\n最低', 'Q2', 'Q3', 'Q4', 'Q5\n最高']
    vals = [4.4, 2.0, 1.7, 1.7, 3.2]
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=110)
    colors = [RED, GRAY, GRAY, GRAY, '#F59E0B']
    bars = ax.bar(labels, vals, color=colors, width=.62)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + .12, f'{v}%', ha='center',
                fontsize=17, fontweight='bold',
                color=RED if v == 4.4 else ('#B45309' if v == 3.2 else '#4B5563'))
    ax.set_ylim(0, 5.6)
    ax.set_ylabel('腰斩率（12 个月跌掉一半以上）', fontsize=14)
    ax.set_xlabel('ROE 分位', fontsize=14)
    ax.set_title('最低分位的腰斩率是中间的 2.6 倍，而最高分位也不低',
                 fontsize=18, fontweight='bold', color=DARK, pad=14)
    ax.tick_params(labelsize=13)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#E5E7EB', lw=.9)
    ax.set_axisbelow(True)
    ax.annotate('两端都不安全，中间最稳', xy=(2, 2.0), xytext=(2.1, 4.7),
                fontsize=14.5, color=RED, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.8))
    ax.text(.02, -.24, '差异 t = −3.10（Newey-West）· 门槛 −50% · 2019–2026 · 91 次月度调仓',
            transform=ax.transAxes, fontsize=11.5, color=GRAY)
    fig.tight_layout()
    fig.savefig(f'{OUT}/imgs/02-腰斩率U型.png', facecolor='white')
    plt.close()


def fig3():
    """信息图③：RankIC 修正轨迹（底稿：退市收益约定 §二 + 中性化复核 §二）。"""
    xs = np.arange(5)
    ys = [-0.0589, -0.0479, -0.0473, -0.0400, 0.0292]
    names = ['原始', '+ 退市股', '+ 零值修正', '+ 退市约定', '+ 市值中性']
    fig, ax = plt.subplots(figsize=(11.6, 6.4), dpi=110)
    ax.axhline(0, color='#111827', lw=1.2)
    ax.plot(xs, ys, '-o', color=BLUE, lw=2.8, ms=9, zorder=4,
            markerfacecolor='white', markeredgewidth=2.6)
    ax.plot(xs[3:], ys[3:], '-o', color=RED, lw=2.8, ms=9, zorder=5,
            markerfacecolor='white', markeredgewidth=2.6)
    # 标签偏移逐个手调：原来 "+0.0292" 和 "-0.0400" 压在线上
    offs = [(-4, -26), (0, 16), (0, -28), (-6, 18), (4, 16)]
    has = ['right', 'center', 'center', 'right', 'left']
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.annotate(f'{y:+.4f}', (x, y), textcoords='offset points',
                    xytext=offs[i], ha=has[i], fontsize=14.5,
                    fontweight='bold', color=RED if i >= 3 else '#1E3A8A')
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=13.5)
    ax.set_ylabel('RankIC（12 个月持有期）', fontsize=14)
    ax.set_title('每补一道修正，结论就往反方向翻一次', fontsize=18,
                 fontweight='bold', color=DARK, pad=14)
    ax.tick_params(labelsize=12.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#E5E7EB', lw=.9); ax.set_axisbelow(True)
    ax.set_ylim(-.075, .048)
    ax.annotate('符号翻转', xy=(4, 0.0292), xytext=(2.85, 0.0425),
                fontsize=15, color=RED, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color=RED, lw=1.8))
    ax.text(.02, -.25, '负向效应主要是规模效应：市值中性化后转正 (+0.0292)',
            transform=ax.transAxes, fontsize=12, color=GRAY)
    fig.tight_layout()
    fig.savefig(f'{OUT}/imgs/03-RankIC轨迹.png', facecolor='white')
    plt.close()


if __name__ == '__main__':
    cover(); fig1(); fig2(); fig3()
    for f in ('cover.jpg', 'imgs/01-退市股断层.png', 'imgs/02-腰斩率U型.png',
              'imgs/03-RankIC轨迹.png'):
        p = os.path.join(OUT, f)
        print(f'  {f}  {os.path.getsize(p)/1024:.0f} KB')
