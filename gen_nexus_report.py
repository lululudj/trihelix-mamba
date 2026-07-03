# -*- coding: utf-8 -*-
"""Generate Nexus 消融实验验证报告 PDF (reportlab Platypus)."""
import os
import glob
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from PIL import Image as PILImage

# ---------- font ----------
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
CN = 'STSong-Light'

# ---------- palette ----------
DARK_BG = colors.HexColor('#0a0a1a')
GOLD = colors.HexColor('#ffd700')
WHITE = colors.Color(1, 1, 1)
HEADER_BG = colors.HexColor('#1a3a6e')   # deep blue
ALT_ROW = colors.HexColor('#e8f0fc')     # light blue
TEXT = colors.HexColor('#1a1a1a')
ACCENT = colors.HexColor('#0a3a7a')
HILITE = colors.HexColor('#c0392b')      # highlight red
GRID = colors.HexColor('#9bb3d4')
GREY = colors.HexColor('#666666')

# ---------- page ----------
PAGE_W, PAGE_H = A4
MARGIN = 2 * cm
FRAME_W = PAGE_W - 2 * MARGIN
OUT_PATH = r'e:\沐曦基金申请文件\Nexus消融实验验证报告.pdf'
FIG_DIR = r'e:\沐曦基金申请文件\figures'


# ---------- paragraph styles ----------
def style(name, **kw):
    base = dict(fontName=CN, fontSize=10.5, leading=15, textColor=TEXT)
    base.update(kw)
    return ParagraphStyle(name, **base)


S_H1 = style('h1', fontSize=15, leading=20, textColor=ACCENT, spaceBefore=10, spaceAfter=8)
S_H2 = style('h2', fontSize=12, leading=17, textColor=HEADER_BG, spaceBefore=6, spaceAfter=4)
S_BODY = style('body', alignment=TA_JUSTIFY, spaceAfter=4)
S_BULLET = style('bullet', leftIndent=16, bulletIndent=2, spaceAfter=3, alignment=TA_LEFT)
S_CELL = style('cell', fontSize=8.8, leading=12, alignment=TA_LEFT)
S_CELL_C = style('cellc', fontSize=8.8, leading=12, alignment=TA_CENTER)
S_HDR = style('hdr', fontSize=9.2, leading=12, alignment=TA_CENTER, textColor=WHITE)
S_CAP = style('cap', fontSize=9, leading=12, alignment=TA_CENTER, textColor=colors.HexColor('#444444'))
S_META = style('meta', fontSize=9.5, leading=14, textColor=GREY, alignment=TA_LEFT)


def hi(text):
    """wrap text in red font tag for emphasis (since CID bold is unreliable)."""
    return f'<font color="#c0392b">{text}</font>'


# ---------- cover (canvas) ----------
def draw_cover(canvas, doc):
    c = canvas
    c.saveState()
    # dark background, full page
    c.setFillColor(DARK_BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # decorative gold rules
    c.setStrokeColor(GOLD)
    c.setLineWidth(1.2)
    c.line(MARGIN, PAGE_H * 0.70, PAGE_W - MARGIN, PAGE_H * 0.70)
    c.setLineWidth(0.5)
    c.line(MARGIN, PAGE_H * 0.71 + 4, PAGE_W - MARGIN, PAGE_H * 0.71 + 4)
    # title (gold, large, fake-bold via double draw to honour 加粗)
    c.setFillColor(GOLD)
    c.setFont(CN, 27)
    tx = 'Nexus 消融实验验证报告'
    cx = PAGE_W / 2
    ty = PAGE_H * 0.755
    c.drawCentredString(cx, ty, tx)
    c.drawCentredString(cx + 0.6, ty, tx)
    c.drawCentredString(cx - 0.6, ty, tx)
    # subtitle (white)
    c.setFillColor(WHITE)
    c.setFont(CN, 14)
    c.drawCentredString(PAGE_W / 2, PAGE_H * 0.67, 'ThreeChainMamba2 在 SDD 多智能体真实数据上的双场景验证')
    # meta block
    c.setFont(CN, 12)
    y = PAGE_H * 0.58
    for line in [
        '申报方向:沐曦青年开源专项基金 · MXMACA 软件栈生态适配',
        '日期:2026-07-03',
        '仓库:github.com/lulululudj/trihelix-mamba',
    ]:
        c.drawCentredString(PAGE_W / 2, y, line)
        y -= 22
    # bottom accent text
    c.setFillColor(GOLD)
    c.setFont(CN, 9.5)
    c.drawCentredString(PAGE_W / 2, 1.7 * cm, '— 数据可追溯 · 实验可复现 · 三链贡献可验证 —')
    c.restoreState()


# ---------- footer (canvas) ----------
def draw_footer(canvas, doc):
    c = canvas
    c.saveState()
    c.setStrokeColor(colors.HexColor('#cccccc'))
    c.setLineWidth(0.5)
    c.line(MARGIN, 1.3 * cm, PAGE_W - MARGIN, 1.3 * cm)
    c.setFillColor(GREY)
    c.setFont(CN, 8.5)
    c.drawString(MARGIN, 0.9 * cm, 'ThreeChainMamba2 · 沐曦基金申报附件')
    c.drawRightString(PAGE_W - MARGIN, 0.9 * cm, f'第 {doc.page} 页')
    c.restoreState()


# ---------- image sizing ----------
def sized_image(path, max_w, max_h):
    with PILImage.open(path) as im:
        iw, ih = im.size
    ratio = iw / ih
    w = max_w
    h = w / ratio
    if h > max_h:
        h = max_h
        w = h * ratio
    return Image(path, width=w, height=h)


# ---------- table builder ----------
def make_table(header, rows, col_widths, aligns=None):
    if aligns is None:
        aligns = ['C'] * len(header)
    cell_styles = [S_CELL_C if a == 'C' else S_CELL for a in aligns]
    data = [[Paragraph(str(h), S_HDR) for h in header]]
    for r in rows:
        data.append([Paragraph(str(c), cell_styles[i]) for i, c in enumerate(r)])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    ts = [
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, GRID),
        ('LINEBELOW', (0, 0), (-1, 0), 1.2, HEADER_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            ts.append(('BACKGROUND', (0, i), (-1, i), ALT_ROW))
        else:
            ts.append(('BACKGROUND', (0, i), (-1, i), colors.white))
    t.setStyle(TableStyle(ts))
    return t


def bullet(text):
    return Paragraph('• ' + text, S_BULLET)


def lab(label, rest=''):
    """coloured label for emphasis (CID font has no real bold)."""
    return f'<font color="#0a3a7a">{label}</font>{rest}'


# ---------- build story ----------
story = []

# Cover page: content drawn by canvas; tiny spacer + page break to advance
story.append(Spacer(1, 0.1))
story.append(PageBreak())

# ---------- figures (2 per page, half page each) ----------
fig_captions = {
    'fig_shuffle_sanity.png': '图 1:shuffle_labels sanity 对照 —— agent_id 在双场景下完全坍缩(=0.000)',
    'fig_pos_iou_decay_compare.png': '图 2:pos_iou 长程外推衰减对比(@100 → @150),揭示长程退化',
    'fig_agent_id_learning.png': '图 3:agent_id 学习能力 —— nexus SSM 显著高于随机基线',
    'fig_negative_shadow_compare.png': '图 4:负面分身消融三链贡献对比(双场景)',
}
fig_order = [
    'fig_shuffle_sanity.png',
    'fig_pos_iou_decay_compare.png',
    'fig_agent_id_learning.png',
    'fig_negative_shadow_compare.png',
]

fig_max_w = FRAME_W - 6
fig_max_h = (PAGE_H - 2 * MARGIN) / 2 - 40  # half page minus caption & spacing

fig_count = 0
for idx, fn in enumerate(fig_order):
    fp = os.path.join(FIG_DIR, fn)
    if not os.path.exists(fp):
        continue
    story.append(Spacer(1, 6))
    story.append(sized_image(fp, fig_max_w, fig_max_h))
    story.append(Spacer(1, 4))
    story.append(Paragraph(fig_captions.get(fn, fn), S_CAP))
    story.append(Spacer(1, 8))
    fig_count += 1
    if fig_count % 2 == 0:  # page break after every 2 figures
        story.append(PageBreak())

# ensure Section 1 starts on a fresh page (only if last figure didn't already break)
if fig_count > 0 and fig_count % 2 != 0:
    story.append(PageBreak())

# ---------- Section 1 ----------
story.append(Paragraph('1. 实验概述', S_H1))
story.append(Paragraph(
    '本报告验证 ThreeChainMamba2 三链状态空间模型(SSM)在 Stanford Drone Dataset(SDD)'
    '真实行人轨迹上的长程外推能力与三链贡献。实验严格控制变量,采用零模型改动配置,'
    '在两类难度差异显著的真实多智能体场景上完成双场景对照验证。', S_BODY))
story.append(Spacer(1, 3))
story.append(bullet(lab('目标', ':验证 ThreeChainMamba2 三链 SSM 在 SDD 真实行人轨迹上的长程外推能力与三链贡献。')))
story.append(bullet(lab('双场景', ':bookstore(7 视频,391 窗口,简单场景)+ nexus(12 视频,531 窗口,多 agent 交互复杂场景)。')))
story.append(bullet(lab('模型', ':30M 参数 ThreeChainMamba2(d_model=768, n_layers=2),零模型改动,严格控制变量。')))
story.append(bullet(lab('训练', ':SSM 10k 步 + shuffle_labels 3k 步 sanity 对照。')))

# ---------- Section 2 ----------
story.append(Paragraph('2. 核心指标定义', S_H1))
sec2_header = ['指标', '定义', '判别力']
sec2_rows = [
    ['changed_acc', '变化 cell 准确率', 'SDD 上失效(被"预测空" shortcut 污染)'],
    ['position_iou', '非零 cell 位置 IoU', '简单场景有效,复杂场景减弱'],
    ['agent_id_acc', 'agent 身份预测准确率', hi('双场景都有效') + '(shuffle 完全坍缩)'],
    ['enter_acc', '进入 cell 准确率', '辅助'],
    ['leave_acc', '离开 cell 准确率', '揭示"预测空" shortcut'],
]
story.append(make_table(sec2_header, sec2_rows, [95, 175, 212], aligns=['C', 'L', 'L']))
story.append(Spacer(1, 6))

# ---------- Section 3 ----------
story.append(Paragraph('3. 双场景 OOD 结果(核心数据)', S_H1))

story.append(Paragraph('3.1 bookstore(简单场景)', S_H2))
s3a_header = ['模型', 'pos_iou@100', 'pos_iou@150', 'decay%', 'agent_id@100']
s3a_rows = [
    ['SSM 10k', '0.166', '0.134', '-19.4%(单点)/ -10.9%(window)', '0.055(<随机,未学到)'],
    ['shuffle', '0.000', '0.000', '—', '0.000(完全坍缩)'],
]
story.append(make_table(s3a_header, s3a_rows, [80, 78, 78, 168, 98], aligns=['C', 'C', 'C', 'C', 'L']))

story.append(Paragraph('3.2 nexus(复杂场景,12 视频)', S_H2))
s3b_rows = [
    ['SSM 10k', '0.244', '0.159', '-34.7%(单点)/ -6.9%(window)', hi('0.153(>随机 0.067,学到了!)')],
    ['shuffle', '0.076', '0.167', '—', hi('0.000(完全坍缩)')],
]
story.append(make_table(s3a_header, s3b_rows, [80, 78, 78, 168, 98], aligns=['C', 'C', 'C', 'C', 'L']))
story.append(Spacer(1, 6))

# ---------- Section 4 ----------
story.append(Paragraph('4. 三大科学发现', S_H1))

story.append(Paragraph('发现 1:指标层次性(反直觉)', S_H2))
story.append(bullet('简单场景(bookstore):position_iou 是有效判别(shuffle=0.000 vs SSM=0.166)。'))
story.append(bullet('复杂场景(nexus):position_iou 判别力减弱(shuffle=0.167 vs SSM=0.188),但 ' + hi('agent_id_acc 双场景都完全坍缩(shuffle=0.000)') + '。'))
story.append(bullet('结论:agent_id_acc 是更稳健的判别指标,揭示"位置预测"与"身份预测"的解耦。'))

story.append(Paragraph('发现 2:多场景数据让 SSM 学到 agent 身份', S_H2))
story.append(bullet('nexus:agent_id=0.153 > 随机 0.067(' + hi('2.3 倍,学到了') + ')。'))
story.append(bullet('bookstore:agent_id=0.055 < 随机(未学到)。'))
story.append(bullet('结论:数据多样性是 SSM 学到丰富动力学的关键。'))

story.append(Paragraph('发现 3:三链 SSM 在真实数据上是必需的', S_H2))
story.append(bullet('负面分身 ablate_all:bookstore pos_iou 降 71%,nexus 降 45%。'))
story.append(bullet('对比 GridWorld:ablate_all decay=-0.66%(三链锦上添花)。'))
story.append(bullet('结论:真实数据复杂度高,' + hi('三链从"锦上添花"升级为"必需"') + '。'))

# ---------- Section 5 ----------
story.append(Paragraph('5. 三链贡献负面分身消融(pos_iou@100, window_avg)', S_H1))
s5_header = ['消融模式', 'bookstore', 'nexus', '解读']
s5_rows = [
    ['normal(三链全开)', '0.156', '0.188', 'nexus 起点更高'],
    ['ablate_s(空间链)', '0.249(+60%)', '0.399(+112%)', '反常:空间链置零反升'],
    ['ablate_t(时间链)', '0.072(-54%)', '0.089(-53%)', hi('时间链贡献最大')],
    ['ablate_c(因果链)', '0.190(+22%)', '0.085(-55%)', 'bookstore 噪声,nexus 重要'],
    ['ablate_all(三链全消融)', '0.046(-71%)', '0.103(-45%)', '三链必需,AnchorInit2 保留 29%/55%'],
]
story.append(make_table(s5_header, s5_rows, [125, 78, 70, 209], aligns=['L', 'C', 'C', 'L']))
story.append(Spacer(1, 6))

# ---------- Section 6 ----------
story.append(Paragraph('6. MXMACA 适配与落地', S_H1))
story.append(bullet('mamba_ssm 基于 Triton,沐曦 Triton-MXMACA 编译后端零修改适配。'))
story.append(bullet('端侧人形机器人多智能体实时推演场景。'))
story.append(bullet('SSM O(L) 线性内存适合端侧 GPU。'))

# ---------- Appendix ----------
story.append(Paragraph('附录:数据可追溯性', S_H1))
story.append(bullet('原始 metrics JSON:results_stage2/{sdd_30m_seed0_10k, nexus_30m_seed0_10k, sdd_shuffle_3k, nexus_shuffle_3k}/ood_metrics_advanced.json'))
story.append(bullet('负面分身:results_stage2/{sdd_30m_seed0_10k, nexus_30m_seed0_10k}/negative_shadow.json'))
story.append(bullet('评估脚本:scripts_sdd/eval_ood_advanced.py + scripts_sdd/eval_negative_shadow.py'))
story.append(bullet('图表:e:\\沐曦基金申请文件\\figures\\*.png'))
story.append(Spacer(1, 8))
story.append(Paragraph('— 报告结束 —', S_CAP))


# ---------- build ----------
def build():
    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title='Nexus 消融实验验证报告',
        author='ThreeChainMamba2',
    )
    doc.build(story, onFirstPage=draw_cover, onLaterPages=draw_footer)
    size = os.path.getsize(OUT_PATH)
    print(f'OK: {OUT_PATH}  size={size} bytes ({size/1024:.1f} KB)')
    assert size > 50 * 1024, f'PDF too small: {size} bytes'
    print('size > 50KB check PASSED')


if __name__ == '__main__':
    build()
