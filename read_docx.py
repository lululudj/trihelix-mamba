"""提取 docx 文本 (docx 本质是 zip, word/document.xml 里是正文)"""
import zipfile
import re
import sys

def read_docx(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read('word/document.xml').decode('utf-8', errors='replace')
    # 提取 <w:t>...</w:t> 文本, 段落用 <w:p> 分隔
    # 先按段落切
    paragraphs = re.split(r'</w:p>', xml)
    out = []
    for p in paragraphs:
        texts = re.findall(r'<w:t[^>]*>([^<]*)</w:t>', p)
        line = ''.join(texts)
        if line.strip():
            out.append(line)
    return '\n'.join(out)

for f in [r"C:\Users\Administrator\Downloads\三链DNA.docx",
          r"C:\Users\Administrator\Downloads\ManbaDNA 三螺旋架构 —— DeepSeek 技术自荐求职信.docx"]:
    print("=" * 78)
    print(f"文件: {f}")
    print("=" * 78)
    try:
        print(read_docx(f))
    except Exception as e:
        print(f"[读取失败] {type(e).__name__}: {e}")
    print()
